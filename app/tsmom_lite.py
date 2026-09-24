"""
Lightweight, dependency-free (pandas/numpy only) copy of the classical TSMOM
signal + performance functions from functions.py, used by app.py.

functions.py pulls in TensorFlow/keras-tuner at import time for the LSTM/MLP
research code, which is unnecessary weight for a live demo that only runs the
rule-based strategies (SGN/Long/MACD) and would slow down or risk running out
of memory on Streamlit Community Cloud's free tier. Keep this file in sync with
the corresponding functions in functions.py if that logic changes.
"""

import numpy as np
import pandas as pd


def process_data(file_path, sheet_name, column_names):
    """
    Lightweight copy of functions.process_data, restricted to selecting columns
    by ticker-prefix match (no top_n_columns/least-NaN mode, which the app doesn't need).
    """
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=4)
    data = data.iloc[2:, :]
    data.rename(columns={data.columns[0]: "Date"}, inplace=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    data.columns = data.columns.str.strip()
    if isinstance(column_names, str):
        column_names = [column_names]
    cols = [c for c in data.columns if any(c.upper().startswith(n.upper()) for n in column_names)]
    selected = data[cols].apply(pd.to_numeric, errors="coerce")
    returns = selected.pct_change().dropna()
    return selected, returns


def long(returns):
    cul_returns = (1 + returns).rolling(window=252).apply(np.prod, raw=True) - 1
    position_sign = np.sign(cul_returns).dropna()
    return position_sign.map(lambda x: x if x > 0 else 0)


def SGN(returns):
    cul_returns = (1 + returns).rolling(window=252).apply(np.prod, raw=True) - 1
    return np.sign(cul_returns).dropna()


def MACD(prices, short_window, long_window):
    lambda_short = (short_window - 1) / float(short_window)
    lambda_long = (long_window - 1) / float(long_window)
    hl_short = np.log(0.5) / np.log(lambda_short)
    hl_long = np.log(0.5) / np.log(lambda_long)
    ewma_short = prices.ewm(halflife=hl_short, ignore_na=True).mean()
    ewma_long = prices.ewm(halflife=hl_long, ignore_na=True).mean()
    macd = ewma_short - ewma_long
    rolling_std_63 = prices.rolling(window=63).std()
    q = macd / rolling_std_63
    y = (q / q.rolling(window=252).std()).dropna()
    return (y * np.exp(-y**2 / 4)) / 0.89


def calculate_cta_momentum_signal(prices):
    span_pairs = [(8, 24), (16, 48), (32, 96)]
    position_signs = []
    for short_span, long_span in span_pairs:
        position_signs.append(MACD(prices, short_span, long_span))
    average_macd = np.mean(np.stack(position_signs), axis=0)
    return pd.DataFrame(average_macd, columns=position_signs[0].columns, index=position_signs[0].index)


def calculate_cta_momentum_signal_long_only(prices):
    macd = calculate_cta_momentum_signal(prices)
    return macd.map(lambda x: x if x > 0 else 0)


def _trading_signal(returns, prices, trading_str):
    if trading_str == "LONG":
        return long(returns)
    if trading_str == "SGN":
        return SGN(returns)
    if trading_str == "MACD":
        return calculate_cta_momentum_signal(prices)
    if trading_str == "MACD_LONG":
        return calculate_cta_momentum_signal_long_only(prices)
    raise ValueError(f"Unknown trading_str: {trading_str}")


def TSMOM_daily(returns, prices, trading_str):
    sigma_tar = 0.15
    sigma_t = returns.ewm(span=60, min_periods=60, adjust=False).std().dropna()
    trading_signal = _trading_signal(returns, prices, trading_str)

    aligned_sigma = sigma_t.iloc[:-1]
    aligned_trading = trading_signal.iloc[:-1]
    common_index = aligned_trading.index.intersection(aligned_sigma.index)
    aligned_sigma_t = sigma_t.loc[common_index]
    aligned_trading_signal = trading_signal.loc[common_index]

    pos_in_returns = returns.index.get_indexer(aligned_trading_signal.index)
    next_pos = pos_in_returns + 1
    valid_mask = (pos_in_returns != -1) & (next_pos < len(returns.index))

    aligned_trading_signal = aligned_trading_signal.loc[valid_mask]
    aligned_sigma_t = aligned_sigma_t.loc[valid_mask]
    shifted_index = returns.index[next_pos[valid_mask]]
    aligned_next_return = returns.loc[shifted_index]

    results = aligned_trading_signal.values * (sigma_tar / (aligned_sigma_t.values * np.sqrt(252))) * aligned_next_return.values
    results_df = pd.DataFrame(results, index=aligned_next_return.index, columns=aligned_next_return.columns).dropna()
    return np.mean(results_df, axis=1)


def weight_series(returns, prices, trading_str, freq_balance):
    """Position weight (signal * vol-scale) that TSMOM_rebalance holds fixed between
    rebalance dates, averaged across assets - exposed here so it can be plotted directly."""
    sigma_tar = 0.15
    sigma_t = returns.ewm(span=60, min_periods=60, adjust=False).std().dropna()
    trading_signal = _trading_signal(returns, prices, trading_str)

    trading_signal_freq = trading_signal.iloc[::freq_balance]
    trading_signal_freq_with_sigma = trading_signal_freq.loc[trading_signal_freq.index >= sigma_t.index[0]]

    dates, weights = [], []
    for i in range(len(trading_signal_freq_with_sigma) - 1):
        start_date = trading_signal_freq_with_sigma.index[i]
        signal_i = trading_signal_freq_with_sigma.loc[start_date]
        sigma_at_start = sigma_t.loc[start_date]
        weight_i = signal_i.values * (sigma_tar / (sigma_at_start.values * np.sqrt(252)))
        dates.append(start_date)
        weights.append(weight_i.mean())

    return pd.Series(weights, index=dates)


def TSMOM_rebalance(returns, prices, trading_str, freq_balance):
    sigma_tar = 0.15
    sigma_t = returns.ewm(span=60, min_periods=60, adjust=False).std().dropna()
    if trading_str == "LONG":
        trading_signal = long(returns)
    elif trading_str == "SGN":
        trading_signal = SGN(returns)
    elif trading_str == "MACD":
        trading_signal = calculate_cta_momentum_signal(prices)
    elif trading_str == "MACD_LONG":
        trading_signal = calculate_cta_momentum_signal_long_only(prices)
    else:
        raise ValueError(f"Unknown trading_str: {trading_str}")

    trading_signal_freq = trading_signal.iloc[::freq_balance]
    trading_signal_freq_with_sigma = trading_signal_freq.loc[trading_signal_freq.index >= sigma_t.index[0]]
    if len(trading_signal_freq_with_sigma) < 2:
        raise ValueError("Not enough data after the 252-day/60-day warmup windows to run this strategy.")

    returns_series = []
    for i in range(len(trading_signal_freq_with_sigma) - 1):
        start_date = trading_signal_freq_with_sigma.index[i]
        end_date = trading_signal_freq_with_sigma.index[i + 1]
        signal_i = trading_signal_freq_with_sigma.loc[start_date]
        sigma_at_start = sigma_t.loc[start_date]
        window_dates = sigma_t.loc[start_date:end_date].iloc[:-1].index
        pos_in_returns = returns.index.get_indexer(window_dates)
        next_pos = pos_in_returns + 1
        valid_mask = (pos_in_returns != -1) & (next_pos < len(returns.index))
        shifted_index = returns.index[next_pos[valid_mask]]
        period_returns = returns.loc[shifted_index]
        weight_i = signal_i.values * (sigma_tar / (sigma_at_start.values * np.sqrt(252)))
        results = weight_i * period_returns.values
        returns_series.append(pd.DataFrame(results, columns=period_returns.columns, index=period_returns.index))

    returns_series_df = pd.concat(returns_series)
    return np.mean(returns_series_df, axis=1)


def cumulative_returns(selected_returns):
    return (1 + selected_returns).cumprod()


def annualized_return(selected_returns, freq):
    t1 = cumulative_returns(selected_returns)
    return t1.iloc[-1] ** (freq / len(t1)) - 1


def annualized_std(selected_returns, freq):
    return np.sqrt(freq) * selected_returns.std()


def sharp_ratio(series_returns):
    SR = series_returns.mean() / series_returns.std()
    return SR * (252 / np.sqrt(252))


def max_drawdown(cum_returns):
    drawdowns = cum_returns / cum_returns.cummax() - 1
    return -drawdowns.min()


def downside_deviation(selected_returns):
    MAR = 0
    downside_returns = selected_returns[selected_returns < MAR] - MAR
    n = len(downside_returns)
    if n > 1:
        return np.sqrt((downside_returns ** 2).mean() * (n / (n - 1)))
    return 0


def sortino_ratio(selected_returns, MAR=0):
    downside_std = downside_deviation(selected_returns)
    if downside_std == 0:
        return np.nan
    excess_return = selected_returns.mean() - MAR
    return (excess_return / downside_std) * np.sqrt(252)


def per_positive_return(selected_returns):
    num_positive = (selected_returns > 0).sum()
    return (num_positive / len(selected_returns)) * 100


def StatV1(vec, freq):
    """Summary statistics table (annualized mean/std/Sharpe, skew, kurtosis, max
    drawdown, downside deviation, Sortino, % positive returns) for each column of
    a returns DataFrame - same table used to report the research notebook's
    performance metrics."""
    Resultats = pd.DataFrame(index=vec.columns, columns=["Mean", "Std", "Sharpe", "Skew", "Kurt", "MaxDD", "DownsideDev", "Sortino", "PerPosReturn"])

    Ret = annualized_return(vec, freq)
    STD = annualized_std(vec, freq)
    Sharpe = Ret / STD
    SKEW = vec.skew()
    Kurt = vec.kurt()

    for cc in vec.columns:
        Resultats.loc[cc, "MaxDD"] = 100 * max_drawdown((1 + vec[cc].dropna()).astype(float).cumprod())
        Resultats.loc[cc, "DownsideDev"] = 100 * downside_deviation(vec[cc].dropna())
        Resultats.loc[cc, "Sortino"] = sortino_ratio(vec[cc].dropna())
        Resultats.loc[cc, "PerPosReturn"] = per_positive_return(vec[cc].dropna())

    Resultats["Mean"] = 100 * Ret.values
    Resultats["Std"] = 100 * STD.values
    Resultats["Sharpe"] = Sharpe.values
    Resultats["Skew"] = SKEW.values
    Resultats["Kurt"] = Kurt.values

    return Resultats
