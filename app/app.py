"""
TSMOM Strategy Explorer — portfolio/resume-facing Streamlit app.

Four tabs:
  1. Results      — precomputed research results (clickable hero metrics)
  2. Methodology   — the math, the model comparison, the tech stack
  3. Live Demo     — upload your own data, run the classical strategies live
  4. About         — author info, skills demonstrated, links

Edit the CONFIG block below with your own name/links before sharing this.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tsmom_lite as functions
from neural_section import render_neural_section
from methodology_section import render_methodology
from about_section import render_about

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "Data_V1.xlsx"

# =============================================================================
# CONFIG — edit these before sharing the deployed link
# =============================================================================
CONFIG = {
    "author_name": "Xiao Xue",
    "author_tagline": "Quantitative Finance / ML",
    "github_url": "https://github.com/xiao001/momentum-learned",
    "linkedin_url": "https://www.linkedin.com/in/xiao-xue-9a5b88103/",
    "email": "xuexiao1631@hotmail.com",
    "notebook_note": "Full research notebooks (feature engineering, Lasso/MLP/LSTM "
                      "training, per-index diagnostics) are in the linked GitHub repo.",
}

st.set_page_config(page_title=f"TSMOM Strategy Explorer — {CONFIG['author_name']}", page_icon="📈", layout="wide")

# =============================================================================
# Styling
# =============================================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .hero-card {
        border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
        padding: 16px 18px; text-align: left; height: 100%;
    }
    .hero-card .val {font-size: 1.6rem; font-weight: 700; line-height: 1.1;}
    .hero-card .label {font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em; opacity: 0.65; margin-top: 4px;}
    .hero-card .sub {font-size: 0.8rem; opacity: 0.55; margin-top: 6px;}
    .skill-pill {
        display: inline-block; padding: 4px 12px; margin: 3px 4px 3px 0;
        border-radius: 999px; background: rgba(31,111,84,0.12); color: #1f6f54;
        font-size: 0.8rem; font-weight: 600;
    }
    .subtle-divider {border: none; border-top: 1px solid rgba(128,128,128,0.2); margin: 28px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

STRATEGIES = {
    "Long Only": "LONG",
    "SGN (Long & Short)": "SGN",
    "MACD": "MACD",
    "MACD Long Only": "MACD_LONG",
}
FREQS = {"Daily": 1, "Weekly": 4, "Monthly": 20, "Quarterly": 62}

UNIVERSE = [
    ("ES1 Index", "S&P 500", "Equity", "Americas"),
    ("NQ1 Index", "Nasdaq 100", "Equity", "Americas"),
    ("PT1 Index", "S&P TSX 60", "Equity", "Americas"),
    ("VG1 Index", "Euro Stoxx 50", "Equity", "EMEA"),
    ("Z 1 Index", "FTSE 100", "Equity", "EMEA"),
    ("GX1 Index", "DAX", "Equity", "EMEA"),
    ("SM1 Index", "SMI", "Equity", "EMEA"),
    ("CF1 Index", "CAC 40", "Equity", "EMEA"),
    ("TP1 Index", "TOPIX", "Equity", "APAC"),
    ("NK1 Index", "Nikkei 225", "Equity", "APAC"),
    ("HI1 Index", "Hang Seng", "Equity", "APAC"),
    ("QZ1 Index", "MSCI Singapore", "Equity", "APAC"),
    ("KM1 Index", "KOSPI 200", "Equity", "APAC"),
    ("FT1 Index", "TAIEX", "Equity", "APAC"),
    ("CO1 Comdty", "Brent Crude", "Commodity", "Energy"),
    ("CL1 Comdty", "WTI", "Commodity", "Energy"),
    ("NG1 Comdty", "US Natural Gas", "Commodity", "Energy"),
    ("HO1 Comdty", "Heating Oil", "Commodity", "Energy"),
    ("QS1 Comdty", "Low Sulphur Gasoil", "Commodity", "Energy"),
    ("HG1 Comdty", "Copper", "Commodity", "Base Metals"),
    ("LN1 Comdty", "Nickel", "Commodity", "Base Metals"),
    ("LA1 Comdty", "Aluminum", "Commodity", "Base Metals"),
    ("GC1 Comdty", "Gold", "Commodity", "Precious Metals"),
    ("SI1 Comdty", "Silver", "Commodity", "Precious Metals"),
    ("S 1 Comdty", "Soybean", "Commodity", "Agriculture"),
    ("C 1 Comdty", "Corn", "Commodity", "Agriculture"),
    ("W 1 Comdty", "Wheat", "Commodity", "Agriculture"),
    ("TU1 Comdty", "2y UST", "Fixed Income", "Americas"),
    ("FV1 Comdty", "5y UST", "Fixed Income", "Americas"),
    ("TY1 Comdty", "10y UST", "Fixed Income", "Americas"),
    ("RX1 Comdty", "Germany 10y (Bund)", "Fixed Income", "EMEA"),
    ("JB1 Comdty", "Japan 10y", "Fixed Income", "APAC"),
    ("FF1 Comdty", "1M Fed Funds", "Interest Rates", "Global"),
]

RESEARCH_RESULTS = pd.DataFrame([
    {"Strategy": "Long Only",            "E[Return] %": 3.85,  "Vol. %": 4.19, "Sharpe Ratio": 0.92,  "Sortino Ratio": 0.86,  "% Positive Days": 54.60},
    {"Strategy": "Long+Short (SGN)",     "E[Return] %": 0.33,  "Vol. %": 3.94, "Sharpe Ratio": 0.08,  "Sortino Ratio": 0.08,  "% Positive Days": 52.15},
    {"Strategy": "MACD",                 "E[Return] %": -0.08, "Vol. %": 2.61, "Sharpe Ratio": -0.03, "Sortino Ratio": -0.03, "% Positive Days": 51.23},
    {"Strategy": "Buy & Hold",           "E[Return] %": 7.80,  "Vol. %": 7.22, "Sharpe Ratio": 1.08,  "Sortino Ratio": 1.00,  "% Positive Days": 57.98},
    {"Strategy": "Lasso + Loss",         "E[Return] %": 5.78,  "Vol. %": 4.59, "Sharpe Ratio": 1.26,  "Sortino Ratio": 1.22,  "% Positive Days": 56.13},
    {"Strategy": "Lasso + Sharpe-Loss",  "E[Return] %": 8.96,  "Vol. %": 2.86, "Sharpe Ratio": 3.13,  "Sortino Ratio": 3.78,  "% Positive Days": 55.83},
    {"Strategy": "MLP + Loss",           "E[Return] %": 10.83, "Vol. %": 3.12, "Sharpe Ratio": 3.47,  "Sortino Ratio": 3.76,  "% Positive Days": 58.59},
    {"Strategy": "MLP + Sharpe-Loss",    "E[Return] %": 8.46,  "Vol. %": 2.68, "Sharpe Ratio": 3.16,  "Sortino Ratio": 3.36,  "% Positive Days": 59.20},
    {"Strategy": "LSTM + Loss",          "E[Return] %": -4.98, "Vol. %": 5.14, "Sharpe Ratio": -0.97, "Sortino Ratio": -0.96, "% Positive Days": 47.24},
    {"Strategy": "LSTM + Sharpe-Loss",   "E[Return] %": -1.77, "Vol. %": 2.34, "Sharpe Ratio": -0.76, "Sortino Ratio": -0.77, "% Positive Days": 45.40},
]).set_index("Strategy")


@st.cache_data(show_spinner="Loading the 33-index universe from Data_V1.xlsx...")
def load_universe_panel() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    """Real price + return panels for the 33-index universe, computed from Data_V1.xlsx.
    Returns (prices, returns, None) on success, or (None, None, error_message) on failure -
    surfacing the real cause instead of swallowing it, since a bare except here
    previously hid genuine bugs behind one generic "file missing" message even
    when the file was present but something else (e.g. a bad sheet name, a
    ticker mismatch) was the actual problem."""
    if not DATA_PATH.exists():
        return None, None, f"Data_V1.xlsx not found at {DATA_PATH}"
    try:
        # Strip only the trailing "Index"/"Comdty" suffix (not split on every space -
        # tickers like "Z 1 Index" or "S 1 Comdty" contain a space in the code itself).
        tickers = [t.rsplit(" ", 1)[0] for t, *_ in UNIVERSE]
        data, returns = functions.process_data(str(DATA_PATH), "Series", tickers)
        short = {c: c.rsplit(" ", 1)[0] for c in data.columns}
        data = data.rename(columns=short)
        returns = returns.rename(columns=short)
        # Guard against the same prefix-match ambiguity leaking into the result -
        # keep only the 33 intended tickers, in the app's canonical order.
        wanted = [t for t in tickers if t in data.columns]
        return data[wanted], returns[wanted], None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(buf, index_col=0, parse_dates=True)
    else:
        df = pd.read_excel(buf, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Price-history downloads usually come as Open/High/Low/Close/Volume. Those are not separate
    # assets, so keep only the closing price (adjusted close if that is what the file has).
    lower = {str(c).strip().lower(): c for c in df.columns}
    ohlc_markers = {"open", "high", "low", "volume", "vol"}
    close_col = lower.get("close") or lower.get("adj close") or lower.get("adj_close") or lower.get("close price")
    note = None
    if close_col is not None and ohlc_markers & set(lower):
        stem = filename.rsplit(".", 1)[0]
        df = df[[close_col]].rename(columns={close_col: stem})
        note = f"Price-history format detected (Open/High/Low/Close/Volume) — using **{close_col}** only."

    # Strip thousands separators / currency symbols before converting to numbers (e.g. "1,234.50", "$12").
    df = df.apply(lambda s: pd.to_numeric(s.astype(str).str.replace(r"[,$\s]", "", regex=True), errors="coerce"))
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.attrs["note"] = note
    return df


def performance_row(name: str, r: pd.Series) -> dict:
    r = r.dropna()
    cum = functions.cumulative_returns(r)
    return {
        "Strategy": name,
        "E[Return] %": round(functions.annualized_return(r, 252) * 100, 2),
        "Vol. %": round(functions.annualized_std(r, 252) * 100, 2),
        "Sharpe Ratio": round(functions.sharp_ratio(r), 2),
        "Max Drawdown %": round(functions.max_drawdown(cum) * 100, 2),
        "Sortino Ratio": round(functions.sortino_ratio(r), 2),
        "% Positive Days": round(functions.per_positive_return(r), 2),
    }


REBALANCE_FREQS = {"Daily": 1, "Weekly": 4, "Monthly": 20, "Quarterly": 62}
BENCHMARK_STRATS = {"LONG": "Long Only", "SGN": "SGN", "MACD": "MACD", "MACD_LONG": "MACD Long Only"}


@st.cache_data(show_spinner="Running the 4 classical strategies across every rebalancing frequency...")
def load_rebalance_comparison(panel_data: pd.DataFrame, panel_returns: pd.DataFrame):
    """Cumulative return + position weight (signal x vol-scale) of the 4 classical
    strategies (Long Only, SGN, MACD, MACD Long Only) plus Buy & Hold, computed live
    from the real 33-index panel at every rebalancing frequency."""
    buy_and_hold = (panel_returns.mean(axis=1) + 1).cumprod()

    cum_returns = {}
    weight_tables = {}
    for freq_name, freq in REBALANCE_FREQS.items():
        strat_cum = {}
        weight_cols = {}
        for code, label in BENCHMARK_STRATS.items():
            if freq == 1:
                r = functions.TSMOM_daily(panel_returns, panel_data, trading_str=code)
            else:
                r = functions.TSMOM_rebalance(panel_returns, panel_data, trading_str=code, freq_balance=freq)
            strat_cum[label] = functions.cumulative_returns(r.dropna())
            weight_cols[label] = functions.weight_series(panel_returns, panel_data, code, freq)
        strat_cum["Buy & Hold"] = buy_and_hold
        cum_returns[freq_name] = strat_cum

        weight_df = pd.DataFrame(weight_cols)
        weight_df.index.name = "Date"
        weight_tables[freq_name] = weight_df

    return cum_returns, weight_tables


# =============================================================================
# Header
# =============================================================================
h1, h2 = st.columns([3, 1])
with h1:
    st.title("📈 Momentum, Learned")
    st.caption(
        "Can neural networks beat classical trend-following? Lasso, MLP and LSTM signals "
        "vs. SGN and MACD across 33 futures markets."
    )
with h2:
    st.markdown(
        f"""
        <div style="text-align:right; padding-top: 8px;">
        <b>{CONFIG['author_name']}</b><br>
        <span style="opacity:0.65; font-size:0.85rem">{CONFIG['author_tagline']}</span><br>
        <a href="{CONFIG['github_url']}" target="_blank">GitHub</a> ·
        <a href="{CONFIG['linkedin_url']}" target="_blank">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_results, tab_method, tab_demo, tab_about = st.tabs(
    ["📊 Results", "🧮 Methodology", "🚀 Live Demo", "ℹ️ About"]
)

# =============================================================================
# TAB 1 — Results
# =============================================================================
with tab_results:
    st.caption(
        "Precomputed results from the research notebook, held-out test period, "
        "quarterly rebalancing. Click a card for detail."
    )

    if "active_card" not in st.session_state:
        st.session_state["active_card"] = "sharpe"

    card_specs = [
        ("sharpe", "6", "Neural TSMOM Strategies", "Lasso, MLP, LSTM × Return/Sharpe-Loss"),
        ("benchmark", "4", "Benchmark Strategies", "SGN, MACD, Long Only & Buy & Hold"),
        ("universe", "33", "Universe", "Equity / Commodity / Rates"),
    ]

    cols = st.columns(len(card_specs))
    for col, (key, val, label, sub) in zip(cols, card_specs):
        with col:
            st.markdown(
                f"""<div class="hero-card"><div class="val">{val}</div>
                <div class="label">{label}</div><div class="sub">{sub}</div></div>""",
                unsafe_allow_html=True,
            )
            if st.button("View detail", key=f"card_{key}", use_container_width=True):
                st.session_state["active_card"] = key

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    active = st.session_state["active_card"]
    if active == "sharpe":
        render_neural_section(RESEARCH_RESULTS)
    elif active == "benchmark":
        st.markdown("**Classical benchmark strategies — rebalancing frequency comparison**")
        st.caption(
            "The rule-based strategies (no ML) that every Lasso/MLP/LSTM signal model is compared "
            "against: Long Only, SGN trend signal, MACD, MACD Long Only, and Buy & Hold — computed "
            "live from the real 33-index panel, at 4 different rebalancing frequencies."
        )

        bench_data, bench_returns, bench_error = load_universe_panel()
        if bench_data is None:
            st.info(f"Live computation unavailable: {bench_error}")
        else:
            cum_returns, weight_tables = load_rebalance_comparison(bench_data, bench_returns)

            st.markdown("#### Cumulative return by rebalancing frequency")
            st.caption(
                "At each rebalance date, both the position signal and the volatility-scaling term are "
                "fixed and held constant until the next rebalance — that's what changes between the "
                "4 panels below, not the signal logic itself. Buy & Hold is included as a reference "
                "line; it doesn't depend on rebalancing frequency since it's just the equal-weighted "
                "average of the 33 raw asset returns."
            )
            fig_perf = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                subplot_titles=[f"{name} Rebalance" for name in REBALANCE_FREQS],
                vertical_spacing=0.06,
            )
            colors = {"Long Only": "#1f77b4", "SGN": "#ff7f0e", "MACD": "#2ca02c", "MACD Long Only": "#d62728", "Buy & Hold": "#7f7f7f"}
            for row, freq_name in enumerate(REBALANCE_FREQS, start=1):
                for label, series in cum_returns[freq_name].items():
                    fig_perf.add_trace(
                        go.Scatter(
                            x=series.index, y=series.values, name=label, mode="lines",
                            line=dict(color=colors[label], width=1.3, dash="dot" if label == "Buy & Hold" else "solid"),
                            legendgroup=label, showlegend=(row == 1),
                        ),
                        row=row, col=1,
                    )
            fig_perf.update_layout(height=900, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.06))
            fig_perf.update_yaxes(title_text="Growth of $1")
            st.plotly_chart(fig_perf, use_container_width=True)

            st.markdown("#### Summary statistics — Daily rebalance")
            st.caption(
                "Full performance breakdown for the daily-rebalanced strategies: annualized mean "
                "return and volatility, Sharpe ratio, skew and kurtosis of the daily return "
                "distribution, max drawdown, downside deviation, Sortino ratio, and the share of "
                "days with a positive return. Skew/kurtosis matter here because trend-following "
                "return streams are famously non-normal — a strategy can look fine on Sharpe alone "
                "while still carrying fat-tail crash risk that Sharpe doesn't price in."
            )
            r_long_d = functions.TSMOM_daily(bench_returns, bench_data, trading_str="LONG")
            r_sgn_d = functions.TSMOM_daily(bench_returns, bench_data, trading_str="SGN")
            r_macd_d = functions.TSMOM_daily(bench_returns, bench_data, trading_str="MACD")
            bh_d = bench_returns.mean(axis=1)
            daily_tbl = pd.concat([r_long_d, r_sgn_d, r_macd_d, bh_d], axis=1).dropna()
            daily_tbl.columns = ["Long Only", "Long+Short", "MACD", "Buy+Hold"]
            stat_v1 = functions.StatV1(daily_tbl, freq=252)
            st.dataframe(
                stat_v1.style.background_gradient(subset=["Sharpe"], cmap="RdYlGn").format(precision=2),
                use_container_width=True,
            )
    elif active == "universe":
        st.markdown("**The 33-ticker cross-asset universe**")
        st.caption("14 Equity, 13 Commodity, 5 Fixed Income + 1 Interest Rate. Full daily history 1999-12-30 to 2024-06-20, zero missing values.")
        universe_df = pd.DataFrame(UNIVERSE, columns=["Ticker", "Name", "Asset Class", "Region"])
        u1, u2 = st.columns([2, 1])
        with u1:
            st.dataframe(universe_df, use_container_width=True, hide_index=True, height=420)
        with u2:
            counts = universe_df["Asset Class"].value_counts()
            fig_u = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55))
            fig_u.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
            st.plotly_chart(fig_u, use_container_width=True)

        panel_data, panel_returns, panel_error = load_universe_panel()

        st.markdown("**Index price over time**")
        if panel_data is None:
            st.info(f"Price panel unavailable: {panel_error}")
        else:
            fig_p = go.Figure()
            for col in panel_data.columns:
                fig_p.add_trace(go.Scatter(x=panel_data.index, y=panel_data[col], name=col, mode="lines", line=dict(width=1)))
            fig_p.update_layout(
                title="Index Price Over Time", height=480,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Date", yaxis_title="Index Value",
                legend=dict(font=dict(size=9)),
            )
            st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("**Cumulative return over time**")
        if panel_returns is None:
            st.info(f"Return panel unavailable: {panel_error}")
        else:
            cum_returns = (panel_returns.dropna() + 1).cumprod()
            fig_cr = go.Figure()
            for col in cum_returns.columns:
                fig_cr.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns[col], name=col, mode="lines", line=dict(width=1)))
            fig_cr.update_layout(
                title="Cumulative Return Over Time", height=480,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Date", yaxis_title="Growth of $1",
                legend=dict(font=dict(size=9)),
            )
            st.plotly_chart(fig_cr, use_container_width=True)

        st.markdown("**Return correlation across the 33 indexes**")
        corr = None
        if panel_returns is not None:
            corr = panel_returns.corr()
        corr_error = panel_error
        if corr is None:
            st.info(f"Correlation matrix unavailable: {corr_error}")
        else:
            st.caption(
                "Strong within-asset-class clustering (equities, bonds, energy, metals, grains each move "
                "together), but low or negative correlation across classes — the diversification a "
                "cross-asset TSMOM strategy is built to exploit."
            )
            n = len(corr)
            text = corr.map(lambda v: f"{v:.2f}").values
            fig_c = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                text=text, texttemplate="%{text}", textfont=dict(size=8),
                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                colorbar=dict(title="ρ"),
                xgap=1, ygap=1,
            ))
            fig_c.update_layout(
                title="Correlation between the returns",
                height=22 * n + 120, width=22 * n + 220,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(tickfont=dict(size=9), side="bottom", tickangle=-90, scaleanchor="y", constrain="domain"),
                yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_c, use_container_width=False)

# =============================================================================
# TAB 2 — Methodology
# =============================================================================
with tab_method:
    render_methodology()

# =============================================================================
# TAB 3 — Live Demo
# =============================================================================
with tab_demo:
    st.caption(
        "Upload your own price data (CSV/XLSX) and run the classical rule-based "
        "strategies live. Model-based (Lasso/MLP/LSTM) strategies are trained "
        "offline — see the research notebooks — and are not retrained here."
    )

    d1, d2 = st.columns([1, 2])
    with d1:
        st.markdown("**1. Data**")
        uploaded = st.file_uploader(
            "Price CSV/XLSX (dates in the first column, one column per asset)",
            type=["csv", "xlsx", "xls"],
        )
        st.caption("Built-in samples — tick any combination:")
        use_es1 = st.checkbox("Use built-in sample: ES1 Index (S&P 500 futures)", value=uploaded is None)
        use_fv1 = st.checkbox("Use built-in sample: FV1 Index (5y US Treasury futures)", value=False)
        use_spx = st.checkbox("Use built-in sample: S&P 500 index (HistoricalPrices_SP500.csv)", value=False)
        use_sample = use_es1 or use_fv1 or use_spx

        st.markdown("**2. Strategy**")
        chosen_labels = st.multiselect(
            "Strategies to compare", list(STRATEGIES.keys()), default=list(STRATEGIES.keys())
        )
        freq_label = st.selectbox("Rebalancing frequency", list(FREQS.keys()), index=0)
        sigma_tar_pct = st.slider("Target annualized volatility (%)", 5, 30, 15)
        run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)

    with d2:
        data = None
        if uploaded is not None:
            try:
                data = load_data(uploaded.getvalue(), uploaded.name)
                if data.attrs.get("note"):
                    st.info(data.attrs["note"])
            except Exception as e:
                st.error(f"Could not read the uploaded file: {e}")
        elif use_sample:
            # Each ticked sample is added as its own column. ES1 / FV1 come from the same loader the
            # rest of the app uses; the S&P 500 index comes from HistoricalPrices_SP500.csv (Close).
            pieces, problems = [], []
            wanted = [c for c, on in (("ES1", use_es1), ("FV1", use_fv1)) if on]
            if wanted:
                sample_prices, _, sample_error = load_universe_panel()
                if sample_prices is not None and all(c in sample_prices.columns for c in wanted):
                    pieces.append(sample_prices[wanted])
                else:
                    problems.append(f"ES1/FV1 sample: {sample_error}")
            if use_spx:
                spx_path = DATA_DIR / "HistoricalPrices_SP500.csv"
                try:
                    spx = load_data(spx_path.read_bytes(), spx_path.name)
                    spx.columns = ["SPX"]
                    pieces.append(spx)
                except Exception as e:
                    problems.append(f"S&P 500 index file: {e}")
            if problems:
                st.error("Could not load: " + "; ".join(problems) + ". Upload your own file instead.")
            if pieces:
                data = pd.concat(pieces, axis=1, join="inner").dropna()
                if len(pieces) > 1:
                    st.caption(
                        "Several samples selected: they are aligned to their common dates "
                        f"({data.index.min().date()} to {data.index.max().date()})."
                    )

        if data is None or data.empty:
            st.info("Upload a file, or tick a built-in sample (ES1, FV1 and/or S&P 500), then click **Run Backtest**.")
        else:
            st.markdown("**Preview**")
            fig = go.Figure()
            for col in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col, mode="lines"))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{data.shape[0]} rows x {data.shape[1]} asset(s), {data.index.min().date()} to {data.index.max().date()}")

            if len(data) < 320:
                st.warning(
                    "Fewer than ~320 rows — the strategies use a 252-day trend lookback "
                    "and a 60-day volatility window, so results will be sparse until you "
                    "have at least ~1.5 years of daily data."
                )

    if data is not None and not data.empty and run_clicked:
        returns = data.pct_change().dropna(how="all")
        freq = FREQS[freq_label]
        vol_scale = (sigma_tar_pct / 100.0) / 0.15  # TSMOM_rebalance hardcodes sigma_tar=0.15

        results, errors = {}, []
        for label in chosen_labels:
            try:
                results[label] = functions.TSMOM_rebalance(returns, data, STRATEGIES[label], freq) * vol_scale
            except Exception as e:
                errors.append(f"{label}: {e}")

        if errors:
            st.warning("Some strategies could not be computed on this dataset:\n\n" + "\n".join(f"- {e}" for e in errors))

        if not results:
            st.error("No strategy produced results — the uploaded data is likely too short or has too many gaps.")
        else:
            results["Buy & Hold"] = returns.mean(axis=1)

            st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
            st.subheader(f"Cumulative return — {freq_label} rebalancing")
            fig2 = go.Figure()
            for label, r in results.items():
                cum = functions.cumulative_returns(r.dropna())
                fig2.add_trace(go.Scatter(x=cum.index, y=cum.values, name=label, mode="lines"))
            fig2.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Growth of $1")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Performance metrics")
            metrics_df = pd.DataFrame([performance_row(label, r) for label, r in results.items()]).set_index("Strategy")
            st.dataframe(
                metrics_df.style.background_gradient(subset=["Sharpe Ratio"], cmap="RdYlGn"),
                use_container_width=True,
            )
            csv = metrics_df.to_csv().encode("utf-8")
            st.download_button("Download metrics as CSV", csv, "tsmom_performance_metrics.csv", "text/csv")

# =============================================================================
# TAB 4 — About
# =============================================================================
with tab_about:
    render_about(CONFIG)
