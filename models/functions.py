import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from keras_tuner.tuners import RandomSearch
import tensorflow.keras.backend as K
import os 
import random 
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1


##########################Data Process#############################################
# Process the data -> currenctly select the top 10 most length of the data 
def process_data(file_path, sheet_name, top_n_columns, column_names=None):
    '''
    Process the data from the given excel sheets
    Parameter: file_path-> identify the path of the folder,
    sheet_name -> identify which sheet to extract the data
    top_n_columns -> remove the top of the n rows to read the data directly
    column_names -> optional; a name (or list of names) to select specific index/indices by,
    matched against the column headers by prefix (e.g. "FV1" matches "FV1 Comdty"). When given,
    this overrides the least-NaN top_n_columns selection below.
    '''
    # Read the Excel File and skip the first 4 rows
    data = pd.read_excel(file_path, sheet_name= sheet_name, header=4)
    # Skip the first tow rows
    data = data.iloc[2:, :]
    data.rename(columns={data.columns[0]:'Date'}, inplace= True)
    data['Date'] =pd.to_datetime(data['Date'])
    data.set_index('Date', inplace= True )
    data.columns = data.columns.str.strip()
    if column_names is not None:
        if isinstance(column_names, str):
            column_names = [column_names]
        column_name = [c for c in data.columns if any(c.upper().startswith(name.upper()) for name in column_names)]
    else:
        # Calculate the number of NaNs in each column and rank them from least to most missing data,
        # so that requesting top_n_columns=N returns the N columns with the least NaNs overall
        # (rather than only the columns tied for the single smallest NaN count)
        nan_counts = data.isna().sum().sort_values(kind='stable')
        column_name = nan_counts.index[:top_n_columns].tolist()
    selected_data = data[column_name].apply(pd.to_numeric, errors='coerce')
    # Calculate the relative returns
    returns = selected_data.pct_change()
    returns = returns.dropna()
    # we can have the selected_data with the prices and relative returns 
    return selected_data, returns

# calculate the drawdown to illustrate the prices changes compared to the peak 
def calculate_drawdown(prices):
    peak = prices.expanding(min_periods =1).max()
    drawdown = (prices - peak) / peak 
    return drawdown 


###########################Trend Estimation########################################
# Define all the Trned Estimations 
# Long Only Singal 
def long(returns):
    '''
    calculate the long-only trading signal based on cumulative returns 
    '''
    # Calculate the cumulative returns 
    cul_returns = (1 + returns).rolling(window=252).apply(np.prod, raw = True) -1 
    # Based on the cumulative returns, we can get the signal 
    position_sign = np.sign(cul_returns)
    position_sign = position_sign.dropna()
    # here only the positive signal, if it is negative, it will turn into 0
    long_only = position_sign.map(lambda x: x if x>0  else 0)
    return long_only 

# Moskowitz signal 
def SGN(returns): 
    '''
    calculate the Moskowitz signal based on the cumulative returns 
    ''' 
    # cumulative with 252 trading days returns
    cul_returns = (1 + returns).rolling(window=252).apply(np.prod, raw = True) -1 
    # Based on the cumulative returns, we can get the signal 
    position_sign = np.sign(cul_returns)
    position_sign = position_sign.dropna()
    return position_sign

# Baz Signal 
def MACD(prices, short_window, long_window): 
    '''
    calculate the Baz signal based on he MACD(Moving Average Convergence Divergence) method
    ''' 
    # Calculate the lambda values for the short and long window 
    lambda_short = (short_window -1)/ float(short_window)
    lambda_long = (long_window -1)/ float(long_window)
    # Calculate the half-life values for the short and long window
    hl_short = np.log(0.5)/np.log(lambda_short)
    hl_long = np.log(0.5)/np.log(lambda_long)
    # Calculate the exponentially weighted moving average for short and long window 
    ewma_short = prices.ewm(halflife=hl_short,ignore_na=True).mean()
    ewma_long = prices.ewm(halflife=hl_long,ignore_na=True).mean()
    macd = ewma_short - ewma_long
    rolling_std_63 = prices.rolling(window=63).std()
    q = macd / rolling_std_63
    y = (q / q.rolling(window=252).std()).dropna()
    position_sign = (y * np.exp(-y**2/4))/0.89
    return position_sign


# Average the Baz Signal 
def calculate_cta_momentum_signal(prices): 
    '''
    calculate the Baz signal by averaging 3 different pairs of the short and long windows 
    ''' 
    span_pairs = [(8,24), (16,48), (32,96)] # In the literature, it is given by pairs 
    # Create MACD indicators to pass through all the short spans and long spans 
    position_signs = [] 
    for short_span, long_span in span_pairs: 
        macd_signal = MACD(prices, short_span, long_span)
        position_signs.append(macd_signal)
    # Average the results to get the MACD signals 
    average_macd = np.mean(np.stack(position_signs), axis =0)
    macd = pd.DataFrame(average_macd, columns = macd_signal.columns, index = macd_signal.index)
    return macd


# Long-only variant of the Baz/MACD signal: negative position sizes clipped to 0
def calculate_cta_momentum_signal_long_only(prices):
    '''
    calculate the long-only Baz signal - same as calculate_cta_momentum_signal but
    negative position sizes are clipped to 0 (no short positions)
    '''
    macd = calculate_cta_momentum_signal(prices)
    macd_long_only = macd.map(lambda x: x if x > 0 else 0)
    return macd_long_only


##########################Times Series Mometum Strategy's Return #############################################
def TSMOM_daily(returns, prices, trading_str): 
    '''
    calculate the Time Series Momentum Strategy's daily return
    ''' 
    sigma_tar = 0.15 # 15% as the target volatlity (adjustment depending on the situation)
    # Calculate the exponential moving standarddeviation over the 60-day window -> daily standard deviation 
    sigma_t = returns.ewm(span=60, min_periods =60, adjust=False).std().dropna()
    # select the trading strategies 
    if trading_str == 'LONG': 
        trading_signal = long(returns) 
    if trading_str == 'SGN': 
        trading_signal = SGN(returns)
    if trading_str == 'MACD':
        trading_signal = calculate_cta_momentum_signal(prices)
    if trading_str == 'MACD_LONG':
        trading_signal = calculate_cta_momentum_signal_long_only(prices)
    # remove the last row of the sigma and trading signal in order to match the length of the returns
    algined_sigma = sigma_t.iloc[:-1]
    aligned_trading = trading_signal.iloc[:-1]
    # To get the common index 
    common_index = aligned_trading.index.intersection(algined_sigma.index)
    aligned_sigma_t = sigma_t.loc[common_index]
    aligned_trading_signal = trading_signal.loc[common_index]
    # The returns will be 1 day more than the sigma and the trading signals -> that's why we need the next day's return.
    # We look up the next row positionally in returns.index (instead of returns.index + BusinessDay(1)) because
    # BusinessDay only skips weekends, not market holidays, and would otherwise look up dates that don't exist
    # in returns.index (e.g. July 4th, Thanksgiving) and raise a KeyError.
    pos_in_returns = returns.index.get_indexer(aligned_trading_signal.index)
    next_pos = pos_in_returns + 1
    valid_mask = (pos_in_returns != -1) & (next_pos < len(returns.index))

    aligned_trading_signal = aligned_trading_signal.loc[valid_mask]
    aligned_sigma_t = aligned_sigma_t.loc[valid_mask]
    shifted_index = returns.index[next_pos[valid_mask]]
    aligned_next_return = returns.loc[shifted_index]

    # Based on the math functions calcualte the Time Series Momentum Strategy's returns 
    results = aligned_trading_signal.values * (sigma_tar/ (aligned_sigma_t.values * np.sqrt(252))) * aligned_next_return.values
    results_df = pd.DataFrame(results, index=aligned_next_return.index, columns = aligned_next_return.columns)
    results_df = results_df.dropna()
    avg_results = np.mean(results_df, axis=1)
    return avg_results 

# Instead of the daily reblance portfolios, we are looking for weekly, monthly, quarterly rebalancing of the portfolios 
# Constructing the portfolios based on the trading signal on day 04th, day 20th and day 62th 
def TSMOM_rebalance(returns, prices, trading_str, freq_balance): 
    sigma_tar = 0.15 # 15% as the target volatlity (adjustment depending on the situation)
    sigma_t = returns.ewm(span=60, min_periods =60, adjust=False).std().dropna()
    # select the trading strategies 
    if trading_str == 'LONG': 
        trading_signal = long(returns) 
    if trading_str == 'SGN': 
        trading_signal = SGN(returns)
    if trading_str == 'MACD':
        trading_signal = calculate_cta_momentum_signal(prices)
    if trading_str == 'MACD_LONG':
        trading_signal = calculate_cta_momentum_signal_long_only(prices)
    # calculate the frequency of the balance - weekly, monthly, quarterly or yearly
    trading_signal_freq = trading_signal.iloc[::freq_balance]
    trading_signal_freq_with_sigma = trading_signal_freq.loc[trading_signal_freq.index >= sigma_t.index[0]]
    returns_series = []
    # Using the loop to set up the range of the sigma and returns based on which trading signal days
    for i in range(len(trading_signal_freq_with_sigma)-1):
        # At the beginning of the signal
        start_date = trading_signal_freq_with_sigma.index[i]
        # Setting the end of the date at the next signal (inclusive), then dropping it below so the
        # period covers every day up to (but not including) the next rebalance date
        end_date = trading_signal_freq_with_sigma.index[i+1]
        signal_i = trading_signal_freq_with_sigma.loc[start_date]
        # Fix the vol-scaling term at start_date too (not recomputed daily) - the position weight
        # (signal * vol-scale) is only supposed to change at rebalance dates, so both pieces of the
        # weight must be fixed at start_date and held constant for the whole holding period, otherwise
        # the notional would silently be resized every day even though we're only "rebalancing" every
        # freq_balance days.
        sigma_at_start = sigma_t.loc[start_date]
        window_dates = sigma_t.loc[start_date: end_date].iloc[:-1].index
        # Look up each day's next trading day positionally in returns.index (instead of + BusinessDay(1))
        # because BusinessDay only skips weekends, not market holidays, and would otherwise look up dates
        # that don't exist in returns.index (e.g. July 4th, Thanksgiving) and raise a KeyError or silently
        # misalign the window.
        pos_in_returns = returns.index.get_indexer(window_dates)
        next_pos = pos_in_returns + 1
        valid_mask = (pos_in_returns != -1) & (next_pos < len(returns.index))
        shifted_index = returns.index[next_pos[valid_mask]]
        period_returns = returns.loc[shifted_index]
        # Based on the math functions calcualte the Time Series Momentum Strategy's returns
        weight_i = signal_i.values * (sigma_tar / (sigma_at_start.values * np.sqrt(252)))
        results = weight_i * period_returns.values
        results_df = pd.DataFrame(results, columns = period_returns.columns, index= period_returns.index)
        returns_series.append(results_df)
    returns_series_df = pd.concat(returns_series)
    average_returns = np.mean(returns_series_df, axis =1)
    return average_returns

def weight_series(returns, prices, trading_str, freq_balance):
    '''
    Returns the position weight (signal * vol-scale) that TSMOM_rebalance computes at each
    rebalance date and holds fixed until the next rebalance - i.e. the same weight used inside
    TSMOM_rebalance, exposed here (averaged across assets) so it can be inspected/plotted directly.
    '''
    sigma_tar = 0.15
    sigma_t = returns.ewm(span=60, min_periods=60, adjust=False).std().dropna()
    if trading_str == 'LONG':
        trading_signal = long(returns)
    if trading_str == 'SGN':
        trading_signal = SGN(returns)
    if trading_str == 'MACD':
        trading_signal = calculate_cta_momentum_signal(prices)
    if trading_str == 'MACD_LONG':
        trading_signal = calculate_cta_momentum_signal_long_only(prices)

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

def weight_series_per_asset(returns, prices, trading_str, freq_balance):
    '''
    Same as weight_series, but keeps the weight per asset instead of averaging across assets -
    returns a DataFrame indexed by rebalance date, one column per index, so the per-index position
    weight (signal * vol-scale) can be inspected/plotted individually.
    '''
    sigma_tar = 0.15
    sigma_t = returns.ewm(span=60, min_periods=60, adjust=False).std().dropna()
    if trading_str == 'LONG':
        trading_signal = long(returns)
    if trading_str == 'SGN':
        trading_signal = SGN(returns)
    if trading_str == 'MACD':
        trading_signal = calculate_cta_momentum_signal(prices)
    if trading_str == 'MACD_LONG':
        trading_signal = calculate_cta_momentum_signal_long_only(prices)

    trading_signal_freq = trading_signal.iloc[::freq_balance]
    trading_signal_freq_with_sigma = trading_signal_freq.loc[trading_signal_freq.index >= sigma_t.index[0]]

    dates, rows = [], []
    for i in range(len(trading_signal_freq_with_sigma) - 1):
        start_date = trading_signal_freq_with_sigma.index[i]
        signal_i = trading_signal_freq_with_sigma.loc[start_date]
        sigma_at_start = sigma_t.loc[start_date]
        weight_i = signal_i.values * (sigma_tar / (sigma_at_start.values * np.sqrt(252)))
        dates.append(start_date)
        rows.append(weight_i)

    weight_df = pd.DataFrame(rows, index=dates, columns=returns.columns)
    weight_df.index.name = 'Date'
    return weight_df

def weight_series_per_asset_signal(returns, signal, freq_balance):
    '''
    Per-asset counterpart to TSMOM_rebalance_signal - takes an already-computed position signal
    DataFrame (e.g. Lasso/MLP predicted X_t) instead of a trading_str rule, and returns the
    per-index weight (signal * vol-scale) as a DataFrame indexed by rebalance date, one column
    per index, instead of averaging across assets.
    '''
    sigma_tar = 0.15
    sigma_t = returns.ewm(span=60, min_periods=60, adjust=False).std().dropna()

    signal_freq = signal.iloc[::freq_balance]
    signal_freq_with_sigma = signal_freq.loc[signal_freq.index >= sigma_t.index[0]]

    dates, rows = [], []
    for i in range(len(signal_freq_with_sigma) - 1):
        start_date = signal_freq_with_sigma.index[i]
        signal_i = signal_freq_with_sigma.loc[start_date]
        sigma_at_start = sigma_t.loc[start_date]
        weight_i = signal_i.values * (sigma_tar / (sigma_at_start.values * np.sqrt(252)))
        dates.append(start_date)
        rows.append(weight_i)

    weight_df = pd.DataFrame(rows, index=dates, columns=returns.columns)
    weight_df.index.name = 'Date'
    return weight_df

def TSMOM_rebalance_signal(returns, signal, freq_balance):
    '''
    Same fixed-weight rebalancing logic as TSMOM_rebalance, but takes an already-computed position
    signal DataFrame (e.g. Lasso/MLP predicted X_t, date-indexed, one column per asset) instead of
    deriving the signal internally from trading_str - needed for strategies whose signal comes from
    a fitted model rather than one of the classical rules.
    '''
    sigma_tar = 0.15
    sigma_t = returns.ewm(span=60, min_periods=60, adjust=False).std().dropna()

    signal_freq = signal.iloc[::freq_balance]
    signal_freq_with_sigma = signal_freq.loc[signal_freq.index >= sigma_t.index[0]]

    returns_series = []
    for i in range(len(signal_freq_with_sigma) - 1):
        start_date = signal_freq_with_sigma.index[i]
        end_date = signal_freq_with_sigma.index[i + 1]
        signal_i = signal_freq_with_sigma.loc[start_date]
        sigma_at_start = sigma_t.loc[start_date]
        window_dates = sigma_t.loc[start_date: end_date].iloc[:-1].index
        pos_in_returns = returns.index.get_indexer(window_dates)
        next_pos = pos_in_returns + 1
        valid_mask = (pos_in_returns != -1) & (next_pos < len(returns.index))
        shifted_index = returns.index[next_pos[valid_mask]]
        period_returns = returns.loc[shifted_index]
        weight_i = signal_i.values * (sigma_tar / (sigma_at_start.values * np.sqrt(252)))
        results = weight_i * period_returns.values
        results_df = pd.DataFrame(results, columns=period_returns.columns, index=period_returns.index)
        returns_series.append(results_df)
    returns_series_df = pd.concat(returns_series)
    average_returns = np.mean(returns_series_df, axis=1)
    return average_returns


##########################Features of the Neural Layers #############################################
def calculate_ex_ante_volatility(returns, period):
    return returns.rolling(window=period, min_periods=period).std()

# Feature I: the normalized returns 
def normalized_lag_returns(returns, period):
    period = int(period)
    cumla_return = (1+returns).rolling(period).apply(np.prod, raw = True)-1
    # Normalized the cumulative returns using the same 60-day EWM sigma_t as Eq. 1, scaled to the return's horizon
    sigma_t = returns.ewm(span=60, min_periods =60, adjust=False).std().dropna()
    normalized_lag_returns = cumla_return / (sigma_t * np.sqrt(period))
    return normalized_lag_returns.dropna()

# Feature II: the exponential moving average 
def EMA_ind(short_window, long_window, prices):
    lambda_short = (short_window -1)/ float(short_window)
    lambda_long = (long_window -1)/ float(long_window)
    hl_short = np.log(0.5)/np.log(lambda_short)
    hl_long = np.log(0.5)/np.log(lambda_long)
    ewma_short = prices.ewm(halflife=hl_short,ignore_na=True).mean()
    ewma_long = prices.ewm(halflife=hl_long,ignore_na=True).mean()
    macd = ewma_short - ewma_long
    rolling_std_63 = prices.rolling(window=63).std()
    q = macd / rolling_std_63
    y = (q / q.rolling(window=252).std()).dropna()
    return y



##########################Target values to calculate the normalized returns #############################################
def target_value(returns, features): 
    '''
    getting orignal sigma and returns for minimizing the loss function in order to train the neural layers 
    ''' 
    sigma_t = returns.ewm(span=60, min_periods =60, adjust=False).std().dropna()
    # Calcualte the common index 
    common_index = features.index.intersection(returns.index)
    aligned_sigma_t = sigma_t.loc[common_index]
    # trading_signal_monthly = trading_signal.iloc[::62]
    algined_sigma_t_new = aligned_sigma_t.iloc[:-1]
    # Look up the next row positionally in returns.index (instead of index + BusinessDay(1)) because
    # BusinessDay only skips weekends, not market holidays, and would otherwise look up dates that don't
    # exist in returns.index (e.g. July 4th, Thanksgiving) and raise a KeyError.
    pos_in_returns = returns.index.get_indexer(algined_sigma_t_new.index)
    next_pos = pos_in_returns + 1
    valid_mask = (pos_in_returns != -1) & (next_pos < len(returns.index))

    algined_sigma_t_new = algined_sigma_t_new.loc[valid_mask]
    shifted_index = returns.index[next_pos[valid_mask]]
    aligned_next_return = returns.loc[shifted_index]

    # Ensure we have the same length for the final multiplation
    # algined_sigma_t = aligned_sigma_t.iloc[:-1]
    target_returns = aligned_next_return
    target_sigma =algined_sigma_t_new
    return target_returns, target_sigma



##########################Train values to train the neural layers#############################################
def train_value(returns, features):
    '''
    getting the features from Feature I and Feature II to have the same length with the sigma and returns,
    in order to go through all the neural layers 
    ''' 
    # sperate data into Training, Validation and Testing
    target_returns, target_sigma = target_value(returns, features)
    common_index = features.index.intersection(target_sigma.index)
    adj_features = features.loc[common_index]
    return adj_features 


# create time-sequences for LSTM for training the neural layers 
def create_sequences(data, target_return, target_sigma, seq_length=63): 
    '''
    getting the time-sequences features to make the input data (time series, time_length, num_features)
    in order to go through the LSTM layers to contruct the time series data 
    ''' 
    X_seq = []
    y_seq = []
    for i in range(len(data) - seq_length+1): 
        X_seq.append(data[i:i+seq_length])
        combined_target = np.hstack((target_return[i+seq_length-1], target_sigma[i+seq_length-1]))
        y_seq.append(combined_target) 
    return np.array(X_seq), np.array(y_seq)

     
##########################Neural Layers#############################################
#Building the Lasso Regression 
def lasso_regression_tanh(X_train_tf, y_train_combined_tf, X_val_tf, y_val_combined_tf, output_size, alpha = 0.001, learning_rate = 0.001, epochs=100, loss_function='loss_return'):
        # Setting the seed for the reproductivity 
    def set_seeds(seed =42):
        os.environ['PYTHONHASHSEED']= str(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

    # Ensure deterministic operations 
    def set_deterministic_ops():
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.config.experimental.enable_op_determinism()

    # Forcing the CPU usage 
    def disable_gpu():
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    set_seeds(42)
    set_deterministic_ops()
    disable_gpu()
    if loss_function == 'loss_return': 
        input_dim = X_train_tf.shape[1]
        inputs = Input(shape= (input_dim, ))
        outputs = Dense(output_size, activation = 'tanh', kernel_regularizer=None)(inputs)
        model = Model(inputs, outputs)
        # Get the weights of the Dense Layer 
        weights = model.trainable_weights[0]
    # Choose the loss function to train the model 
        model.compile(optimizer = Adam(learning_rate=learning_rate), loss=lambda y_true_combined, y_pred:lasso_loss_return(y_true_combined, y_pred, weights, alpha) )
        early_stop = EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train_tf, y_train_combined_tf, epochs = epochs, validation_data= [X_val_tf, y_val_combined_tf],  callbacks = [early_stop], verbose =1 )
    elif loss_function == 'SR': 
        input_dim = X_train_tf.shape[1]
        inputs = Input(shape= (input_dim, ))
        outputs = Dense(output_size, activation = 'tanh', kernel_regularizer=None)(inputs)
        model = Model(inputs, outputs)
        # Get the weights of the Dense Layer 
        weights = model.trainable_weights[0]
        model.compile(optimizer = Adam(learning_rate=learning_rate), loss=lambda y_true_combined, y_pred:lasso_return_SR(y_true_combined, y_pred, weights, alpha) )
        early_stop = EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train_tf, y_train_combined_tf, epochs = epochs, validation_data= [X_val_tf, y_val_combined_tf],  callbacks = [early_stop], verbose =1 )
    return model 


# Building the MLP layers 
def build_mlp(input_size, output_size, loss_function):
    # Setting the seed for the reproductivity 
    def set_seeds(seed =42):
        os.environ['PYTHONHASHSEED']= str(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

    # Ensure deterministic operations 
    def set_deterministic_ops():
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.config.experimental.enable_op_determinism()

    # Forcing the CPU usage 
    def disable_gpu():
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    set_seeds(42)
    set_deterministic_ops()
    disable_gpu()

    # Define 2 different loss functions - one is Time Series Momentum Return, the other is Sharp Ratio 
    if loss_function == "loss_return": 
    # MLP constructions for the loss function: loss return (layer size will be changed based on the random search)
        model = Sequential()
        model.add(Dense(80, input_shape=(input_size,), activation = 'tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(40, input_shape=(input_size,), activation = 'tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(output_size, activation='tanh'))
    
    # MLP constructions for the loss function: Sharp Ratio (layer size will be changed based on the random search)
    elif loss_function == "SR": 
        model = Sequential()
        model.add(Dense(80, input_shape=(input_size,), activation = 'tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(40, input_shape=(input_size,), activation = 'tanh'))
        model.add(Dropout(0.2))
        # model.add(Dense(10, input_shape=(input_size,), activation = 'tanh'))
        # model.add(Dropout(0.2))
        model.add(Dense(output_size, activation='tanh'))
        
    return model

# Building the LSTM layers 
def build_LSTM(input_size, output_size, dropout, seq_length): 
    # Setting the seed for the reproductivity 
    def set_seeds(seed =42):
        os.environ['PYTHONHASHSEED']= str(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

    # Ensure deterministic operations 
    def set_deterministic_ops():
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.config.experimental.enable_op_determinism()

    # Forcing the CPU usage 
    def disable_gpu():
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    set_seeds(42)
    set_deterministic_ops()
    disable_gpu()

    # LSTM constructions 
    model = Sequential()
    model.add(LSTM(128, input_shape = (seq_length, input_size.shape[2]), return_sequences = True, recurrent_dropout = 0.4))
    model.add(Dropout(dropout))
    model.add(LSTM(80, return_sequences = False, dropout= 0.4, recurrent_dropout = 0.4))
    model.add(Dense(40, activation = 'tanh'))
    model.add(Dropout(dropout))
    model.add(Dense(output_size, activation='tanh'))
    return model  



##########################Loss functions for the Neural Layers - Pred#############################################
def loss_return(y_true_combined, y_pred): 
    sigma_tar = 0.15 # 15% as the target volatlity (adjustment depending on the situation)
    y_true_returns, y_true_sigma = tf.split(y_true_combined, num_or_size_splits=2, axis =1)
    # Calculate the TSMOM results 
    results = y_pred * (sigma_tar/ (y_true_sigma * np.sqrt(252))) * y_true_returns 
    # results = results.dropna()
    loss = -tf.reduce_mean(results)
    return loss  

def loss_return_SR(y_true_combined, y_pred): 
    sigma_tar = 0.15 # 15% as the target volatlity (adjustment depending on the situation)
    y_true_returns, y_true_sigma = tf.split(y_true_combined, num_or_size_splits=2, axis =1)
    # Calculate the TSMOM results 
    results = y_pred * (sigma_tar/ (y_true_sigma * np.sqrt(252))) * y_true_returns 
    mean_ret = tf.reduce_mean(results)
    mean_ret_2 = tf.reduce_mean(results ** 2)
    sharp_ratio = ( np.sqrt(252)  * mean_ret) / K.sqrt(mean_ret_2 - K.square(mean_ret))
    loss = -sharp_ratio
    return loss 

##########################Loss functions for the Lasso Regression#############################################
def lasso_loss_return(y_true_combined, y_pred, weights, alpha):
    loss_return_1 = loss_return(y_true_combined, y_pred)
    l1_regularization = tf.reduce_sum(tf.abs(weights))
    return loss_return_1 + alpha * l1_regularization


def lasso_return_SR(y_true_combined, y_pred, weights, alpha = 0.01): 
    SR_loss = loss_return_SR(y_true_combined, y_pred)
    l1_regularization = tf.reduce_sum(tf.abs(weights))
    return SR_loss + alpha * l1_regularization



##########################Train and evaluate the neural layers #############################################
def evaluate_model(deep_model, X_train_tf,  y_train_combined_tf, X_val_tf, y_val_combined_tf, X_test_tf, y_test_combined_tf,  epochs, loss_function): 
    '''
    Based on which loss function, the neural layers will be optimized.
    And the early stopping will be added based on the results of the validation data 
    '''     
    tar_sigma = 0.15
    # based on the loss function, the neural layer will be trained with the Adam optimizer with the learning rate
    if loss_function == 'loss_return':
        deep_model.compile(optimizer= Adam(learning_rate=0.001), loss = loss_return)
    if loss_function == 'SR':
        deep_model.compile(optimizer= Adam(learning_rate=0.001), loss = loss_return_SR)
    # Early stopping is set
    early_stopping = EarlyStopping(monitor='val_loss', patience =10, restore_best_weights= True)
    deep_model.fit(X_train_tf,y_train_combined_tf, epochs =epochs, validation_data = (X_val_tf, y_val_combined_tf), callbacks= [early_stopping], verbose =1)
    val_loss = deep_model.evaluate(X_val_tf, y_val_combined_tf)
    print(F"Validation Loss:{val_loss}")
    # predict the signal based on the neural layer 
    pred_signal = deep_model.predict(X_test_tf)
    y_test_returns, y_test_sigma = tf.split(y_test_combined_tf, num_or_size_splits=2, axis =1)
    # Calculate the predicted mometum strategy return based on the predicted signal and testing data 
    TSMOM_returns = pred_signal * ( tar_sigma/ (y_test_sigma*np.sqrt(252)) ) * y_test_returns 
    avg_TSMOM =tf.reduce_mean(TSMOM_returns, axis =1)
    avg_TSMOM = avg_TSMOM.numpy()
    return pred_signal, avg_TSMOM, TSMOM_returns

# define the TSMOM returns for the lasso regression
def TSMOM_return_lasso(pred_signal, y_test_combined_tf): 
    '''
    Since the Lasso Regression has different loss functions because of the alpha - regularization parameters, 
    it will be another way to calculate the average of the time series momentum return.
    '''  
    tar_sigma = 0.15
    y_test_returns, y_test_sigma = tf.split(y_test_combined_tf, num_or_size_splits=2, axis =1)
    # Calculate the predicted mometum strategy return based on the predicted signal and testing data 
    TSMOM_returns = pred_signal * ( tar_sigma / (y_test_sigma*np.sqrt(252)) ) * y_test_returns 
    avg_TSMOM =tf.reduce_mean(TSMOM_returns, axis =1)
    avg_TSMOM = avg_TSMOM.numpy()
    return avg_TSMOM, TSMOM_returns

##########################Calculate the portfolio Performance Metrics #############################################
# calculate the annulized the Sharp Ratio 
def sharp_ratio(series_returns):
    SR = series_returns.mean()/ series_returns.std()
    return SR * (252 / np.sqrt(252))


def cumulative_returns(selected_returns):
    return (1 + selected_returns).cumprod()

def annualized_return(selected_returns, freq):
    t1 = cumulative_returns(selected_returns)
    return t1.iloc[-1]**(freq/len(t1)) - 1

def annualized_std(selected_returns, freq):
    return np.sqrt(freq) * selected_returns.std()

def max_drawdown(cumulative_returns):
    drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
    return -drawdowns.min()

def downside_deviation(selected_returns):
    MAR = 0  # Minimum Acceptable Return, can be adjusted as needed
    downside_returns = selected_returns[selected_returns < MAR] - MAR
    square_downside_deviation = downside_returns ** 2
    mean_square_dd = square_downside_deviation.mean()
    n = len(downside_returns)
    if n > 1:
        downside_deviation_result = np.sqrt(mean_square_dd * (n / (n - 1)))
    else:
        downside_deviation_result = 0  # Handling the case with no downside returns
    return downside_deviation_result

def sortino_ratio(selected_returns, MAR=0):
    downside_std = downside_deviation(selected_returns)
    mean_selected_returns = selected_returns.mean()
    excess_return = mean_selected_returns - MAR
    sortino_ratio = excess_return / downside_std
    return sortino_ratio * np.sqrt(252)

def per_positive_return(selected_returns):
    # Count the number of positive returns
    num_positive = (selected_returns > 0).sum()
    # Calculate the number of total returns
    total_returns = len(selected_returns)
    per_positive = (num_positive / total_returns) * 100
    return per_positive

def StatV1(vec, freq):
    Resultats = pd.DataFrame(index=vec.columns, columns=['Mean', 'Std', 'Sharpe', 'Skew', 'Kurt', 'MaxDD', 'DownsideDev', 'Sortino', 'PerPosReturn'])

    t1 = cumulative_returns(vec)
    Ret = annualized_return(vec, freq)
    STD = annualized_std(vec, freq)
    Sharpe = Ret / STD
    SKEW = vec.skew()
    Kurt = vec.kurt()
    
    for cc in vec.columns:
        Resultats.loc[cc, 'MaxDD'] = 100 * max_drawdown((1 + vec[cc].dropna()).astype(float).cumprod())
        Resultats.loc[cc, 'DownsideDev'] = 100 * downside_deviation(vec[cc].dropna())
        Resultats.loc[cc, 'Sortino'] = sortino_ratio(vec[cc].dropna())
        Resultats.loc[cc, 'PerPosReturn'] = per_positive_return(vec[cc].dropna())
    
    Resultats['Mean'] = 100 * Ret.values
    Resultats['Std'] = 100 * STD.values
    Resultats['Sharpe'] = Sharpe.values
    Resultats['Skew'] = SKEW.values
    Resultats['Kurt'] = Kurt.values

    start_date = vec.index[0].strftime("%Y-%m-%d")
    end_date = vec.index[-1].strftime("%Y-%m-%d")
    
    print(start_date)
    print(end_date)

    return Resultats




