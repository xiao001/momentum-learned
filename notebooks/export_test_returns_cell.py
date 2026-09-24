# Paste this into a new cell at the END of neural_tsmom_research.ipynb (after section 4.1 has run) and execute it.
# It saves the 11 daily test-period return series that the Streamlit comparison chart plots.
export = pd.DataFrame({
    'Long Only': r_Long_test,
    'SGN': r_SGN_test,
    'MACD': r_MACD_test,
    'MACD Long Only': r_MACD_LONG_test,
    'Buy and Hold': buy_and_hold_test,
    'Lasso (Return Loss)': lasso_loss_test,
    'Lasso (Sharpe Loss)': lasso_sr_test,
    'MLP (Return Loss)': mlp_loss_test,
    'MLP (Sharpe Loss)': mlp_sr_test,
    'LSTM (Return Loss)': lstm_loss_test,
    'LSTM (Sharpe Loss)': lstm_sr_test,
})
export.index.name = 'Date'
from pathlib import Path
_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'app' / 'app.py').exists())
(_root / 'results').mkdir(exist_ok=True)
export.to_csv(_root / 'results' / 'test_period_returns.csv')
export.shape, export.isna().sum().sum()
