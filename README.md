# Momentum, Learned — TSMOM Strategy Explorer

**[Live app](https://momentum-learned-beue9unwnobiztgbshks5x.streamlit.app/)** · [Repository](https://github.com/xiao001/momentum-learned) · MIT licence

Can neural networks beat classical trend-following? This project compares rule-based Time Series
Momentum (SGN, MACD and their long-only variants) with **Lasso, MLP and LSTM** position signals,
each trained with a **return loss** and a **Sharpe loss**, across a **33-market futures universe**
(equities, commodities, fixed income and interest rates, 1999–2024). The formulation follows
*Enhancing Time Series Momentum Strategies Using Deep Neural Networks* (Lim, Zohren & Roberts).

The Streamlit app shows the research results, the methodology (with the paper's equations), and a
live demo that runs the classical strategies on your own price data (CSV or Excel) or on built-in
samples. The neural models are trained offline in `notebooks/neural_tsmom_research.ipynb`; the app
does not retrain them.

![Daily cumulative returns of all strategies on the held-out test period](results/figures/Daily%20rebalancing%20of%20the%20strategy.png)

## Results

Held-out test period, daily rebalancing, 15% volatility target (about 326 trading days,
2023-03 to 2024-06). All strategies are aligned to the LSTM's shorter test window.

| Strategy | Annual return | Volatility | Sharpe |
|---|---|---|---|
| MLP · Return-Loss | 10.83% | 3.12% | 3.47 |
| MLP · Sharpe-Loss | 8.46% | 2.68% | 3.16 |
| Lasso · Sharpe-Loss | 8.96% | 2.86% | 3.13 |
| Lasso · Return-Loss | 5.78% | 4.59% | 1.26 |
| Buy & Hold (equal-weight) | 7.80% | 7.22% | 1.08 |
| Long Only | 3.85% | 4.19% | 0.92 |
| SGN (Long & Short) | 0.33% | 3.94% | 0.08 |
| MACD | -0.08% | 2.61% | -0.03 |
| LSTM · Sharpe-Loss | -1.77% | 2.34% | -0.76 |
| LSTM · Return-Loss | -4.98% | 5.14% | -0.97 |

- The MLP and Lasso (Sharpe-Loss) signals beat every rule-based benchmark and Buy & Hold on a
  risk-adjusted basis.
- Training on the Sharpe ratio lowered volatility for Lasso and MLP.
- The LSTM lost money out of sample; more capacity did not help on this amount of data.

### Limitations

- The test window is short (about 14 months), so a Sharpe of 3+ is encouraging, not proof.
- A single fixed 70/20/10 chronological split, not walk-forward; one random seed.
- No transaction costs or slippage.
- Fixed architectures with little hyperparameter search.
- Differences from the paper: 33 assets instead of 88, a 5-day (not 1-day) shortest lookback, and
  no WaveNet model.

## Project layout

```
app/                        Streamlit app
  app.py                    entry point
  neural_section.py         neural strategies view
  methodology_section.py    math (paper equations)
  about_section.py          about page
  tsmom_lite.py             light classical engine (pandas/numpy only)
data/
  Data_V1.xlsx              33-asset price panel (1999-12-30 to 2024-06-20)
  Data_V1 copy.xlsx         extended copy used by the ES1 notebook
  HistoricalPrices_SP500.csv  S&P 500 index (Live Demo sample)
models/
  functions.py              features, models and losses (TensorFlow)
notebooks/
  neural_tsmom_research.ipynb   main research notebook
  export_test_returns_cell.py   cell that exports daily test returns
  archive/                  earlier notebooks
results/
  test_period_returns.csv   daily test-period returns for all 11 strategies
  figures/                  saved charts
papers/                     reference papers
requirements.txt            app dependencies
```

## Run locally

Clone the repository (or download it as a ZIP), then from the project folder:

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
```

Then open the printed `http://localhost:8501` URL.

To rerun the notebooks and retrain the models you also need TensorFlow, scikit-learn,
keras-tuner and seaborn. The notebooks locate `models/functions.py` and `data/` on their own, so
they work from any folder inside the project. Training uses a fixed seed (42), deterministic
TensorFlow operations and CPU-only execution.

## Deploy for free (Streamlit Community Cloud)

1. Push the repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick the repo and branch, and set the main file to `app/app.py`.
4. Deploy. The app sleeps after a period of inactivity on the free tier and wakes on the next visit
   (about 30 seconds).

`app/app.py` imports `app/tsmom_lite.py`, a small pandas/numpy-only module, and not
`models/functions.py` (which pulls in TensorFlow), which keeps the deployed install light.

## Regenerating the results file

After retraining, run the cell in `notebooks/export_test_returns_cell.py` at the end of the
notebook. It writes `results/test_period_returns.csv`, which the app's comparison chart and table
read.

## Data and disclaimer

- The price data in `data/` is included for reproducibility. It is **not** covered by the MIT
  licence and remains subject to the terms of its original providers; check those terms before
  reusing it.
- The reference papers in `papers/` belong to their authors and are not covered by the MIT licence.
- This project is for research and education only. It is **not investment advice**, and past or
  back-tested performance does not indicate future results.

## References

1. Lim, B., Zohren, S. & Roberts, S. — *Enhancing Time Series Momentum Strategies Using Deep Neural Networks*.
2. Wood, K., Roberts, S. & Zohren, S. — *Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection*.
3. Wood, K., Giegerich, S., Roberts, S. & Zohren, S. — *Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture*.
4. Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012) — *Time Series Momentum*, Journal of Financial Economics.
5. Baz, J. et al. (2015) — *Dissecting Investment Strategies in the Cross Section and Time Series*.

## Author

Xiao Xue · [LinkedIn](https://www.linkedin.com/in/xiao-xue-9a5b88103/) · xuexiao1631@hotmail.com

## Licence

The code is released under the [MIT licence](LICENSE).
