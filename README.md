# Momentum, Learned — TSMOM Strategy Explorer

Can neural networks beat classical trend-following? This project compares rule-based
Time Series Momentum (SGN, MACD and their long-only variants) with Lasso, MLP and LSTM
position signals, each trained with a return loss and a Sharpe loss, across a 33-market
futures universe. The formulation follows *Enhancing Time Series Momentum Strategies
Using Deep Neural Networks* (Lim, Zohren & Roberts).

The Streamlit app shows the research results, the methodology (with the paper's
equations), and a live demo that runs the classical strategies on your own price data
(CSV or Excel) or on built-in samples.

The neural models are trained offline in `notebooks/neural_tsmom_research.ipynb`; the app
does not retrain them.

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

To rerun the notebooks and retrain the models you also need TensorFlow,
scikit-learn, keras-tuner and seaborn. The notebooks locate `models/functions.py` and
`data/` on their own, so they work from any folder inside the project.

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (public or private).
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click **New app**, pick the repo/branch, and set the main file to `app/app.py`.
4. Deploy. You get a free `https://<your-app-name>.streamlit.app` URL. The app sleeps
   after a period of inactivity on the free tier and wakes on the next visit (~30 s).

`app/app.py` imports `app/tsmom_lite.py`, a small pandas/numpy-only module, and not
`models/functions.py` (which pulls in TensorFlow). This keeps the deployed install light.

## Regenerating the results file

After retraining in the notebook, run the cell in `notebooks/export_test_returns_cell.py`
at the end of the notebook. It writes `results/test_period_returns.csv`, which the app's
comparison chart and table read.
