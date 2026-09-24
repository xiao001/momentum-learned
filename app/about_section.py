"""About tab — project story, findings, caveats, references, structure."""

import streamlit as st


def _pill(text):
    return f'<span class="skill-pill">{text}</span>'


def _stat(value, label, sub=""):
    return (f'<div class="hero-card"><div class="val">{value}</div>'
            f'<div class="label">{label}</div><div class="sub">{sub}</div></div>')


def render_about(config: dict):
    left, right = st.columns([2, 1])

    with left:
        st.subheader("About this project")
        st.markdown(
            """
Trend-following (time series momentum) has long been done with hand-built rules: look at
the past year's return, or a moving-average crossover, and go long or short. This project
asks a simple question: **can a neural network learn a better rule directly from the data?**

It builds a cross-asset TSMOM research pipeline over **33 futures markets** (equities,
commodities, fixed income and interest rates, 1999–2024) and compares classical rules —
SGN, MACD and their long-only variants — against learned position signals from **Lasso,
MLP and LSTM** models. Each model is trained twice: once to maximise **return**, once to
maximise the **Sharpe ratio**. The formulation follows *Enhancing Time Series Momentum
Strategies Using Deep Neural Networks* (Lim, Zohren & Roberts).
"""
        )

        st.markdown("#### Headline results")
        c = st.columns(4)
        stats = [
            ("3.47", "Best Sharpe", "MLP · Return-Loss"),
            ("10.83%", "Best return", "MLP · Return-Loss"),
            ("1.08", "Buy & Hold Sharpe", "equal-weight reference"),
            ("< 0", "LSTM Sharpe", "both loss variants"),
        ]
        for col, s in zip(c, stats):
            with col:
                st.markdown(_stat(*s), unsafe_allow_html=True)
        st.markdown(
            """
- The **MLP and Lasso (Sharpe-Loss)** signals clearly beat every rule-based benchmark and
  Buy & Hold on a risk-adjusted basis.
- **Training on the Sharpe ratio** cut volatility for Lasso and MLP, and lifted Lasso's Sharpe
  from 1.26 to 3.13.
- **The LSTM lost money out of sample.** More capacity and a 252-day memory did not help on
  this amount of data.
"""
        )

        st.markdown("#### What this demonstrates")
        st.markdown(
            """
- **End-to-end quant research:** data cleaning, feature engineering, signal construction,
  backtesting and performance evaluation on a 6,300+ day, 33-asset panel.
- **Volatility-targeted position sizing** (15% target, 60-day EWM volatility) with configurable
  daily / weekly / monthly / quarterly rebalancing.
- **Custom loss functions:** return and Sharpe objectives written directly in TensorFlow/Keras, so
  the network learns trend estimation and position sizing jointly.
- **Leak-aware ML practice:** chronological 70/20/10 split, scaler fit on training data only,
  early stopping on validation loss, targets aligned by position so holidays cannot cause look-ahead.
- **Research turned into a tool:** the notebook results are packaged as this interactive Streamlit
  app, with live benchmark calculations and a demo that runs on your own data.
"""
        )

        st.markdown("#### How the results were produced")
        st.markdown(
            """
- **Trained offline, displayed here.** The Lasso, MLP and LSTM models are trained in the research
  notebook (`neural_tsmom_research.ipynb`). This app does **not** retrain them. Their daily test-period returns are
  exported once to `results/test_period_returns.csv`, and the Results and Compare charts are drawn from that file.
- **Classical benchmarks are computed live** from the price data, so they can be recalculated on the fly.
- **Data:** a 33-asset futures panel (`data/Data_V1.xlsx`, 1999-12-30 to 2024-06-20, no missing values) for the
  research results. The Live Demo also offers the ES1 futures contract, the FV1 5-year Treasury future, and
  the S&P 500 cash index (`data/HistoricalPrices_SP500.csv`, 2015–2026). ES1 and the S&P 500 index are
  different series.
"""
        )

        st.markdown("#### Reproducibility")
        st.markdown(
            """
- **Fixed seed (42)** for Python's `random` module, the hash seed and TensorFlow, with deterministic
  TensorFlow operations and **CPU-only** execution, so a rerun on the same data should reproduce the same weights.
- **Chronological 70 / 20 / 10** train / validation / test split — no shuffling.
- **No leakage:** the scaler is fit on the training block only, and next-day targets are aligned by
  position in the return index, so market holidays cannot cause look-ahead.
- **Early stopping** on validation loss (patience 10, best weights restored); the test block is used once.
- All six neural strategies use the same features, split and volatility target, so differences come from
  the model and the loss function only.
"""
        )

        st.markdown("#### Try it or run it yourself")
        st.markdown(
            "- **Just look around:** open the hosted app link — no installation needed.\n"
            "- **Run it locally:** first **clone the repository or download it as a ZIP** from GitHub "
            "(the app reads local data files, so all of them must be present), then, from that folder:"
        )
        st.code(
            "pip install -r requirements.txt\n"
            "python -m streamlit run app/app.py",
            language="bash",
        )
        st.caption(
            "The app needs only the packages in requirements.txt. To rerun the notebook and retrain the models "
            "you also need TensorFlow, scikit-learn, keras-tuner and seaborn."
        )

        st.markdown("#### Using the Live Demo")
        st.markdown(
            """
- **Built-in samples:** tick any of ES1, FV1 or the S&P 500 index, or upload your own file.
- **File layout:** CSV, XLSX or XLS with **dates in the first column** and **one price column per asset**
  (prices, not returns). For Excel files only the first sheet is read.
- **Price-history downloads** with Open / High / Low / Close / Volume columns are detected automatically and
  reduced to the **Close** price.
- **Length:** at least about 320 daily rows — the trend rules need a 252-day lookback plus a 60-day volatility window.
- **What you get:** cumulative-return chart and a performance table for the classical strategies
  (Long Only, SGN, MACD, MACD Long Only) plus Buy & Hold, with a CSV download.
- **Not included:** Lasso, MLP and LSTM — they are trained offline (see above).
- Your uploaded data is processed in memory for your session only and is not saved.
"""
        )

        st.markdown("#### Limitations — read these before drawing conclusions")
        st.markdown(
            """
- **The test window is short:** about 326 trading days (2023-03 to 2024-06) after aligning to the
  LSTM's start. A Sharpe of 3+ on a window this small is encouraging, not proof.
- **One fixed split**, not walk-forward. The paper recalibrates every five years; this project does not.
- **No transaction costs or slippage** are modelled.
- **Little hyperparameter search:** architectures are fixed (MLP 80→40, LSTM 128→80→40) rather than
  tuned by random search, and results come from a single seed.
- **Differences from the paper:** 33 assets instead of 88, a 5-day (not 1-day) shortest lookback,
  and no WaveNet model.
"""
        )

        st.markdown("#### Roadmap")
        st.markdown(
            """
- Walk-forward recalibration and multiple seeds, with confidence intervals on Sharpe
- Transaction-cost-aware backtests and turnover analysis
- Feature importance (SHAP) for the Lasso and MLP signals
- Momentum Transformer and change-point-detection variants (see references)
"""
        )

        st.markdown("#### References")
        st.markdown(
            """
1. Lim, B., Zohren, S. & Roberts, S. — *Enhancing Time Series Momentum Strategies Using Deep Neural Networks*.
2. Wood, K., Roberts, S. & Zohren, S. — *Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection*.
3. Wood, K., Giegerich, S., Roberts, S. & Zohren, S. — *Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture*.
4. Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012) — *Time Series Momentum*, Journal of Financial Economics.
5. Baz, J. et al. (2015) — *Dissecting Investment Strategies in the Cross Section and Time Series*.
"""
        )
        st.caption(config["notebook_note"])

    with right:
        st.markdown(
            f"""
**Contact**

{config['author_name']}
{config['author_tagline']}

[GitHub]({config['github_url']})
[LinkedIn]({config['linkedin_url']})
[{config['email']}](mailto:{config['email']})
"""
        )
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("**Skills**")
        st.markdown(
            "".join(_pill(s) for s in [
                "Quantitative research", "Time series momentum", "Volatility targeting",
                "Backtesting", "Feature engineering", "Custom loss functions",
                "TensorFlow / Keras", "LSTM", "scikit-learn", "pandas / NumPy",
                "Plotly", "Streamlit",
            ]),
            unsafe_allow_html=True,
        )
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("**Project files**")
        st.code(
            "app/                          Streamlit app\n"
            "  app.py                      entry point\n"
            "  neural_section.py           neural strategies view\n"
            "  methodology_section.py      math (paper equations)\n"
            "  about_section.py            this page\n"
            "  tsmom_lite.py               light classical engine\n"
            "data/\n"
            "  Data_V1.xlsx                33-asset price panel\n"
            "  Data_V1 copy.xlsx           extended copy (ES1 notebook)\n"
            "  HistoricalPrices_SP500.csv  S&P 500 index (demo)\n"
            "models/\n"
            "  functions.py                features, models, losses\n"
            "notebooks/\n"
            "  neural_tsmom_research.ipynb research notebook\n"
            "  export_test_returns_cell.py exports daily returns\n"
            "  archive/                    earlier notebooks\n"
            "results/\n"
            "  test_period_returns.csv     daily test returns\n"
            "  figures/                    saved charts\n"
            "papers/                       reference papers\n"
            "requirements.txt              app dependencies",
            language="text",
        )
