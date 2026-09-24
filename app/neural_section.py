"""
Neural TSMOM Strategies — detail view for the "6 Neural TSMOM Strategies" card.

Renders: hero banner -> feature-engineering walkthrough (notebook 3.1) ->
one tab per model family (notebook 3.2 Lasso, 3.3 MLP, 3.5 LSTM), each showing
architecture, loss maths, and results for Return-Loss vs Sharpe-Loss.

Usage in app.py:
    from neural_section import render_neural_section
    render_neural_section(RESEARCH_RESULTS)
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ACCENT = {"Lasso": "#2a9d8f", "MLP": "#6c63ff", "LSTM": "#e76f51"}
LOSS_COLOR = {"Return-Loss": "#1f6f54", "Sharpe-Loss": "#c9892b"}

CSS = """
<style>
.nt-hero {
    border-radius: 18px; padding: 28px 32px; margin-bottom: 8px; color: #fff;
    background: linear-gradient(120deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
    position: relative; overflow: hidden;
}
.nt-hero:after {
    content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
    border-radius: 50%; background: radial-gradient(circle, rgba(108,99,255,.55), transparent 70%);
}
.nt-hero .big {font-size: 3.2rem; font-weight: 800; line-height: 1;}
.nt-hero .ttl {font-size: 1.35rem; font-weight: 700; margin-top: 4px;}
.nt-hero .sub {opacity: .8; font-size: .95rem; margin-top: 4px;}
.nt-chip {
    display: inline-block; padding: 3px 12px; margin: 10px 6px 0 0; border-radius: 999px;
    font-size: .75rem; font-weight: 700; background: rgba(255,255,255,.14); color: #fff;
    border: 1px solid rgba(255,255,255,.25);
}
.nt-flow {display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 14px 0;}
.nt-step {
    flex: 1 1 150px; border: 1px solid rgba(128,128,128,.28); border-radius: 12px;
    padding: 12px 14px; position: relative;
}
.nt-step .n {
    display: inline-block; width: 24px; height: 24px; line-height: 24px; text-align: center;
    border-radius: 50%; background: #6c63ff; color: #fff; font-weight: 700; font-size: .8rem;
}
.nt-step .h {font-weight: 700; margin-top: 6px; font-size: .92rem;}
.nt-step .d {opacity: .7; font-size: .78rem; margin-top: 2px;}
.nt-card {
    border: 1px solid rgba(128,128,128,.28); border-left-width: 5px; border-radius: 12px;
    padding: 14px 18px; margin: 8px 0;
}
.nt-card .k {font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; opacity: .6;}
.nt-card .v {font-size: 1.7rem; font-weight: 800; line-height: 1.15;}
.nt-card .s {font-size: .8rem; opacity: .65;}
.nt-layer {
    text-align: center; border-radius: 10px; padding: 8px 10px; margin: 0 auto; color: #fff;
    font-size: .85rem; font-weight: 600;
}
.nt-arrow {text-align: center; opacity: .45; line-height: 1; margin: 3px 0; font-size: .9rem;}
.nt-callout {
    border-radius: 10px; padding: 12px 16px; margin: 10px 0; font-size: .9rem;
    background: rgba(108,99,255,.09); border: 1px solid rgba(108,99,255,.3);
}
</style>
"""


def _card(label, value, sub, color):
    return (f'<div class="nt-card" style="border-left-color:{color}">'
            f'<div class="k">{label}</div><div class="v" style="color:{color}">{value}</div>'
            f'<div class="s">{sub}</div></div>')


def _stack(layers, color):
    """Vertical architecture diagram. layers = [(label, width_pct, opacity)]."""
    html = []
    for i, (label, width, alpha) in enumerate(layers):
        if i:
            html.append('<div class="nt-arrow">▼</div>')
        html.append(
            f'<div class="nt-layer" style="width:{width}%;background:{color};opacity:{alpha}">{label}</div>'
        )
    return "".join(html)


def _network_fig(layers, color, note=None):
    """Node-and-edge network diagram.
    layers = [(title, subtitle, n_nodes_drawn, role, symbol)]; role in input|hidden|output.
    Only a subset of nodes is drawn per layer (real widths are in the titles)."""
    n_max = max(l[2] for l in layers)
    xs = [i * 2.2 for i in range(len(layers))]
    node_col = {"input": "#1f6bff", "hidden": color, "output": "#8fa8ff"}
    fig = go.Figure()

    ys = []
    for n in (l[2] for l in layers):
        ys.append([(n - 1) / 2 - k for k in range(n)])

    ex, ey = [], []
    for i in range(len(layers) - 1):
        for y0 in ys[i]:
            for y1 in ys[i + 1]:
                ex += [xs[i], xs[i + 1], None]
                ey += [y0, y1, None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip", showlegend=False,
                             line=dict(color=color, width=0.7), opacity=0.28))

    for i, (title, sub, n, role, symbol) in enumerate(layers):
        fig.add_trace(go.Scatter(
            x=[xs[i]] * n, y=ys[i], mode="markers", showlegend=False,
            marker=dict(size=26, color=node_col[role], symbol=symbol,
                        line=dict(color="white", width=2)),
            hovertext=f"{title} — {sub}", hoverinfo="text",
        ))
        top = (n_max - 1) / 2 + 1.05
        fig.add_annotation(x=xs[i], y=top, text=f"<b>{title}</b>", showarrow=False,
                           font=dict(size=12), yanchor="bottom")
        fig.add_annotation(x=xs[i], y=-top + 0.15, text=sub, showarrow=False,
                           font=dict(size=11, color="#888"), yanchor="top")
    if note:
        fig.add_annotation(x=(xs[0] + xs[-1]) / 2, y=(n_max - 1) / 2 + 2.3, text=note,
                           showarrow=False, font=dict(size=11, color=color))
    fig.update_layout(
        height=360, margin=dict(l=5, r=5, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-0.8, xs[-1] + 0.8]),
        yaxis=dict(visible=False, range=[-(n_max - 1) / 2 - 2.2, (n_max - 1) / 2 + 3.0]),
    )
    return fig


def _metric_pair(results, family, color_note=True):
    cols = st.columns(2)
    for col, (loss, row) in zip(cols, [("Return-Loss", f"{family} + Loss"),
                                        ("Sharpe-Loss", f"{family} + Sharpe-Loss")]):
        r = results.loc[row]
        with col:
            st.markdown(
                _card(f"{family} · {loss}", f'{r["Sharpe Ratio"]:.2f}',
                      f'Sharpe · Return {r["E[Return] %"]:.2f}% · Vol {r["Vol. %"]:.2f}% · '
                      f'Sortino {r["Sortino Ratio"]:.2f}',
                      LOSS_COLOR[loss]),
                unsafe_allow_html=True,
            )


def _loss_block():
    st.markdown("**Training objective — both losses act on the strategy's own P&L**")
    st.caption("Same notation as the Methodology tab (Eqs. 15–16, Lim, Zohren & Roberts).")
    st.latex(r"R(i,t)=X^{(i)}_t\,\frac{\sigma_{tgt}}{\sigma^{(i)}_t}\,r^{(i)}_{t,t+1},\qquad \sigma_{tgt}=15\%")
    st.latex(r"\mu_R=\frac{1}{M}\sum_{\Omega}R(i,t),\qquad \sigma_R=\sqrt{\frac{1}{M}\sum_{\Omega}R(i,t)^{2}-\mu_R^{2}}")
    st.markdown("🟢 **Return-Loss** — maximise the average return")
    st.latex(r"\mathcal{L}_{returns}(\boldsymbol{\theta})=-\mu_R\qquad\text{(15)}")
    st.caption("Simple, but can take large, noisy positions.")
    st.markdown("🟠 **Sharpe-Loss** — maximise the annualised Sharpe ratio")
    st.latex(r"\mathcal{L}_{sharpe}(\boldsymbol{\theta})=-\,\frac{\mu_R}{\sigma_R}\sqrt{252}\qquad\text{(16)}")
    st.caption("Penalizes volatile signals, so positions stay calmer.")


def _feature_engineering():
    st.markdown("### 🧬 Feature engineering")
    st.caption("Notebook §3.1 — how raw prices become model inputs. Open each block for detail.")

    st.markdown(
        """
<div class="nt-flow">
  <div class="nt-step"><span class="n">1</span><div class="h">Normalized returns</div><div class="d">5 lookbacks per asset</div></div>
  <div class="nt-step"><span class="n">2</span><div class="h">MACD signals</div><div class="d">3 half-life pairs</div></div>
  <div class="nt-step"><span class="n">3</span><div class="h">Targets</div><div class="d">next-day return &amp; σ</div></div>
  <div class="nt-step"><span class="n">4</span><div class="h">Split &amp; scale</div><div class="d">70 / 20 / 10, train-only scaler</div></div>
  <div class="nt-step"><span class="n">5</span><div class="h">Sequences</div><div class="d">252-day windows (LSTM)</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("① Normalized lagged returns — momentum at 5 horizons", expanded=False):
        st.markdown(
            "Returns over the past week, 1-month, 3-month, 6-month and 1-year periods "
            "($k = 5, 21, 63, 126, 252$ days) are normalised by a measure of daily volatility "
            "scaled to an appropriate time scale (Lim, Zohren & Roberts):"
        )
        st.latex(r"\frac{r^{(i)}_{t-k,t}}{\sigma^{(i)}_t\sqrt{k}}\qquad\text{e.g. annual: }\;\frac{r^{(i)}_{t-252,t}}{\sigma^{(i)}_t\sqrt{252}}")
        st.markdown(
            "- $r^{(i)}_{t-k,t}$ is the $k$-day return of asset $i$ (compounded daily returns) and "
            "$\\sigma^{(i)}_t$ is the **60-day exponentially-weighted daily volatility** \u2014 the same estimate used for position sizing.\n"
            "- Dividing by $\\sigma_t\\sqrt{k}$ rescales a $k$-day move to daily-vol units, "
            "so a copper move and a T-note move are **comparable** and one model can learn across 33 assets.\n"
            "- Five horizons give the model a term-structure of trend, from short-term reversal to "
            "long-term momentum.\n"
            "- *Difference from the paper:* it uses a 1-day lookback as the shortest horizon; this "
            "model uses 5 days (one week)."
        )

    with st.expander("② MACD trend features — smoothed crossover signals", expanded=False):
        st.markdown(
            "Three short/long pairs $(S,L) \\in \\{(8,24),(16,48),(32,96)\\}$ (Baz et al. 2015), "
            "each converted to a half-life and normalized twice:"
        )
        st.latex(r"q_t=\frac{\text{EWMA}_S(p)-\text{EWMA}_L(p)}{\text{std}_{63}(p)},\qquad y_t=\frac{q_t}{\text{std}_{252}(q)}")
        st.markdown(
            "- The 63-day price std removes price-level effects; the 252-day std of $q$ puts every asset on a unit scale.\n"
            "- Unlike the raw-return features these are **smooth**, so they add trend information "
            "with less day-to-day noise.\n"
            "- Concatenated with block ①: **5 + 3 = 8 features per asset**, stacked across the whole universe."
        )

    with st.expander("③ Targets — what the model is trying to predict", expanded=False):
        st.markdown(
            "The model does **not** predict a price. For every day $t$ it receives features and "
            "outputs a position signal $X_t \\in [-1,1]$ per asset. The loss is then evaluated on:\n"
            "- the **next-day return** $r_{t+1}$, and\n"
            "- the **current volatility** $\\sigma_t$ (used for position sizing).\n\n"
            "Both are packed side-by-side into one target tensor `[returns | sigma]`. "
            "Because $r_{t+1}$ is looked up **positionally** in the return index, market holidays "
            "can never cause a mis-aligned or look-ahead label."
        )

    with st.expander("④ Chronological split & leak-free scaling", expanded=False):
        st.markdown(
            "- **70 / 20 / 10** train / validation / test, in time order — no shuffling.\n"
            "- `StandardScaler` is **fit on the training block only**, then applied to validation and test, "
            "so no future statistics leak backwards.\n"
            "- Validation loss drives **early stopping** (patience 10, best weights restored); "
            "the test block is touched once, at the end."
        )
        split = go.Figure()
        for name, w, c in [("Train 70%", 70, "#2a9d8f"), ("Validation 20%", 20, "#e9c46a"), ("Test 10%", 10, "#e76f51")]:
            split.add_trace(go.Bar(y=["Timeline"], x=[w], orientation="h", name=name,
                                   marker_color=c, text=name, textposition="inside"))
        split.update_layout(barmode="stack", height=90, margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(split, use_container_width=True)

    with st.expander("⑤ 252-day sequences — LSTM input only", expanded=False):
        st.markdown(
            "Lasso and MLP see **one flat feature vector per day**. The LSTM instead receives a "
            "rolling window of the last `seq_length = 252` days:"
        )
        st.code("X_seq[i] = features[i : i+252]     # shape (252, n_features)\n"
                "y_seq[i] = [return, sigma][i+251]  # label of the window's last day", language="python")
        st.markdown(
            "The first prediction therefore needs 252 days of history, so **LSTM's test window starts "
            "251 days later** than the other models. Every comparison involving LSTM is re-aligned to "
            "that shorter window."
        )


BENCH_ROWS = ["Long Only", "Long+Short (SGN)", "MACD", "Buy & Hold"]

# name in test_period_returns.csv -> (colour, dash, width)
SERIES_STYLE = {
    "Long Only": ("#8d99ae", "dot", 1.6),
    "SGN": ("#adb5bd", "dot", 1.6),
    "MACD": ("#6c757d", "dot", 1.6),
    "MACD Long Only": ("#495057", "dot", 1.6),
    "Buy and Hold": ("#000000", "dash", 2.2),
    "Lasso (Return Loss)": ("#2a9d8f", "solid", 2.2),
    "Lasso (Sharpe Loss)": ("#2a9d8f", "dash", 2.2),
    "MLP (Return Loss)": ("#6c63ff", "solid", 2.6),
    "MLP (Sharpe Loss)": ("#6c63ff", "dash", 2.2),
    "LSTM (Return Loss)": ("#e76f51", "solid", 2.2),
    "LSTM (Sharpe Loss)": ("#e76f51", "dash", 2.2),
}
RETURNS_CSV = Path(__file__).resolve().parent.parent / "results" / "test_period_returns.csv"


@st.cache_data(show_spinner=False)
def _load_test_returns(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Performance table computed from the same daily series the chart plots."""
    cum = (1 + df).cumprod()
    down = df.clip(upper=0)
    out = pd.DataFrame({
        "Cum. Return %": (cum.iloc[-1] - 1) * 100,
        "E[Return] %": df.mean() * 252 * 100,
        "Vol. %": df.std() * (252 ** 0.5) * 100,
        "Sharpe": df.mean() / df.std() * (252 ** 0.5),
        "Sortino": df.mean() / (down.pow(2).mean() ** 0.5) * (252 ** 0.5),
        "Max DD %": (cum / cum.cummax() - 1).min() * 100,
        "% Positive Days": (df > 0).mean() * 100,
    })
    return out.sort_values("Sharpe", ascending=False)


def render_comparison(key="nt_cum", default=None, title="#### Daily cumulative return — test period, all 11 strategies"):
    """Cumulative-return chart + metrics table. Call twice with different `key`s to show it in two places."""
    st.markdown(title)
    if not RETURNS_CSV.exists():
        st.info(
            "Daily strategy returns not found. Run the export cell from the notebook once "
            "(it writes `results/test_period_returns.csv`), then refresh this page."
        )
        return
    df = _load_test_returns(str(RETURNS_CSV), RETURNS_CSV.stat().st_mtime)
    cum = (1 + df).cumprod()

    picked = st.multiselect(
        "Strategies shown", list(cum.columns), default=[c for c in (default or cum.columns) if c in cum.columns], key=f"{key}_pick",
    )
    fig = go.Figure()
    for name in picked:
        color, dash, width = SERIES_STYLE.get(name, ("#888", "solid", 1.8))
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[name], name=name, mode="lines",
            line=dict(color=color, dash=dash, width=width),
            hovertemplate="%{y:.3f}<extra>" + name + "</extra>",
        ))
    fig.add_hline(y=1, line_width=1, line_color="rgba(128,128,128,.5)")
    fig.update_layout(
        height=520, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Cumulative return (start = 1)", xaxis_title="Date",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_fig")
    st.markdown("#### Performance comparison — same window, same series")
    st.dataframe(
        _metrics_table(df[picked]).style
        .format("{:.2f}")
        .background_gradient(subset=["Sharpe", "Sortino"], cmap="RdYlGn")
        .background_gradient(subset=["Max DD %"], cmap="RdYlGn"),
        use_container_width=True,
    )
    st.caption(
        "Dotted grey = rule-based benchmarks · dashed black = Buy & Hold · "
        "solid = Return-Loss models · dashed colour = Sharpe-Loss models. All series are aligned "
        "to the LSTM test window (shortest) for a like-for-like comparison."
    )


def render_neural_section(results: pd.DataFrame):
    st.markdown(CSS, unsafe_allow_html=True)

    best = results.loc[[f"{m} + {l}" for m in ("Lasso", "MLP", "LSTM") for l in ("Loss", "Sharpe-Loss")]] \
        ["Sharpe Ratio"].idxmax()
    st.markdown(
        f"""
<div class="nt-hero">
  <div class="big">6</div>
  <div class="ttl">Neural TSMOM Strategies</div>
  <div class="sub">Lasso, MLP, LSTM × Return/Sharpe-Loss — a learned signal replaces the rule-based one</div>
  <span class="nt-chip">3 model families</span><span class="nt-chip">2 loss functions</span>
  <span class="nt-chip">33 assets</span><span class="nt-chip">Top Sharpe · {best}</span>
</div>
""",
        unsafe_allow_html=True,
    )

    _feature_engineering()

    st.markdown("### 🤖 Models")
    tab_lasso, tab_mlp, tab_lstm, tab_cmp = st.tabs(
        ["3.2 · Lasso", "3.3 · MLP", "3.5 · LSTM", "⚖️ Compare All Strategies"]
    )

    # ---- Lasso ------------------------------------------------------------
    with tab_lasso:
        c = ACCENT["Lasso"]
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**Linear baseline with L1 sparsity**")
            st.markdown(
                "In the simplest case a standard linear model generates the prediction. For direct "
                "position outputs the activation $g(\\cdot)$ is $\\tanh$, so the signal stays in $[-1,1]$:"
            )
            st.latex(r"Z^{(i)}_t=g\big(w^{\top}u^{(i)}_{t-\tau:t}+b\big),\quad Z^{(i)}_t=X^{(i)}_t,\; g=\tanh\qquad\text{(17)}")
            st.markdown(
                "$w$ is the weight vector, $b$ the bias and $u^{(i)}_t$ the input features. Training adds an "
                "**L1 regulariser** to the Return-Loss (15) or Sharpe-Loss (16), which drives uninformative "
                "features to zero:"
            )
            st.latex(r"\tilde{\mathcal{L}}(\boldsymbol{\theta})=\mathcal{L}(\boldsymbol{\theta})+\alpha\,\lVert w\rVert_1\qquad\text{(18)}")
            st.markdown(
                "Here $\\alpha=0.001$. The paper concatenates the past $\\tau=5$ days of inputs, "
                "$u_{t-\\tau:t}=[u_{t-\\tau}^{\\top},\\dots,u_t^{\\top}]^{\\top}$; this project uses the features of "
                "day $t$ only."
            )
            st.caption("Adam · lr 0.001 · ≤100 epochs · early stopping (patience 10)")
        with right:
            st.plotly_chart(_network_fig([
                ("Input", "flat features", 6, "input", "circle"),
                ("Dense · tanh", "one signal per asset", 4, "hidden", "circle"),
                ("Signal X", "∈ [-1, 1]", 4, "output", "circle"),
            ], c, note="L1 penalty on weights W  (α = 0.001)"), use_container_width=True)
        _metric_pair(results, "Lasso")
        st.markdown('<div class="nt-callout">💡 <b>Read:</b> the Sharpe-loss version nearly triples the '
                    'Sharpe of the return-loss version — even a linear model benefits from being told '
                    'to care about risk.</div>', unsafe_allow_html=True)
        _loss_block()

    # ---- MLP --------------------------------------------------------------
    with tab_mlp:
        c = ACCENT["MLP"]
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**Adding non-linearity with a hidden layer**")
            st.markdown(
                "Increasing model complexity slightly, a 2-layer neural network (one hidden layer plus "
                "an output layer) can capture non-linear effects that a linear model cannot."
            )
            st.latex(r"h^{(i)}_t=\tanh\left(W_h\,u^{(i)}_{t-\tau:t}+b_h\right)\qquad\text{(19)}")
            st.latex(r"Z^{(i)}_t=g\left(W_z\,h^{(i)}_t+b_z\right),\quad Z^{(i)}_t=X^{(i)}_t,\; g=\tanh\qquad\text{(20)}")
            st.markdown(
                "- $h^{(i)}_t$ is the **hidden state**: the inputs are mixed by $W_h$ and passed through an "
                "internal $\\tanh$, so each hidden unit can respond to a *combination* of features.\n"
                "- $W_\\cdot$ and $b_\\cdot$ are the layer **weight matrices and biases**.\n"
                "- The output activation $g$ depends on the task; for direct position outputs it is "
                "$\\tanh$, keeping $X_t\\in[-1,1]$.\n"
                "- *Example:* a strong quarterly trend with a weak weekly bounce can mean something "
                "different from either signal alone — the hidden layer can represent that."
            )
            st.caption(
                "Project vs. paper: this model stacks **two** hidden layers (80 → 40 units) with "
                "dropout 0.2 after each, uses the features of day t only, and is trained with the "
                "Return-Loss (15) or Sharpe-Loss (16)."
            )
            st.caption("Adam · lr 0.001 · ≤100 epochs · early stopping (patience 10)")
        with right:
            st.plotly_chart(_network_fig([
                ("Input", "flat features", 6, "input", "circle"),
                ("Dense 80", "tanh · dropout 0.2", 6, "hidden", "circle"),
                ("Dense 40", "tanh · dropout 0.2", 5, "hidden", "circle"),
                ("Dense · tanh", "n_assets", 4, "hidden", "circle"),
                ("Signal X", "∈ [-1, 1]", 4, "output", "circle"),
            ], c), use_container_width=True)
        _metric_pair(results, "MLP")
        st.markdown('<div class="nt-callout">🏆 <b>Read:</b> MLP + Return-Loss delivers the best '
                    'annualized return (10.83%) and Sharpe of the study; the Sharpe-loss variant '
                    'trades a little return for the lowest volatility and the most positive days.</div>',
                    unsafe_allow_html=True)
        _loss_block()

    # ---- LSTM -------------------------------------------------------------
    with tab_lstm:
        c = ACCENT["LSTM"]
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**A recurrent memory over a full trading year**")
            st.markdown(
                "Traditionally used for sequence prediction in language processing, recurrent networks "
                "— specifically **long short-term memory (LSTM)** cells — are increasingly used for "
                "time series. Instead of a single day's snapshot, this model reads the last **252 days** of "
                "features one step at a time and updates an internal memory at each step, so it can, "
                "in principle, learn *how* a trend evolved."
            )
            st.markdown(
                "- The **cell state** $c_t$ is a compact summary of the past.\n"
                "- The **forget gate** controls memory retention; the **input gate** adds new information.\n"
                "- Two stacked LSTM layers with recurrent dropout (0.4) guard against overfitting on a "
                "small sample."
            )
            st.caption("Input shape (252, n_features) · Adam · lr 0.001 · early stopping (patience 10)")
            st.markdown('<div class="nt-callout">⚠️ Predictions start 251 days later than the other '
                        'models and are scored on a shorter, later test window — so LSTM is compared '
                        'on a different (harder) slice.</div>', unsafe_allow_html=True)
        with right:
            st.plotly_chart(_network_fig([
                ("Input", "252 days × features", 6, "input", "circle"),
                ("LSTM 128", "return sequences · dropout 0.2", 6, "hidden", "square"),
                ("LSTM 80", "dropout 0.4", 5, "hidden", "square"),
                ("Dense 40", "tanh · dropout 0.2", 5, "hidden", "circle"),
                ("Dense · tanh", "n_assets", 4, "hidden", "circle"),
                ("Signal X", "∈ [-1, 1]", 4, "output", "circle"),
            ], c, note="■ = recurrent (LSTM) units · ● = dense units"), use_container_width=True)
        st.markdown("**LSTM equations** (Lim, Zohren & Roberts, Eqs. 28–33)")
        for eq, note in [
            (r"f^{(i)}_t=\sigma\left(W_f u^{(i)}_t+V_f h^{(i)}_{t-1}+b_f\right)\qquad\text{(28)}",
             "Forget gate: how much of the old memory to keep (0 = erase, 1 = keep)."),
            (r"i^{(i)}_t=\sigma\left(W_i u^{(i)}_t+V_i h^{(i)}_{t-1}+b_i\right)\qquad\text{(29)}",
             "Input gate: how much of the new candidate information to write."),
            (r"o^{(i)}_t=\sigma\left(W_o u^{(i)}_t+V_o h^{(i)}_{t-1}+b_o\right)\qquad\text{(30)}",
             "Output gate: how much of the memory to reveal as the hidden state."),
            (r"c^{(i)}_t=f^{(i)}_t\odot c^{(i)}_{t-1}+i^{(i)}_t\odot\tanh\left(W_c u^{(i)}_t+V_c h^{(i)}_{t-1}+b_c\right)\qquad\text{(31)}",
             "Cell state: old memory scaled by the forget gate, plus new information scaled by the input gate."),
            (r"h^{(i)}_t=o^{(i)}_t\odot\tanh\left(c^{(i)}_t\right)\qquad\text{(32)}",
             "Hidden state: the filtered view of memory passed to the next step and to the output."),
            (r"Z^{(i)}_t=g\left(W_z h^{(i)}_t+b_z\right),\quad Z^{(i)}_t=X^{(i)}_t,\; g=\tanh\qquad\text{(33)}",
             "Output: the position signal, kept in [-1, 1] by tanh."),
        ]:
            st.latex(eq)
            st.caption(note)
        st.markdown(
            "$\\odot$ is the Hadamard (element-wise) product, $\\sigma(\\cdot)$ the sigmoid, and $W_\\cdot$, $V_\\cdot$ "
            "the weight matrices applied to the input $u_t$ and to the previous hidden state $h_{t-1}$. "
            "Sequentially updating these memory states at every step lets the LSTM learn "
            "representations of long-term relationships relevant to the prediction task."
        )
        st.caption(
            "Project vs. paper: this model stacks two LSTM layers (128 → 80 units), followed by a dense "
            "layer (40) before the output."
        )
        _metric_pair(results, "LSTM")
        st.markdown('<div class="nt-callout">📉 <b>Read:</b> both LSTM variants lose money out of '
                    'sample. More capacity and a longer memory did not help here — with ~one '
                    'independent regime per decade, a simpler model generalizes better.</div>',
                    unsafe_allow_html=True)
        _loss_block()

    # ---- Compare ----------------------------------------------------------
    with tab_cmp:
        render_comparison()
