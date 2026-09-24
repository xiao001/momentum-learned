"""
Methodology tab — follows Lim, Zohren & Roberts, "Enhancing Time Series Momentum
Strategies Using Deep Neural Networks" (Deep Momentum Networks). Equation numbers
in the tags refer to that paper. Where this project differs from the paper, it is
called out in a "Project vs. paper" note.
"""

import pandas as pd
import streamlit as st

PAPER = "Lim, Zohren & Roberts — *Enhancing Time Series Momentum Strategies Using Deep Neural Networks*"


def _note(text):
    st.markdown(
        f'<div style="border-left:4px solid #6c63ff;background:rgba(108,99,255,.08);'
        f'border-radius:8px;padding:10px 14px;margin:8px 0;font-size:.88rem">'
        f'<b>Project vs. paper:</b> {text}</div>',
        unsafe_allow_html=True,
    )


def render_methodology():
    st.caption(f"Formulation follows {PAPER}. Equation tags (1), (2)… match the paper.")

    # ------------------------------------------------------------------ 1
    st.subheader("1 · Time series momentum strategy")
    st.markdown(
        "A TSMOM strategy is defined by a trading rule, or signal, $X_t \\in [-1, 1]$ for each asset. "
        "The combined return from day $t$ to $t+1$ is the volatility-scaled average over the $N_t$ assets:"
    )
    st.latex(r"r^{TSMOM}_{t,t+1}=\frac{1}{N_t}\sum_{i=1}^{N_t}X^{(i)}_t\,\frac{\sigma_{tgt}}{\sigma^{(i)}_t}\,r^{(i)}_{t,t+1}\qquad\text{(1)}")
    st.markdown(
        "- $r^{(i)}_{t,t+1}$ is the one-day return of asset $i$.\n"
        "- $\\sigma_{tgt}=15\\%$ is the annualised volatility target.\n"
        "- $\\sigma^{(i)}_t$ is an ex-ante volatility estimate: an **exponentially weighted moving standard "
        "deviation with a 60-day span** on $r^{(i)}_{t,t+1}$, annualised with $\\sqrt{252}$."
    )
    st.latex(r"\text{weight}^{(i)}_t=X^{(i)}_t\cdot\frac{\sigma_{tgt}}{\sigma^{(i)}_t\sqrt{252}}")
    _note("33 assets (equities, commodities, rates) instead of 88 continuous futures; data 1999–2024 instead of 1990–2015. "
          "At each rebalance date the signal and the volatility term are both held fixed until the next rebalance.")

    # ------------------------------------------------------------------ 2
    st.subheader("2 · Standard trading rules (benchmarks)")
    st.markdown("Each rule has two steps: **trend estimation** $Y_t$, then **position sizing** $X_t$.")

    st.markdown("**Moskowitz et al. (2012) — SGN**")
    st.latex(r"Y^{(i)}_t=r^{(i)}_{t-252,t}\qquad\text{(2)}")
    st.latex(r"X^{(i)}_t=\operatorname{sgn}\big(Y^{(i)}_t\big)\qquad\text{(3)}")
    st.caption("Fully long when the past year's return is positive, fully short when negative.")

    st.markdown("**Baz et al. (2015) — volatility-normalised MACD**")
    st.latex(r"Y^{(i)}_t=\frac{q^{(i)}_t}{\operatorname{std}\big(q^{(i)}_{t-252:t}\big)}\qquad\text{(4)}")
    st.latex(r"q^{(i)}_t=\frac{\text{MACD}(i,t,S,L)}{\operatorname{std}\big(p^{(i)}_{t-63:t}\big)}\qquad\text{(5)}")
    st.latex(r"\text{MACD}(i,t,S,L)=m(i,S)-m(i,L)\qquad\text{(6)}")
    st.markdown(
        "$m(i,S)$ is the exponentially weighted moving average of prices with time-scale $S$, "
        "i.e. half-life $HL=\\log(0.5)/\\log\\!\\big(1-\\tfrac{1}{S}\\big)$. The trend strength is then mapped to a position:"
    )
    st.latex(r"X^{(i)}_t=\varphi\big(Y^{(i)}_t\big),\qquad \varphi(y)=\frac{y\,e^{-y^{2}/4}}{0.89}\qquad\text{(7)}")
    st.caption("Positions grow until |Y| = √2 ≈ 1.41, then shrink back toward zero for larger moves — "
               "the rule de-risks when an asset looks overbought or oversold.")
    st.markdown("Multiple time-scales are combined, with $S_k\\in\\{8,16,32\\}$ and $L_k\\in\\{24,48,96\\}$:")
    st.latex(r"\tilde Y^{(i)}_t=\sum_{k=1}^{3}Y^{(i)}_t(S_k,L_k)\qquad\text{(8)}")
    _note("the benchmark averages the three position sizes $\\varphi\\big(Y(S_k,L_k)\\big)$ rather than averaging the "
          "trend estimates first. Long-only variants clip negative signals to 0.")

    # ------------------------------------------------------------------ 3
    st.subheader("3 · Machine-learning extensions")
    st.markdown(
        "Instead of hand-specifying the trend estimate and the position size, a network learns them "
        "from data. The paper contrasts two routes."
    )
    st.markdown("**Standard supervised learning** — predict the volatility-normalised next return, then take its sign:")
    st.latex(r"Y^{(i)}_t=f\big(u^{(i)}_t;\boldsymbol\theta\big)\qquad\text{(9)}")
    st.latex(r"\mathcal L_{reg}(\boldsymbol\theta)=\frac1M\sum_{\Omega}\Big(Y^{(i)}_t-\frac{r^{(i)}_{t,t+1}}{\sigma^{(i)}_t}\Big)^{2}\qquad\text{(10)}")
    st.latex(r"X^{(i)}_t=\operatorname{sgn}\big(Y^{(i)}_t\big)\qquad\text{(12)}")
    st.caption("Drawback: accuracy is not profit — the loss ignores the size of returns and the risk taken.")

    st.markdown("**Direct outputs** — the network outputs the position itself, and is trained on a "
                "performance metric:")
    st.latex(r"X^{(i)}_t=f\big(u^{(i)}_t;\boldsymbol\theta\big)\qquad\text{(14)}")
    st.markdown("Let $R(i,t)$ be the return captured by the rule, $\Omega$ the set of all $M$ asset–day tuples, "
                "and $\mu_R$, $\sigma_R$ the mean and standard deviation of $R$ over $\Omega$:")
    st.latex(r"R(i,t)=X^{(i)}_t\,\frac{\sigma_{tgt}}{\sigma^{(i)}_t}\,r^{(i)}_{t,t+1}")
    st.latex(r"\mu_R=\frac{1}{M}\sum_{\Omega}R(i,t),\qquad \sigma_R=\sqrt{\frac{1}{M}\sum_{\Omega}R(i,t)^{2}-\mu_R^{2}}")
    st.markdown("🟢 **Return-Loss** — maximise the average return")
    st.latex(r"\mathcal{L}_{returns}(\boldsymbol{\theta})=-\mu_R\qquad\text{(15)}")
    st.markdown("🟠 **Sharpe-Loss** — maximise the annualised Sharpe ratio")
    st.latex(r"\mathcal{L}_{sharpe}(\boldsymbol{\theta})=-\,\frac{\mu_R}{\sigma_R}\sqrt{252}\qquad\text{(16)}")
    st.markdown("Because $X_t=\\tanh(\\cdot)\\in[-1,1]$, gradients flow through the P&L itself and the "
                "network learns trend estimation and position sizing **jointly**.")

    # ------------------------------------------------------------------ 4
    st.subheader("4 · Network architectures")
    st.markdown("**Lasso regression** — a linear model with a $\\tanh$ output and an L1 penalty:")
    st.latex(r"X^{(i)}_t=\tanh\!\big(w^{\top}u^{(i)}_{t-\tau:t}+b\big)\qquad\text{(17)}")
    st.latex(r"\tilde{\mathcal L}(\boldsymbol\theta)=\mathcal L(\boldsymbol\theta)+\alpha\,\lVert w\rVert_1\qquad\text{(18)}")

    st.markdown("**Multilayer perceptron (MLP)** — hidden non-linear layer(s) capture interactions between horizons:")
    st.latex(r"h^{(i)}_t=\tanh\!\big(W_h\,u^{(i)}_{t-\tau:t}+b_h\big)\qquad\text{(19)}")
    st.latex(r"X^{(i)}_t=\tanh\!\big(W_z\,h^{(i)}_t+b_z\big)\qquad\text{(20)}")

    st.markdown("**Long short-term memory (LSTM)** — a recurrent cell keeps a compact summary of the past:")
    for eq, note in [
        (r"f_t=\sigma\left(W_f u_t+V_f h_{t-1}+b_f\right)\qquad\text{(28)}", "Forget gate — what to erase from memory"),
        (r"i_t=\sigma\left(W_i u_t+V_i h_{t-1}+b_i\right)\qquad\text{(29)}", "Input gate — what new information to store"),
        (r"o_t=\sigma\left(W_o u_t+V_o h_{t-1}+b_o\right)\qquad\text{(30)}", "Output gate — what to expose"),
        (r"c_t=f_t\odot c_{t-1}+i_t\odot\tanh\left(W_c u_t+V_c h_{t-1}+b_c\right)\qquad\text{(31)}", "Cell state — the long-term memory"),
        (r"h_t=o_t\odot\tanh\left(c_t\right)\qquad\text{(32)}", "Hidden state"),
        (r"X_t=\tanh\left(W_z h_t+b_z\right)\qquad\text{(33)}", "Position signal in [-1, 1]"),
    ]:
        col_eq, col_note = st.columns([3, 2])
        with col_eq:
            st.latex(eq)
        with col_note:
            st.markdown(f"<div style='padding-top:14px;opacity:.7;font-size:.85rem'>{note}</div>",
                        unsafe_allow_html=True)
    st.caption("⊙ is the element-wise (Hadamard) product and σ the sigmoid. The forget gate controls memory "
               "retention; the input gate adds new information.")

    st.markdown("**Model inputs $u^{(i)}_t$** — two feature families, per asset:")
    st.latex(r"\underbrace{\frac{r^{(i)}_{t-k,t}}{\sigma^{(i)}_t\sqrt{k}}}_{\text{normalised returns}}\;,\qquad "
             r"\underbrace{Y^{(i)}_t(S_k,L_k)}_{\text{MACD indicators, Eq. (4)}}")
    _note("the paper also evaluates a WaveNet (dilated CNN) — not implemented here. It concatenates the past "
          "$\\tau=5$ days of inputs for Lasso/MLP; this project feeds the features of day $t$ only, and gives the LSTM a "
          "252-day window. The shortest normalised-return lookback is 5 days here versus 1 day in the paper.")

    # ------------------------------------------------------------------ 5
    st.subheader("5 · Training and backtest design")
    st.dataframe(
        pd.DataFrame([
            {"Item": "Optimiser", "Paper": "Adam, minibatch SGD, ≤100 epochs", "This project": "Adam (lr 0.001), ≤100 epochs"},
            {"Item": "Validation & stopping", "Paper": "last 10% of each block; early stop after 25 epochs", "This project": "chronological 70/20/10 split; early stop after 10 epochs, best weights restored"},
            {"Item": "Regularisation", "Paper": "dropout as a tuned hyperparameter; L1 for Lasso", "This project": "dropout 0.2 (MLP, LSTM) + recurrent dropout 0.4 (LSTM); L1 α = 0.001 (Lasso)"},
            {"Item": "Hyperparameters", "Paper": "50-iteration random search", "This project": "fixed architectures (MLP 80→40; LSTM 128→80→40)"},
            {"Item": "Recalibration", "Paper": "every 5 years, out-of-sample", "This project": "single fixed split; test set touched once"},
            {"Item": "Feature scaling", "Paper": "—", "This project": "StandardScaler fit on the training block only"},
        ]),
        use_container_width=True, hide_index=True,
    )
    st.markdown(
        "- Each model family is trained twice — **Return-Loss** (15) and **Sharpe-Loss** (16) — to compare "
        "\"maximise return\" with \"maximise risk-adjusted return\".\n"
        "- The LSTM needs a full 252-day window before its first prediction, so its test period starts later; "
        "every comparison including it is aligned to that shorter window."
    )

    st.subheader("6 · Performance metrics")
    st.latex(r"\text{Sharpe}=\frac{\mu_R\sqrt{252}}{\sigma_R},\qquad "
             r"\text{Sortino}=\frac{\mu_R\sqrt{252}}{\text{DD}},\quad \text{DD}=\sqrt{\tfrac1T\sum\min(R_t,0)^2}")
    st.latex(r"\text{Max drawdown}=\min_t\Big(\frac{C_t}{\max_{s\le t}C_s}-1\Big),\qquad C_t=\prod_{s\le t}(1+R_s)")

    st.subheader("Tech stack")
    st.markdown(
        "".join(
            f'<span class="skill-pill">{s}</span>'
            for s in ["Python", "pandas / NumPy", "scikit-learn", "TensorFlow / Keras",
                      "LSTM", "Time Series Momentum", "Volatility Targeting",
                      "Streamlit", "Plotly"]
        ),
        unsafe_allow_html=True,
    )
