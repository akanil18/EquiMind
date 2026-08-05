# EquiMind v1.0: Deterministic Quantitative & Advanced Institutional Engines (`equimind.quantitative`)

EquiMind v1.0 strictly eliminates LLM mathematical hallucinations by running 100% deterministic Python math calculators using `numpy`, `scipy`, and `pandas`.

---

## 📐 Technical Analysis Engine (`equimind.quantitative.technical`)

Calculates technical indicators using `pandas` and `numpy`:
- **RSI (14)**: Relative Strength Index evaluating momentum and overbought vs oversold boundaries.
- **MACD**: 12-fast EMA, 26-slow EMA, 9-signal EMA, and MACD Histogram.
- **Bollinger Bands**: 20-day SMA, Upper band (+2 $\sigma$), Lower band (-2 $\sigma$), and Bandwidth percentage.
- **Moving Averages**: SMA and EMA for 20, 50, 200 periods.
- **ATR (14)**: Average True Range measuring price volatility.
- **Support & Resistance**: Pivot point calculations ($P = (H + L + C) / 3$), $S_1, S_2, R_1, R_2$.

---

## 📊 Fundamental Metrics Engine (`equimind.quantitative.fundamental`)

Computes financial valuation, profitability, health, and bankruptcy risk metrics:
- **Valuation Ratios**: PE ratio, PB ratio, PEG ratio, Free Cash Flow Yield ($FCF / MarketCap$).
- **Profitability Metrics**: Return on Equity (ROE), Return on Assets (ROA), Operating Margin, Net Profit Margin.
- **Financial Health**: Current Ratio ($Assets / Liabilities$), Debt-to-Equity ratio.

### Piotroski F-Score (0 to 9 integer score)
Evaluates 9 discrete financial criteria:
1. Positive Net Income
2. Positive Return on Assets (ROA)
3. Positive Operating Cash Flow
4. Cash Flow > Net Income (Earnings Quality)
5. Lower Long-Term Debt YoY
6. Higher Current Ratio YoY
7. No Shares Dilution
8. Higher Gross Margin YoY
9. Higher Asset Turnover YoY

### Altman Z-Score (Bankruptcy Risk Classification)
$$Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 0.999 X_5$$
- **Safe Zone**: $Z > 2.99$ (Low bankruptcy risk)
- **Grey Zone**: $1.81 \le Z \le 2.99$ (Moderate risk)
- **Distress Zone**: $Z < 1.81$ (High risk)

---

## 🎲 Risk & Return Engine (`equimind.quantitative.risk`)

Calculates statistical risk distributions:
- **Annualized Volatility & Return**: Standard deviation scaled by $\sqrt{252}$.
- **Sharpe Ratio**: $\frac{R_a - R_f}{\sigma_a}$
- **Sortino Ratio**: $\frac{R_a - R_f}{\sigma_{downside}}$
- **Max Drawdown**: Maximum peak-to-trough decline.
- **Daily Value at Risk (VaR 95% & 99%)**: 5th and 1st percentile daily return boundaries.
- **Conditional VaR (CVaR 95%)**: Expected tail loss beyond VaR.

---

## ⏳ Advanced Time Series Research Engine (`equimind.quantitative.time_series`)

- **1D Kalman Filter**: State-space signal noise reduction (`apply_kalman_filter`).
- **Hidden Markov Model (HMM) Market Regime Classifier**: Classifies states into `BULL_TREND`, `BEAR_TREND`, `HIGH_VOLATILITY_SIDEWAYS`.
- **GARCH(1,1) Volatility Model**: Computes conditional volatility estimates.
- **Ensemble Forecast Framework**: Combines trend drift, Kalman filter, and GARCH volatility bounds to output 95% confidence intervals ($\mu \pm 1.96 \sigma$).

---

## 🧪 Alpha Research Laboratory (`equimind.quantitative.alpha_lab`)

- **Information Coefficient (IC)**: Pearson correlation between factor signals and forward asset returns.
- **Rank IC**: Spearman rank correlation for non-linear alpha relationships.
- **Factor Evaluation & Decay**: Evaluates signal decay half-life and flags statistical significance ($|Rank IC| \ge 0.05$ and $|Sharpe| \ge 1.0$).
- **Alpha Factor Ranker**: Ranks candidate alpha factors across `MOMENTUM`, `VALUE`, `QUALITY`, `ALTERNATIVE`, `DEVELOPER_VELOCITY`, `MACRO_SENSITIVITY`.

---

## 🔬 Feature Engineering Platform & FeatureStore (`equimind.features`)

- **Evidence Feature Extractor**: Transforms evidence nodes into numerical feature vectors (`avg_sentiment_score`, `bullish_sentiment_ratio`, `avg_credibility`, `verified_official_count`).
- **Price Feature Extractor**: Converts price series into rolling statistical features (`price_return_1d`, `rolling_volatility`, `price_momentum_ratio`).
- **Z-Score Normalization**: Standardizes raw features into zero-mean, unit-variance vectors.

---

## 🎯 Structural Causal Reasoning Engine (`equimind.quantitative.causal_engine`)

- **Do-Calculus Interventions ($P(Y | \text{do}(X))$)**: Partial regression controlling for confounding variable $Z$.
- **Spurious Correlation Classifier**: Eliminates false market correlations driven by shared macro/sector confounders rather than direct mechanisms.

---

## 🎲 Monte Carlo Stochastic Simulator (`equimind.quantitative.monte_carlo`)

- **Geometric Brownian Motion & Jump-Diffusion Simulator**: Simulates 1,000+ stochastic price paths incorporating drift, volatility, and jump shocks.
- **Risk/Reward Boundaries**: Computes downside risk (P05) and upside reward (P95) thresholds.
- **Probability of Profit ($P(S_T > S_0)$)**: Probability of positive final return.

---

## 💼 Portfolio Construction & Optimization Engine (`equimind.quantitative.portfolio_optimizer`)

- **Markowitz Mean-Variance Tangency Solver (`MEAN_VARIANCE`)**: Maximizes portfolio Sharpe ratio ($\max_w \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}$).
- **Inverse-Volatility Risk Parity Allocator (`RISK_PARITY`)**: Equalizes risk contribution ($w_i \propto 1 / \sigma_i$).
- **Black-Litterman Model (`BLACK_LITTERMAN`)**: Blends equilibrium market returns with investor/agent view vectors.
- **Kelly Criterion Position Sizer (`KELLY_CRITERION`)**: Fractional Kelly allocation ($f^* = \frac{p \cdot b - q}{b}$).
- **Herfindahl Diversification Score**: Measures portfolio concentration ($1 - \sum w_i^2$).
