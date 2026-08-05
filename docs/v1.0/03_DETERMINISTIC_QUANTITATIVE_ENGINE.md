# EquiMind v1.0: Deterministic Quantitative Engine (`equimind.quantitative`)

EquiMind v1.0 strictly eliminates LLM mathematical hallucinations by running 100% deterministic Python math calculators.

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
- **Alpha & Beta**: Covariance relative to benchmark index returns.
- **30-Day Projected Return Distribution**: Expected return with 95% confidence intervals ($\mu \pm 1.96 \sigma$).
