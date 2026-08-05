# EquiMind v1.0: Reasoning Planner & Research Subagents

EquiMind v1.0 dynamically constructs custom research execution plans tailored to company sector and investment query.

---

## 🎯 Dynamic Reasoning Planner Agent (`equimind.planner`)

The `ReasoningPlanner` analyzes user query, ticker symbol, and sector heuristics before any research agent runs. It generates a custom `ResearchPlan`:

```python
class ResearchPlan(BaseModel):
    ticker: str
    sector: SectorType                # SEMICONDUCTOR_TECH, BANKING_FINANCE, SAAS_SOFTWARE, etc.
    horizon: InvestmentHorizon        # DAY_TRADING, SHORT_TERM, MEDIUM_TERM, LONG_TERM
    active_teams: List[str]           # ["market_data", "fundamentals", "macro", "web_intelligence"]
    active_adapters: List[str]        # ["sec_filings", "financial_news", "reddit", "twitter_x", "github_commits"]
    focus_areas: List[str]            # ["Capex & Fab Utilization", "GPU/AI Demand", "R&D Spend"]
```

### Sector Customization Rules
- **Semiconductor / Tech**: Activates `github_commits`, `job_postings`, focusing on Fab utilization, GPU demand, and developer adoption.
- **Banking & Financials**: Focuses on Net Interest Margin (NIM), Tier 1 Capital Ratio, Non-Performing Assets (NPA), and Fed rate sensitivity. Skips developer commit monitoring.
- **Airlines & Transport**: Focuses on Jet Fuel prices (Brent crude), Load Factor, and Debt servicing.
- **SaaS & Cloud**: Focuses on Annual Recurring Revenue (ARR), Net Retention Rate (NRR), CAC payback, and GitHub repository star growth.

---

## 👥 Specialized Research Teams (`equimind.teams`)

1. **`MarketDataTeam`**: Fetches prices, orderbook/liquidity, benchmark indices, computes `TechnicalEngine` metrics, and builds `MARKET_PRICES` EvidenceNodes.
2. **`FundamentalTeam`**: Financial statements, balance sheets, cash flows, SEC 10-K/10-Q metrics, Piotroski F-Score (0-9), and Altman Z-Score bankruptcy zones.
3. **`MacroTeam`**: CPI inflation, Fed funds rate, GDP growth, Brent Crude Oil, Gold, VIX index, and FX rates.
4. **`WebIntelligenceTeam`**: Operates platform-specific adapters for SEC EDGAR filings, Bloomberg/Reuters news, Reddit r/stocks, Twitter/X analyst feeds, GitHub developer commits, and Earnings Call transcripts.

---

## ⏳ Temporal Guard / Time-Machine Backtesting (`equimind.time_machine`)

The `TemporalGuard` enforces temporal cutoffs for historical simulation:

```python
with TemporalGuard(as_of_date=as_of_dt) as guard:
    raw_nodes = team.research(ticker="TSLA", query=query)
    filtered_nodes = guard.filter_evidence(raw_nodes)
```

Any evidence node published after `as_of_date` (e.g. `2024-01-01`) is strictly pruned to evaluate research quality without future data leakage.
