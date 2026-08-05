# EquiMind: Autonomous AI Equity Research Firm & Financial Research Orchestration Framework

## Core Philosophy & Vision
EquiMind is a model-agnostic, deterministic-quantitative, multi-agent financial research orchestration framework. Rather than a simple stock prediction bot that outputs "Buy" or "Sell", EquiMind operates like an institutional investment committee and quantitative research firm.

It systematically gathers multi-source evidence, computes deterministic mathematical models, executes adversarial debate (Bull vs. Bear vs. Judge), prunes context without extra LLM overhead, maintains hierarchical temporal memory, and delivers 100% traceable, explainable investment recommendations backed by verifiable proof.

---

## Key System Design Pillars

### 1. Model-Agnostic LLM Provider Layer
- Unified abstraction over OpenAI, Anthropic Claude, Google Gemini, DeepSeek, Qwen, Llama, Ollama, OpenRouter, and custom local models.
- Swappable reasoning engines without touching orchestration, memory, tools, or debate logic.

### 2. Dynamic Reasoning Planner Agent
- Intelligently constructs custom research DAGs based on asset type, sector, user investment horizon, and company profile.
- Skips irrelevant tools/agents (e.g. avoiding mining metrics for a SaaS stock or SaaS ARR metrics for a bank).

### 3. Specialized Research Teams
- **Market Data Team**: Price history, volume, liquidity, order book, benchmark indices.
- **Fundamental Analysis Team**: Balance sheets, income statements, cash flow, SEC filings (10-K, 10-Q), ratios, insider trades, institutional ownership.
- **Macroeconomic Team**: Inflation (CPI), Interest rates (Fed/Central Banks), GDP, Unemployment, Oil, Gold, FX rates, VIX, global macro events.
- **Alternative & Internet Intelligence Team**: Multi-platform adapters for Reddit, Twitter/X, Stocktwits, SEC filings, Earnings Call transcripts, News feeds, Govt announcements, Tech/GitHub signals, LinkedIn/Job posting trends.

### 4. Deterministic Quantitative Engine (No LLM Math Hallucinations)
- Technical Indicators: RSI, MACD, Bollinger Bands, Moving Averages (SMA/EMA), ATR, Support/Resistance, Volatility, Liquidity.
- Fundamental Metrics: PE, PB, PEG, ROE, ROA, EPS Growth, Free Cash Flow Yield, Debt-to-Equity, Operating Margins, Piotroski F-Score, Altman Z-Score.
- Probabilistic Risk/Return Models: Expected return distributions, Volatility/VaR, Confidence intervals.
- The LLM *never* performs raw mathematical calculations—it strictly interprets deterministic numerical outputs.

### 5. Investment Committee & Adversarial Debate Architecture
- **Bull Agent**: Formulates evidence-backed thesis supporting investment.
- **Bear Agent**: Formulates evidence-backed thesis opposing investment & highlighting risk.
- **Judge Agent**: Critically evaluates evidence strength, filters unbacked claims, weighs conflicting signals, and synthesizes a balanced recommendation.

### 6. Hierarchical Memory & Delta-Research Engine
- Multi-tier memory abstraction: Raw Observations -> Daily Summaries -> Weekly Syntheses -> Monthly Investment Theses -> Quarterly Reports -> Persistent Entity Knowledge Base.
- Delta Research: For existing tickers, inspects previous reports, identifies timestamp diff, and updates only new/changed evidence.

### 7. Runtime Context Optimization & Compression
- In-memory deterministic compression: deduplication, semantic clustering, time-decay scoring, relevance ranking, and context-budget packing.
- Avoids extra LLM summarization API calls while preventing context window overflow.

### 8. Structured Evidence Graph & Provenance Traceability
- Every observation is a structured node with source URL, timestamp, author credibility, confidence score, sentiment polarity, vector embedding, and relational edges.
- 100% claim traceability back to verified primary sources.

### 9. Time-Machine / Historical Simulation & Backtesting
- Temporal cutoff filter (`as_of_date="YYYY-MM-DD"`).
- All agents, feeds, and memory enforce strict cutoff to evaluate research quality without future data leakage.

### 10. Generalized Domain Adaptability
- Abstract orchestration architecture reusable for Legal Research, Medical/Healthcare Analysis, Cybersecurity, and Enterprise Decision Support.
