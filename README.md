# EquiMind: AI-Powered Financial Research Operating System

<div align="center">

![EquiMind Banner](https://img.shields.io/badge/System-Institutional_Equity_Research-06b6d4?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.12+-38bdf8?style=for-the-badge&logo=python&logoColor=white)
![Build & Tests](https://img.shields.io/badge/Tests-46%2F46_PASSED-10b981?style=for-the-badge)
![Architecture](https://img.shields.io/badge/Architecture-Model_Agnostic-f59e0b?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-6366f1?style=for-the-badge)

</div>

---

## 🏛️ Vision & Philosophy

**EquiMind** is an autonomous AI-Powered Financial Research Operating System that functions like an institutional equity research firm and quantitative investment committee.

Unlike typical AI stock prediction bots that output unbacked "Buy" or "Sell" signals, EquiMind enforces the strict separation between **language reasoning** and **deterministic numerical computation**:

> **Core Axiom**: Language models are *never* the source of truth for numerical data or financial calculations. All mathematical, statistical, time-series, alpha factor, causal, and portfolio optimization models are calculated 100% deterministically outside the LLM using reproducible mathematical libraries (`numpy`, `scipy`, `pandas`). The LLM acts solely as a research analyst synthesizing verified evidence without inventing unsupported facts or hallucinating numerical metrics.

---

## 🏗️ System Architecture

```mermaid
graph TD
    User[User / CLI / API / Web Dashboard] --> Engine[EquiMindEngine Orchestrator]
    
    Engine --> Delta[Delta Research Engine & Hierarchical Memory]
    Delta --> Planner[Reasoning Planner Agent]
    
    Planner --> DAG[Dynamic Research DAG Pipeline]

    subgraph LLM Provider Layer (Model Agnostic)
        LLM[Unified LLMProvider Interface]
        LLM --> OpenAI[OpenAI: gpt-4o, o3-mini]
        LLM --> Claude[Anthropic: claude-3-5-sonnet]
        LLM --> Gemini[Google Gemini: gemini-1.5-pro]
        LLM --> DeepSeek[DeepSeek / Qwen]
        LLM --> Ollama[Ollama / Local Llama-3]
        LLM --> OpenRouter[OpenRouter Provider]
        LLM --> Mock[Mock Offline Engine]
    end

    DAG --> MarketTeam[Market Data Research Team]
    DAG --> FundTeam[Fundamental Analysis Team]
    DAG --> MacroTeam[Macroeconomic Research Team]
    DAG --> WebTeam[Web & Alternative Intelligence Team]

    subgraph Deterministic Quantitative Engine (Pure Math - No LLM)
        TechEngine[Technical: RSI, MACD, BB, ATR, S/R]
        FundEngine[Fundamental: PE, PB, ROE, Piotroski F-Score, Altman Z-Score]
        RiskEngine[Risk: VaR 95%, CVaR, Sharpe, Sortino, MaxDD]
        TimeSeries[Time Series: 1D Kalman Filter, HMM Regimes, GARCH 1,1]
        AlphaLab[Alpha Lab: IC, Rank IC, Sharpe Ranking, Factor Decay]
        CausalEngine[Causal Engine: Structural Causal Models & Confounder Adjustment]
        MonteCarlo[Monte Carlo: 1,000 Stochastic Paths, P05/P95 Risk]
        PortfolioOpt[Portfolio Optimizer: Markowitz, Risk Parity, Black-Litterman, Kelly]
    end

    MarketTeam & FundTeam & MacroTeam & WebTeam --> FeatureStore[Feature Engineering Platform & FeatureStore]
    FeatureStore --> TechEngine & FundEngine & RiskEngine & TimeSeries & AlphaLab & CausalEngine & MonteCarlo & PortfolioOpt
    
    TechEngine & FundEngine & RiskEngine & TimeSeries & AlphaLab & CausalEngine & MonteCarlo & PortfolioOpt --> EvidenceGraph[Structured Evidence Provenance Graph]
    
    subgraph Context Optimization & Memory
        EvidenceGraph --> ContextCompressor[Context Optimization & Compressor Engine]
        ContextCompressor --> Deduplication[Exact MD5 & Jaccard Fuzzy Clustering]
        ContextCompressor --> Ranking[Time-Decay e^-0.05t & Relevance Scoring]
        ContextCompressor --> BudgetPacker[Token Budget Packer]
    end

    BudgetPacker --> Committee[Adversarial Investment Committee]
    
    subgraph Debate Arena
        Committee --> BullAgent[Bull Research Agent]
        Committee --> BearAgent[Bear Research Agent]
        BullAgent & BearAgent --> JudgeAgent[Debate Judge Agent]
    end

    JudgeAgent --> FinalRec[Structured Explainable Investment Recommendation]
    FinalRec --> User
```

---

## ⚡ Core Engine Modules

### 1. Model-Agnostic LLM Provider Layer (`equimind.providers`)
- Decoupled `LLMProvider` interface supporting OpenAI, Anthropic Claude, Google Gemini, DeepSeek, Qwen, Ollama (Local), OpenRouter, and MockProvider.
- `ProviderFactory` with automated fallback execution chains to guarantee zero research downtime.

### 2. Intelligent Research Planner (`equimind.planner`)
- Dynamic sector classification (`SEMICONDUCTOR_TECH`, `BANKING_FINANCE`, `SAAS_SOFTWARE`, `PHARMACEUTICALS`, `AIRLINES_TRANSPORT`, etc.) and horizon detection (`DAY_TRADING` to `LONG_TERM`).
- Generates custom execution DAG pipelines, invoking relevant subagents while skipping irrelevant scrapers.

### 3. Specialized Research Teams & Adapters (`equimind.teams`)
- `MarketDataTeam`: Prices, volume, liquidity, orderbook, benchmark indices.
- `FundamentalTeam`: Financial statements, balance sheet ratios, Piotroski F-Score (0-9), Altman Z-Score bankruptcy risk zones.
- `MacroTeam`: CPI inflation, Fed funds rate, GDP, Brent crude oil, Gold, VIX, FX rates.
- `WebIntelligenceTeam`: Platform-specific adapters for SEC EDGAR filings (10-K/10-Q), Bloomberg/Reuters news, Reddit r/stocks, Twitter/X analyst feeds, GitHub developer commits, and Earnings Call transcripts.

### 4. Deterministic Quantitative & Advanced Institutional Engines (`equimind.quantitative` & `equimind.features`)
- **Technical Analysis**: RSI 14, MACD, Bollinger Bands, Moving Averages (SMA/EMA), ATR 14, Support/Resistance pivots.
- **Fundamental Metrics**: PE, PB, PEG, ROE, ROA, FCF Yield, Debt-to-Equity, **Piotroski F-Score (0-9)**, **Altman Z-Score**.
- **Advanced Time Series Engine**: 1D Kalman Filter noise reduction, Hidden Markov Model (HMM) market regime classifier (`BULL_TREND`, `BEAR_TREND`, `HIGH_VOLATILITY_SIDEWAYS`), GARCH(1,1) volatility, ensemble forecast bounds ($\mu \pm 1.96 \sigma$).
- **Alpha Research Laboratory**: Information Coefficient (IC), Rank IC (Spearman correlation), factor Sharpe ratio, factor decay half-life, and statistical significance ranking.
- **Feature Engineering Platform & FeatureStore**: Converts evidence nodes and price series into standardized numerical feature vectors with Z-score normalization.
- **Structural Causal Reasoning Engine**: Do-calculus intervention ($P(Y | \text{do}(X))$) partialling out confounder $Z$ to eliminate spurious market correlations.
- **Monte Carlo Stochastic Simulator**: Generates 1,000+ stochastic price paths using Geometric Brownian Motion (GBM) with jump-diffusion, calculating downside risk (P05) and upside reward (P95) boundaries.
- **Portfolio Construction & Risk Optimization Engine**: Markowitz Tangency Mean-Variance Optimization, Inverse-Volatility Risk Parity, Black-Litterman model, Fractional Kelly Criterion position sizing ($f^* = \frac{p \cdot b - q}{b}$), and Herfindahl diversification scoring.

### 5. Adversarial Investment Committee Debate Engine (`equimind.committee`)
- Tri-agent debate (`BullAgent` vs `BearAgent` evaluated by `JudgeAgent`).
- Strips unbacked claims, evaluates evidence weight ratios, resolves contradictions, and calculates ratings (`STRONG_BUY`, `BUY`, `HOLD`, `SELL`), conviction scores, target entry price ranges, and risk-reward ratios.

### 6. Hierarchical Memory & Delta-Research Engine (`equimind.memory`)
- Tiers 1 through 5 multi-stage memory store (Raw -> Daily -> Weekly -> Monthly -> Quarterly Persistent Knowledge).
- `DeltaResearchEngine`: Timestamp diffing that reuses validated historical evidence and fetches *only* fresh/modified signals.

### 7. Non-LLM Context Optimization & Compressor (`equimind.context`)
- MD5 exact deduplication, Jaccard token set fuzzy clustering, exponential time-decay scoring ($e^{-0.05 \Delta t}$), and token budget packing.

### 8. Historical Simulation & Backtesting Time Machine (`equimind.time_machine`)
- `TemporalGuard` enforcing `as_of_date` temporal cutoff to prevent look-ahead bias during backtesting.

### 9. Multi-Domain Adaptability Suite (`equimind.domain_adapter`)
- Reusable domain adapters for **Legal Case Research**, **Healthcare/Medical Literature Review**, and **Cybersecurity Threat Intelligence**.

---

## 💻 Installation & Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/akanil18/EquiMind.git
cd EquiMind

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Complete Unit Test Suite (46 Tests)
```bash
python3 -m unittest discover tests -v
```

### 3. Run Command Line Research Query (CLI)
```bash
# Execute query using offline zero-cost Mock engine
python3 -m equimind.cli --ticker NVDA --query "Should I invest in NVIDIA today?" --provider mock

# Execute query with temporal cutoff backtest date
python3 -m equimind.cli --ticker TSLA --query "Analyze Tesla" --provider openai --as-of-date 2024-01-01
```

### 4. Launch FastAPI Server & Web Dashboard UI
```bash
python3 -m uvicorn equimind.api.server:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser to interact with the custom dark-mode dashboard styled with `Syne` & `DM Sans` typography inspired by `ayushjha.online`.

### 5. Production Docker Compose
```bash
docker-compose up --build -d
```

---

## 📚 Documentation Hub

Complete versioned technical documentation is available in the [`docs/`](file:///home/anil-paliwal/Documents/Development/Quant_project/docs/) directory:
- [docs/v1.0/01_SYSTEM_ARCHITECTURE.md](file:///home/anil-paliwal/Documents/Development/Quant_project/docs/v1.0/01_SYSTEM_ARCHITECTURE.md)
- [docs/v1.0/02_MODEL_AGNOSTIC_PROVIDERS.md](file:///home/anil-paliwal/Documents/Development/Quant_project/docs/v1.0/02_MODEL_AGNOSTIC_PROVIDERS.md)
- [docs/v1.0/03_DETERMINISTIC_QUANTITATIVE_ENGINE.md](file:///home/anil-paliwal/Documents/Development/Quant_project/docs/v1.0/03_DETERMINISTIC_QUANTITATIVE_ENGINE.md)
- [docs/v1.0/04_EVIDENCE_GRAPH_AND_CONTEXT_COMPRESSION.md](file:///home/anil-paliwal/Documents/Development/Quant_project/docs/v1.0/04_EVIDENCE_GRAPH_AND_CONTEXT_COMPRESSION.md)
- [docs/v1.0/05_REASONING_PLANNER_AND_RESEARCH_TEAMS.md](file:///home/anil-paliwal/Documents/Development/Quant_project/docs/v1.0/05_REASONING_PLANNER_AND_RESEARCH_TEAMS.md)
- [docs/v1.0/06_ADVERSARIAL_DEBATE_COMMITTEE.md](file:///home/anil-paliwal/Documents/Development/Quant_project/docs/v1.0/06_ADVERSARIAL_DEBATE_COMMITTEE.md)
- [docs/v1.0/07_HIERARCHICAL_MEMORY_AND_DELTA_ENGINE.md](file:///home/anil-paliwal/Documents/Development/Quant_project/docs/v1.0/07_HIERARCHICAL_MEMORY_AND_DELTA_ENGINE.md)
- [docs/v1.0/08_SELF_REFLECTION_AND_MULTI_DOMAIN.md](file:///home/anil-paliwal/Documents/Development/Quant_project/docs/v1.0/08_SELF_REFLECTION_AND_MULTI_DOMAIN.md)
- [docs/v1.0/09_DEPLOYMENT_AND_API_GUIDE.md](file:///home/anil-paliwal/Documents/Development/Quant_project/docs/v1.0/09_DEPLOYMENT_AND_API_GUIDE.md)

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
