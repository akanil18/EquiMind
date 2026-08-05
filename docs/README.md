# EquiMind Documentation Hub

Welcome to the official documentation hub for **EquiMind**, an autonomous AI Equity Research Firm and generalized financial research orchestration framework.

---

## 📁 Versioned Documentation Suites

### 🟢 **Version 1.0 (Current Release)**

| Document Module | Description |
| :--- | :--- |
| 🏗️ **[System Architecture](v1.0/01_SYSTEM_ARCHITECTURE.md)** | High-level system architecture, layer breakdown, Mermaid diagrams, data flows. |
| 🤖 **[Model-Agnostic LLM Provider Layer](v1.0/02_MODEL_AGNOSTIC_PROVIDERS.md)** | OpenAI, Claude, Gemini, DeepSeek, Qwen, Ollama, OpenRouter, and ProviderFactory with fallback execution. |
| 🧮 **[Deterministic Quantitative Engine](v1.0/03_DETERMINISTIC_QUANTITATIVE_ENGINE.md)** | 100% non-LLM math calculations (RSI, MACD, BB, ATR, S/R, PE, PB, Piotroski F-Score 0-9, Altman Z-Score, VaR, Sharpe, Sortino). |
| 🕸️ **[Structured Evidence Graph & Context Compression](v1.0/04_EVIDENCE_GRAPH_AND_CONTEXT_COMPRESSION.md)** | Provenance tracking (`EvidenceNode`), Graph edges, MD5 exact deduplication, Jaccard fuzzy clustering, $e^{-0.05 \Delta t}$ time decay, and token budget packing. |
| 🎯 **[Reasoning Planner & Research Teams](v1.0/05_REASONING_PLANNER_AND_RESEARCH_TEAMS.md)** | Dynamic planner sector rules (`SEMICONDUCTOR_TECH`, `BANKING_FINANCE`, `SAAS_SOFTWARE`), subagent teams (`MarketDataTeam`, `FundamentalTeam`, `MacroTeam`, `WebIntelligenceTeam`), and `TemporalGuard` backtesting isolation. |
| ⚔️ **[Adversarial Investment Committee Debate Engine](v1.0/06_ADVERSARIAL_DEBATE_COMMITTEE.md)** | Tri-agent debate (`BullAgent` vs `BearAgent` evaluated by `JudgeAgent`), evidence weighting, rating rules, conviction scoring, entry target calculation, and citations. |
| 🧠 **[Hierarchical Memory & Delta-Research Engine](v1.0/07_HIERARCHICAL_MEMORY_AND_DELTA_ENGINE.md)** | Multi-tier memory store (Tiers 1-5), `EntityKnowledgeEntry`, `DeltaResearchEngine` timestamp diffing and incremental update mechanics. |
| 🪞 **[Self-Reflection & Multi-Domain Adaptability](v1.0/08_SELF_REFLECTION_AND_MULTI_DOMAIN.md)** | `SelfReflectionAgent` bias detection, conviction calibration factor, and domain adapters for Legal Case Research, Healthcare/Medical Review, and Cybersecurity Threat Intelligence. |
| 🚀 **[Deployment, API & CLI Guide](v1.0/09_DEPLOYMENT_AND_API_GUIDE.md)** | FastAPI REST/WebSocket endpoints, CLI command syntax, multi-stage Docker containerization, and production deployment guide. |

---

### 🔮 **[Version 2.0 (Planned Feature Roadmap)](v2.0/PLANNED_FEATURES.md)**
- Vector DB integration (Qdrant / ChromaDB / Pinecone).
- Real-time WebSocket streaming of agent DAG thoughts.
- Live market data feed connectors (Polygon.io / AlphaVantage).


---

## 🚀 Quick Execution Check

Run the test suite across all 9 modules:
```bash
python3 -m unittest discover tests -v
```

Execute a sample research CLI query:
```bash
python3 -m equimind.cli --ticker NVDA --query "Should I invest in NVIDIA today?" --provider mock
```
