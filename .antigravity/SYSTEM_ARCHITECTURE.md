# EquiMind System Architecture

```mermaid
graph TD
    User[User Query / CLI / API / Web UI] --> Orchestrator[Orchestration Engine / EquiMindEngine]
    
    Orchestrator --> Planner[Dynamic Reasoning Planner Agent]
    Planner --> DAG[Custom Dynamic Execution DAG]
    
    subgraph LLM Provider Layer (Model Agnostic)
        UnifiedLLM[LLMProvider Interface]
        UnifiedLLM --> OpenAIProvider[OpenAI Provider]
        UnifiedLLM --> AnthropicProvider[Anthropic Claude Provider]
        UnifiedLLM --> GeminiProvider[Gemini Provider]
        UnifiedLLM --> DeepSeekProvider[DeepSeek / Qwen Provider]
        UnifiedLLM --> OllamaProvider[Ollama / Local Provider]
        UnifiedLLM --> OpenRouterProvider[OpenRouter Provider]
    end

    DAG --> MarketTeam[Market Data Research Team]
    DAG --> FundTeam[Fundamental Analysis Team]
    DAG --> MacroTeam[Macroeconomic Analysis Team]
    DAG --> AltTeam[Alternative & Web Intelligence Team]
    
    subgraph Alternative Intelligence Adapters
        AltTeam --> RedditAdapter[Reddit Adapter]
        AltTeam --> XAdapter[X / Twitter Adapter]
        AltTeam --> SECAdapter[SEC Filings Adapter]
        AltTeam --> NewsAdapter[Financial News Adapter]
        AltTeam --> DevAdapter[GitHub / Job Trends Adapter]
        AltTeam --> CallAdapter[Earnings Transcripts Adapter]
    end
    
    subgraph Deterministic Quantitative Engine (Pure Math - No LLM)
        TechEngine[Technical Analysis Module: RSI, MACD, BB, ATR, S/R]
        FundQuantEngine[Fundamental Metrics Module: PE, PB, ROE, FCF, Z-Score]
        RiskEngine[Probabilistic Risk Engine: VaR, Volatility, Dist]
    end

    MarketTeam & FundTeam & MacroTeam & AltTeam --> QuantEngineBridge[Quantitative & Evidence Bridge]
    QuantEngineBridge --> TechEngine & FundQuantEngine & RiskEngine
    
    TechEngine & FundQuantEngine & RiskEngine --> EvidenceGraph[Structured Evidence Graph]
    
    subgraph Context Optimization & Memory
        EvidenceGraph --> ContextCompressor[Context Optimization & Compression Engine]
        ContextCompressor --> Deduplication[Deduplication & Clustering]
        ContextCompressor --> Ranking[Time Decay & Relevance Ranking]
        ContextCompressor --> BudgetPacker[Context Budget Packer]
        
        EvidenceGraph --> MemoryPipeline[Hierarchical Memory Pipeline]
        MemoryPipeline --> Tier1[Raw Observations]
        MemoryPipeline --> Tier2[Daily Summaries]
        MemoryPipeline --> Tier3[Weekly Syntheses]
        MemoryPipeline --> Tier4[Monthly Investment Theses]
        MemoryPipeline --> Tier5[Quarterly Persistent Knowledge]
    end

    BudgetPacker --> DebateEngine[Investment Committee & Debate Engine]
    
    subgraph Investment Committee
        DebateEngine --> BullAgent[Bull Research Agent]
        DebateEngine --> BearAgent[Bear Research Agent]
        BullAgent & BearAgent --> JudgeAgent[Debate Judge Agent]
    end

    JudgeAgent --> RecEngine[Explainable Recommendation Generator]
    RecEngine --> Output[Final Research Report with Provenance Citations]
```

## Layer Architecture & Component Breakdown

1. **`equimind.providers`**: Unified LLM abstraction layer supporting OpenAI, Claude, Gemini, DeepSeek, Qwen, Ollama, OpenRouter, etc.
2. **`equimind.evidence`**: `EvidenceNode`, `EvidenceGraph`, `EvidenceSource`, provenance metadata, and JSON/vector serialization.
3. **`equimind.context`**: In-memory context optimization engine (deduplication, fuzzy clustering, time-decay scoring, relevance ranking, budget packing).
4. **`equimind.quantitative`**: Deterministic technical indicator calculator, fundamental metric suite, risk/return distribution engine (pure math).
5. **`equimind.planner`**: Dynamic Reasoning Planner Agent for adaptive DAG generation based on query, sector, ticker, and asset type.
6. **`equimind.teams`**: Specialized research subagent teams (Market Data, Fundamentals, Macroeconomics, Web/Alternative Signals).
7. **`equimind.adapters`**: Platform-specific adapters (Reddit, X, Stocktwits, SEC filings, Earnings transcripts, News feeds, GitHub, LinkedIn).
8. **`equimind.committee`**: Investment Committee Debate Engine (Bull Agent, Bear Agent, Judge Agent).
9. **`equimind.memory`**: Hierarchical multi-tier memory store (Raw -> Daily -> Weekly -> Monthly -> Quarterly Persistent Knowledge) with delta updates.
10. **`equimind.time_machine`**: Backtesting and temporal isolation guard (`as_of_date` context manager).
11. **`equimind.orchestrator`**: Core engine coordinating planner, teams, quantitative calculations, debate, memory, and final output generation.
