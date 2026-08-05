# EquiMind v1.0: Hierarchical Memory & Delta Engine (`equimind.memory`)

Financial information changes continuously. EquiMind stores completed research as multi-tiered persistent memory and performs incremental delta updates.

---

## 🏛️ Multi-Tier Memory Abstraction

1. **Tier 1 (Raw Observations)**: Uncompressed raw `EvidenceNode` objects.
2. **Tier 2 (Daily Summaries)**: Compressed observations aggregated per ticker per day.
3. **Tier 3 (Weekly Syntheses)**: Weekly trends and channel check syntheses.
4. **Tier 4 (Monthly Investment Theses)**: Monthly macro and valuation thesis updates.
5. **Tier 5 (Quarterly Persistent Knowledge)**: Long-term persistent entity repository per company (`EntityKnowledgeEntry`).

---

## ⚡ Delta-Research Engine (`DeltaResearchEngine`)

When a user requests research for a ticker previously analyzed:
1. `DeltaResearchEngine.compute_delta_research_plan` checks `HierarchicalMemoryStore` for previous `EntityKnowledgeEntry` records.
2. Identifies the timestamp diff since the last research report (`last_report.timestamp`).
3. Reuses all previously validated evidence nodes from cumulative memory.
4. Instructs research subagent teams to fetch *only* fresh/modified evidence published after `last_report.timestamp`.
5. Updates persistent memory with the delta result, reducing computational cost and enabling long-term continuous reasoning.
