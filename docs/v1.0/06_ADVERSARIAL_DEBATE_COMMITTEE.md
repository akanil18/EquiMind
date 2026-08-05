# EquiMind v1.0: Adversarial Investment Committee Debate Engine (`equimind.committee`)

Instead of relying on a single AI analyst's opinion, EquiMind executes tri-agent adversarial debate.

---

## ⚔️ Debate Architecture

1. **`BullAgent`**: Collects all evidence supporting investment. Formulates growth thesis, identifies catalysts (e.g. SEC 10-Q datacenter revenue surge, GitHub star velocity), and calculates upside price targets.
2. **`BearAgent`**: Collects all evidence opposing investment. Formulates downside risk thesis, identifies headwinds (e.g. elevated valuation multiples, CoWoS supply constraints, macro rate sensitivity), and calculates downside price targets.
3. **`JudgeAgent`**: Evaluates evidence strength ratio ($\text{Bull Weight} / \text{Bear Weight}$), strips unbacked claims, resolves evidence contradictions, and calculates final ratings (`STRONG_BUY`, `BUY`, `HOLD`, `SELL`), conviction scores, target entry price ranges, and citation lists.

---

## 📊 Recommendation Output Schema

```python
class InvestmentRecommendation(BaseModel):
    ticker: str
    rating: InvestmentRating                         # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    conviction_score: float                          # 0.0 to 1.0
    current_price: float
    target_entry_range: Tuple[float, float]
    expected_risk_reward_ratio: float
    recommended_portfolio_allocation: str
    bull_case: BullCase
    bear_case: BearCase
    debate_synthesis: DebateSynthesis
    provenance_citations: List[Dict[str, Any]]
    assumptions_and_risks: List[str]
```
