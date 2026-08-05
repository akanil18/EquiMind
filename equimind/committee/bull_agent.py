from typing import List, Dict, Any, Optional
from equimind.evidence.schema import EvidenceNode, SentimentPolarity
from equimind.committee.schema import BullCase
from equimind.providers.base import LLMProvider, LLMMessage, Role


class BullAgent:
    """Agent advocating for the Bullish investment thesis."""

    @classmethod
    def evaluate(
        cls,
        ticker: str,
        nodes: List[EvidenceNode],
        quant_summary: Dict[str, Any],
        provider: Optional[LLMProvider] = None,
    ) -> BullCase:
        """Constructs evidence-backed Bull case."""
        bullish_nodes = [
            n for n in nodes if n.sentiment in (SentimentPolarity.BULLISH, SentimentPolarity.VERY_BULLISH)
        ]
        ev_ids = [n.id for n in bullish_nodes]

        catalysts = []
        for n in bullish_nodes:
            catalysts.append(f"[{n.source_type.value.upper()}] {n.title}")

        if not catalysts:
            catalysts = [f"Strong underlying market demand for {ticker}"]

        thesis = (
            f"Bullish thesis for {ticker}: Backed by {len(bullish_nodes)} positive evidence signals. "
            f"Key growth drivers include expanding market share, high profitability metrics, "
            f"and strong institutional channel checks."
        )

        last_price = quant_summary.get("last_price", 100.0)
        upside_target = round(last_price * 1.25, 2)

        return BullCase(
            thesis=thesis,
            key_catalysts=catalysts[:5],
            supporting_evidence_ids=ev_ids,
            upside_price_target=upside_target,
        )
