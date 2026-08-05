from typing import List, Dict, Any, Optional
from equimind.evidence.schema import EvidenceNode, SentimentPolarity
from equimind.committee.schema import BearCase
from equimind.providers.base import LLMProvider, LLMMessage, Role


class BearAgent:
    """Agent advocating for the Bearish investment thesis."""

    @classmethod
    def evaluate(
        cls,
        ticker: str,
        nodes: List[EvidenceNode],
        quant_summary: Dict[str, Any],
        provider: Optional[LLMProvider] = None,
    ) -> BearCase:
        """Constructs evidence-backed Bear case."""
        bearish_nodes = [
            n for n in nodes if n.sentiment in (SentimentPolarity.BEARISH, SentimentPolarity.VERY_BEARISH)
        ]
        ev_ids = [n.id for n in bearish_nodes]

        headwinds = []
        for n in bearish_nodes:
            headwinds.append(f"[{n.source_type.value.upper()}] {n.title}")

        if not headwinds:
            headwinds = [f"Valuation risk & potential macroeconomic headwinds for {ticker}"]

        thesis = (
            f"Bearish thesis for {ticker}: Highlighted by {len(bearish_nodes)} risk signals. "
            f"Key risk factors include elevated valuation multiples, potential supply constraints, "
            f"and macroeconomic rate sensitivity."
        )

        last_price = quant_summary.get("last_price", 100.0)
        downside_target = round(last_price * 0.85, 2)

        return BearCase(
            thesis=thesis,
            key_headwinds=headwinds[:5],
            supporting_evidence_ids=ev_ids,
            downside_price_target=downside_target,
        )
