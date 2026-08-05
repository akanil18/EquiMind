import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from equimind.evidence.schema import EvidenceNode
from equimind.memory.schema import MemoryTier, EntityKnowledgeEntry, ResearchReportRecord

logger = logging.getLogger(__name__)


class HierarchicalMemoryStore(BaseModel):
    """Multi-tiered memory store (Raw -> Daily -> Weekly -> Monthly -> Quarterly Persistent Knowledge)."""

    ticker_knowledge: Dict[str, EntityKnowledgeEntry] = Field(default_factory=dict)

    def get_or_create_entity(self, ticker: str, sector: str = "general_equity") -> EntityKnowledgeEntry:
        """Retrieves persistent entity entry or initializes a new one."""
        ticker_upper = ticker.upper()
        if ticker_upper not in self.ticker_knowledge:
            self.ticker_knowledge[ticker_upper] = EntityKnowledgeEntry(
                ticker=ticker_upper,
                company_name=ticker_upper,
                sector=sector,
                last_updated=datetime.now(timezone.utc),
            )
        return self.ticker_knowledge[ticker_upper]

    def store_research_report(
        self,
        ticker: str,
        user_query: str,
        rating: str,
        conviction_score: float,
        summary: str,
        evidence_nodes: List[EvidenceNode],
    ) -> ResearchReportRecord:
        """Stores research report and integrates raw evidence into hierarchical memory."""
        entity = self.get_or_create_entity(ticker)

        report = ResearchReportRecord(
            ticker=ticker.upper(),
            timestamp=datetime.now(timezone.utc),
            user_query=user_query,
            rating=rating,
            conviction_score=conviction_score,
            summary=summary,
            evidence_count=len(evidence_nodes),
            raw_evidence_ids=[n.id for n in evidence_nodes],
        )

        entity.historical_reports.append(report)
        for node in evidence_nodes:
            entity.cumulative_evidence_nodes[node.id] = node

        entity.last_updated = datetime.now(timezone.utc)
        entity.persistent_thesis = f"Updated thesis for {ticker.upper()}: Rated {rating} with conviction {conviction_score:.2f} based on {len(entity.cumulative_evidence_nodes)} cumulative observations."

        return report

    def get_last_report(self, ticker: str) -> Optional[ResearchReportRecord]:
        """Returns the most recent research report for a ticker."""
        entity = self.ticker_knowledge.get(ticker.upper())
        if entity and entity.historical_reports:
            return entity.historical_reports[-1]
        return None

    def to_json(self) -> str:
        """Serializes store to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "HierarchicalMemoryStore":
        """Deserializes store from JSON string."""
        return cls.model_validate_json(json_str)
