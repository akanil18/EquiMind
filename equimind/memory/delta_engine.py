import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from equimind.evidence.schema import EvidenceNode
from equimind.memory.schema import EntityKnowledgeEntry, ResearchReportRecord
from equimind.memory.hierarchical_store import HierarchicalMemoryStore

logger = logging.getLogger(__name__)


class DeltaResearchEngine:
    """Engine computing delta research updates relative to persistent historical memory."""

    @classmethod
    def compute_delta_research_plan(
        cls,
        ticker: str,
        memory_store: HierarchicalMemoryStore,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[datetime], List[EvidenceNode]]:
        """Determines if previous research exists and returns (has_previous, last_updated_time, cached_evidence_nodes)."""
        ticker_upper = ticker.upper()
        entity = memory_store.ticker_knowledge.get(ticker_upper)

        if not entity or not entity.historical_reports:
            logger.info(f"No prior persistent research found for {ticker_upper}. Executing full baseline research.")
            return False, None, []

        last_report = entity.historical_reports[-1]
        last_time = last_report.timestamp
        cached_nodes = list(entity.cumulative_evidence_nodes.values())

        logger.info(
            f"Prior research found for {ticker_upper} (Last research: {last_time.strftime('%Y-%m-%d %H:%M UTC')}). "
            f"Reusing {len(cached_nodes)} validated evidence nodes and executing delta fetch for updates after {last_time.strftime('%Y-%m-%d')}."
        )

        return True, last_time, cached_nodes
