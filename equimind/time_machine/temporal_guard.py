import logging
from datetime import datetime, timezone
from typing import List, Optional
from equimind.evidence.schema import EvidenceNode

logger = logging.getLogger(__name__)


class TemporalGuard:
    """Enforces strict temporal cutoffs to eliminate future data leakage during historical simulations & backtests."""

    def __init__(self, as_of_date: Optional[datetime] = None):
        self.as_of_date = as_of_date

    def __enter__(self):
        if self.as_of_date:
            logger.info(f"TemporalGuard active: Backtesting cutoff enforced as_of_date={self.as_of_date.isoformat()}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def filter_evidence(self, nodes: List[EvidenceNode]) -> List[EvidenceNode]:
        """Filters out any evidence published after the configured cutoff date."""
        if not self.as_of_date:
            return nodes

        valid_nodes: List[EvidenceNode] = []
        for node in nodes:
            # Ensure comparison is timezone-aware
            node_ts = node.publication_timestamp
            if node_ts.tzinfo is None:
                node_ts = node_ts.replace(tzinfo=timezone.utc)
            
            cutoff_ts = self.as_of_date
            if cutoff_ts.tzinfo is None:
                cutoff_ts = cutoff_ts.replace(tzinfo=timezone.utc)

            if node_ts <= cutoff_ts:
                valid_nodes.append(node)
            else:
                logger.debug(f"TemporalGuard pruned future observation: '{node.title}' ({node_ts.isoformat()}) > {cutoff_ts.isoformat()}")

        return valid_nodes

    @classmethod
    def parse_as_of_date(cls, date_str: Optional[str]) -> Optional[datetime]:
        """Utility helper to parse YYYY-MM-DD string into UTC datetime."""
        if not date_str:
            return None
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")
            return None
