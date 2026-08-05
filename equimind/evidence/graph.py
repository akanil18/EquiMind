import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from equimind.evidence.schema import EvidenceNode, EvidenceEdge, EdgeType, EvidenceSource

logger = logging.getLogger(__name__)


class EvidenceGraph(BaseModel):
    """Graph structure maintaining all structured evidence nodes and relational provenance edges."""
    
    nodes: Dict[str, EvidenceNode] = {}
    edges: List[EvidenceEdge] = []

    def add_node(self, node: EvidenceNode) -> str:
        """Add an evidence node to the graph."""
        self.nodes[node.id] = node
        return node.id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        description: Optional[str] = None,
    ) -> Optional[EvidenceEdge]:
        """Add a directed relationship edge between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add edge between missing nodes: {source_id} -> {target_id}")
            return None
        
        edge = EvidenceEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            description=description,
        )
        self.edges.append(edge)
        return edge

    def get_nodes_for_ticker(
        self,
        ticker: str,
        source_type: Optional[EvidenceSource] = None,
        as_of_date: Optional[datetime] = None,
    ) -> List[EvidenceNode]:
        """Query nodes filtered by ticker, optional source type, and optional backtesting temporal cutoff."""
        results = []
        ticker_upper = ticker.upper()
        
        for node in self.nodes.values():
            if node.affected_ticker.upper() != ticker_upper:
                continue
            if source_type and node.source_type != source_type:
                continue
            if as_of_date and node.publication_timestamp > as_of_date:
                # Temporal backtesting isolation: skip future observations
                continue
            results.append(node)
            
        return results

    def to_json(self) -> str:
        """Serialize graph to JSON representation."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EvidenceGraph":
        """Deserialize graph from JSON representation."""
        return cls.model_validate_json(json_str)
