"""
RAG-specific Pydantic schemas for the Agentic RAG Orchestration pipeline.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RAGCriticResult(BaseModel):
    """Output of the RAGCriticAgent after evaluating a retrieval round.

    The critic acts as a pre-debate judge: it checks whether the retrieved
    evidence is diverse, timely, sentiment-balanced, and query-relevant
    before the Bull/Bear/Judge committee debate begins.
    """

    iteration: int
    is_sufficient: bool
    coverage_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite score: source diversity + sentiment balance + query relevance + recency",
    )
    source_diversity_score: float = Field(ge=0.0, le=1.0)
    sentiment_balance_score: float = Field(ge=0.0, le=1.0)
    query_relevance_score: float = Field(ge=0.0, le=1.0)
    recency_score: float = Field(ge=0.0, le=1.0)
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Evidence gaps the critic identified (e.g. 'no SEC filings', 'only bullish sentiment')",
    )
    refined_query: str = Field(
        description="Query refined by the critic to target missing aspects in the next retrieval round",
    )
    discard_node_ids: List[str] = Field(
        default_factory=list,
        description="Node IDs the critic determined are irrelevant or low-quality and should be dropped",
    )
    rationale: str = Field(
        default="",
        description="Human-readable explanation of the critic's evaluation",
    )


class RAGIterationLog(BaseModel):
    """Immutable log entry capturing state of a single retrieval iteration."""

    iteration: int
    query_used: str
    nodes_retrieved: int
    nodes_after_filter: int
    coverage_score: float
    is_sufficient: bool
    missing_aspects: List[str] = Field(default_factory=list)
    hnsw_search_time_ms: float = 0.0
    embed_time_ms: float = 0.0
    critic_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgenticRAGResult(BaseModel):
    """Final output of the AgenticRAGOrchestrator after all retrieval iterations.

    Wraps the semantically curated evidence nodes with full observability
    metadata for every retrieval round.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str
    original_query: str
    final_query: str
    total_iterations: int
    final_coverage_score: float = Field(ge=0.0, le=1.0)
    converged: bool = Field(
        description="True if RAGCriticAgent declared sufficiency before max_iterations"
    )
    curated_node_ids: List[str] = Field(
        description="Ordered list of EvidenceNode IDs selected by the agentic RAG loop (best first)"
    )
    discarded_node_ids: List[str] = Field(
        default_factory=list,
        description="Node IDs dropped by the critic across all iterations",
    )
    iteration_logs: List[RAGIterationLog] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
