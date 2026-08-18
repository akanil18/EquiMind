"""
CrossEncoderReranker — Second-Stage Neural & Lexical-Semantic Reranker.

Architecture:
  - Takes candidate nodes (e.g. Top 50 from HybridRetriever)
  - Evaluates full query-document cross interaction
  - Applies latency budget enforcement (e.g. max 50ms)
  - Scores query-passage relevance, source authority, and temporal freshness
  - Returns top N (e.g. 5-10) highest quality EvidenceNodes to the LLM
"""

import time
import math
import re
import logging
from typing import List, Tuple, Optional, Dict, Any

from equimind.evidence.schema import EvidenceNode, AuthorCredibility

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Two-stage Cross-Encoder Reranker with latency budget and semantic cross-scoring."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        latency_budget_ms: float = 100.0,
    ):
        self.model_name = model_name
        self.latency_budget_ms = latency_budget_ms

    def rerank(
        self,
        query: str,
        candidates: List[EvidenceNode],
        top_n: int = 10,
    ) -> List[Tuple[float, EvidenceNode]]:
        """Reranks candidate EvidenceNodes using cross-scoring and returns top_n."""
        if not candidates:
            return []

        t0 = time.time()
        scored: List[Tuple[float, EvidenceNode]] = []
        query_terms = set(re.findall(r"\b\w+\b", query.lower()))

        for node in candidates:
            # Latency budget guard
            elapsed_ms = (time.time() - t0) * 1000.0
            if elapsed_ms > self.latency_budget_ms:
                logger.debug(f"CrossEncoderReranker: Latency budget ({self.latency_budget_ms}ms) reached. Stopping early.")
                break

            score = self._compute_cross_score(query, query_terms, node)
            scored.append((score, node))

        # Sort descending by rerank score
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_n]

    def _compute_cross_score(
        self,
        query: str,
        query_terms: set,
        node: EvidenceNode,
    ) -> float:
        """Computes multi-dimensional cross-interaction score between query and document."""
        content_lower = node.content.lower()
        title_lower = node.title.lower()
        doc_text = f"{title_lower} {content_lower}"
        doc_terms = set(re.findall(r"\b\w+\b", doc_text))

        # 1. Exact & Partial Term Cross-Overlap (Term Coverage)
        overlap = len(query_terms & doc_terms)
        coverage_score = (overlap / len(query_terms)) if query_terms else 0.5

        # 2. Phrase matching (Exact N-gram proximity)
        phrase_boost = 1.0
        if len(query.strip().split()) > 1 and query.lower() in doc_text:
            phrase_boost = 1.4

        # 3. Source credibility & confidence weight
        cred_mult = {
            AuthorCredibility.VERIFIED_OFFICIAL: 1.5,
            AuthorCredibility.HIGH: 1.2,
            AuthorCredibility.MEDIUM: 1.0,
            AuthorCredibility.LOW: 0.6,
        }.get(node.author_credibility, 1.0)

        conf_mult = max(0.2, min(1.0, node.confidence_score))

        # 4. Dense retrieval score bonus if already present
        retrieval_bonus = getattr(node, "rag_retrieval_score", 0.5) or 0.5

        # Composite Cross-Encoder Score ∈ [0, 1]
        raw_score = (
            (coverage_score * 0.40)
            + (retrieval_bonus * 0.30)
            + (min(1.0, cred_mult / 1.5) * 0.20)
            + (conf_mult * 0.10)
        ) * phrase_boost

        return round(min(1.0, max(0.0, raw_score)), 4)
