"""
Agentic RAG Orchestrator — End-to-End Production Agentic Retrieval Loop.

Pipeline:
  1. Multi-query expansion (generates targeted perspective sub-queries)
  2. Hybrid Retrieval (Dense HNSW + Sparse BM25 + Reciprocal Rank Fusion)
  3. Second-stage Cross-Encoder Reranker (top 50 -> top N with latency budget)
  4. RAGCriticAgent Sufficiency Evaluation (checks 5 dimensions: diversity, sentiment balance, relevance, recency, confidence)
  5. Self-RAG Iterative Loop: If insufficient, rewrites query and re-retrieves
  6. Final Curated Evidence returned with complete observability logs
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from equimind.evidence.schema import EvidenceNode
from equimind.providers.base import LLMProvider
from equimind.rag.critic_agent import RAGCriticAgent
from equimind.rag.query_rewriter import RAGQueryRewriter
from equimind.rag.vector_store import HNSWVectorStore, MetadataFilter
from equimind.rag.hybrid_retriever import HybridRetriever
from equimind.rag.reranker import CrossEncoderReranker
from equimind.rag.schema import AgenticRAGResult, RAGIterationLog

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 3
DEFAULT_TOP_K = 25
DEFAULT_MIN_SIMILARITY = 0.10


class AgenticRAGOrchestrator:
    """Production Agentic RAG Orchestrator with Hybrid Search, Reranking, and Self-RAG."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        hnsw_M: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
    ) -> None:
        self.provider = provider
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.hnsw_M = hnsw_M
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

    def orchestrate(
        self,
        query: str,
        ticker: str,
        candidate_nodes: List[EvidenceNode],
        as_of_date: Optional[datetime] = None,
    ) -> AgenticRAGResult:
        """Run the full agentic RAG loop."""
        ticker_upper = ticker.upper().strip()

        logger.info(
            f"AgenticRAGOrchestrator: START | ticker={ticker_upper} | "
            f"corpus={len(candidate_nodes)} nodes | max_iter={self.max_iterations}"
        )

        if not candidate_nodes:
            logger.warning("AgenticRAGOrchestrator: empty corpus — returning empty result")
            return AgenticRAGResult(
                ticker=ticker_upper,
                original_query=query,
                final_query=query,
                total_iterations=0,
                final_coverage_score=0.0,
                converged=False,
                curated_node_ids=[],
                discarded_node_ids=[],
                iteration_logs=[],
            )

        # ── 1. Index full candidate corpus into VectorStore & BM25 ──
        t0 = time.time()
        vector_store = HNSWVectorStore(
            M=self.hnsw_M,
            ef_construction=self.hnsw_ef_construction,
            ef_search=self.hnsw_ef_search,
        )
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            provider=self.provider,
        )
        hybrid_retriever.index_nodes(candidate_nodes)
        build_ms = (time.time() - t0) * 1000.0

        reranker = CrossEncoderReranker(latency_budget_ms=100.0)

        # ── 2. Self-RAG Retrieval Loop ─────────────────────────────
        current_query = query
        iteration_logs: List[RAGIterationLog] = []
        all_retrieved: Dict[str, float] = {}
        all_discard_ids: Set[str] = set()
        final_critique = None
        converged = False

        filter_spec = MetadataFilter(ticker=ticker_upper)

        for iteration in range(1, self.max_iterations + 1):
            # 2a. Hybrid Retrieval (Dense + BM25 + RRF)
            t_search = time.time()
            hybrid_results = hybrid_retriever.retrieve(
                current_query,
                top_k=self.top_k,
                filters=filter_spec if len(candidate_nodes) > 10 else None,
            )
            search_ms = (time.time() - t_search) * 1000.0

            # 2b. Cross-Encoder Reranking
            retrieved_candidates = [node for _, node in hybrid_results]
            reranked_results = reranker.rerank(
                current_query,
                retrieved_candidates,
                top_n=min(len(retrieved_candidates), 15),
            )

            # Update score map
            for score, node in reranked_results:
                node.rag_retrieval_score = score
                if node.id not in all_retrieved or all_retrieved[node.id] < score:
                    all_retrieved[node.id] = score

            nodes_for_eval = [node for _, node in reranked_results]
            retrieval_scores = {node.id: score for score, node in reranked_results}

            # 2c. RAGCriticAgent Sufficiency Evaluation
            t_critic = time.time()
            critique = RAGCriticAgent.evaluate(
                query=current_query,
                retrieved_nodes=nodes_for_eval,
                iteration=iteration,
                provider=self.provider,
                retrieval_scores=retrieval_scores,
                as_of_date=as_of_date,
            )
            critic_ms = (time.time() - t_critic) * 1000.0

            all_discard_ids.update(critique.discard_node_ids)
            final_critique = critique

            log = RAGIterationLog(
                iteration=iteration,
                query_used=current_query,
                nodes_retrieved=len(hybrid_results),
                nodes_after_filter=len(nodes_for_eval),
                coverage_score=critique.coverage_score,
                is_sufficient=critique.is_sufficient,
                missing_aspects=critique.missing_aspects,
                hnsw_search_time_ms=round(search_ms, 2),
                embed_time_ms=round(build_ms, 2),
                critic_time_ms=round(critic_ms, 2),
            )
            iteration_logs.append(log)

            if critique.is_sufficient:
                converged = True
                break

            # 2d. Query Rewriting for next iteration
            if iteration < self.max_iterations:
                current_query = RAGQueryRewriter.rewrite(
                    original_query=query,
                    missing_aspects=critique.missing_aspects,
                    iteration=iteration,
                    ticker=ticker_upper,
                    provider=self.provider,
                )

        # ── 3. Merge & Final Rank ──────────────────────────────────
        curated = [
            (score, nid)
            for nid, score in all_retrieved.items()
            if nid not in all_discard_ids
        ]
        curated.sort(key=lambda x: x[0], reverse=True)
        curated_ids = [nid for _, nid in curated]

        final_coverage = final_critique.coverage_score if final_critique else 0.0

        return AgenticRAGResult(
            ticker=ticker_upper,
            original_query=query,
            final_query=current_query,
            total_iterations=len(iteration_logs),
            final_coverage_score=final_coverage,
            converged=converged,
            curated_node_ids=curated_ids,
            discarded_node_ids=list(all_discard_ids),
            iteration_logs=iteration_logs,
            metadata={
                "vector_store_stats": vector_store.stats(),
                "corpus_size": len(candidate_nodes),
                "unique_retrieved": len(all_retrieved),
                "build_time_ms": build_ms,
            },
        )

    def curate_nodes_from_result(
        self,
        rag_result: AgenticRAGResult,
        node_map: Dict[str, EvidenceNode],
    ) -> List[EvidenceNode]:
        """Convert AgenticRAGResult's ordered node IDs back to EvidenceNode objects."""
        curated_nodes = []
        for node_id in rag_result.curated_node_ids:
            if node_id in node_map:
                curated_nodes.append(node_map[node_id])
        return curated_nodes
