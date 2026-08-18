"""
Agentic RAG Orchestrator — the main retrieval loop.

This is the heart of the Agentic RAG system. It drives an iterative
retrieve-critique-rewrite loop until the evidence is sufficient for
the downstream Bull/Bear/Judge committee debate.

Loop pseudocode
---------------
  query ← original_query
  for iteration in 1..max_iterations:
      nodes ← HNSW_retriever.retrieve(query, k=top_k)
      critique ← RAGCriticAgent.evaluate(query, nodes, iteration)
      log_iteration(critique)

      if critique.is_sufficient:
          break   ← early termination (converged)

      query ← RAGQueryRewriter.rewrite(query, critique.missing_aspects, iteration)

  curated_nodes ← merge_and_rank(all_retrieved_nodes, discard=discard_ids)
  return AgenticRAGResult(curated_nodes, iteration_logs, ...)

Key design decisions
--------------------
- The HNSW index is built ONCE on the full raw evidence corpus (all teams' output).
  Each iteration re-queries the same index with a refined query — this avoids
  redundant I/O while allowing the query vector to shift in semantic space.
- Retrieved nodes are merged across iterations (union of all retrieved sets).
  Nodes flagged for discard by the critic are removed from the final set.
- Final ranking: nodes are re-ranked by their best HNSW similarity score
  across all iterations, combined with the existing ContextCompressor scoring.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from equimind.evidence.schema import EvidenceNode
from equimind.providers.base import LLMProvider
from equimind.rag.critic_agent import RAGCriticAgent
from equimind.rag.query_rewriter import RAGQueryRewriter
from equimind.rag.retriever import HNSWRetriever
from equimind.rag.schema import AgenticRAGResult, RAGIterationLog

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MAX_ITERATIONS = 3
DEFAULT_TOP_K = 25          # nodes retrieved per HNSW search
DEFAULT_MIN_SIMILARITY = 0.10  # minimum HNSW similarity to include a node


class AgenticRAGOrchestrator:
    """Drives the iterative Retrieve → Critique → Rewrite loop.

    Usage
    -----
    rag = AgenticRAGOrchestrator(provider=provider, max_iterations=3)
    result = rag.orchestrate(
        query="Is NVDA a buy for long-term AI infrastructure play?",
        ticker="NVDA",
        candidate_nodes=raw_evidence_nodes,   # from research teams
    )
    # result.curated_node_ids: ordered list of best node IDs
    # result.iteration_logs: full per-iteration observability
    """

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
        """Run the full agentic RAG loop.

        Parameters
        ----------
        query : str
            Original investment research query.
        ticker : str
            Stock ticker symbol.
        candidate_nodes : List[EvidenceNode]
            Raw evidence nodes collected by all research teams.
            The HNSW index is built on this full corpus.
        as_of_date : datetime, optional
            Temporal reference for recency scoring.

        Returns
        -------
        AgenticRAGResult
            Curated, ranked node IDs + full iteration observability.
        """
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

        # ── Step 1: Build HNSW Index (once, on full corpus) ──────────────────
        logger.info("AgenticRAGOrchestrator: Building HNSW index...")
        t_build = time.time()
        retriever = HNSWRetriever(
            provider=self.provider,
            M=self.hnsw_M,
            ef_construction=self.hnsw_ef_construction,
            ef_search=self.hnsw_ef_search,
        )
        retriever.build(candidate_nodes)
        build_ms = (time.time() - t_build) * 1000
        logger.info(
            f"AgenticRAGOrchestrator: HNSW index built in {build_ms:.0f}ms | "
            f"stats={retriever.build_stats}"
        )

        # ── Step 2: Agentic Retrieve → Critique → Rewrite Loop ───────────────
        current_query = query
        iteration_logs: List[RAGIterationLog] = []
        all_retrieved: Dict[str, float] = {}    # node_id → best similarity score
        all_discard_ids: Set[str] = set()
        final_critique = None
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            logger.info(
                f"AgenticRAGOrchestrator: Iteration {iteration}/{self.max_iterations} | "
                f"query='{current_query[:80]}...'"
            )

            # ── 2a. HNSW Retrieval ─────────────────────────────────────────
            t_search = time.time()
            raw_results = retriever.retrieve(current_query, k=self.top_k)
            search_ms = (time.time() - t_search) * 1000

            # Filter by minimum similarity threshold
            filtered_results = [
                (score, node) for score, node in raw_results
                if score >= self.min_similarity
            ]

            # Update best similarity scores across iterations
            for score, node in filtered_results:
                if node.id not in all_retrieved or all_retrieved[node.id] < score:
                    all_retrieved[node.id] = score

            retrieved_nodes = [node for _, node in filtered_results]
            retrieval_scores = {node.id: score for score, node in filtered_results}

            # ── 2b. RAGCriticAgent Evaluation ─────────────────────────────
            t_critic = time.time()
            critique = RAGCriticAgent.evaluate(
                query=current_query,
                retrieved_nodes=retrieved_nodes,
                iteration=iteration,
                provider=self.provider,
                retrieval_scores=retrieval_scores,
                as_of_date=as_of_date,
            )
            critic_ms = (time.time() - t_critic) * 1000

            # Collect discard IDs
            all_discard_ids.update(critique.discard_node_ids)
            final_critique = critique

            # ── 2c. Log Iteration ──────────────────────────────────────────
            log = RAGIterationLog(
                iteration=iteration,
                query_used=current_query,
                nodes_retrieved=len(raw_results),
                nodes_after_filter=len(filtered_results),
                coverage_score=critique.coverage_score,
                is_sufficient=critique.is_sufficient,
                missing_aspects=critique.missing_aspects,
                hnsw_search_time_ms=round(search_ms, 2),
                embed_time_ms=retriever.build_stats.get("embed_time_ms", 0.0),
                critic_time_ms=round(critic_ms, 2),
            )
            iteration_logs.append(log)

            logger.info(
                f"AgenticRAGOrchestrator: Iteration {iteration} complete | "
                f"coverage={critique.coverage_score:.3f} | "
                f"sufficient={critique.is_sufficient} | "
                f"retrieved={len(filtered_results)} | "
                f"search={search_ms:.0f}ms | critic={critic_ms:.0f}ms"
            )

            # ── 2d. Early Termination ──────────────────────────────────────
            if critique.is_sufficient:
                converged = True
                logger.info(
                    f"AgenticRAGOrchestrator: Evidence sufficient — converged at iteration {iteration}"
                )
                break

            # ── 2e. Query Rewriting for Next Round ────────────────────────
            if iteration < self.max_iterations:
                current_query = RAGQueryRewriter.rewrite(
                    original_query=query,
                    missing_aspects=critique.missing_aspects,
                    iteration=iteration,
                    ticker=ticker_upper,
                    provider=self.provider,
                )
                logger.info(
                    f"AgenticRAGOrchestrator: Rewritten query for iter {iteration + 1}: "
                    f"'{current_query[:100]}'"
                )

        # ── Step 3: Merge & Rank Final Node Set ───────────────────────────────
        # Remove discarded nodes, rank by best HNSW similarity across iterations
        curated = [
            (score, node_id)
            for node_id, score in all_retrieved.items()
            if node_id not in all_discard_ids
        ]
        curated.sort(key=lambda x: x[0], reverse=True)  # best similarity first
        curated_ids = [node_id for _, node_id in curated]

        final_coverage = final_critique.coverage_score if final_critique else 0.0

        logger.info(
            f"AgenticRAGOrchestrator: DONE | "
            f"iterations={len(iteration_logs)} | "
            f"converged={converged} | "
            f"curated_nodes={len(curated_ids)} | "
            f"discarded={len(all_discard_ids)} | "
            f"final_coverage={final_coverage:.3f}"
        )

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
                "hnsw_stats": retriever.build_stats,
                "corpus_size": len(candidate_nodes),
                "unique_retrieved": len(all_retrieved),
                "hnsw_build_time_ms": build_ms,
            },
        )

    def curate_nodes_from_result(
        self,
        rag_result: AgenticRAGResult,
        node_map: Dict[str, EvidenceNode],
    ) -> List[EvidenceNode]:
        """Convert AgenticRAGResult's ordered node IDs back to EvidenceNode objects.

        Parameters
        ----------
        rag_result : AgenticRAGResult
            The result of the orchestrate() call.
        node_map : Dict[str, EvidenceNode]
            Mapping from node_id to EvidenceNode (original corpus).

        Returns
        -------
        List[EvidenceNode]
            Curated, RAG-ranked EvidenceNode objects in score-descending order.
        """
        curated_nodes = []
        for node_id in rag_result.curated_node_ids:
            if node_id in node_map:
                curated_nodes.append(node_map[node_id])
        return curated_nodes
