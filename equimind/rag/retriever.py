"""
HNSW-backed Evidence Retriever.

Wraps the HNSWIndex and EmbeddingRouter into a single interface that accepts
a list of EvidenceNodes, builds the vector index, and exposes `retrieve(query, k)`
to return semantically ranked EvidenceNodes with their HNSW similarity scores.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from equimind.evidence.schema import EvidenceNode
from equimind.rag.embedder import EmbeddingRouter
from equimind.rag.hnsw_index import HNSWIndex

logger = logging.getLogger(__name__)


class HNSWRetriever:
    """HNSW-powered semantic retriever for EvidenceNode collections.

    Lifecycle
    ---------
    1. Instantiate with provider (optional, for LLM embeddings)
    2. Call `build(nodes)` to embed all nodes and construct the HNSW index
    3. Call `retrieve(query, k)` to get top-k nodes for a query

    The retriever stores HNSW similarity scores on each retrieved node via the
    `rag_retrieval_score` field so downstream components can use them.
    """

    def __init__(
        self,
        provider=None,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> None:
        self.provider = provider
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._embedder: Optional[EmbeddingRouter] = None
        self._index: Optional[HNSWIndex] = None
        self._node_map: Dict[str, EvidenceNode] = {}   # node_id → EvidenceNode
        self._is_built = False
        self._build_stats: Dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Build
    # ──────────────────────────────────────────────────────────────────────────

    def build(self, nodes: List[EvidenceNode]) -> "HNSWRetriever":
        """Embed all nodes and construct the HNSW index.

        Parameters
        ----------
        nodes : List[EvidenceNode]
            All candidate evidence nodes from the research pipeline.
        """
        if not nodes:
            logger.warning("HNSWRetriever.build: received empty node list")
            self._is_built = True
            return self

        t0 = time.time()
        self._node_map = {n.id: n for n in nodes}

        # ── 1. Prepare corpus for embedding ──
        # Concatenate title + content for richer representation
        corpus_texts = [
            f"{n.title}. {n.content[:512]}" for n in nodes
        ]

        # ── 2. Fit embedder on corpus ──
        self._embedder = EmbeddingRouter(provider=self.provider)
        self._embedder.fit_corpus(corpus_texts)
        embed_dim = self._embedder.output_dim

        # ── 3. Embed all nodes ──
        t_embed = time.time()
        embeddings = self._embedder.embed_batch(corpus_texts)   # (N, dim)
        embed_ms = (time.time() - t_embed) * 1000

        # ── 4. Build HNSW index ──
        t_build = time.time()
        self._index = HNSWIndex(
            dim=embed_dim,
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
        )
        node_vectors = {
            nodes[i].id: embeddings[i] for i in range(len(nodes))
        }
        inserted = self._index.build_from_vectors(node_vectors)
        build_ms = (time.time() - t_build) * 1000

        # ── 5. Store pre-computed embeddings on nodes (if field exists) ──
        for i, node in enumerate(nodes):
            if hasattr(node, "vector_embedding") and node.vector_embedding is None:
                node.vector_embedding = embeddings[i].tolist()

        total_ms = (time.time() - t0) * 1000
        self._is_built = True
        self._build_stats = {
            "nodes_indexed": inserted,
            "embed_dim": embed_dim,
            "embed_time_ms": round(embed_ms, 2),
            "hnsw_build_time_ms": round(build_ms, 2),
            "total_build_time_ms": round(total_ms, 2),
        }
        logger.info(
            f"HNSWRetriever built: {inserted} nodes, dim={embed_dim}, "
            f"embed={embed_ms:.0f}ms, hnsw_build={build_ms:.0f}ms"
        )
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieve
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 20,
        ef: Optional[int] = None,
    ) -> List[Tuple[float, EvidenceNode]]:
        """Retrieve top-k evidence nodes semantically closest to the query.

        Parameters
        ----------
        query : str
            The search query (investment question, sub-query, etc.)
        k : int
            Number of nodes to return.
        ef : int, optional
            Search beam width override. Defaults to max(k, ef_search).

        Returns
        -------
        List[Tuple[float, EvidenceNode]]
            Sorted list of (similarity_score, node) tuples — score in [0, 1],
            1 being most similar. Best matches first.
        """
        if not self._is_built:
            raise RuntimeError("HNSWRetriever.retrieve: call build() first")
        if not self._node_map:
            return []

        t0 = time.time()

        # Embed query
        query_vec = self._embedder.embed(query)
        search_ef = ef or max(k, self.ef_search)

        # HNSW search: returns (distance, node_id) where distance ∈ [0, 2]
        raw_results = self._index.search(query_vec, k=k, ef=search_ef)

        # Convert cosine distance → similarity score ∈ [0, 1]
        results: List[Tuple[float, EvidenceNode]] = []
        for dist, node_id in raw_results:
            if node_id not in self._node_map:
                continue
            # cosine_similarity = 1 - distance (distance ∈ [0, 2])
            similarity = max(0.0, min(1.0, 1.0 - dist))
            node = self._node_map[node_id]

            # Annotate retrieval score on the node
            if hasattr(node, "rag_retrieval_score"):
                node.rag_retrieval_score = round(similarity, 4)

            results.append((similarity, node))

        # Sort descending by similarity
        results.sort(key=lambda x: x[0], reverse=True)

        search_ms = (time.time() - t0) * 1000
        logger.debug(
            f"HNSWRetriever.retrieve: query='{query[:60]}...' "
            f"k={k}, results={len(results)}, search={search_ms:.1f}ms"
        )
        return results

    def retrieve_nodes(self, query: str, k: int = 20) -> List[EvidenceNode]:
        """Convenience wrapper returning just the EvidenceNode objects."""
        return [node for _, node in self.retrieve(query, k=k)]

    @property
    def build_stats(self) -> Dict:
        return self._build_stats

    @property
    def is_built(self) -> bool:
        return self._is_built

    def __len__(self) -> int:
        return len(self._node_map)
