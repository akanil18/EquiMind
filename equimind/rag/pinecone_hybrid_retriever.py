"""
PineconeHybridRetriever — Dense (Pinecone) + Sparse (BM25) + RRF for EquiMind.

This wires together:
  - PineconeVectorStore (persistent dense search)
  - BM25Index (in-memory sparse search built fresh per request)
  - Reciprocal Rank Fusion to merge both result sets
"""

import logging
from typing import Dict, List, Optional, Tuple

from equimind.evidence.schema import EvidenceNode, EvidenceSource
from equimind.rag.hybrid_retriever import BM25Index
from equimind.rag.pinecone_store import PineconeVectorStore
from equimind.rag.embedder import EmbeddingRouter

logger = logging.getLogger(__name__)


class PineconeHybridRetriever:
    """Production Hybrid Retriever: Pinecone Dense + BM25 Sparse + RRF.

    Interview talking points:
      - Dense: Pinecone Serverless (HNSW managed, cosine similarity, metadata filter)
      - Sparse: BM25 over current session's EvidenceNodes (exact keyword match)
      - RRF: score = 1/(60 + rank_dense) + 1/(60 + rank_sparse)
    """

    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        index_name: str = "equimind",
        rrf_k: int = 60,
        provider=None,
    ):
        self.embedder = EmbeddingRouter(provider=provider)
        self.pinecone_store = PineconeVectorStore(
            api_key=pinecone_api_key,
            index_name=index_name,
            embedder=self.embedder,
        )
        self.bm25_index = BM25Index()
        self.rrf_k = rrf_k
        self._local_node_map: Dict[str, EvidenceNode] = {}
        self._is_indexed = False

    def index_nodes(self, nodes: List[EvidenceNode]) -> int:
        """Persists nodes to Pinecone (dense) and builds BM25 (sparse) for current session."""
        if not nodes:
            return 0

        # Update local node map for BM25 + result reconstruction
        for n in nodes:
            self._local_node_map[n.id] = n

        # Upsert to Pinecone (persistent)
        pinecone_count = self.pinecone_store.upsert(nodes)

        # Build BM25 on current session nodes (in-memory, fast)
        self.bm25_index.index(list(self._local_node_map.values()))
        self._is_indexed = True

        if self.pinecone_store.is_connected:
            logger.info(
                f"PineconeHybridRetriever: Upserted {pinecone_count} vectors to Pinecone | "
                f"BM25 indexed {len(self._local_node_map)} nodes"
            )
        else:
            logger.warning(
                f"PineconeHybridRetriever: Pinecone not connected — "
                f"BM25 only for {len(self._local_node_map)} nodes"
            )
        return pinecone_count

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        ticker: Optional[str] = None,
        source_types: Optional[List[EvidenceSource]] = None,
    ) -> List[Tuple[float, EvidenceNode]]:
        """Hybrid dense + sparse retrieval with RRF fusion.

        Steps:
          1. Dense: query Pinecone → top_k * 2 candidates
          2. Sparse: query BM25 → top_k * 2 candidates
          3. Merge: Reciprocal Rank Fusion
          4. Reconstruct EvidenceNode objects from local map
        """
        if not self._is_indexed:
            logger.warning("PineconeHybridRetriever: index_nodes() not called.")
            return []

        # 1. Dense retrieval from Pinecone
        dense_matches = self.pinecone_store.search(
            query=query,
            top_k=top_k * 2,
            ticker=ticker,
            source_types=source_types,
        )

        # 2. Sparse BM25 retrieval (in-memory, current session)
        sparse_matches = self.bm25_index.search(query, top_k=top_k * 2)

        # 3. RRF fusion
        rrf_scores: Dict[str, float] = {}

        # Pinecone returns (score, metadata_dict) — use _pinecone_id for node lookup
        for rank, (score, meta) in enumerate(dense_matches):
            nid = meta.get("_pinecone_id", "")
            if nid and nid in self._local_node_map:
                rrf_scores[nid] = rrf_scores.get(nid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # BM25 returns (score, EvidenceNode)
        for rank, (score, node) in enumerate(sparse_matches):
            rrf_scores[node.id] = rrf_scores.get(node.id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # 4. Sort by RRF score and reconstruct EvidenceNodes
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = sorted_ids[0][1] if sorted_ids else 1.0

        results: List[Tuple[float, EvidenceNode]] = []
        for nid, score in sorted_ids:
            if nid in self._local_node_map:
                norm_score = score / max_score
                results.append((norm_score, self._local_node_map[nid]))

        return results

    def expand_queries(self, query: str, ticker: str = "") -> List[str]:
        """Multi-query expansion for better recall."""
        queries = [query]
        t = ticker.upper()
        queries.append(f"{t} {query} SEC 10-K quarterly earnings balance sheet valuation")
        queries.append(f"{t} {query} risk factors competition margin headwinds downside")
        return queries

    @property
    def pinecone_stats(self) -> dict:
        return self.pinecone_store.stats()
