"""
HybridRetriever — Combining Dense Semantic Search (HNSW) + Sparse Lexical Search (BM25) + RRF.

Key Features:
  - Dense Search: Embeds queries into semantic space, queries HNSW index
  - Sparse Search: Pure Python BM25 index over tokenized document texts
  - Reciprocal Rank Fusion (RRF):
      Score(d) = (1 / (60 + rank_dense)) + (1 / (60 + rank_sparse))
  - Query Rewriting & Multi-Query Expansion (HyDE / Sub-Query generation)
  - Metadata filtering support
"""

import math
import re
import logging
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple, Set

from equimind.evidence.schema import EvidenceNode
from equimind.rag.vector_store import VectorStore, HNSWVectorStore, MetadataFilter
from equimind.providers.base import LLMProvider, LLMMessage, Role

logger = logging.getLogger(__name__)


class BM25Index:
    """In-memory BM25 Okapi sparse search implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: Dict[str, EvidenceNode] = {}
        self._doc_lens: Dict[str, int] = {}
        self._avg_dl: float = 0.0
        self._doc_freqs: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._inverted_index: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: term_freq}

    def index(self, nodes: List[EvidenceNode]) -> int:
        """Build BM25 inverted index on EvidenceNodes."""
        self._corpus = {n.id: n for n in nodes}
        self._doc_lens = {}
        self._doc_freqs = Counter()
        self._inverted_index = {}

        total_len = 0
        for node in nodes:
            tokens = self._tokenize(f"{node.title} {node.content}")
            doc_len = len(tokens)
            self._doc_lens[node.id] = doc_len
            total_len += doc_len

            counts = Counter(tokens)
            for term, count in counts.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = {}
                self._inverted_index[term][node.id] = count
                self._doc_freqs[term] += 1

        n_docs = len(nodes)
        self._avg_dl = total_len / n_docs if n_docs > 0 else 1.0

        # Precompute IDF: log((N - df + 0.5) / (df + 0.5) + 1.0)
        self._idf = {}
        for term, df in self._doc_freqs.items():
            self._idf[term] = math.log(((n_docs - df + 0.5) / (df + 0.5)) + 1.0)

        return len(nodes)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[MetadataFilter] = None,
    ) -> List[Tuple[float, EvidenceNode]]:
        """Search BM25 index with query tokens."""
        if not self._corpus:
            return []

        tokens = self._tokenize(query)
        scores: Dict[str, float] = Counter()

        for term in tokens:
            if term not in self._inverted_index:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_id, tf in self._inverted_index[term].items():
                node = self._corpus[doc_id]
                if filters and not filters.matches(node):
                    continue
                doc_len = self._doc_lens.get(doc_id, self._avg_dl)
                # BM25 term weighting formula
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avg_dl))
                scores[doc_id] += idf * (numerator / denominator)

        if not scores:
            return []

        # Sort and return top_k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = sorted_docs[0][1] if sorted_docs and sorted_docs[0][1] > 0 else 1.0
        
        return [(score / max_score, self._corpus[doc_id]) for doc_id, score in sorted_docs]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"\b[a-zA-Z0-9_\-\$]{2,30}\b", text)]


class HybridRetriever:
    """Production Hybrid Retriever merging Dense (HNSW) + Sparse (BM25) via RRF."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        provider: Optional[LLMProvider] = None,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store or HNSWVectorStore()
        self.bm25_index = BM25Index()
        self.provider = provider
        self.rrf_k = rrf_k
        self._is_indexed = False

    def index_nodes(self, nodes: List[EvidenceNode]) -> int:
        """Indexes nodes into both Dense HNSW vector store and Sparse BM25 index."""
        if not nodes:
            return 0
        dense_count = self.vector_store.upsert(nodes)
        sparse_count = self.bm25_index.index(nodes)
        self._is_indexed = True
        logger.info(f"HybridRetriever: Indexed {dense_count} dense nodes and {sparse_count} BM25 nodes.")
        return dense_count

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[MetadataFilter] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[Tuple[float, EvidenceNode]]:
        """Executes hybrid dense + sparse retrieval and merges with Reciprocal Rank Fusion."""
        if not self._is_indexed:
            logger.warning("HybridRetriever: index_nodes() was not called before retrieve().")
            return []

        # 1. Dense retrieval
        dense_results = self.vector_store.search(query, top_k=top_k * 2, filters=filters)
        
        # 2. Sparse BM25 retrieval
        sparse_results = self.bm25_index.search(query, top_k=top_k * 2, filters=filters)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        node_map: Dict[str, EvidenceNode] = {}

        for rank, (score, node) in enumerate(dense_results):
            node_map[node.id] = node
            rrf_scores[node.id] = rrf_scores.get(node.id, 0.0) + dense_weight * (1.0 / (self.rrf_k + rank + 1))

        for rank, (score, node) in enumerate(sparse_results):
            node_map[node.id] = node
            rrf_scores[node.id] = rrf_scores.get(node.id, 0.0) + sparse_weight * (1.0 / (self.rrf_k + rank + 1))

        # Sort by final RRF score descending
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        max_rrf = sorted_rrf[0][1] if sorted_rrf and sorted_rrf[0][1] > 0 else 1.0
        normalized_results = [(score / max_rrf, node_map[nid]) for nid, score in sorted_rrf]
        
        return normalized_results

    def expand_queries(self, query: str, ticker: str = "") -> List[str]:
        """Multi-query expansion: generates alternative financial query perspectives."""
        queries = [query]
        t_upper = ticker.upper()

        if self.provider and self.provider.is_available():
            try:
                prompt = (
                    f"Given the financial research query: '{query}' for ticker '{t_upper}', "
                    "generate 2 distinct search queries targeting: "
                    "1) SEC financial filings & valuation fundamentals, "
                    "2) Macro headwinds, industry competition & downside risks. "
                    "Output one query per line."
                )
                resp = self.provider.generate([LLMMessage(role=Role.USER, content=prompt)], max_tokens=100)
                for line in resp.content.strip().split("\n"):
                    clean = line.strip().strip("1234567890.- ")
                    if clean and len(clean) > 5:
                        queries.append(clean)
            except Exception as ex:
                logger.debug(f"Query expansion fallback: {ex}")

        if len(queries) == 1:
            # Deterministic multi-query expansion
            queries.append(f"{t_upper} {query} SEC 10-K quarterly earnings balance sheet valuation")
            queries.append(f"{t_upper} {query} risk factors competition margin headwinds")

        return queries
