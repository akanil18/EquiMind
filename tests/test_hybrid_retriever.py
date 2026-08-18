"""
Tests for HybridRetriever (Dense HNSW + Sparse BM25 + RRF).
"""

import unittest
from equimind.evidence.schema import EvidenceNode, EvidenceSource
from equimind.rag.hybrid_retriever import HybridRetriever, BM25Index
from equimind.rag.vector_store import HNSWVectorStore, MetadataFilter


class TestHybridRetriever(unittest.TestCase):

    def setUp(self):
        self.nodes = [
            EvidenceNode(
                source_type=EvidenceSource.SEC_FILING,
                title="NVDA 10-K Annual Report",
                content="Annual report detailing GPU architecture, Hopper and Blackwell datacenter revenue.",
                affected_ticker="NVDA",
            ),
            EvidenceNode(
                source_type=EvidenceSource.FINANCIAL_NEWS,
                title="Semiconductor Industry Market Share",
                content="NVIDIA dominates AI hardware accelerator market with over 80% market share.",
                affected_ticker="NVDA",
            ),
            EvidenceNode(
                source_type=EvidenceSource.FINANCIAL_STATEMENTS,
                title="AAPL Balance Sheet Q4",
                content="Apple reports cash reserves of $160B and long-term debt of $95B.",
                affected_ticker="AAPL",
            ),
        ]
        self.retriever = HybridRetriever()
        self.retriever.index_nodes(self.nodes)

    def test_hybrid_retrieve(self):
        results = self.retriever.retrieve("NVIDIA GPU Blackwell AI market share", top_k=2)
        self.assertGreater(len(results), 0)
        top_score, top_node = results[0]
        self.assertEqual(top_node.affected_ticker, "NVDA")
        self.assertGreater(top_score, 0.0)

    def test_bm25_exact_keyword_matching(self):
        bm25 = BM25Index()
        bm25.index(self.nodes)
        results = bm25.search("Blackwell Hopper", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("10-K", results[0][1].title)

    def test_expand_queries(self):
        queries = self.retriever.expand_queries("NVDA valuation risks", ticker="NVDA")
        self.assertGreaterEqual(len(queries), 2)
        self.assertIn("NVDA valuation risks", queries)


if __name__ == "__main__":
    unittest.main()
