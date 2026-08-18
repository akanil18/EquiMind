"""
Tests for VectorStore, HNSWVectorStore, and MetadataFilter.
"""

import unittest
from datetime import datetime, timezone

from equimind.evidence.schema import EvidenceNode, EvidenceSource, AuthorCredibility, SentimentPolarity
from equimind.rag.vector_store import HNSWVectorStore, MetadataFilter


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.store = HNSWVectorStore(M=8, ef_construction=50, ef_search=20)
        self.nodes = [
            EvidenceNode(
                source_type=EvidenceSource.SEC_FILING,
                title="NVDA 10-Q Filing Q3",
                content="NVIDIA quarterly revenue reached $18.1B driven by datacenter AI chips.",
                affected_ticker="NVDA",
                tags=["sec", "revenue", "gpu"],
                confidence_score=0.95,
                author_credibility=AuthorCredibility.VERIFIED_OFFICIAL,
            ),
            EvidenceNode(
                source_type=EvidenceSource.FINANCIAL_NEWS,
                title="AAPL iPhone Sales Surge in China",
                content="Apple reported strong iPhone 15 sales rebound in Asian markets.",
                affected_ticker="AAPL",
                tags=["news", "iphone"],
                confidence_score=0.85,
            ),
            EvidenceNode(
                source_type=EvidenceSource.MACRO_DATA,
                title="Fed Rate Decision",
                content="Federal Reserve keeps benchmark interest rates unchanged at 5.25%.",
                affected_ticker="GENERAL",
                tags=["macro", "interest_rate"],
                confidence_score=0.90,
            ),
        ]

    def test_upsert_and_count(self):
        inserted = self.store.upsert(self.nodes)
        self.assertEqual(inserted, 3)
        self.assertEqual(self.store.count(), 3)

    def test_search_basic(self):
        self.store.upsert(self.nodes)
        results = self.store.search("NVDA AI datacenter chips", top_k=2)
        self.assertGreater(len(results), 0)
        top_score, top_node = results[0]
        self.assertEqual(top_node.affected_ticker, "NVDA")
        self.assertGreater(top_score, 0.0)

    def test_search_with_ticker_filter(self):
        self.store.upsert(self.nodes)
        filter_spec = MetadataFilter(ticker="AAPL")
        results = self.store.search("revenue and growth", top_k=5, filters=filter_spec)
        for _, node in results:
            self.assertEqual(node.affected_ticker, "AAPL")

    def test_delete_tombstone(self):
        self.store.upsert(self.nodes)
        del_count = self.store.delete([self.nodes[0].id])
        self.assertEqual(del_count, 1)
        self.assertEqual(self.store.count(), 2)

        results = self.store.search("NVDA", top_k=5)
        returned_ids = [n.id for _, n in results]
        self.assertNotIn(self.nodes[0].id, returned_ids)


if __name__ == "__main__":
    unittest.main()
