"""
Tests for CrossEncoderReranker.
"""

import unittest
from equimind.evidence.schema import EvidenceNode, EvidenceSource, AuthorCredibility
from equimind.rag.reranker import CrossEncoderReranker


class TestReranker(unittest.TestCase):

    def setUp(self):
        self.reranker = CrossEncoderReranker(latency_budget_ms=200.0)
        self.candidates = [
            EvidenceNode(
                source_type=EvidenceSource.FINANCIAL_NEWS,
                title="Generic tech market article",
                content="Tech stocks moved sideways today with light trading volume across indices.",
                affected_ticker="GENERAL",
                author_credibility=AuthorCredibility.LOW,
            ),
            EvidenceNode(
                source_type=EvidenceSource.SEC_FILING,
                title="NVDA SEC 10-Q Segment Revenue",
                content="NVIDIA reported Data Center segment revenue of $14.5 billion, representing 279% YoY growth.",
                affected_ticker="NVDA",
                author_credibility=AuthorCredibility.VERIFIED_OFFICIAL,
            ),
        ]

    def test_rerank_prioritizes_relevant_verified_node(self):
        results = self.reranker.rerank("NVIDIA Data Center revenue growth", self.candidates, top_n=2)
        self.assertEqual(len(results), 2)
        top_score, top_node = results[0]
        self.assertEqual(top_node.affected_ticker, "NVDA")
        self.assertIn("10-Q", top_node.title)


if __name__ == "__main__":
    unittest.main()
