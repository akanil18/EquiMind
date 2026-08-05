import unittest
from datetime import datetime, timezone, timedelta

from equimind.evidence.schema import (
    EvidenceNode,
    EvidenceSource,
    AuthorCredibility,
    SentimentPolarity,
    EdgeType,
)
from equimind.evidence.graph import EvidenceGraph
from equimind.context.compressor import ContextCompressor


class TestEvidenceAndContext(unittest.TestCase):

    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.node1 = EvidenceNode(
            source_type=EvidenceSource.FINANCIAL_NEWS,
            title="NVIDIA Announces New Blackwell GPU Sales Surge",
            content="NVIDIA reported extraordinary demand for its Blackwell AI architecture.",
            publication_timestamp=self.now,
            author="Reuters Tech",
            author_credibility=AuthorCredibility.VERIFIED_OFFICIAL,
            confidence_score=0.95,
            sentiment=SentimentPolarity.VERY_BULLISH,
            affected_ticker="NVDA",
            tags=["AI", "GPU", "Blackwell"],
        )

        self.node2 = EvidenceNode(
            source_type=EvidenceSource.REDDIT,
            title="NVIDIA Blackwell GPU Sales Surge Big",
            content="NVIDIA reported extraordinary demand for its Blackwell AI architecture.",  # exact content match
            publication_timestamp=self.now,
            author="RedditUser123",
            author_credibility=AuthorCredibility.LOW,
            confidence_score=0.6,
            sentiment=SentimentPolarity.BULLISH,
            affected_ticker="NVDA",
        )

        self.node3 = EvidenceNode(
            source_type=EvidenceSource.TWITTER_X,
            title="Nvidia Blackwell demand is insane",
            content="NVIDIA reported extraordinary demand for Blackwell chips across cloud providers.",  # fuzzy match
            publication_timestamp=self.now - timedelta(days=20),  # older
            author="SemiAnalyst",
            author_credibility=AuthorCredibility.HIGH,
            confidence_score=0.85,
            sentiment=SentimentPolarity.BULLISH,
            affected_ticker="NVDA",
        )

    def test_evidence_graph_and_edges(self):
        graph = EvidenceGraph()
        id1 = graph.add_node(self.node1)
        id3 = graph.add_node(self.node3)

        edge = graph.add_edge(
            source_id=id3,
            target_id=id1,
            edge_type=EdgeType.CORROBORATES,
            description="Twitter analyst corroborates Reuters news report",
        )

        self.assertIsNotNone(edge)
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)

        nvda_nodes = graph.get_nodes_for_ticker("NVDA")
        self.assertEqual(len(nvda_nodes), 2)

        # Test JSON serialization & deserialization
        json_str = graph.to_json()
        loaded_graph = EvidenceGraph.from_json(json_str)
        self.assertEqual(len(loaded_graph.nodes), 2)
        self.assertEqual(len(loaded_graph.edges), 1)

    def test_exact_deduplication(self):
        nodes = [self.node1, self.node2]
        deduped = ContextCompressor.exact_deduplicate(nodes)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].id, self.node1.id)

    def test_fuzzy_clustering(self):
        nodes = [self.node1, self.node3]
        clustered = ContextCompressor.fuzzy_cluster_deduplicate(nodes, similarity_threshold=0.4)
        self.assertEqual(len(clustered), 1)
        self.assertEqual(clustered[0].author_credibility, AuthorCredibility.VERIFIED_OFFICIAL)

    def test_context_compressor_budget_packing(self):
        nodes = [self.node1, self.node2, self.node3]
        compressed = ContextCompressor.compress(
            nodes=nodes,
            query_context="Blackwell AI GPU demand NVDA",
            max_token_budget=500,
        )
        self.assertTrue(len(compressed) >= 1)
        # Should prioritize node1 (highest credibility and recent timestamp)
        self.assertEqual(compressed[0].id, self.node1.id)


if __name__ == "__main__":
    unittest.main()
