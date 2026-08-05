import unittest
from datetime import datetime, timezone

from equimind.evidence.schema import (
    EvidenceNode,
    EvidenceSource,
    AuthorCredibility,
    SentimentPolarity,
)
from equimind.memory.hierarchical_store import HierarchicalMemoryStore
from equimind.memory.delta_engine import DeltaResearchEngine


class TestHierarchicalMemoryAndDelta(unittest.TestCase):

    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.store = HierarchicalMemoryStore()
        self.node1 = EvidenceNode(
            source_type=EvidenceSource.FINANCIAL_NEWS,
            title="NVIDIA Q2 Revenue Beat",
            content="NVIDIA reported record quarterly revenue.",
            publication_timestamp=self.now,
            author="Reuters",
            affected_ticker="NVDA",
        )

    def test_store_and_retrieve_report(self):
        report = self.store.store_research_report(
            ticker="NVDA",
            user_query="Analyze NVDA",
            rating="STRONG_BUY",
            conviction_score=0.90,
            summary="Strong AI datacenter growth",
            evidence_nodes=[self.node1],
        )

        self.assertIsNotNone(report.id)
        self.assertEqual(report.ticker, "NVDA")
        
        last_report = self.store.get_last_report("NVDA")
        self.assertIsNotNone(last_report)
        self.assertEqual(last_report.rating, "STRONG_BUY")

        entity = self.store.get_or_create_entity("NVDA")
        self.assertEqual(len(entity.cumulative_evidence_nodes), 1)
        self.assertIn("Updated thesis", entity.persistent_thesis)

        # Serialization test
        json_str = self.store.to_json()
        loaded_store = HierarchicalMemoryStore.from_json(json_str)
        self.assertIn("NVDA", loaded_store.ticker_knowledge)

    def test_delta_research_engine(self):
        # 1. First research call (no prior data)
        has_prev, last_time, cached = DeltaResearchEngine.compute_delta_research_plan("AAPL", self.store)
        self.assertFalse(has_prev)
        self.assertIsNone(last_time)
        self.assertEqual(len(cached), 0)

        # 2. Store a report for AAPL
        self.store.store_research_report(
            ticker="AAPL",
            user_query="Analyze Apple",
            rating="BUY",
            conviction_score=0.80,
            summary="iPhone upgrade cycle",
            evidence_nodes=[self.node1],
        )

        # 3. Second research call (prior data exists)
        has_prev2, last_time2, cached2 = DeltaResearchEngine.compute_delta_research_plan("AAPL", self.store)
        self.assertTrue(has_prev2)
        self.assertIsNotNone(last_time2)
        self.assertEqual(len(cached2), 1)


if __name__ == "__main__":
    unittest.main()
