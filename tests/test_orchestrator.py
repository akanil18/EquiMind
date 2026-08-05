import unittest
from equimind.orchestrator.engine import EquiMindEngine


class TestMasterOrchestrator(unittest.TestCase):

    def test_end_to_end_equity_research(self):
        engine = EquiMindEngine()
        results = engine.analyze_equity(
            ticker="NVDA",
            query="Should I invest in NVIDIA today for AI growth?",
            provider_name="mock",
        )

        self.assertEqual(results["ticker"], "NVDA")
        self.assertIn("recommendation", results)
        rec = results["recommendation"]
        
        self.assertIn("rating", rec)
        self.assertGreater(rec["conviction_score"], 0.0)
        self.assertTrue(len(rec["target_entry_range"]) == 2)
        self.assertIn("bull_case", rec)
        self.assertIn("bear_case", rec)
        self.assertIn("debate_synthesis", rec)
        self.assertTrue(len(rec["provenance_citations"]) >= 1)

        # Check persistent memory integration
        entity = engine.memory_store.ticker_knowledge.get("NVDA")
        self.assertIsNotNone(entity)
        self.assertEqual(len(entity.historical_reports), 1)


if __name__ == "__main__":
    unittest.main()
