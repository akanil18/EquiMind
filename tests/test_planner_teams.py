import unittest
from datetime import datetime, timezone, timedelta

from equimind.planner.reasoning_planner import ReasoningPlanner, SectorType, InvestmentHorizon
from equimind.teams.market_data_team import MarketDataTeam
from equimind.teams.fundamental_team import FundamentalTeam
from equimind.teams.macro_team import MacroTeam
from equimind.teams.web_intelligence_team import WebIntelligenceTeam
from equimind.time_machine.temporal_guard import TemporalGuard
from equimind.evidence.schema import EvidenceNode, EvidenceSource, AuthorCredibility, SentimentPolarity


class TestPlannerAndTeams(unittest.TestCase):

    def test_reasoning_planner_sector_detection(self):
        plan_nvda = ReasoningPlanner.plan(query="Should I invest in NVIDIA today?", ticker="NVDA")
        self.assertEqual(plan_nvda.sector, SectorType.SEMICONDUCTOR_TECH)
        self.assertIn("github_commits", plan_nvda.active_adapters)
        self.assertIn("GPU/AI Demand", plan_nvda.focus_areas)

        plan_jpm = ReasoningPlanner.plan(query="Analyze JPMorgan for long-term investment", ticker="JPM")
        self.assertEqual(plan_jpm.sector, SectorType.BANKING_FINANCE)
        self.assertIn("Net Interest Margin (NIM)", plan_jpm.focus_areas)

        plan_dal = ReasoningPlanner.plan(query="Analyze Delta Air Lines", ticker="DAL")
        self.assertEqual(plan_dal.sector, SectorType.AIRLINE_TRANSPORT)

    def test_market_data_team(self):
        team = MarketDataTeam()
        nodes = team.research(ticker="NVDA", query="Technical prices")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].source_type, EvidenceSource.MARKET_PRICES)
        self.assertIn("rsi_14", nodes[0].metadata)

    def test_fundamental_team(self):
        team = FundamentalTeam()
        nodes = team.research(ticker="NVDA", query="Financial statements")
        self.assertEqual(len(nodes), 1)
        self.assertIn("valuation", nodes[0].metadata)
        self.assertIn("profitability", nodes[0].metadata)

    def test_macro_team(self):
        team = MacroTeam()
        nodes = team.research(ticker="NVDA", query="Inflation and interest rates")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].source_type, EvidenceSource.MACRO_DATA)
        self.assertIn("us_cpi_yoy_pct", nodes[0].metadata)

    def test_web_intelligence_team(self):
        team = WebIntelligenceTeam()
        nodes = team.research(ticker="NVDA", query="SEC filings and financial news")
        self.assertTrue(len(nodes) >= 2)
        source_types = {n.source_type for n in nodes}
        self.assertIn(EvidenceSource.SEC_FILING, source_types)
        self.assertIn(EvidenceSource.FINANCIAL_NEWS, source_types)

    def test_temporal_guard_future_pruning(self):
        cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
        guard = TemporalGuard(as_of_date=cutoff)

        past_node = EvidenceNode(
            source_type=EvidenceSource.FINANCIAL_NEWS,
            title="Past News 2023",
            content="Historical news item",
            publication_timestamp=datetime(2023, 12, 15, tzinfo=timezone.utc),
            affected_ticker="TSLA",
        )

        future_node = EvidenceNode(
            source_type=EvidenceSource.FINANCIAL_NEWS,
            title="Future News 2024",
            content="Future news item published after cutoff",
            publication_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            affected_ticker="TSLA",
        )

        nodes = [past_node, future_node]
        filtered = guard.filter_evidence(nodes)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].title, "Past News 2023")


if __name__ == "__main__":
    unittest.main()
