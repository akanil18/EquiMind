import unittest
from datetime import datetime, timezone

from equimind.evidence.schema import (
    EvidenceNode,
    EvidenceSource,
    AuthorCredibility,
    SentimentPolarity,
)
from equimind.committee.schema import InvestmentRating, InvestmentRecommendation
from equimind.committee.bull_agent import BullAgent
from equimind.committee.bear_agent import BearAgent
from equimind.committee.judge_agent import JudgeAgent


class TestInvestmentCommittee(unittest.TestCase):

    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.bull_node = EvidenceNode(
            source_type=EvidenceSource.SEC_FILING,
            title="NVIDIA 10-Q Revenue Surge",
            content="Official quarterly filing confirms 122% revenue growth.",
            publication_timestamp=self.now,
            author="SEC EDGAR",
            author_credibility=AuthorCredibility.VERIFIED_OFFICIAL,
            confidence_score=0.98,
            sentiment=SentimentPolarity.VERY_BULLISH,
            affected_ticker="NVDA",
            url="https://www.sec.gov",
        )

        self.bear_node = EvidenceNode(
            source_type=EvidenceSource.TWITTER_X,
            title="Valuation Multiple Caution",
            content="PE multiple expanded rapidly over past month.",
            publication_timestamp=self.now,
            author="MarketAnalyst",
            author_credibility=AuthorCredibility.MEDIUM,
            confidence_score=0.70,
            sentiment=SentimentPolarity.BEARISH,
            affected_ticker="NVDA",
        )

        self.nodes = [self.bull_node, self.bear_node]
        self.quant_summary = {"last_price": 125.0}
        self.risk_summary = {"annualized_volatility_pct": 25.0}

    def test_bull_agent(self):
        bull_case = BullAgent.evaluate(ticker="NVDA", nodes=self.nodes, quant_summary=self.quant_summary)
        self.assertIn("NVDA", bull_case.thesis)
        self.assertTrue(len(bull_case.key_catalysts) >= 1)
        self.assertGreater(bull_case.upside_price_target, 125.0)

    def test_bear_agent(self):
        bear_case = BearAgent.evaluate(ticker="NVDA", nodes=self.nodes, quant_summary=self.quant_summary)
        self.assertIn("NVDA", bear_case.thesis)
        self.assertTrue(len(bear_case.key_headwinds) >= 1)
        self.assertLess(bear_case.downside_price_target, 125.0)

    def test_judge_agent_debate_synthesis(self):
        bull_case = BullAgent.evaluate(ticker="NVDA", nodes=self.nodes, quant_summary=self.quant_summary)
        bear_case = BearAgent.evaluate(ticker="NVDA", nodes=self.nodes, quant_summary=self.quant_summary)

        rec = JudgeAgent.evaluate_debate(
            ticker="NVDA",
            bull_case=bull_case,
            bear_case=bear_case,
            nodes=self.nodes,
            quant_summary=self.quant_summary,
            risk_summary=self.risk_summary,
        )

        self.assertIsInstance(rec, InvestmentRecommendation)
        self.assertEqual(rec.ticker, "NVDA")
        self.assertIn(rec.rating, [InvestmentRating.BUY, InvestmentRating.STRONG_BUY])
        self.assertGreater(rec.conviction_score, 0.5)
        self.assertTrue(len(rec.provenance_citations) == 2)
        self.assertIn("sec.gov", rec.provenance_citations[0]["url"])


if __name__ == "__main__":
    unittest.main()
