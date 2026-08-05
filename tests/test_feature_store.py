import unittest
from datetime import datetime, timezone

from equimind.evidence.schema import EvidenceNode, EvidenceSource, AuthorCredibility, SentimentPolarity
from equimind.features.schema import FeatureVector, FeatureSet
from equimind.features.feature_store import FeatureStore


class TestFeatureStore(unittest.TestCase):

    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.nodes = [
            EvidenceNode(
                source_type=EvidenceSource.SEC_FILING,
                title="10-Q Datacenter Growth",
                content="Datacenter revenue grew 122% YoY",
                publication_timestamp=self.now,
                author_credibility=AuthorCredibility.VERIFIED_OFFICIAL,
                confidence_score=0.98,
                sentiment=SentimentPolarity.VERY_BULLISH,
                affected_ticker="NVDA",
            ),
            EvidenceNode(
                source_type=EvidenceSource.FINANCIAL_NEWS,
                title="Hyperscaler Commitments",
                content="Multi-year infrastructure purchases",
                publication_timestamp=self.now,
                author_credibility=AuthorCredibility.HIGH,
                confidence_score=0.90,
                sentiment=SentimentPolarity.BULLISH,
                affected_ticker="NVDA",
            ),
        ]
        self.prices = [100.0, 102.0, 105.0, 103.0, 108.0, 112.0]

    def test_extract_features_from_evidence(self):
        vector = FeatureStore.extract_features_from_evidence(self.nodes, ticker="NVDA")
        self.assertIsInstance(vector, FeatureVector)
        self.assertEqual(vector.ticker, "NVDA")
        self.assertEqual(vector.raw_features["evidence_count"], 2.0)
        self.assertGreater(vector.raw_features["avg_credibility"], 0.8)
        self.assertIn("avg_sentiment_score", vector.raw_features)
        self.assertEqual(len(vector.normalized_features), vector.feature_count)

    def test_extract_features_from_prices(self):
        vector = FeatureStore.extract_features_from_prices(self.prices, ticker="NVDA")
        self.assertIsInstance(vector, FeatureVector)
        self.assertEqual(vector.raw_features["latest_price"], 112.0)
        self.assertGreater(vector.raw_features["price_momentum_ratio"], 1.0)
        self.assertIn("rolling_volatility", vector.raw_features)

    def test_zscore_normalization(self):
        vector = FeatureStore.extract_features_from_prices(self.prices, ticker="NVDA")
        norm_vals = list(vector.normalized_features.values())
        self.assertTrue(len(norm_vals) > 0)


if __name__ == "__main__":
    unittest.main()
