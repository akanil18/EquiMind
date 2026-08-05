import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from equimind.evidence.schema import EvidenceNode, SentimentPolarity
from equimind.features.schema import FeatureVector, FeatureSet


class FeatureStore:
    """Institutional Feature Engineering Platform & Feature Store."""

    @classmethod
    def extract_features_from_evidence(cls, nodes: List[EvidenceNode], ticker: str) -> FeatureVector:
        """Transforms evidence nodes into structured numerical features."""
        if not nodes:
            return FeatureVector(
                vector_id=str(uuid.uuid4()),
                ticker=ticker,
                raw_features={
                    "evidence_count": 0.0,
                    "avg_credibility": 0.0,
                    "avg_confidence": 0.0,
                    "bullish_sentiment_ratio": 0.0,
                    "bearish_sentiment_ratio": 0.0,
                    "verified_official_count": 0.0,
                },
                normalized_features={},
                feature_count=6,
                lineage_sources=[],
            )

        sentiment_map = {
            SentimentPolarity.VERY_BEARISH: -1.0,
            SentimentPolarity.BEARISH: -0.5,
            SentimentPolarity.NEUTRAL: 0.0,
            SentimentPolarity.BULLISH: 0.5,
            SentimentPolarity.VERY_BULLISH: 1.0,
        }

        credibility_map = {
            "low": 0.25,
            "medium": 0.50,
            "high": 0.75,
            "verified_official": 1.00,
        }

        sentiments = [sentiment_map.get(n.sentiment, 0.0) for n in nodes]
        credibilities = [credibility_map.get(getattr(n.author_credibility, "value", str(n.author_credibility)).lower(), 0.5) for n in nodes]
        confidences = [n.confidence_score for n in nodes]

        bullish_count = sum(1 for s in sentiments if s > 0)
        bearish_count = sum(1 for s in sentiments if s < 0)
        total = len(nodes)

        raw = {
            "evidence_count": float(total),
            "avg_credibility": round(float(np.mean(credibilities)), 4),
            "avg_confidence": round(float(np.mean(confidences)), 4),
            "avg_sentiment_score": round(float(np.mean(sentiments)), 4),
            "bullish_sentiment_ratio": round(bullish_count / total, 4) if total > 0 else 0.0,
            "bearish_sentiment_ratio": round(bearish_count / total, 4) if total > 0 else 0.0,
            "verified_official_count": float(sum(1 for c in credibilities if c == 1.00)),
        }

        sources = list(set([str(n.source_type) for n in nodes]))

        return FeatureVector(
            vector_id=str(uuid.uuid4()),
            ticker=ticker,
            raw_features=raw,
            normalized_features=cls._zscore_normalize(raw),
            feature_count=len(raw),
            lineage_sources=sources,
        )

    @classmethod
    def extract_features_from_prices(cls, prices: List[float], ticker: str) -> FeatureVector:
        """Transforms raw price time series into rolling statistical features."""
        if len(prices) < 2:
            return FeatureVector(
                vector_id=str(uuid.uuid4()),
                ticker=ticker,
                raw_features={"price_return_1d": 0.0, "rolling_volatility_5d": 0.0},
                normalized_features={},
                feature_count=2,
                lineage_sources=["MARKET_DATA"],
            )

        arr = np.array(prices, dtype=float)
        returns = np.diff(arr) / arr[:-1]

        ret_1d = float(returns[-1]) if len(returns) > 0 else 0.0
        ret_mean = float(np.mean(returns))
        vol = float(np.std(returns))

        raw = {
            "latest_price": round(float(arr[-1]), 2),
            "price_return_1d": round(ret_1d, 4),
            "mean_daily_return": round(ret_mean, 4),
            "rolling_volatility": round(vol, 4),
            "price_momentum_ratio": round(float(arr[-1] / arr[0]), 4) if arr[0] > 0 else 1.0,
        }

        return FeatureVector(
            vector_id=str(uuid.uuid4()),
            ticker=ticker,
            raw_features=raw,
            normalized_features=cls._zscore_normalize(raw),
            feature_count=len(raw),
            lineage_sources=["MARKET_DATA"],
        )

    @classmethod
    def _zscore_normalize(cls, features: Dict[str, float]) -> Dict[str, float]:
        """Calculates Z-Score normalization for a feature vector."""
        vals = np.array(list(features.values()), dtype=float)
        mean = np.mean(vals)
        std = np.std(vals)

        normalized = {}
        for k, v in features.items():
            if std == 0:
                normalized[k] = 0.0
            else:
                normalized[k] = round(float((v - mean) / std), 4)

        return normalized
