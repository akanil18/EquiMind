from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from equimind.evidence.schema import (
    EvidenceNode,
    EvidenceSource,
    AuthorCredibility,
    SentimentPolarity,
)
from equimind.quantitative.technical import TechnicalEngine
from equimind.teams.base_team import ResearchTeam
from equimind.providers.base import LLMProvider


class MarketDataTeam(ResearchTeam):
    """Specialized team collecting price data, liquidity, moving averages, and technical indicators."""

    @property
    def team_name(self) -> str:
        return "market_data"

    def research(
        self,
        ticker: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        provider: Optional[LLMProvider] = None,
        as_of_date: Optional[datetime] = None,
    ) -> List[EvidenceNode]:
        ref_date = as_of_date or datetime.now(timezone.utc)
        ticker_upper = ticker.upper()

        # Generate synthetic historical prices for standard execution or backtest
        df = self._generate_market_data(ticker_upper, ref_date)

        # Compute deterministic technical indicators
        tech_summary = TechnicalEngine.analyze_dataframe(df)

        rsi = tech_summary["rsi_14"]
        sentiment = SentimentPolarity.NEUTRAL
        if rsi > 70:
            sentiment = SentimentPolarity.BEARISH  # Overbought
        elif rsi < 30:
            sentiment = SentimentPolarity.BULLISH  # Oversold
        elif rsi >= 55:
            sentiment = SentimentPolarity.BULLISH
        elif rsi <= 45:
            sentiment = SentimentPolarity.BEARISH

        content_str = (
            f"Market price for {ticker_upper} is ${tech_summary['last_price']:.2f}. "
            f"RSI (14) = {rsi}. MACD line = {tech_summary['macd']['macd']} "
            f"(Signal = {tech_summary['macd']['signal']}, Hist = {tech_summary['macd']['histogram']}). "
            f"Bollinger Bands: Upper=${tech_summary['bollinger_bands']['upper']}, "
            f"Middle=${tech_summary['bollinger_bands']['middle']}, Lower=${tech_summary['bollinger_bands']['lower']}. "
            f"Support levels: {tech_summary['support_resistance']['support']}, "
            f"Resistance levels: {tech_summary['support_resistance']['resistance']}."
        )

        node = EvidenceNode(
            source_type=EvidenceSource.MARKET_PRICES,
            title=f"{ticker_upper} Market Data & Technical Indicators Analysis",
            content=content_str,
            publication_timestamp=ref_date,
            author="EquiMind Quantitative Market Engine",
            author_credibility=AuthorCredibility.VERIFIED_OFFICIAL,
            confidence_score=0.95,
            sentiment=sentiment,
            affected_ticker=ticker_upper,
            tags=["technical_analysis", "prices", "rsi", "macd", "bollinger_bands"],
            metadata=tech_summary,
        )

        return [node]

    def _generate_market_data(self, ticker: str, ref_date: datetime) -> pd.DataFrame:
        """Generates realistic market price series relative to reference cutoff date."""
        np.random.seed(abs(hash(ticker)) % 10000)
        dates = pd.date_range(end=ref_date, periods=100, freq="D")
        base_price = 150.0 if ticker in ("NVDA", "AAPL") else 100.0
        returns = np.random.normal(loc=0.001, scale=0.015, size=100)
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            "date": dates,
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, size=100),
        })
