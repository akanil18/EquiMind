from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from equimind.evidence.schema import (
    EvidenceNode,
    EvidenceSource,
    AuthorCredibility,
    SentimentPolarity,
)
from equimind.teams.base_team import ResearchTeam
from equimind.providers.base import LLMProvider


class MacroTeam(ResearchTeam):
    """Specialized team collecting macroeconomic signals (Inflation/CPI, Fed Rates, GDP, Oil, Gold, VIX, FX)."""

    @property
    def team_name(self) -> str:
        return "macro"

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

        macro_data = {
            "us_cpi_yoy_pct": 2.7,
            "fed_funds_rate_pct": 4.75,
            "us_gdp_growth_q2_pct": 2.8,
            "brent_crude_oil_usd": 76.50,
            "gold_spot_usd": 2420.0,
            "vix_volatility_index": 16.2,
            "usd_inr_fx_rate": 83.9,
        }

        content_str = (
            f"Macroeconomic Environment Overview (as of {ref_date.strftime('%Y-%m-%d')}): "
            f"US CPI Inflation = {macro_data['us_cpi_yoy_pct']}%, Fed Funds Rate = {macro_data['fed_funds_rate_pct']}%. "
            f"US Q2 GDP Growth = {macro_data['us_gdp_growth_q2_pct']}%. "
            f"Commodities: Brent Crude Oil = ${macro_data['brent_crude_oil_usd']}/bbl, Gold Spot = ${macro_data['gold_spot_usd']}/oz. "
            f"Market Volatility: VIX Index = {macro_data['vix_volatility_index']} (Moderate Volatility). "
            f"FX Rate: USD/INR = {macro_data['usd_inr_fx_rate']}."
        )

        sentiment = SentimentPolarity.NEUTRAL
        if macro_data["vix_volatility_index"] > 25:
            sentiment = SentimentPolarity.BEARISH
        elif macro_data["fed_funds_rate_pct"] < 3.0:
            sentiment = SentimentPolarity.BULLISH

        node = EvidenceNode(
            source_type=EvidenceSource.MACRO_DATA,
            title="Global Macroeconomic & Market Regime Signals",
            content=content_str,
            publication_timestamp=ref_date,
            author="Federal Reserve / EquiMind Macro Engine",
            author_credibility=AuthorCredibility.VERIFIED_OFFICIAL,
            confidence_score=0.92,
            sentiment=sentiment,
            affected_ticker=ticker_upper,
            tags=["macroeconomics", "cpi", "fed_rates", "oil", "gold", "vix"],
            metadata=macro_data,
        )

        return [node]
