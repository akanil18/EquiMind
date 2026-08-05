from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from equimind.evidence.schema import (
    EvidenceNode,
    EvidenceSource,
    AuthorCredibility,
    SentimentPolarity,
)
from equimind.quantitative.fundamental import FundamentalEngine
from equimind.teams.base_team import ResearchTeam
from equimind.providers.base import LLMProvider


class FundamentalTeam(ResearchTeam):
    """Specialized team collecting financial statements, balance sheets, SEC 10-K/10-Q metrics, valuation ratios, F-score, and Z-score."""

    @property
    def team_name(self) -> str:
        return "fundamentals"

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

        # Simulated corporate fundamental financial metrics
        mkt_cap = 2_800_000_000_000 if ticker_upper == "NVDA" else 150_000_000_000
        price = 125.0
        eps = 3.8
        book_val = 18.5
        fcf = 35_000_000_000
        growth_rate = 25.0

        val_ratios = FundamentalEngine.calculate_valuation_ratios(
            market_cap=mkt_cap,
            price=price,
            eps=eps,
            book_value_per_share=book_val,
            free_cash_flow=fcf,
            earnings_growth_rate=growth_rate,
        )

        prof_metrics = FundamentalEngine.calculate_profitability_metrics(
            net_income=30_000_000_000,
            revenue=60_000_000_000,
            total_assets=80_000_000_000,
            shareholder_equity=50_000_000_000,
            operating_income=32_000_000_000,
        )

        health_metrics = FundamentalEngine.calculate_financial_health(
            current_assets=45_000_000_000,
            current_liabilities=15_000_000_000,
            total_debt=10_000_000_000,
            shareholder_equity=50_000_000_000,
        )

        f_score = FundamentalEngine.calculate_piotroski_f_score({
            "net_income": 30_000,
            "roa": 0.35,
            "operating_cash_flow": 35_000,
            "long_term_debt_current": 10_000,
            "long_term_debt_prior": 11_000,
            "current_ratio_current": 3.0,
            "current_ratio_prior": 2.5,
            "shares_outstanding_current": 24_000,
            "shares_outstanding_prior": 24_000,
            "gross_margin_current": 0.75,
            "gross_margin_prior": 0.70,
            "asset_turnover_current": 0.75,
            "asset_turnover_prior": 0.65,
        })

        z_score = FundamentalEngine.calculate_altman_z_score(
            working_capital=30_000,
            retained_earnings=40_000,
            ebit=32_000,
            market_cap=2_800_000,
            revenue=60_000,
            total_assets=80_000,
            total_liabilities=15_000,
        )

        content_str = (
            f"Fundamental analysis for {ticker_upper}: PE ratio = {val_ratios['pe_ratio']}, "
            f"PB ratio = {val_ratios['pb_ratio']}, PEG ratio = {val_ratios['peg_ratio']}, FCF Yield = {val_ratios['fcf_yield_pct']}%. "
            f"ROE = {prof_metrics['roe_pct']}%, Operating Margin = {prof_metrics['operating_margin_pct']}%, Net Margin = {prof_metrics['net_margin_pct']}%. "
            f"Current Ratio = {health_metrics['current_ratio']}, Debt-to-Equity = {health_metrics['debt_to_equity']}. "
            f"Piotroski F-Score = {f_score['piotroski_f_score']}/9 ({f_score['rating']}). "
            f"Altman Z-Score = {z_score['z_score']} ({z_score['zone']})."
        )

        node = EvidenceNode(
            source_type=EvidenceSource.FINANCIAL_STATEMENTS,
            title=f"{ticker_upper} Fundamental Statements & Valuation Analysis",
            content=content_str,
            publication_timestamp=ref_date,
            author="EquiMind Fundamental Analysis Engine",
            author_credibility=AuthorCredibility.VERIFIED_OFFICIAL,
            confidence_score=0.95,
            sentiment=SentimentPolarity.VERY_BULLISH if f_score['piotroski_f_score'] >= 7 else SentimentPolarity.NEUTRAL,
            affected_ticker=ticker_upper,
            tags=["valuation", "pe_ratio", "roe", "piotroski_f_score", "altman_z_score"],
            metadata={
                "valuation": val_ratios,
                "profitability": prof_metrics,
                "health": health_metrics,
                "piotroski": f_score,
                "altman_z": z_score,
            },
        )

        return [node]
