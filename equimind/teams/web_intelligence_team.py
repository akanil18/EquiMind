from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import logging

from equimind.evidence.schema import (
    EvidenceNode,
    EvidenceSource,
    AuthorCredibility,
    SentimentPolarity,
)
from equimind.teams.base_team import ResearchTeam
from equimind.providers.base import LLMProvider
from equimind.adapters.sec_edgar_adapter import SECEdgarAdapter
from equimind.adapters.news_adapter import NewsRSSAdapter

logger = logging.getLogger(__name__)


class WebIntelligenceTeam(ResearchTeam):
    """Specialized team operating multi-source internet & alternative signal gathering adapters.
    
    Real Data Sources:
      - SEC EDGAR (real 10-K/10-Q filings, XBRL financials) — no API key
      - Financial News RSS (Yahoo Finance, MarketWatch, CNBC) — no API key
      - Reddit (via PRAW) — requires API key (falls back to synthetic)
      - Twitter/X — requires API key (falls back to synthetic)
      - GitHub signals — falls back to synthetic
      - Earnings transcripts — falls back to synthetic
    """

    @property
    def team_name(self) -> str:
        return "web_intelligence"

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
        active_adapters = (context or {}).get("active_adapters", [
            "sec_filings", "financial_news", "reddit", "twitter_x",
            "earnings_transcripts", "github_commits"
        ])

        nodes: List[EvidenceNode] = []

        # ── 1. SEC Filings — REAL DATA via EDGAR API ──────────
        if "sec_filings" in active_adapters:
            nodes.extend(self._fetch_real_sec_data(ticker_upper, ref_date))

        # ── 2. Financial News — REAL DATA via RSS feeds ───────
        if "financial_news" in active_adapters:
            nodes.extend(self._fetch_real_news(ticker_upper, ref_date))

        # ── 3. Reddit — Real if PRAW available, else synthetic
        if "reddit" in active_adapters:
            nodes.extend(self._fetch_reddit(ticker_upper, ref_date))

        # ── 4. Twitter/X — Synthetic (requires paid API) ─────
        if "twitter_x" in active_adapters:
            nodes.append(self._synthetic_twitter(ticker_upper, ref_date))

        # ── 5. GitHub Signals — Synthetic ─────────────────────
        if "github_commits" in active_adapters:
            nodes.append(self._synthetic_github(ticker_upper, ref_date))

        # ── 6. Earnings Transcripts — Synthetic ──────────────
        if "earnings_transcripts" in active_adapters:
            nodes.append(self._synthetic_earnings(ticker_upper, ref_date))

        return nodes

    # ══════════════════════════════════════════════════════════
    # REAL DATA ADAPTERS
    # ══════════════════════════════════════════════════════════

    def _fetch_real_sec_data(self, ticker: str, ref_date: datetime) -> List[EvidenceNode]:
        """Fetch real SEC EDGAR filings and financial data."""
        nodes = []
        
        # Get financial summary
        fin_summary = SECEdgarAdapter.get_financial_summary(ticker)
        
        if fin_summary.get("source") == "SEC EDGAR XBRL":
            income = fin_summary.get("income_statement", {})
            balance = fin_summary.get("balance_sheet", {})
            ratios = fin_summary.get("computed_ratios", {})
            
            # Build content from real financial data
            content_parts = [f"SEC EDGAR XBRL Financial Data for {ticker}:"]
            
            # Revenue
            revenue_entries = income.get("revenue", [])
            if revenue_entries:
                latest_rev = revenue_entries[0]
                content_parts.append(
                    f"Revenue: ${latest_rev['value']/1e9:.2f}B "
                    f"(Period ending {latest_rev.get('end_date', 'N/A')}, "
                    f"Form: {latest_rev.get('form', 'N/A')})"
                )
                # YoY growth if we have previous year
                if len(revenue_entries) >= 2:
                    prev_rev = revenue_entries[1]
                    if prev_rev["value"] and latest_rev["value"]:
                        growth = (latest_rev["value"] / prev_rev["value"] - 1) * 100
                        content_parts.append(f"Revenue YoY Growth: {growth:.1f}%")
            
            # Net Income
            ni_entries = income.get("net_income", [])
            if ni_entries:
                latest_ni = ni_entries[0]
                content_parts.append(f"Net Income: ${latest_ni['value']/1e9:.2f}B")
            
            # EPS
            eps_entries = income.get("eps_diluted", income.get("eps_basic", []))
            if eps_entries:
                content_parts.append(f"EPS (Diluted): ${eps_entries[0]['value']:.2f}")
            
            # Computed ratios
            if ratios:
                if "net_margin" in ratios:
                    content_parts.append(f"Net Margin: {ratios['net_margin']:.1f}%")
                if "roe" in ratios:
                    content_parts.append(f"ROE: {ratios['roe']:.1f}%")
                if "debt_to_equity" in ratios:
                    content_parts.append(f"Debt/Equity: {ratios['debt_to_equity']:.2f}")
            
            # Balance sheet
            cash_entries = balance.get("cash", [])
            if cash_entries:
                content_parts.append(f"Cash & Equivalents: ${cash_entries[0]['value']/1e9:.2f}B")
            
            content_str = "\n".join(content_parts)
            
            nodes.append(EvidenceNode(
                source_type=EvidenceSource.SEC_FILING,
                title=f"SEC EDGAR XBRL Financial Data — {ticker} (Real Data)",
                content=content_str,
                publication_timestamp=ref_date - timedelta(days=1),
                author="SEC EDGAR Official XBRL API",
                author_credibility=AuthorCredibility.VERIFIED_OFFICIAL,
                confidence_score=0.98,
                sentiment=SentimentPolarity.NEUTRAL,
                affected_ticker=ticker,
                tags=["sec_filing", "xbrl", "real_data", "financials"],
                url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K",
                metadata=fin_summary,
            ))
        
        # Get recent filings list
        filings = SECEdgarAdapter.get_recent_filings(ticker, max_filings=5)
        if filings:
            filing_lines = [f"Recent SEC Filings for {ticker}:"]
            for f in filings:
                filing_lines.append(
                    f"  • {f['form']} filed {f['filing_date']}"
                )
            
            nodes.append(EvidenceNode(
                source_type=EvidenceSource.SEC_FILING,
                title=f"Recent SEC Filing Activity — {ticker}",
                content="\n".join(filing_lines),
                publication_timestamp=ref_date,
                author="SEC EDGAR Submissions API",
                author_credibility=AuthorCredibility.VERIFIED_OFFICIAL,
                confidence_score=0.95,
                sentiment=SentimentPolarity.NEUTRAL,
                affected_ticker=ticker,
                tags=["sec_filing", "real_data", "filing_history"],
                url=filings[0]["url"] if filings else None,
            ))

        # If no real data, use fallback
        if not nodes:
            nodes.append(EvidenceNode(
                source_type=EvidenceSource.SEC_FILING,
                title=f"SEC Filing Data — {ticker} (Synthetic Fallback)",
                content=f"SEC EDGAR data unavailable for {ticker}. Using estimated financials.",
                publication_timestamp=ref_date - timedelta(days=5),
                author="EquiMind Synthetic Engine",
                author_credibility=AuthorCredibility.MEDIUM,
                confidence_score=0.60,
                sentiment=SentimentPolarity.NEUTRAL,
                affected_ticker=ticker,
                tags=["sec_filing", "synthetic"],
            ))

        return nodes

    def _fetch_real_news(self, ticker: str, ref_date: datetime) -> List[EvidenceNode]:
        """Fetch real financial news from RSS feeds."""
        nodes = []
        articles = NewsRSSAdapter.fetch_news(ticker=ticker, max_articles=5)
        
        for article in articles:
            if article.get("source") == "Synthetic Feed":
                credibility = AuthorCredibility.LOW
                tags = ["news", "synthetic"]
            else:
                credibility = AuthorCredibility.HIGH
                tags = ["news", "real_data", article.get("source", "").lower().replace(" ", "_")]
            
            # Simple sentiment from title keywords
            title_lower = article.get("title", "").lower()
            sentiment = SentimentPolarity.NEUTRAL
            bullish_words = ["surge", "jump", "rally", "soar", "beat", "upgrade", "growth", "record"]
            bearish_words = ["drop", "fall", "decline", "crash", "miss", "downgrade", "cut", "warning"]
            
            bull_count = sum(1 for w in bullish_words if w in title_lower)
            bear_count = sum(1 for w in bearish_words if w in title_lower)
            if bull_count > bear_count:
                sentiment = SentimentPolarity.BULLISH
            elif bear_count > bull_count:
                sentiment = SentimentPolarity.BEARISH
            
            pub_time = ref_date
            if article.get("published"):
                try:
                    pub_time = datetime.fromisoformat(article["published"])
                    if pub_time.tzinfo is None:
                        pub_time = pub_time.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            
            nodes.append(EvidenceNode(
                source_type=EvidenceSource.FINANCIAL_NEWS,
                title=article.get("title", f"Financial News — {ticker}"),
                content=article.get("summary", "")[:800],
                publication_timestamp=pub_time,
                author=article.get("source", "Financial News Feed"),
                author_credibility=credibility,
                confidence_score=min(0.5 + article.get("relevance_score", 0.3), 0.92),
                sentiment=sentiment,
                affected_ticker=ticker,
                tags=tags,
                url=article.get("link"),
            ))

        return nodes

    def _fetch_reddit(self, ticker: str, ref_date: datetime) -> List[EvidenceNode]:
        """Fetch Reddit sentiment — real if PRAW available, else synthetic."""
        # Try PRAW
        try:
            import praw
            # Would need credentials — fall through to synthetic for now
            raise ImportError("PRAW credentials not configured")
        except ImportError:
            pass

        # Synthetic fallback
        return [EvidenceNode(
            source_type=EvidenceSource.REDDIT,
            title=f"Retail Sentiment Analysis — r/stocks for {ticker} (Synthetic)",
            content=(
                f"Synthetic Reddit sentiment for {ticker}. "
                f"Note: Real Reddit data requires PRAW API credentials. "
                f"Configure REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env to enable."
            ),
            publication_timestamp=ref_date - timedelta(hours=12),
            author="r/stocks Community (Synthetic)",
            author_credibility=AuthorCredibility.LOW,
            confidence_score=0.40,
            sentiment=SentimentPolarity.NEUTRAL,
            affected_ticker=ticker,
            tags=["reddit", "synthetic", "requires_api_key"],
        )]

    # ══════════════════════════════════════════════════════════
    # SYNTHETIC ADAPTERS (require paid APIs)
    # ══════════════════════════════════════════════════════════

    def _synthetic_twitter(self, ticker: str, ref_date: datetime) -> EvidenceNode:
        return EvidenceNode(
            source_type=EvidenceSource.TWITTER_X,
            title=f"X/Twitter Sentiment — {ticker} (Synthetic)",
            content=f"Synthetic Twitter/X data for {ticker}. Real data requires X API credentials.",
            publication_timestamp=ref_date - timedelta(hours=6),
            author="X/Twitter (Synthetic)",
            author_credibility=AuthorCredibility.LOW,
            confidence_score=0.40,
            sentiment=SentimentPolarity.NEUTRAL,
            affected_ticker=ticker,
            tags=["twitter_x", "synthetic", "requires_api_key"],
        )

    def _synthetic_github(self, ticker: str, ref_date: datetime) -> EvidenceNode:
        return EvidenceNode(
            source_type=EvidenceSource.GITHUB_COMMITS,
            title=f"Developer Activity Signals — {ticker} (Synthetic)",
            content=f"Synthetic GitHub developer activity for {ticker}.",
            publication_timestamp=ref_date - timedelta(days=1),
            author="GitHub Activity Tracker (Synthetic)",
            author_credibility=AuthorCredibility.LOW,
            confidence_score=0.40,
            sentiment=SentimentPolarity.NEUTRAL,
            affected_ticker=ticker,
            tags=["github", "synthetic"],
        )

    def _synthetic_earnings(self, ticker: str, ref_date: datetime) -> EvidenceNode:
        return EvidenceNode(
            source_type=EvidenceSource.EARNINGS_TRANSCRIPT,
            title=f"Earnings Call Transcript — {ticker} (Synthetic)",
            content=f"Synthetic earnings transcript for {ticker}. Real transcripts require premium data provider.",
            publication_timestamp=ref_date - timedelta(days=10),
            author="Earnings Transcript (Synthetic)",
            author_credibility=AuthorCredibility.LOW,
            confidence_score=0.40,
            sentiment=SentimentPolarity.NEUTRAL,
            affected_ticker=ticker,
            tags=["earnings_call", "synthetic"],
        )
