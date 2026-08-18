"""
RAG Query Rewriter — generates refined sub-queries targeting evidence gaps.

The QueryRewriter works in tandem with the RAGCriticAgent: when the critic
identifies missing evidence aspects (e.g. "no SEC filings", "only bullish"),
the rewriter generates a focused sub-query that will retrieve the missing piece.

Strategy
--------
- Iteration 1: Original query (no rewrite)
- Iteration 2: Aspect-focused query generated from critic's missing_aspects
- Iteration 3+: Broader fallback query covering all identified gaps

Uses LLM refinement when available, with a deterministic financial-domain
keyword expansion fallback that requires no API calls.
"""

import logging
import re
from typing import Dict, List, Optional

from equimind.providers.base import LLMMessage, LLMProvider, Role

logger = logging.getLogger(__name__)


# ── Domain expansion dictionaries ─────────────────────────────────────────────

_ASPECT_EXPANSIONS: Dict[str, List[str]] = {
    # Official / fundamental data
    "sec": ["SEC 10-K 10-Q annual report", "EDGAR filing quarterly earnings"],
    "filing": ["SEC annual report quarterly filing EDGAR", "balance sheet income statement"],
    "earnings": ["earnings per share EPS guidance revenue beat miss"],
    "fundamental": ["P/E ratio book value free cash flow ROE EBITDA"],
    # Bearish / risk
    "bearish": ["risk factors headwinds downside price target sell analyst"],
    "risk": ["downside risk volatility bear case credit rating downgrade"],
    "downside": ["price target cut bear case risk factor sell-side"],
    # Bullish / growth
    "bullish": ["growth catalyst upside buy rating institutional accumulate"],
    "growth": ["revenue growth market share expansion AI opportunity"],
    "catalyst": ["upcoming catalyst product launch partnership acquisition"],
    # News / sentiment
    "news": ["financial news analyst report Bloomberg Reuters WSJ"],
    "social": ["Reddit WallStreetBets Twitter StockTwits retail sentiment"],
    "sentiment": ["investor sentiment options flow institutional buying"],
    # Macro
    "macro": ["macroeconomic interest rate Fed inflation GDP recession"],
    "rate": ["Federal Reserve rate hike cut monetary policy yield curve"],
    # Recency
    "recent": ["latest Q3 Q4 2024 2025 recent news update"],
    "stale": ["most recent current latest update 2024 2025"],
    # Market data
    "price": ["stock price technical analysis support resistance momentum"],
    "market": ["market cap float short interest options chain"],
}


class RAGQueryRewriter:
    """Generates focused sub-queries targeting evidence coverage gaps.

    Each call produces a single search query string optimized to retrieve
    the specific type of evidence that the RAGCriticAgent flagged as missing.
    """

    @classmethod
    def rewrite(
        cls,
        original_query: str,
        missing_aspects: List[str],
        iteration: int,
        ticker: str = "",
        provider: Optional[LLMProvider] = None,
    ) -> str:
        """Generate a refined query for the next retrieval round.

        Parameters
        ----------
        original_query : str
            The original investment research query.
        missing_aspects : List[str]
            Coverage gaps identified by the RAGCriticAgent.
        iteration : int
            Current iteration (1-indexed). Iteration 1 returns original query.
        ticker : str
            Stock ticker to anchor the query.
        provider : LLMProvider, optional
            If available, uses LLM for a higher-quality rewrite.

        Returns
        -------
        str
            A refined query string for the next HNSW retrieval round.
        """
        if not missing_aspects or iteration <= 1:
            return original_query

        # ── LLM-powered rewrite (preferred) ────────────────────────────────
        if provider and provider.is_available():
            llm_query = cls._llm_rewrite(original_query, missing_aspects, ticker, provider)
            if llm_query:
                return llm_query

        # ── Deterministic fallback ──────────────────────────────────────────
        return cls._deterministic_rewrite(original_query, missing_aspects, ticker, iteration)

    @classmethod
    def _llm_rewrite(
        cls,
        original_query: str,
        missing_aspects: List[str],
        ticker: str,
        provider: LLMProvider,
    ) -> Optional[str]:
        """Use LLM to generate a targeted sub-query."""
        try:
            gaps = "\n".join(f"  • {a}" for a in missing_aspects[:4])
            system_prompt = (
                "You are an expert financial research query optimizer. "
                "Given an original investment research query and a list of evidence gaps, "
                "generate a single precise search query (max 30 words) that will retrieve "
                "the missing evidence types. Include specific financial terminology."
            )
            user_prompt = (
                f"Ticker: {ticker.upper() if ticker else 'N/A'}\n"
                f"Original query: '{original_query}'\n\n"
                f"Evidence gaps to address:\n{gaps}\n\n"
                "Generate ONE refined search query targeting these gaps. "
                "Return ONLY the query text, no explanation."
            )
            resp = provider.generate(
                messages=[
                    LLMMessage(role=Role.SYSTEM, content=system_prompt),
                    LLMMessage(role=Role.USER, content=user_prompt),
                ],
                temperature=0.15,
                max_tokens=80,
            )
            if resp.content and len(resp.content.strip()) > 8:
                refined = resp.content.strip().strip('"').strip("'").split("\n")[0]
                logger.debug(f"RAGQueryRewriter (LLM): '{refined}'")
                return refined
        except Exception as e:
            logger.warning(f"RAGQueryRewriter: LLM rewrite failed: {e}")
        return None

    @classmethod
    def _deterministic_rewrite(
        cls,
        original_query: str,
        missing_aspects: List[str],
        ticker: str,
        iteration: int,
    ) -> str:
        """Build a domain-keyword-expanded query from the missing aspect list."""
        keyword_expansions: List[str] = []

        for aspect in missing_aspects[:3]:
            aspect_lower = aspect.lower()
            for key, expansions in _ASPECT_EXPANSIONS.items():
                if key in aspect_lower:
                    keyword_expansions.extend(expansions[:1])  # take first expansion per key
                    break

        # Remove duplicate terms while preserving order
        all_terms = re.findall(r"\b\w+\b", " ".join(keyword_expansions).lower())
        existing_terms = set(re.findall(r"\b\w+\b", original_query.lower()))
        new_terms = [t for t in all_terms if t not in existing_terms]
        new_terms = list(dict.fromkeys(new_terms))  # deduplicate order-preserving

        ticker_prefix = f"{ticker.upper()} " if ticker else ""

        # Iteration 2: targeted aspect query
        if iteration == 2 and keyword_expansions:
            refined = f"{ticker_prefix}{original_query} {' '.join(keyword_expansions[:2])}"
        # Iteration 3+: broad sweep incorporating all gaps
        elif keyword_expansions:
            gap_summary = " ".join(new_terms[:12])
            refined = f"{ticker_prefix}{original_query} {gap_summary} fundamental analysis"
        else:
            # Last resort: add broad financial research keywords
            refined = f"{ticker_prefix}{original_query} SEC filing analyst rating fundamental macro risk"

        # Trim to reasonable length
        refined = refined.strip()[:250]
        logger.debug(f"RAGQueryRewriter (deterministic): '{refined}'")
        return refined
