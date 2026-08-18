"""
RAG Critic Agent — evaluates evidence sufficiency after each retrieval round.

The RAGCriticAgent plays the same role in the RAG loop as the JudgeAgent
plays in the committee debate: it objectively evaluates whether the retrieved
evidence is sufficient to support a high-quality investment analysis, or if
more targeted retrieval is needed.

Evaluation dimensions
---------------------
1. Source diversity    — are multiple source types represented? (SEC, news, social, macro)
2. Sentiment balance   — does evidence cover both bullish and bearish signals?
3. Query relevance     — do the top nodes actually relate to the query?
4. Recency             — is the evidence fresh enough for the investment horizon?
5. Confidence quality  — average confidence score of retrieved nodes

When all dimensions exceed their thresholds, the critic declares sufficiency
and the agentic loop terminates early. Otherwise it identifies missing aspects
and hands them to RAGQueryRewriter for the next iteration.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from equimind.evidence.schema import (
    AuthorCredibility,
    EvidenceNode,
    EvidenceSource,
    SentimentPolarity,
)
from equimind.providers.base import LLMMessage, LLMProvider, Role
from equimind.rag.schema import RAGCriticResult

logger = logging.getLogger(__name__)


# ── Coverage thresholds ───────────────────────────────────────────────────────

SUFFICIENCY_THRESHOLD = 0.65          # Overall coverage_score to declare sufficiency
SOURCE_DIVERSITY_THRESHOLD = 0.50     # Fraction of source type buckets covered
SENTIMENT_BALANCE_THRESHOLD = 0.40   # Min fraction for minority sentiment
QUERY_RELEVANCE_THRESHOLD = 0.40     # Min avg HNSW similarity for top-k nodes
RECENCY_THRESHOLD = 0.50             # Min recency score (half-life = 30 days)
CONFIDENCE_THRESHOLD = 0.55          # Min avg confidence score

# Source "buckets" for diversity scoring
_OFFICIAL_SOURCES = {
    EvidenceSource.SEC_FILING,
    EvidenceSource.EARNINGS_TRANSCRIPT,
    EvidenceSource.FINANCIAL_STATEMENTS,
    EvidenceSource.GOVT_ANNOUNCEMENT,
}
_NEWS_SOURCES = {
    EvidenceSource.FINANCIAL_NEWS,
    EvidenceSource.COMPANY_BLOG,
}
_SOCIAL_SOURCES = {
    EvidenceSource.REDDIT,
    EvidenceSource.TWITTER_X,
    EvidenceSource.STOCKTWITS,
}
_ALT_DATA_SOURCES = {
    EvidenceSource.GITHUB_COMMITS,
    EvidenceSource.JOB_POSTINGS,
    EvidenceSource.MARKET_PRICES,
    EvidenceSource.MACRO_DATA,
}
_SOURCE_BUCKETS = [_OFFICIAL_SOURCES, _NEWS_SOURCES, _SOCIAL_SOURCES, _ALT_DATA_SOURCES]


class RAGCriticAgent:
    """Evaluates whether retrieved evidence is sufficient for high-quality analysis.

    Can use LLM for nuanced evaluation when a provider is available;
    falls back to deterministic heuristics otherwise.
    """

    @classmethod
    def evaluate(
        cls,
        query: str,
        retrieved_nodes: List[EvidenceNode],
        iteration: int,
        provider: Optional[LLMProvider] = None,
        retrieval_scores: Optional[Dict[str, float]] = None,
        as_of_date: Optional[datetime] = None,
    ) -> RAGCriticResult:
        """Evaluate evidence sufficiency and identify gaps.

        Parameters
        ----------
        query : str
            The current retrieval query being evaluated.
        retrieved_nodes : List[EvidenceNode]
            Nodes returned by the HNSW retriever for this iteration.
        iteration : int
            Current iteration number (1-indexed).
        provider : LLMProvider, optional
            LLM provider for enhanced critique. Uses heuristics if None.
        retrieval_scores : Dict[str, float], optional
            node_id → HNSW similarity score from the retriever.
        as_of_date : datetime, optional
            Temporal reference for recency scoring.

        Returns
        -------
        RAGCriticResult
            Full evaluation result including sufficiency verdict and refined query.
        """
        if not retrieved_nodes:
            return RAGCriticResult(
                iteration=iteration,
                is_sufficient=False,
                coverage_score=0.0,
                source_diversity_score=0.0,
                sentiment_balance_score=0.0,
                query_relevance_score=0.0,
                recency_score=0.0,
                missing_aspects=["No evidence nodes retrieved — check data adapters"],
                refined_query=f"{query} financial data fundamentals SEC filing",
                rationale="Empty retrieval result.",
            )

        retrieval_scores = retrieval_scores or {}
        ref_time = as_of_date or datetime.now(timezone.utc)

        # ── Dimension 1: Source Diversity ──────────────────────────────────
        source_diversity_score, missing_sources = cls._score_source_diversity(retrieved_nodes)

        # ── Dimension 2: Sentiment Balance ─────────────────────────────────
        sentiment_balance_score, sentiment_gaps = cls._score_sentiment_balance(retrieved_nodes)

        # ── Dimension 3: Query Relevance ───────────────────────────────────
        query_relevance_score = cls._score_query_relevance(
            query, retrieved_nodes, retrieval_scores
        )

        # ── Dimension 4: Recency ───────────────────────────────────────────
        recency_score = cls._score_recency(retrieved_nodes, ref_time)

        # ── Dimension 5: Confidence Quality ───────────────────────────────
        avg_confidence = (
            sum(n.confidence_score for n in retrieved_nodes) / len(retrieved_nodes)
            if retrieved_nodes else 0.0
        )

        # ── Composite Coverage Score ────────────────────────────────────────
        # Weighted: official source diversity matters most, then relevance
        coverage_score = (
            source_diversity_score * 0.25
            + sentiment_balance_score * 0.20
            + query_relevance_score * 0.30
            + recency_score * 0.15
            + min(1.0, avg_confidence / 0.8) * 0.10
        )
        coverage_score = round(min(1.0, coverage_score), 4)

        # ── Identify missing aspects ────────────────────────────────────────
        missing_aspects: List[str] = []
        missing_aspects.extend(missing_sources)
        missing_aspects.extend(sentiment_gaps)
        if query_relevance_score < QUERY_RELEVANCE_THRESHOLD:
            missing_aspects.append(
                f"Low query relevance ({query_relevance_score:.2f}) — retrieved nodes may be off-topic"
            )
        if recency_score < RECENCY_THRESHOLD:
            missing_aspects.append("Evidence is stale — need more recent news and filings")
        if avg_confidence < CONFIDENCE_THRESHOLD:
            missing_aspects.append(
                f"Low average confidence ({avg_confidence:.2f}) — need higher credibility sources"
            )

        is_sufficient = coverage_score >= SUFFICIENCY_THRESHOLD

        # ── Identify low-quality nodes to discard ──────────────────────────
        discard_ids = cls._identify_discard_nodes(retrieved_nodes, retrieval_scores)

        # ── Generate refined query ──────────────────────────────────────────
        refined_query = cls._generate_refined_query(query, missing_aspects, iteration, provider)

        # ── Build rationale string ─────────────────────────────────────────
        rationale = (
            f"Iteration {iteration}: coverage={coverage_score:.2f} "
            f"[diversity={source_diversity_score:.2f}, "
            f"balance={sentiment_balance_score:.2f}, "
            f"relevance={query_relevance_score:.2f}, "
            f"recency={recency_score:.2f}, "
            f"confidence={avg_confidence:.2f}]. "
            f"{'SUFFICIENT ✓' if is_sufficient else 'INSUFFICIENT — re-querying.'}"
        )
        logger.info(f"RAGCriticAgent: {rationale}")
        if missing_aspects:
            logger.info(f"RAGCriticAgent: missing aspects: {missing_aspects}")

        return RAGCriticResult(
            iteration=iteration,
            is_sufficient=is_sufficient,
            coverage_score=coverage_score,
            source_diversity_score=round(source_diversity_score, 4),
            sentiment_balance_score=round(sentiment_balance_score, 4),
            query_relevance_score=round(query_relevance_score, 4),
            recency_score=round(recency_score, 4),
            missing_aspects=missing_aspects,
            refined_query=refined_query,
            discard_node_ids=discard_ids,
            rationale=rationale,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Scoring helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_source_diversity(
        nodes: List[EvidenceNode],
    ) -> tuple[float, List[str]]:
        """Score source diversity: fraction of source buckets covered."""
        covered_sources: Set[EvidenceSource] = {n.source_type for n in nodes}
        covered_buckets = sum(
            1 for bucket in _SOURCE_BUCKETS
            if any(src in covered_sources for src in bucket)
        )
        score = covered_buckets / len(_SOURCE_BUCKETS)

        missing: List[str] = []
        if not any(src in covered_sources for src in _OFFICIAL_SOURCES):
            missing.append("Missing official sources (SEC filings, earnings transcripts)")
        if not any(src in covered_sources for src in _NEWS_SOURCES):
            missing.append("Missing financial news coverage")
        if not any(src in covered_sources for src in _SOCIAL_SOURCES):
            missing.append("Missing social sentiment signals (Reddit, Twitter)")

        return score, missing

    @staticmethod
    def _score_sentiment_balance(
        nodes: List[EvidenceNode],
    ) -> tuple[float, List[str]]:
        """Score sentiment balance: both bullish and bearish views should be represented."""
        bullish_count = sum(
            1 for n in nodes
            if n.sentiment in (SentimentPolarity.BULLISH, SentimentPolarity.VERY_BULLISH)
        )
        bearish_count = sum(
            1 for n in nodes
            if n.sentiment in (SentimentPolarity.BEARISH, SentimentPolarity.VERY_BEARISH)
        )
        total = len(nodes)
        if total == 0:
            return 0.0, ["No nodes to assess sentiment"]

        bull_frac = bullish_count / total
        bear_frac = bearish_count / total
        minority_frac = min(bull_frac, bear_frac)

        # Score peaks at 0.5/0.5 balance, penalizes extreme skew
        score = min(1.0, minority_frac / SENTIMENT_BALANCE_THRESHOLD) if SENTIMENT_BALANCE_THRESHOLD > 0 else 1.0

        gaps: List[str] = []
        if bear_frac < 0.10:
            gaps.append("Insufficient bearish/risk evidence — need downside analysis")
        if bull_frac < 0.10:
            gaps.append("Insufficient bullish evidence — need growth catalyst data")

        return round(score, 4), gaps

    @staticmethod
    def _score_query_relevance(
        query: str,
        nodes: List[EvidenceNode],
        retrieval_scores: Dict[str, float],
    ) -> float:
        """Score query relevance: average HNSW similarity score of top nodes."""
        if retrieval_scores:
            scores = [retrieval_scores.get(n.id, 0.0) for n in nodes]
            return round(sum(scores) / len(scores), 4) if scores else 0.0

        # Fallback: keyword overlap with query
        import re
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return 0.5

        overlaps = []
        for node in nodes:
            node_tokens = set(re.findall(r"\w+", (node.title + " " + node.content).lower()))
            if query_tokens and node_tokens:
                overlap = len(query_tokens & node_tokens) / len(query_tokens)
                overlaps.append(overlap)
        return round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0

    @staticmethod
    def _score_recency(
        nodes: List[EvidenceNode],
        ref_time: datetime,
    ) -> float:
        """Score recency: exponential decay with 30-day half-life."""
        if not nodes:
            return 0.0
        decay_scores = []
        for node in nodes:
            pub = node.publication_timestamp
            # Ensure timezone-aware comparison
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            diff_seconds = (ref_time - pub).total_seconds()
            days_old = max(0.0, diff_seconds / 86400.0)
            decay = math.exp(-0.023 * days_old)    # half-life ≈ 30 days
            decay_scores.append(decay)
        return round(sum(decay_scores) / len(decay_scores), 4)

    @staticmethod
    def _identify_discard_nodes(
        nodes: List[EvidenceNode],
        retrieval_scores: Dict[str, float],
    ) -> List[str]:
        """Identify nodes with very low retrieval scores or low credibility for discarding."""
        discard_ids = []
        for node in nodes:
            score = retrieval_scores.get(node.id, 1.0)
            # Discard if: very low similarity AND low credibility AND low confidence
            if (
                score < 0.15
                and node.author_credibility == AuthorCredibility.LOW
                and node.confidence_score < 0.5
            ):
                discard_ids.append(node.id)
        return discard_ids

    @staticmethod
    def _generate_refined_query(
        original_query: str,
        missing_aspects: List[str],
        iteration: int,
        provider: Optional[LLMProvider],
    ) -> str:
        """Generate a refined search query targeting the identified coverage gaps."""
        if not missing_aspects:
            return original_query

        if provider and provider.is_available():
            try:
                gaps_text = "\n".join(f"- {a}" for a in missing_aspects[:4])
                prompt = (
                    f"Original research query: '{original_query}'\n\n"
                    f"Missing evidence aspects identified by the RAG Critic:\n{gaps_text}\n\n"
                    "Generate a single refined search query (max 25 words) that specifically "
                    "targets the missing evidence aspects while staying relevant to the original query. "
                    "Return ONLY the refined query text, no explanation."
                )
                resp = provider.generate(
                    messages=[LLMMessage(role=Role.USER, content=prompt)],
                    temperature=0.2,
                    max_tokens=60,
                )
                if resp.content and len(resp.content.strip()) > 5:
                    refined = resp.content.strip().strip('"').strip("'")
                    logger.debug(f"RAGCriticAgent: LLM refined query: '{refined}'")
                    return refined
            except Exception as e:
                logger.warning(f"RAGCriticAgent: LLM query refinement failed: {e}")

        # Deterministic fallback: append key aspect terms to original query
        aspect_keywords = []
        for aspect in missing_aspects[:3]:
            if "SEC" in aspect or "official" in aspect.lower():
                aspect_keywords.extend(["SEC filing", "10-Q", "earnings"])
            elif "bearish" in aspect.lower() or "risk" in aspect.lower():
                aspect_keywords.extend(["risk", "downside", "headwind"])
            elif "bullish" in aspect.lower() or "growth" in aspect.lower():
                aspect_keywords.extend(["growth", "catalyst", "upside"])
            elif "news" in aspect.lower():
                aspect_keywords.extend(["financial news", "analyst report"])
            elif "stale" in aspect.lower() or "recent" in aspect.lower():
                aspect_keywords.extend(["latest", "Q3 2024", "recent"])

        extras = " ".join(dict.fromkeys(aspect_keywords))[:80]
        refined = f"{original_query} {extras}".strip()
        logger.debug(f"RAGCriticAgent: deterministic refined query: '{refined}'")
        return refined
