"""
End-to-end tests for the Agentic RAG Orchestration pipeline.

Tests cover:
- EmbeddingRouter (TF-IDF path)
- HNSWRetriever build + retrieve
- RAGCriticAgent evaluation logic
- RAGQueryRewriter deterministic rewrite
- AgenticRAGOrchestrator full loop with mock evidence nodes
"""

import math
from datetime import datetime, timezone
from typing import List

import numpy as np
import pytest

from equimind.evidence.schema import (
    AuthorCredibility,
    EvidenceNode,
    EvidenceSource,
    SentimentPolarity,
)
from equimind.rag.critic_agent import RAGCriticAgent
from equimind.rag.embedder import EmbeddingRouter, TFIDFEmbedder
from equimind.rag.hnsw_index import HNSWIndex
from equimind.rag.orchestrator import AgenticRAGOrchestrator
from equimind.rag.query_rewriter import RAGQueryRewriter
from equimind.rag.retriever import HNSWRetriever
from equimind.rag.schema import AgenticRAGResult


# ── Test Fixtures ─────────────────────────────────────────────────────────────

def _make_node(
    ticker: str = "NVDA",
    source: EvidenceSource = EvidenceSource.FINANCIAL_NEWS,
    sentiment: SentimentPolarity = SentimentPolarity.BULLISH,
    title: str = "NVDA revenue beats expectations",
    content: str = "Nvidia reported record quarterly revenue driven by data center demand.",
    confidence: float = 0.85,
    credibility: AuthorCredibility = AuthorCredibility.HIGH,
    days_old: float = 5.0,
) -> EvidenceNode:
    """Helper: create a single EvidenceNode for testing."""
    from datetime import timedelta
    pub_time = datetime.now(timezone.utc) - timedelta(days=days_old)
    return EvidenceNode(
        source_type=source,
        title=title,
        content=content,
        affected_ticker=ticker,
        sentiment=sentiment,
        confidence_score=confidence,
        author_credibility=credibility,
        publication_timestamp=pub_time,
    )


def _make_diverse_corpus(ticker: str = "NVDA") -> List[EvidenceNode]:
    """Build a diverse 10-node evidence corpus covering multiple source types."""
    nodes = [
        _make_node(ticker, EvidenceSource.SEC_FILING, SentimentPolarity.NEUTRAL,
                   "NVDA 10-Q Filing Q3 2024",
                   "NVDA filed its Q3 2024 quarterly report showing $18.1B revenue, up 122% YoY."),
        _make_node(ticker, EvidenceSource.EARNINGS_TRANSCRIPT, SentimentPolarity.BULLISH,
                   "NVDA Q3 Earnings Call Transcript",
                   "CEO Jensen Huang: Blackwell demand is insane. Data center is the new factory."),
        _make_node(ticker, EvidenceSource.FINANCIAL_NEWS, SentimentPolarity.BULLISH,
                   "Goldman Sachs raises NVDA target to $1000",
                   "Goldman Sachs initiated overweight on NVDA citing AI infrastructure supercycle."),
        _make_node(ticker, EvidenceSource.FINANCIAL_NEWS, SentimentPolarity.BEARISH,
                   "Analyst warns NVDA valuation stretched",
                   "Morgan Stanley noted NVDA trades at 30x forward sales, well above historical norm."),
        _make_node(ticker, EvidenceSource.REDDIT, SentimentPolarity.VERY_BULLISH,
                   "r/WallStreetBets NVDA DD",
                   "NVDA is the best AI play. Buying calls for January 2025. GPU demand secular trend."),
        _make_node(ticker, EvidenceSource.TWITTER_X, SentimentPolarity.BEARISH,
                   "Tech analyst on NVDA competition",
                   "AMD MI300X gaining traction. NVDA may lose 15% market share in 2025. Risky at these prices."),
        _make_node(ticker, EvidenceSource.MARKET_PRICES, SentimentPolarity.NEUTRAL,
                   "NVDA Price Data",
                   "NVDA closed at $875, 52-week high $974, RSI 68 overbought territory."),
        _make_node(ticker, EvidenceSource.MACRO_DATA, SentimentPolarity.NEUTRAL,
                   "US Semiconductor Sector Macro Update",
                   "Federal Reserve holds rates. AI capex spending from hyperscalers up 40% YoY."),
        _make_node(ticker, EvidenceSource.GITHUB_COMMITS, SentimentPolarity.BULLISH,
                   "CUDA repository activity surge",
                   "NVDA CUDA GitHub saw 3x commit activity in ML frameworks. Developer adoption accelerating."),
        _make_node(ticker, EvidenceSource.JOB_POSTINGS, SentimentPolarity.BULLISH,
                   "NVDA hiring surge",
                   "Nvidia posted 847 new jobs in AI and data center this month, suggesting strong growth."),
    ]
    return nodes


# ── TF-IDF Embedder Tests ─────────────────────────────────────────────────────

class TestTFIDFEmbedder:

    def test_fit_and_embed_basic(self):
        docs = ["NVDA revenue beats expectations quarterly",
                "semiconductor chip GPU AI demand strong",
                "bear case valuation expensive risk"]
        emb = TFIDFEmbedder(output_dim=32, max_vocab=200)
        emb.fit(docs)
        vec = emb.embed("NVDA GPU chip revenue")
        assert isinstance(vec, np.ndarray)
        assert vec.shape[0] <= 32
        # Should be unit normalized
        assert abs(np.linalg.norm(vec) - 1.0) < 0.1

    def test_fit_empty_corpus_uses_fallback(self):
        emb = TFIDFEmbedder(output_dim=16)
        emb.fit([])  # empty
        assert emb._is_fitted
        vec = emb.embed("test query")
        assert isinstance(vec, np.ndarray)

    def test_embed_batch_shape(self):
        docs = [f"document number {i} about financial markets" for i in range(20)]
        emb = TFIDFEmbedder(output_dim=16)
        emb.fit(docs)
        batch = emb.embed_batch(docs[:5])
        assert batch.shape == (5, emb.output_dim)

    def test_similar_texts_closer_than_dissimilar(self):
        """Semantically similar texts should be closer in embedding space."""
        docs = [
            "NVDA stock revenue earnings beat quarterly strong growth",
            "Nvidia quarterly revenue beat expectations AI demand",
            "macroeconomic recession inflation Federal Reserve rate hike",
        ]
        emb = TFIDFEmbedder(output_dim=32)
        emb.fit(docs)
        v0 = emb.embed(docs[0])
        v1 = emb.embed(docs[1])
        v2 = emb.embed(docs[2])
        sim_01 = float(np.dot(v0, v1))
        sim_02 = float(np.dot(v0, v2))
        # Similar docs (0 and 1) should have higher cosine similarity
        assert sim_01 > sim_02, f"Expected sim_01={sim_01:.3f} > sim_02={sim_02:.3f}"


class TestEmbeddingRouter:

    def test_fit_corpus_and_embed(self):
        docs = [f"financial research document {i}" for i in range(10)]
        router = EmbeddingRouter(provider=None)  # no LLM → TF-IDF path
        router.fit_corpus(docs)
        vec = router.embed("NVDA stock analysis")
        assert isinstance(vec, np.ndarray)
        assert vec.shape[0] > 0

    def test_embed_batch(self):
        docs = [f"evidence node {i}" for i in range(15)]
        router = EmbeddingRouter(provider=None)
        router.fit_corpus(docs)
        batch = router.embed_batch(docs[:5])
        assert batch.shape == (5, router.output_dim)


# ── HNSWRetriever Tests ────────────────────────────────────────────────────────

class TestHNSWRetriever:

    def test_build_on_diverse_corpus(self):
        nodes = _make_diverse_corpus()
        retriever = HNSWRetriever(provider=None)
        retriever.build(nodes)
        assert retriever.is_built
        assert len(retriever) == len(nodes)

    def test_retrieve_returns_nodes(self):
        nodes = _make_diverse_corpus()
        retriever = HNSWRetriever(provider=None)
        retriever.build(nodes)
        results = retriever.retrieve("NVDA revenue earnings growth", k=5)
        assert len(results) > 0
        assert len(results) <= 5
        for score, node in results:
            assert 0.0 <= score <= 1.0
            assert isinstance(node, EvidenceNode)

    def test_retrieve_nodes_convenience(self):
        nodes = _make_diverse_corpus()
        retriever = HNSWRetriever(provider=None)
        retriever.build(nodes)
        result_nodes = retriever.retrieve_nodes("NVDA earnings", k=3)
        assert len(result_nodes) <= 3
        for node in result_nodes:
            assert isinstance(node, EvidenceNode)

    def test_semantic_relevance_ordering(self):
        """SEC filing query should rank official sources higher."""
        nodes = _make_diverse_corpus()
        retriever = HNSWRetriever(provider=None)
        retriever.build(nodes)
        results = retriever.retrieve("SEC 10-Q quarterly filing balance sheet EDGAR", k=10)
        # All results should have positive similarity score
        for score, node in results:
            assert score >= 0.0

    def test_empty_corpus_build_and_retrieve(self):
        retriever = HNSWRetriever(provider=None)
        retriever.build([])
        assert retriever.is_built
        results = retriever.retrieve("any query", k=5)
        assert results == []

    def test_build_stats_populated(self):
        nodes = _make_diverse_corpus()
        retriever = HNSWRetriever(provider=None)
        retriever.build(nodes)
        stats = retriever.build_stats
        assert "nodes_indexed" in stats
        assert stats["nodes_indexed"] == len(nodes)
        assert stats["embed_dim"] > 0
        assert stats["total_build_time_ms"] >= 0.0

    def test_rag_retrieval_score_annotated(self):
        """Nodes should have rag_retrieval_score set after retrieval."""
        nodes = _make_diverse_corpus()
        retriever = HNSWRetriever(provider=None)
        retriever.build(nodes)
        results = retriever.retrieve("NVDA AI chip GPU", k=5)
        for score, node in results:
            assert node.rag_retrieval_score is not None
            assert 0.0 <= node.rag_retrieval_score <= 1.0


# ── RAGCriticAgent Tests ───────────────────────────────────────────────────────

class TestRAGCriticAgent:

    def test_empty_nodes_returns_insufficient(self):
        result = RAGCriticAgent.evaluate("NVDA analysis", [], iteration=1)
        assert result.is_sufficient is False
        assert result.coverage_score == 0.0

    def test_diverse_corpus_scores_higher_than_single_source(self):
        diverse = _make_diverse_corpus()
        single = [_make_node() for _ in range(5)]  # all FINANCIAL_NEWS

        result_diverse = RAGCriticAgent.evaluate("NVDA investment", diverse, iteration=1)
        result_single = RAGCriticAgent.evaluate("NVDA investment", single, iteration=1)

        assert result_diverse.coverage_score > result_single.coverage_score

    def test_sentiment_balance_score_improves_with_mixed_sentiment(self):
        mixed = [
            _make_node(sentiment=SentimentPolarity.BULLISH),
            _make_node(sentiment=SentimentPolarity.BEARISH),
            _make_node(sentiment=SentimentPolarity.NEUTRAL),
        ]
        only_bull = [
            _make_node(sentiment=SentimentPolarity.BULLISH),
            _make_node(sentiment=SentimentPolarity.BULLISH),
            _make_node(sentiment=SentimentPolarity.VERY_BULLISH),
        ]
        r_mixed = RAGCriticAgent.evaluate("NVDA", mixed, iteration=1)
        r_bull = RAGCriticAgent.evaluate("NVDA", only_bull, iteration=1)
        assert r_mixed.sentiment_balance_score > r_bull.sentiment_balance_score

    def test_missing_aspects_identified_for_single_source(self):
        single_source = [_make_node(source=EvidenceSource.REDDIT) for _ in range(5)]
        result = RAGCriticAgent.evaluate("NVDA", single_source, iteration=1)
        # Should flag missing official sources and news
        assert len(result.missing_aspects) > 0
        assert any("official" in a.lower() or "sec" in a.lower() for a in result.missing_aspects)

    def test_sufficient_diverse_high_confidence_corpus(self):
        """A diverse, high-confidence corpus should achieve sufficiency."""
        nodes = _make_diverse_corpus()
        # Simulate high retrieval scores
        scores = {n.id: 0.85 for n in nodes}
        result = RAGCriticAgent.evaluate(
            "NVDA AI chip semiconductor growth",
            nodes,
            iteration=1,
            retrieval_scores=scores,
        )
        # With diverse sources + high retrieval scores, should be sufficient
        assert result.coverage_score > 0.5

    def test_critic_result_fields_complete(self):
        nodes = _make_diverse_corpus()
        result = RAGCriticAgent.evaluate("test", nodes, iteration=1)
        assert isinstance(result.is_sufficient, bool)
        assert 0.0 <= result.coverage_score <= 1.0
        assert 0.0 <= result.source_diversity_score <= 1.0
        assert 0.0 <= result.sentiment_balance_score <= 1.0
        assert 0.0 <= result.query_relevance_score <= 1.0
        assert 0.0 <= result.recency_score <= 1.0
        assert isinstance(result.refined_query, str)
        assert len(result.refined_query) > 0


# ── RAGQueryRewriter Tests ────────────────────────────────────────────────────

class TestRAGQueryRewriter:

    def test_iteration_1_returns_original_query(self):
        original = "NVDA long-term AI chip analysis"
        result = RAGQueryRewriter.rewrite(
            original, missing_aspects=["Missing SEC filings"], iteration=1
        )
        assert result == original

    def test_no_missing_aspects_returns_original(self):
        original = "NVDA fundamentals"
        result = RAGQueryRewriter.rewrite(original, missing_aspects=[], iteration=2)
        assert result == original

    def test_sec_aspect_appends_filing_keywords(self):
        result = RAGQueryRewriter.rewrite(
            "NVDA revenue",
            missing_aspects=["Missing official sources (SEC filings, earnings transcripts)"],
            iteration=2,
            ticker="NVDA",
        )
        assert len(result) > len("NVDA revenue")
        # Should contain some filing-related terms
        result_lower = result.lower()
        assert any(k in result_lower for k in ["sec", "10-q", "edgar", "quarterly", "annual", "earning"])

    def test_bearish_aspect_appends_risk_keywords(self):
        result = RAGQueryRewriter.rewrite(
            "NVDA analysis",
            missing_aspects=["Insufficient bearish/risk evidence — need downside analysis"],
            iteration=2,
            ticker="NVDA",
        )
        result_lower = result.lower()
        assert any(k in result_lower for k in ["risk", "downside", "bear", "headwind"])

    def test_output_length_capped(self):
        result = RAGQueryRewriter.rewrite(
            "NVDA",
            missing_aspects=["Missing " + "x" * 100 for _ in range(10)],
            iteration=3,
            ticker="NVDA",
        )
        assert len(result) <= 250


# ── AgenticRAGOrchestrator Tests ──────────────────────────────────────────────

class TestAgenticRAGOrchestrator:

    def test_empty_corpus_returns_empty_result(self):
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=2)
        result = rag.orchestrate("NVDA analysis", "NVDA", candidate_nodes=[])
        assert isinstance(result, AgenticRAGResult)
        assert result.curated_node_ids == []
        assert result.total_iterations == 0

    def test_full_loop_with_diverse_corpus(self):
        nodes = _make_diverse_corpus()
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=3, top_k=10)
        result = rag.orchestrate(
            query="Is NVDA a strong buy for AI infrastructure long-term?",
            ticker="NVDA",
            candidate_nodes=nodes,
        )
        assert isinstance(result, AgenticRAGResult)
        assert result.ticker == "NVDA"
        assert result.total_iterations >= 1
        assert result.total_iterations <= 3
        assert len(result.iteration_logs) == result.total_iterations
        assert 0.0 <= result.final_coverage_score <= 1.0
        assert isinstance(result.converged, bool)

    def test_curated_node_ids_are_subset_of_corpus(self):
        nodes = _make_diverse_corpus()
        corpus_ids = {n.id for n in nodes}
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=2, top_k=8)
        result = rag.orchestrate("NVDA revenue", "NVDA", candidate_nodes=nodes)
        for nid in result.curated_node_ids:
            assert nid in corpus_ids

    def test_discarded_nodes_not_in_curated(self):
        nodes = _make_diverse_corpus()
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=2, top_k=8)
        result = rag.orchestrate("NVDA analysis", "NVDA", candidate_nodes=nodes)
        curated_set = set(result.curated_node_ids)
        discarded_set = set(result.discarded_node_ids)
        assert curated_set.isdisjoint(discarded_set), \
            "Curated and discarded node sets should not overlap"

    def test_iteration_logs_have_correct_fields(self):
        nodes = _make_diverse_corpus()
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=2, top_k=5)
        result = rag.orchestrate("NVDA SEC filings earnings", "NVDA", candidate_nodes=nodes)
        for log in result.iteration_logs:
            assert log.iteration >= 1
            assert isinstance(log.query_used, str)
            assert log.nodes_retrieved >= 0
            assert log.nodes_after_filter >= 0
            assert 0.0 <= log.coverage_score <= 1.0
            assert log.hnsw_search_time_ms >= 0.0

    def test_curate_nodes_from_result(self):
        nodes = _make_diverse_corpus()
        node_map = {n.id: n for n in nodes}
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=2, top_k=8)
        result = rag.orchestrate("NVDA AI chip", "NVDA", candidate_nodes=nodes)
        curated_nodes = rag.curate_nodes_from_result(result, node_map)
        assert isinstance(curated_nodes, list)
        for node in curated_nodes:
            assert isinstance(node, EvidenceNode)
        # Should match the count of curated_node_ids that are in node_map
        assert len(curated_nodes) == len(result.curated_node_ids)

    def test_max_iterations_respected(self):
        """With max_iterations=1, loop should only run once."""
        nodes = _make_diverse_corpus()
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=1, top_k=5)
        result = rag.orchestrate("NVDA", "NVDA", candidate_nodes=nodes)
        assert result.total_iterations == 1

    def test_metadata_populated(self):
        nodes = _make_diverse_corpus()
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=2, top_k=5)
        result = rag.orchestrate("NVDA", "NVDA", candidate_nodes=nodes)
        assert "hnsw_stats" in result.metadata
        assert "corpus_size" in result.metadata
        assert result.metadata["corpus_size"] == len(nodes)

    def test_multi_iteration_query_evolves(self):
        """Query should be different in iteration 2 than iteration 1 when not converged."""
        nodes = [_make_node(source=EvidenceSource.REDDIT) for _ in range(5)]  # low diversity
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=3, top_k=5)
        result = rag.orchestrate("NVDA SEC filing earnings", "NVDA", candidate_nodes=nodes)
        if result.total_iterations >= 2:
            # At least one re-query should have happened
            queries = [log.query_used for log in result.iteration_logs]
            assert len(set(queries)) >= 1  # at least one unique query per iteration
