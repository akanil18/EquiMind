"""
End-to-end tests for the Agentic RAG Orchestration pipeline (unittest compatible).
"""

import unittest
from datetime import datetime, timezone
from typing import List

import numpy as np

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
    return [
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
    ]


class TestTFIDFEmbedder(unittest.TestCase):

    def test_fit_and_embed_basic(self):
        docs = [
            "NVDA revenue beats expectations quarterly",
            "semiconductor chip GPU AI demand strong",
            "bear case valuation expensive risk",
        ]
        emb = TFIDFEmbedder(output_dim=32, max_vocab=200)
        emb.fit(docs)
        vec = emb.embed("NVDA GPU chip revenue")
        self.assertIsInstance(vec, np.ndarray)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=2)

    def test_fit_empty_corpus_uses_fallback(self):
        emb = TFIDFEmbedder(output_dim=16)
        emb.fit([])
        self.assertTrue(emb._is_fitted)
        vec = emb.embed("test query")
        self.assertIsInstance(vec, np.ndarray)


class TestRAGCriticAgent(unittest.TestCase):

    def test_empty_nodes_returns_insufficient(self):
        result = RAGCriticAgent.evaluate("NVDA analysis", [], iteration=1)
        self.assertFalse(result.is_sufficient)
        self.assertEqual(result.coverage_score, 0.0)

    def test_diverse_corpus_scores_higher(self):
        diverse = _make_diverse_corpus()
        single = [_make_node() for _ in range(5)]
        result_diverse = RAGCriticAgent.evaluate("NVDA investment", diverse, iteration=1)
        result_single = RAGCriticAgent.evaluate("NVDA investment", single, iteration=1)
        self.assertGreater(result_diverse.coverage_score, result_single.coverage_score)


class TestRAGQueryRewriter(unittest.TestCase):

    def test_iteration_1_returns_original_query(self):
        original = "NVDA long-term AI chip analysis"
        result = RAGQueryRewriter.rewrite(
            original, missing_aspects=["Missing SEC filings"], iteration=1
        )
        self.assertEqual(result, original)

    def test_sec_aspect_appends_filing_keywords(self):
        result = RAGQueryRewriter.rewrite(
            "NVDA revenue",
            missing_aspects=["Missing official sources (SEC filings, earnings transcripts)"],
            iteration=2,
            ticker="NVDA",
        )
        self.assertGreater(len(result), len("NVDA revenue"))


class TestAgenticRAGOrchestrator(unittest.TestCase):

    def test_empty_corpus_returns_empty_result(self):
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=2)
        result = rag.orchestrate("NVDA analysis", "NVDA", candidate_nodes=[])
        self.assertIsInstance(result, AgenticRAGResult)
        self.assertEqual(result.curated_node_ids, [])

    def test_full_loop_with_diverse_corpus(self):
        nodes = _make_diverse_corpus()
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=3, top_k=10)
        result = rag.orchestrate(
            query="Is NVDA a strong buy for AI infrastructure long-term?",
            ticker="NVDA",
            candidate_nodes=nodes,
        )
        self.assertIsInstance(result, AgenticRAGResult)
        self.assertEqual(result.ticker, "NVDA")
        self.assertGreaterEqual(result.total_iterations, 1)
        self.assertLessEqual(result.total_iterations, 3)
        self.assertEqual(len(result.iteration_logs), result.total_iterations)

    def test_curate_nodes_from_result(self):
        nodes = _make_diverse_corpus()
        node_map = {n.id: n for n in nodes}
        rag = AgenticRAGOrchestrator(provider=None, max_iterations=2, top_k=8)
        result = rag.orchestrate("NVDA AI chip", "NVDA", candidate_nodes=nodes)
        curated_nodes = rag.curate_nodes_from_result(result, node_map)
        self.assertIsInstance(curated_nodes, list)
        self.assertEqual(len(curated_nodes), len(result.curated_node_ids))


if __name__ == "__main__":
    unittest.main()
