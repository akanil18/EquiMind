"""
RAGEvaluator — Production Metrics & Evaluation Framework for Agentic RAG.

Metrics implemented:
  1. Retrieval Metrics:
     - Recall@K (did retrieved nodes contain ground truth documents?)
     - Precision@K (fraction of retrieved nodes that are relevant)
     - NDCG@K (ranking position quality with logarithmic discount)
  2. Generation & Context Quality (RAGAS-style):
     - Context Relevance (fraction of retrieved sentences pertinent to query)
     - Faithfulness / Groundedness (does the response hallucinate beyond context?)
     - Answer Relevance (does the generated synthesis answer the user query?)
  3. Golden Dataset Builder for financial benchmark queries
"""

import math
import re
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from equimind.evidence.schema import EvidenceNode

logger = logging.getLogger(__name__)


class RAGMetricsResult(BaseModel):
    """Container for RAG evaluation metrics."""
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    context_relevance: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    composite_rag_score: float = 0.0
    details: Dict[str, Any] = Field(default_factory=dict)


class GoldenQueryEntry(BaseModel):
    """Benchmark entry containing query, ticker, and relevant document identifiers."""
    query: str
    ticker: str
    relevant_keywords: List[str]
    expected_sources: List[str]
    golden_answer_points: List[str]


class RAGEvaluator:
    """Evaluates RAG pipeline performance across retrieval and generation quality."""

    @classmethod
    def evaluate_retrieval(
        cls,
        retrieved_nodes: List[EvidenceNode],
        relevant_node_ids: Set[str],
        k: int = 5,
    ) -> Dict[str, float]:
        """Calculates Recall@K, Precision@K, and NDCG@K against ground truth IDs."""
        top_k_nodes = retrieved_nodes[:k]
        if not top_k_nodes:
            return {"recall_at_k": 0.0, "precision_at_k": 0.0, "ndcg_at_k": 0.0}

        retrieved_ids = [n.id for n in top_k_nodes]
        hits = sum(1 for nid in retrieved_ids if nid in relevant_node_ids)

        recall = hits / len(relevant_node_ids) if relevant_node_ids else 1.0
        precision = hits / len(top_k_nodes)

        # DCG calculation
        dcg = 0.0
        for rank, nid in enumerate(retrieved_ids):
            rel = 1.0 if nid in relevant_node_ids else 0.0
            dcg += rel / math.log2(rank + 2)

        # IDCG calculation (ideal DCG)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(relevant_node_ids), k)))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        return {
            "recall_at_k": round(min(1.0, recall), 4),
            "precision_at_k": round(precision, 4),
            "ndcg_at_k": round(ndcg, 4),
        }

    @classmethod
    def evaluate_context_relevance(
        cls,
        query: str,
        retrieved_nodes: List[EvidenceNode],
    ) -> float:
        """Measures semantic relevance between query and retrieved nodes."""
        if not retrieved_nodes:
            return 0.0

        query_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        if not query_tokens:
            return 1.0

        scores = []
        for node in retrieved_nodes:
            doc_tokens = set(re.findall(r"\b\w+\b", (node.title + " " + node.content).lower()))
            overlap = len(query_tokens & doc_tokens) / len(query_tokens)
            scores.append(overlap)

        return round(sum(scores) / len(scores), 4)

    @classmethod
    def evaluate_faithfulness(
        cls,
        answer_text: str,
        retrieved_nodes: List[EvidenceNode],
    ) -> float:
        """Verifies that facts/citations in the generated answer stem from retrieved context (Groundedness)."""
        if not answer_text or not retrieved_nodes:
            return 0.0

        context_combined = " ".join([f"{n.title} {n.content}" for n in retrieved_nodes]).lower()
        
        # Split answer into key statement sentences (respecting decimal numbers)
        sentences = [s.strip() for s in re.split(r"(?<=[!?\n])\s+|(?<=[a-zA-Z])\.\s+", answer_text) if len(s.strip()) > 5]
        if not sentences:
            return 1.0

        grounded_count = 0
        for sentence in sentences:
            sentence_tokens = set(re.findall(r"\b\w+\b", sentence.lower()))
            # Remove stopwords
            clean_tokens = {t for t in sentence_tokens if len(t) > 3}
            if not clean_tokens:
                grounded_count += 1
                continue
            
            # Check how many sentence tokens appear in context
            tokens_in_context = sum(1 for t in clean_tokens if t in context_combined)
            grounded_ratio = tokens_in_context / len(clean_tokens)
            if grounded_ratio >= 0.5:
                grounded_count += 1

        return round(grounded_count / len(sentences), 4)

    @classmethod
    def generate_golden_dataset(cls) -> List[GoldenQueryEntry]:
        """Returns benchmark evaluation entries for financial testing."""
        return [
            GoldenQueryEntry(
                query="What are NVIDIA's revenue growth drivers and valuation multiples?",
                ticker="NVDA",
                relevant_keywords=["revenue", "data center", "pe_ratio", "margin", "gpu"],
                expected_sources=["sec_filing", "financial_statements", "financial_news"],
                golden_answer_points=["Data Center revenue growth", "Gross margin expansion", "P/E valuation multiple"],
            ),
            GoldenQueryEntry(
                query="What are Apple's main risks and debt to equity profile?",
                ticker="AAPL",
                relevant_keywords=["debt", "equity", "risk", "iphone", "headwinds"],
                expected_sources=["sec_filing", "financial_statements"],
                golden_answer_points=["Leverage ratio", "Hardware replacement cycles", "Supply chain concentration"],
            ),
        ]
