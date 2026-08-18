"""
Tests for RAGEvaluator (Recall@K, Precision@K, NDCG@K, Context Relevance, Faithfulness).
"""

import unittest
from equimind.evidence.schema import EvidenceNode, EvidenceSource
from equimind.rag.evaluator import RAGEvaluator


class TestRAGEvaluator(unittest.TestCase):

    def setUp(self):
        self.nodes = [
            EvidenceNode(
                id="doc-1",
                source_type=EvidenceSource.SEC_FILING,
                title="NVDA 10-Q Revenue",
                content="NVIDIA reported record quarterly revenue of $18.1B.",
                affected_ticker="NVDA",
            ),
            EvidenceNode(
                id="doc-2",
                source_type=EvidenceSource.FINANCIAL_NEWS,
                title="Goldman Sachs NVDA target",
                content="Goldman Sachs analyst raised target to $1000 citing AI infrastructure demand.",
                affected_ticker="NVDA",
            ),
            EvidenceNode(
                id="doc-3",
                source_type=EvidenceSource.REDDIT,
                title="WSB Thread",
                content="Just bought calls on NVDA moon.",
                affected_ticker="NVDA",
            ),
        ]

    def test_evaluate_retrieval_metrics(self):
        ground_truth = {"doc-1", "doc-2"}
        metrics = RAGEvaluator.evaluate_retrieval(self.nodes, relevant_node_ids=ground_truth, k=2)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["precision_at_k"], 1.0)
        self.assertEqual(metrics["ndcg_at_k"], 1.0)

    def test_evaluate_context_relevance(self):
        score = RAGEvaluator.evaluate_context_relevance(
            query="NVIDIA quarterly revenue record",
            retrieved_nodes=[self.nodes[0]],
        )
        self.assertGreater(score, 0.5)

    def test_evaluate_faithfulness(self):
        answer = "NVIDIA reported record quarterly revenue of $18.1B based on its 10-Q filing."
        score = RAGEvaluator.evaluate_faithfulness(answer, retrieved_nodes=[self.nodes[0]])
        self.assertGreater(score, 0.7)

    def test_golden_dataset_generation(self):
        dataset = RAGEvaluator.generate_golden_dataset()
        self.assertGreaterEqual(len(dataset), 2)
        self.assertEqual(dataset[0].ticker, "NVDA")


if __name__ == "__main__":
    unittest.main()
