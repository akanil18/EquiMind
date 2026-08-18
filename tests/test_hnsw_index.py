"""
Tests for the HNSW Vector Index implementation (unittest compatible).
"""

import math
import random
import unittest
import numpy as np

from equimind.rag.hnsw_index import HNSWIndex


def make_random_vectors(n: int, dim: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Generate n random unit-normalized vectors."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.where(norms < 1e-10, 1.0, norms)
    return {f"node_{i}": vecs[i] for i in range(n)}


class TestHNSWIndexBasics(unittest.TestCase):
    """Core functionality tests."""

    def test_empty_index_search_returns_empty(self):
        idx = HNSWIndex(dim=8, M=4, ef_construction=10, ef_search=5)
        query = np.random.randn(8).astype(np.float32)
        results = idx.search(query, k=5)
        self.assertEqual(results, [])

    def test_single_node_insert_and_search(self):
        idx = HNSWIndex(dim=4, M=4, ef_construction=10, ef_search=5)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx.insert("a", vec)
        self.assertIn("a", idx)
        self.assertEqual(len(idx), 1)

        results = idx.search(vec, k=1)
        self.assertEqual(len(results), 1)
        dist, node_id = results[0]
        self.assertEqual(node_id, "a")
        self.assertLess(dist, 0.01)

    def test_duplicate_insert_ignored(self):
        idx = HNSWIndex(dim=4, M=4, ef_construction=10)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx.insert("a", vec)
        idx.insert("a", vec)   # duplicate
        self.assertEqual(len(idx), 1)

    def test_len_and_contains(self):
        idx = HNSWIndex(dim=4, M=4, ef_construction=10)
        vecs = make_random_vectors(5, 4)
        for nid, v in vecs.items():
            idx.insert(nid, v)
        self.assertEqual(len(idx), 5)
        for nid in vecs:
            self.assertIn(nid, idx)
        self.assertNotIn("nonexistent", idx)


class TestHNSWIndexDistance(unittest.TestCase):
    """Distance metric correctness tests."""

    def test_identical_vectors_distance_near_zero(self):
        idx = HNSWIndex(dim=8, M=8, ef_construction=20)
        rng = np.random.RandomState(1)
        vec = rng.randn(8).astype(np.float32)
        idx.insert("a", vec)
        idx.insert("b", rng.randn(8).astype(np.float32))
        idx.insert("c", rng.randn(8).astype(np.float32))

        results = idx.search(vec, k=1)
        dist, best = results[0]
        self.assertEqual(best, "a")
        self.assertLess(dist, 0.05)

    def test_opposite_vectors_high_distance(self):
        idx = HNSWIndex(dim=4, M=4, ef_construction=10)
        pos = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        neg = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx.insert("pos", pos)
        idx.insert("neg", neg)

        results_pos = idx.search(pos, k=2)
        self.assertEqual(results_pos[0][1], "pos")
        dist_neg = next(d for d, nid in results_pos if nid == "neg")
        self.assertGreater(dist_neg, 1.5)

    def test_normalize_helper(self):
        vec = np.array([3.0, 4.0], dtype=np.float32)
        norm_vec = HNSWIndex._normalize(vec)
        self.assertAlmostEqual(float(np.linalg.norm(norm_vec)), 1.0, places=5)

    def test_zero_vector_normalize(self):
        vec = np.zeros(4, dtype=np.float32)
        norm_vec = HNSWIndex._normalize(vec)
        self.assertTrue(np.all(norm_vec == 0.0))

    def test_cosine_distance_symmetry(self):
        a = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        b = np.array([0.5, 1.0, 0.0], dtype=np.float32)
        d_ab = HNSWIndex._distance(HNSWIndex._normalize(a), HNSWIndex._normalize(b))
        d_ba = HNSWIndex._distance(HNSWIndex._normalize(b), HNSWIndex._normalize(a))
        self.assertAlmostEqual(d_ab, d_ba, places=6)


class TestHNSWIndexRecall(unittest.TestCase):
    """Recall@k accuracy tests."""

    def _brute_force_search(
        self, query: np.ndarray, node_vectors: dict[str, np.ndarray], k: int
    ) -> list[str]:
        q_norm = HNSWIndex._normalize(query)
        distances = [
            (HNSWIndex._distance(q_norm, HNSWIndex._normalize(v)), nid)
            for nid, v in node_vectors.items()
        ]
        distances.sort()
        return [nid for _, nid in distances[:k]]

    def test_recall_at_k_medium(self):
        node_vectors = make_random_vectors(50, 8, seed=42)
        query_vectors = make_random_vectors(10, 8, seed=999)

        idx = HNSWIndex(dim=8, M=16, ef_construction=100, ef_search=50, seed=42)
        idx.build_from_vectors(node_vectors)

        hits = 0
        total = 0
        k = 5
        for qid, qvec in query_vectors.items():
            true_top_k = set(self._brute_force_search(qvec, node_vectors, k))
            hnsw_results = idx.search(qvec, k=k)
            hnsw_top_k = {nid for _, nid in hnsw_results}
            hits += len(true_top_k & hnsw_top_k)
            total += k

        recall = hits / total
        self.assertGreaterEqual(recall, 0.70)


class TestHNSWIndexBuild(unittest.TestCase):

    def test_build_from_vectors(self):
        node_vecs = make_random_vectors(30, 16, seed=7)
        idx = HNSWIndex(dim=16, M=8, ef_construction=50)
        inserted = idx.build_from_vectors(node_vecs)
        self.assertEqual(inserted, 30)
        self.assertEqual(len(idx), 30)

    def test_stats_after_build(self):
        node_vecs = make_random_vectors(20, 8)
        idx = HNSWIndex(dim=8, M=6, ef_construction=30, ef_search=20)
        idx.build_from_vectors(node_vecs)
        stats = idx.stats()
        self.assertEqual(stats["num_nodes"], 20)
        self.assertEqual(stats["M"], 6)
        self.assertEqual(stats["dim"], 8)
        self.assertGreaterEqual(stats["num_layers"], 1)
        self.assertGreater(stats["total_edges"], 0)


if __name__ == "__main__":
    unittest.main()
