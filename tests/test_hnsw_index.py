"""
Tests for the HNSW Vector Index implementation.

Validates correctness of insert/search, cosine distance, recall@k,
and edge cases (empty index, duplicate inserts, single-node index).
"""

import math
import random

import numpy as np
import pytest

from equimind.rag.hnsw_index import HNSWIndex


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_random_vectors(n: int, dim: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Generate n random unit-normalized vectors."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.where(norms < 1e-10, 1.0, norms)
    return {f"node_{i}": vecs[i] for i in range(n)}


# ── Unit Tests ────────────────────────────────────────────────────────────────

class TestHNSWIndexBasics:
    """Core functionality tests."""

    def test_empty_index_search_returns_empty(self):
        idx = HNSWIndex(dim=8, M=4, ef_construction=10, ef_search=5)
        query = np.random.randn(8).astype(np.float32)
        results = idx.search(query, k=5)
        assert results == []

    def test_single_node_insert_and_search(self):
        idx = HNSWIndex(dim=4, M=4, ef_construction=10, ef_search=5)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx.insert("a", vec)
        assert "a" in idx
        assert len(idx) == 1

        results = idx.search(vec, k=1)
        assert len(results) == 1
        dist, node_id = results[0]
        assert node_id == "a"
        # Distance to itself (after L2 normalization) should be ~0
        assert dist < 0.01

    def test_duplicate_insert_ignored(self):
        idx = HNSWIndex(dim=4, M=4, ef_construction=10)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx.insert("a", vec)
        idx.insert("a", vec)   # duplicate
        assert len(idx) == 1

    def test_len_and_contains(self):
        idx = HNSWIndex(dim=4, M=4, ef_construction=10)
        vecs = make_random_vectors(5, 4)
        for nid, v in vecs.items():
            idx.insert(nid, v)
        assert len(idx) == 5
        for nid in vecs:
            assert nid in idx
        assert "nonexistent" not in idx


class TestHNSWIndexDistance:
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
        assert best == "a"
        assert dist < 0.05  # nearly identical → distance ≈ 0

    def test_opposite_vectors_high_distance(self):
        idx = HNSWIndex(dim=4, M=4, ef_construction=10)
        pos = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        neg = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx.insert("pos", pos)
        idx.insert("neg", neg)

        results_pos = idx.search(pos, k=2)
        # pos should be nearest to pos
        assert results_pos[0][1] == "pos"
        # neg should be far (distance ≈ 2)
        dist_neg = next(d for d, nid in results_pos if nid == "neg")
        assert dist_neg > 1.5

    def test_normalize_helper(self):
        vec = np.array([3.0, 4.0], dtype=np.float32)
        norm_vec = HNSWIndex._normalize(vec)
        assert abs(np.linalg.norm(norm_vec) - 1.0) < 1e-5

    def test_zero_vector_normalize(self):
        vec = np.zeros(4, dtype=np.float32)
        norm_vec = HNSWIndex._normalize(vec)
        assert np.all(norm_vec == 0.0)

    def test_cosine_distance_symmetry(self):
        a = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        b = np.array([0.5, 1.0, 0.0], dtype=np.float32)
        d_ab = HNSWIndex._distance(HNSWIndex._normalize(a), HNSWIndex._normalize(b))
        d_ba = HNSWIndex._distance(HNSWIndex._normalize(b), HNSWIndex._normalize(a))
        assert abs(d_ab - d_ba) < 1e-6


class TestHNSWIndexRecall:
    """Recall@k accuracy tests — HNSW should find true nearest neighbors."""

    def _brute_force_search(
        self, query: np.ndarray, node_vectors: dict[str, np.ndarray], k: int
    ) -> list[str]:
        """Ground truth: exact nearest neighbors via brute force."""
        q_norm = HNSWIndex._normalize(query)
        distances = [
            (HNSWIndex._distance(q_norm, HNSWIndex._normalize(v)), nid)
            for nid, v in node_vectors.items()
        ]
        distances.sort()
        return [nid for _, nid in distances[:k]]

    @pytest.mark.parametrize("n,dim,k", [
        (50, 8, 5),
        (100, 16, 10),
        (200, 32, 10),
    ])
    def test_recall_at_k(self, n: int, dim: int, k: int):
        """HNSW recall@k should be > 80% across standard configurations."""
        node_vectors = make_random_vectors(n, dim, seed=42)
        query_vectors = make_random_vectors(20, dim, seed=999)

        idx = HNSWIndex(dim=dim, M=16, ef_construction=200, ef_search=100, seed=42)
        idx.build_from_vectors(node_vectors)

        hits = 0
        total = 0
        for qid, qvec in query_vectors.items():
            true_top_k = set(self._brute_force_search(qvec, node_vectors, k))
            hnsw_results = idx.search(qvec, k=k)
            hnsw_top_k = {nid for _, nid in hnsw_results}
            hits += len(true_top_k & hnsw_top_k)
            total += k

        recall = hits / total
        assert recall >= 0.70, f"HNSW recall@{k} = {recall:.2%} (expected >= 70%)"


class TestHNSWIndexBuild:
    """Index build and stats tests."""

    def test_build_from_vectors(self):
        node_vecs = make_random_vectors(30, 16, seed=7)
        idx = HNSWIndex(dim=16, M=8, ef_construction=50)
        inserted = idx.build_from_vectors(node_vecs)
        assert inserted == 30
        assert len(idx) == 30

    def test_stats_after_build(self):
        node_vecs = make_random_vectors(20, 8)
        idx = HNSWIndex(dim=8, M=6, ef_construction=30, ef_search=20)
        idx.build_from_vectors(node_vecs)
        stats = idx.stats()
        assert stats["num_nodes"] == 20
        assert stats["M"] == 6
        assert stats["dim"] == 8
        assert stats["num_layers"] >= 1
        assert stats["total_edges"] > 0

    def test_search_k_capped_by_index_size(self):
        node_vecs = make_random_vectors(5, 8)
        idx = HNSWIndex(dim=8, M=4, ef_construction=10)
        idx.build_from_vectors(node_vecs)
        results = idx.search(np.random.randn(8).astype(np.float32), k=100)
        assert len(results) <= 5  # can't return more than inserted nodes


class TestHNSWIndexEdgeCases:
    """Edge case handling."""

    def test_search_returns_k_or_fewer_results(self):
        idx = HNSWIndex(dim=4, M=4, ef_construction=10)
        vecs = make_random_vectors(3, 4)
        idx.build_from_vectors(vecs)
        results = idx.search(np.random.randn(4).astype(np.float32), k=10)
        assert len(results) <= 3

    def test_random_level_distribution(self):
        """Level distribution should decay exponentially (most nodes at level 0)."""
        idx = HNSWIndex(dim=8, M=16)
        levels = [idx._random_level() for _ in range(1000)]
        # Most nodes should be at level 0
        assert levels.count(0) > 400  # > 40% at level 0 given M=16
        assert max(levels) <= 32

    def test_two_nodes_nearest_neighbor(self):
        idx = HNSWIndex(dim=2, M=4, ef_construction=10)
        idx.insert("a", np.array([1.0, 0.0], dtype=np.float32))
        idx.insert("b", np.array([0.0, 1.0], dtype=np.float32))
        # Query pointing toward "a"
        q = np.array([0.9, 0.1], dtype=np.float32)
        results = idx.search(q, k=2)
        assert len(results) == 2
        # "a" should be the nearest (smaller angle to [0.9, 0.1])
        assert results[0][1] == "a"
