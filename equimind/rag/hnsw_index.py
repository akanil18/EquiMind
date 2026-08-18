"""
HNSW (Hierarchical Navigable Small World) Vector Index — Pure NumPy Implementation.

HNSW is the industry-standard approximate nearest neighbor (ANN) algorithm used
in production vector databases (Pinecone, Weaviate, Qdrant, Milvus, pgvector).

Algorithm overview:
  - Multi-layer graph where layer 0 contains all nodes and higher layers are
    progressively sparser (exponential decay with probability 1/ln(M))
  - INSERT: Start from the top layer, greedily descend to layer 0, add bidirectional
    edges to ef_construction nearest neighbors at each layer
  - SEARCH: Greedy beam search from top to bottom, maintaining a candidate heap
    of ef_search nearest candidates at each layer

Complexity:
  - Build: O(N * M * log(N)) time, O(N * M) space
  - Query: O(log(N)) expected time with tunable recall/speed tradeoff via ef_search

References:
  - Malkov & Yashunin (2018): "Efficient and robust approximate nearest neighbor
    search using Hierarchical Navigable Small World graphs"
    https://arxiv.org/abs/1603.09320
"""

import heapq
import logging
import math
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HNSWIndex:
    """Hierarchical Navigable Small World (HNSW) approximate nearest neighbor index.

    Supports cosine similarity distance metric, suitable for text embeddings.

    Parameters
    ----------
    dim : int
        Dimensionality of the embedding vectors.
    M : int
        Maximum number of bidirectional connections per node per layer.
        Typical values: 8-64. Higher M → better recall, more memory.
    ef_construction : int
        Size of the dynamic candidate list during index construction.
        Higher ef_construction → better index quality, slower build.
    ef_search : int
        Size of the dynamic candidate list during search.
        Higher ef_search → better recall, slower queries.
    seed : int
        Random seed for reproducible layer assignments.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.M = M                          # Max connections per layer (except layer 0 uses M * 2)
        self.M0 = M * 2                     # Max connections at layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1.0 / math.log(M)        # Level generation factor (1/ln(M))
        self.rng = random.Random(seed)

        # Core data structures
        self._vectors: Dict[str, np.ndarray] = {}           # node_id → normalized unit vector
        self._levels: Dict[str, int] = {}                   # node_id → max layer level
        self._graph: Dict[int, Dict[str, List[str]]] = {}   # layer → node_id → [neighbor_ids]
        self._entry_point: Optional[str] = None             # Current global entry point
        self._max_level: int = -1                           # Highest populated level
        self._insert_order: List[str] = []                  # Insertion order for reproducibility

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def insert(self, node_id: str, vector: np.ndarray) -> None:
        """Insert a vector into the HNSW index.

        Parameters
        ----------
        node_id : str
            Unique identifier for the node (maps to EvidenceNode.id).
        vector : np.ndarray
            Raw embedding vector of shape (dim,). Will be L2-normalized internally.
        """
        if node_id in self._vectors:
            logger.debug(f"HNSW: node '{node_id}' already in index, skipping.")
            return

        # Normalize to unit sphere for cosine similarity via inner product
        norm_vec = self._normalize(vector)
        self._vectors[node_id] = norm_vec

        # Assign node to a random maximum level
        node_level = self._random_level()
        self._levels[node_id] = node_level
        self._insert_order.append(node_id)

        # Initialize graph layers for this node
        for layer in range(node_level + 1):
            if layer not in self._graph:
                self._graph[layer] = {}
            self._graph[layer][node_id] = []

        # ── Cold start: first node becomes entry point ──
        if self._entry_point is None:
            self._entry_point = node_id
            self._max_level = node_level
            return

        # ── Greedy search from top level down to node_level + 1 ──
        # (single-element nearest-neighbor descent to find best entry for construction)
        ep = self._entry_point
        curr_dist = self._distance(norm_vec, self._vectors[ep])

        for layer in range(self._max_level, node_level, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self._get_neighbors(ep, layer)
                for nb in neighbors:
                    d = self._distance(norm_vec, self._vectors[nb])
                    if d < curr_dist:
                        curr_dist = d
                        ep = nb
                        changed = True

        # ── Layer-by-layer construction from node_level down to 0 ──
        entry_points = [ep]
        for layer in range(min(node_level, self._max_level), -1, -1):
            ef = self.ef_construction
            candidates = self._search_layer(norm_vec, entry_points, ef, layer)

            # Select M neighbors (M0 at layer 0)
            max_links = self.M0 if layer == 0 else self.M
            neighbors = self._select_neighbors(norm_vec, candidates, max_links)

            # Add bidirectional links
            self._graph[layer][node_id] = [nb for _, nb in neighbors]

            for dist, nb in neighbors:
                nb_neighbors = self._get_neighbors(nb, layer)
                nb_neighbors.append(node_id)

                # Prune if over limit
                if len(nb_neighbors) > max_links:
                    nb_vec = self._vectors[nb]
                    pruned = self._select_neighbors(
                        nb_vec,
                        [(self._distance(nb_vec, self._vectors[x]), x) for x in nb_neighbors],
                        max_links,
                    )
                    self._graph[layer][nb] = [x for _, x in pruned]
                else:
                    self._graph[layer][nb] = nb_neighbors

            entry_points = [nb for _, nb in candidates[:1]]

        # Update global entry point if node has higher level
        if node_level > self._max_level:
            self._max_level = node_level
            self._entry_point = node_id

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
    ) -> List[Tuple[float, str]]:
        """Search for k approximate nearest neighbors.

        Parameters
        ----------
        query_vector : np.ndarray
            Query embedding of shape (dim,).
        k : int
            Number of nearest neighbors to return.
        ef : int, optional
            Search beam width. Defaults to max(k, self.ef_search).

        Returns
        -------
        List[Tuple[float, str]]
            List of (distance, node_id) pairs sorted by distance ascending.
            Distance is 1 - cosine_similarity (0 = identical, 2 = opposite).
        """
        if not self._vectors:
            return []

        ef = ef or max(k, self.ef_search)
        query_norm = self._normalize(query_vector)

        ep = self._entry_point
        curr_dist = self._distance(query_norm, self._vectors[ep])

        # Descend from top level to level 1 (greedy single-neighbor)
        for layer in range(self._max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for nb in self._get_neighbors(ep, layer):
                    d = self._distance(query_norm, self._vectors[nb])
                    if d < curr_dist:
                        curr_dist = d
                        ep = nb
                        changed = True

        # Layer 0: full beam search with ef candidates
        candidates = self._search_layer(query_norm, [ep], ef, layer=0)

        # Return top-k
        results = sorted(candidates)[:k]
        return results

    def build_from_vectors(self, node_vectors: Dict[str, np.ndarray]) -> int:
        """Batch insert all node_id → vector mappings.

        Returns the number of nodes successfully inserted.
        """
        inserted = 0
        for node_id, vec in node_vectors.items():
            try:
                self.insert(node_id, vec)
                inserted += 1
            except Exception as e:
                logger.warning(f"HNSW: failed to insert node '{node_id}': {e}")
        logger.info(f"HNSW: built index with {inserted} nodes, max_level={self._max_level}")
        return inserted

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._vectors

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[str],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, str]]:
        """Beam search within a single layer of the HNSW graph.

        Maintains two heaps:
          - candidates (min-heap by distance): nodes to explore
          - found (max-heap by distance): best ef nodes found so far

        Returns list of (distance, node_id) tuples.
        """
        visited: Set[str] = set(entry_points)

        # Initialize with entry points
        candidates: List[Tuple[float, str]] = []
        found: List[Tuple[float, str]] = []     # stored as max-heap via negation

        for ep in entry_points:
            d = self._distance(query, self._vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(found, (-d, ep))     # negate for max-heap behavior

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)

            # Pruning: if current candidate is further than worst in found, stop
            worst_found_dist = -found[0][0] if found else float("inf")
            if c_dist > worst_found_dist:
                break

            # Explore neighbors
            for nb in self._get_neighbors(c_id, layer):
                if nb in visited:
                    continue
                visited.add(nb)

                nb_dist = self._distance(query, self._vectors[nb])
                worst_found_dist = -found[0][0] if found else float("inf")

                if nb_dist < worst_found_dist or len(found) < ef:
                    heapq.heappush(candidates, (nb_dist, nb))
                    heapq.heappush(found, (-nb_dist, nb))

                    # Keep found bounded to ef
                    if len(found) > ef:
                        heapq.heappop(found)

        return [(- d, node_id) for d, node_id in found]

    def _select_neighbors(
        self,
        query: np.ndarray,
        candidates: List[Tuple[float, str]],
        max_neighbors: int,
    ) -> List[Tuple[float, str]]:
        """Select best neighbors using simple distance-based pruning (Algorithm 3 from paper)."""
        return sorted(candidates)[:max_neighbors]

    def _get_neighbors(self, node_id: str, layer: int) -> List[str]:
        """Returns the neighbor list for a node at a given layer (empty if not present)."""
        return self._graph.get(layer, {}).get(node_id, [])

    def _random_level(self) -> int:
        """Generates a random level for a new node using exponential distribution.

        P(level = l) = (1 - exp(-1/M)) * exp(-l/M)
        """
        level = 0
        while self.rng.random() < (1.0 / self.M) and level < 32:
            level += 1
        return level

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2-normalizes a vector to the unit sphere."""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return np.zeros_like(vector, dtype=np.float32)
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance = 1 - cosine_similarity.

        For L2-normalized unit vectors, inner product = cosine similarity.
        Range: [0, 2] where 0 = identical direction, 2 = opposite direction.
        """
        return float(1.0 - np.dot(a, b))

    def stats(self) -> Dict:
        """Returns diagnostic statistics about the index."""
        total_edges = sum(
            len(neighbors)
            for layer_graph in self._graph.values()
            for neighbors in layer_graph.values()
        )
        return {
            "num_nodes": len(self._vectors),
            "max_level": self._max_level,
            "num_layers": len(self._graph),
            "total_edges": total_edges,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "dim": self.dim,
        }
