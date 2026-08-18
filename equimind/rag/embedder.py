"""
Embedding generation for the Agentic RAG pipeline.

Strategy (in priority order):
  1. LLMEmbedder — uses the active LLM provider's embedding API (dense, high-quality)
  2. TFIDFEmbedder — sparse TF-IDF compressed to dense 256-dim via truncated SVD
     (works fully offline, no API key required, deterministic)

EmbeddingRouter automatically selects the best available embedder.
"""

import logging
import math
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TFIDF_DIM = 256         # Output dimension for TF-IDF + SVD dense vectors
MIN_VOCAB_SIZE = 50     # Minimum documents before vocabulary is meaningful
MAX_VOCAB_SIZE = 4096   # Maximum vocabulary size for TF-IDF


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercase."""
    tokens = re.findall(r"\b[a-z][a-z0-9]{1,24}\b", text.lower())
    return tokens


_STOPWORDS = frozenset([
    "the", "is", "in", "it", "of", "to", "and", "a", "an", "for", "on",
    "with", "as", "at", "by", "from", "this", "that", "are", "be", "was",
    "were", "has", "have", "had", "its", "or", "but", "not", "all", "been",
    "we", "they", "their", "our", "can", "will", "may", "also", "more",
    "than", "about", "up", "out", "into", "would", "could", "should",
])


# ── TF-IDF Embedder ───────────────────────────────────────────────────────────

class TFIDFEmbedder:
    """Sparse TF-IDF vectorizer with truncated SVD projection to dense space.

    The corpus must be fitted once via `fit(documents)` before calling `embed()`.
    After fitting on the evidence corpus, the SVD projection matrix maps any
    text into a fixed-dim dense vector regardless of vocabulary size.

    This gives semantic robustness:
      - Documents sharing related terms (e.g. "revenue" / "sales") are placed
        closer in the projected space than raw token overlap suggests.
    """

    def __init__(self, output_dim: int = TFIDF_DIM, max_vocab: int = MAX_VOCAB_SIZE) -> None:
        self.output_dim = output_dim
        self.max_vocab = max_vocab
        self._vocab: Dict[str, int] = {}        # term → column index
        self._idf: np.ndarray = np.array([])    # IDF weights per term
        self._svd_V: Optional[np.ndarray] = None  # Right singular vectors (vocab × output_dim)
        self._is_fitted = False

    def fit(self, documents: List[str]) -> "TFIDFEmbedder":
        """Build vocabulary, compute IDF, and fit SVD projection on documents.

        Parameters
        ----------
        documents : List[str]
            The corpus to fit on (typically all EvidenceNode content strings).
        """
        if not documents:
            logger.warning("TFIDFEmbedder.fit: empty corpus, using fallback vocabulary")
            self._build_fallback_vocab()
            return self

        # ── Build document-term frequency matrix ──
        doc_tokens = [
            [t for t in _tokenize(doc) if t not in _STOPWORDS]
            for doc in documents
        ]

        # Count term-doc frequency for IDF
        doc_freq: Counter = Counter()
        for tokens in doc_tokens:
            doc_freq.update(set(tokens))

        # Select top-N terms by document frequency
        top_terms = [
            term for term, _ in doc_freq.most_common(self.max_vocab)
            if doc_freq[term] >= 1
        ]
        self._vocab = {term: idx for idx, term in enumerate(top_terms)}
        n_docs = len(documents)
        vocab_size = len(self._vocab)

        if vocab_size == 0:
            self._build_fallback_vocab()
            return self

        # ── Compute IDF: log((1 + N) / (1 + df)) + 1 ──
        self._idf = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in self._vocab.items():
            df = doc_freq.get(term, 0)
            self._idf[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        # ── Build TF-IDF matrix for SVD ──
        tf_idf_matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for doc_idx, tokens in enumerate(doc_tokens):
            tf_counts: Counter = Counter(tokens)
            for term, freq in tf_counts.items():
                if term in self._vocab:
                    col = self._vocab[term]
                    tf = 1.0 + math.log(freq) if freq > 0 else 0.0
                    tf_idf_matrix[doc_idx, col] = tf * self._idf[col]

        # L2-normalize rows
        row_norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
        row_norms = np.where(row_norms < 1e-10, 1.0, row_norms)
        tf_idf_matrix /= row_norms

        # ── Truncated SVD to output_dim ──
        actual_dim = min(self.output_dim, min(n_docs, vocab_size) - 1)
        if actual_dim < 2:
            actual_dim = 2

        try:
            # Use randomized SVD for speed (power method approximation)
            U, S, Vt = self._randomized_svd(tf_idf_matrix, n_components=actual_dim)
            self._svd_V = Vt.T  # (vocab_size, actual_dim)
            self.output_dim = actual_dim
        except Exception as e:
            logger.warning(f"SVD failed ({e}), using identity projection")
            self._svd_V = np.eye(vocab_size, self.output_dim, dtype=np.float32)

        self._is_fitted = True
        logger.info(
            f"TFIDFEmbedder fitted: vocab={vocab_size}, "
            f"docs={n_docs}, output_dim={self.output_dim}"
        )
        return self

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a dense vector of shape (output_dim,)."""
        if not self._is_fitted:
            raise RuntimeError("TFIDFEmbedder.embed: call fit() first")

        tokens = [t for t in _tokenize(text) if t not in _STOPWORDS]
        tf_counts: Counter = Counter(tokens)

        # Build sparse TF-IDF vector
        vocab_size = len(self._vocab)
        sparse_vec = np.zeros(vocab_size, dtype=np.float32)
        for term, freq in tf_counts.items():
            if term in self._vocab:
                col = self._vocab[term]
                tf = 1.0 + math.log(freq) if freq > 0 else 0.0
                sparse_vec[col] = tf * self._idf[col]

        # L2-normalize
        norm = np.linalg.norm(sparse_vec)
        if norm > 1e-10:
            sparse_vec /= norm

        # Project to dense space via SVD
        dense_vec = sparse_vec @ self._svd_V  # (output_dim,)
        dense_norm = np.linalg.norm(dense_vec)
        if dense_norm > 1e-10:
            dense_vec /= dense_norm

        return dense_vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts; returns (len(texts), output_dim) array."""
        return np.stack([self.embed(t) for t in texts])

    def _build_fallback_vocab(self) -> None:
        """Minimal fallback when corpus is empty."""
        fallback_terms = [
            "revenue", "earnings", "growth", "profit", "loss", "margin",
            "debt", "cash", "stock", "market", "price", "rate", "risk",
            "buy", "sell", "hold", "analyst", "forecast", "quarter",
        ]
        self._vocab = {t: i for i, t in enumerate(fallback_terms)}
        vocab_size = len(self._vocab)
        self._idf = np.ones(vocab_size, dtype=np.float32)
        self._svd_V = np.eye(vocab_size, min(self.output_dim, vocab_size), dtype=np.float32)
        self.output_dim = min(self.output_dim, vocab_size)
        self._is_fitted = True

    @staticmethod
    def _randomized_svd(
        M: np.ndarray, n_components: int, n_iter: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomized SVD (Halko et al. 2011) — much faster than full SVD for tall/wide matrices."""
        rng = np.random.RandomState(42)
        n_rows, n_cols = M.shape
        k = min(n_components + 10, min(n_rows, n_cols))

        # Random projection
        Q = rng.randn(n_cols, k).astype(np.float32)
        for _ in range(n_iter):
            Q, _ = np.linalg.qr(M @ Q)
            Q, _ = np.linalg.qr(M.T @ Q)

        # Reduced SVD
        B = Q.T @ M
        U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_hat

        return U[:, :n_components], S[:n_components], Vt[:n_components, :]


# ── LLM Embedder ──────────────────────────────────────────────────────────────

class LLMEmbedder:
    """Embedding generator using the active LLM provider's embedding API.

    Falls back gracefully if the provider doesn't support embeddings.
    Supported: OpenAI (text-embedding-3-small), Gemini (text-embedding-004).
    """

    def __init__(self, provider) -> None:
        self.provider = provider
        self._output_dim: Optional[int] = None

    @property
    def output_dim(self) -> int:
        return self._output_dim or 256

    def can_embed(self) -> bool:
        """Check if the provider has embedding capability."""
        if self.provider is None:
            return False
        provider_name = getattr(self.provider, "provider_name", "").lower()
        return provider_name in ("openai", "gemini")

    def embed(self, text: str) -> Optional[np.ndarray]:
        """Call provider embedding API. Returns None if unavailable."""
        if not self.can_embed():
            return None
        try:
            provider_name = self.provider.provider_name.lower()
            if provider_name == "openai":
                return self._embed_openai(text)
            elif provider_name == "gemini":
                return self._embed_gemini(text)
        except Exception as e:
            logger.warning(f"LLMEmbedder: embedding API failed: {e}")
        return None

    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Embed a batch of texts. Returns (N, dim) array or None."""
        vecs = [self.embed(t) for t in texts]
        if any(v is None for v in vecs):
            return None
        arr = np.stack(vecs)
        self._output_dim = arr.shape[1]
        return arr

    def _embed_openai(self, text: str) -> np.ndarray:
        import json
        import urllib.request

        payload = json.dumps({
            "input": text[:8192],   # API truncation limit
            "model": "text-embedding-3-small",
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.provider.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
        self._output_dim = vec.shape[0]
        return vec

    def _embed_gemini(self, text: str) -> np.ndarray:
        import json
        import urllib.request

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"text-embedding-004:embedContent?key={self.provider.api_key}"
        )
        payload = json.dumps({
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text[:8192]}]},
        }).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        vec = np.array(data["embedding"]["values"], dtype=np.float32)
        self._output_dim = vec.shape[0]
        return vec


# ── EmbeddingRouter ───────────────────────────────────────────────────────────

class EmbeddingRouter:
    """Routing embedder: uses LLMEmbedder if available, falls back to TFIDFEmbedder.

    This ensures the HNSW index always has consistent dimensionality embeddings
    regardless of which embedding backend is active.

    Usage
    -----
    router = EmbeddingRouter(provider=provider)
    router.fit_corpus(documents)          # Required before embed()
    vec = router.embed("NVDA revenue")    # (output_dim,)
    """

    def __init__(self, provider=None) -> None:
        self.provider = provider
        self._llm_embedder = LLMEmbedder(provider) if provider else None
        self._tfidf_embedder = TFIDFEmbedder()
        self._use_llm = False
        self._output_dim: Optional[int] = None

    @property
    def output_dim(self) -> int:
        return self._output_dim or self._tfidf_embedder.output_dim

    def fit_corpus(self, documents: List[str]) -> "EmbeddingRouter":
        """Fit TF-IDF on the corpus and probe LLM embedding availability.

        Always fits TF-IDF as fallback. If LLM embedding works, switches to it.
        """
        # Always fit TF-IDF
        t0 = time.time()
        self._tfidf_embedder.fit(documents)
        logger.info(f"TF-IDF fitted in {(time.time() - t0)*1000:.1f}ms")

        # Probe LLM embedding availability
        if self._llm_embedder and self._llm_embedder.can_embed() and documents:
            probe = self._llm_embedder.embed(documents[0][:200])
            if probe is not None:
                self._use_llm = True
                self._output_dim = probe.shape[0]
                logger.info(f"EmbeddingRouter: using LLM embeddings (dim={self._output_dim})")
            else:
                logger.info("EmbeddingRouter: LLM embedding probe failed, using TF-IDF")

        if not self._use_llm:
            self._output_dim = self._tfidf_embedder.output_dim
            logger.info(f"EmbeddingRouter: using TF-IDF (dim={self._output_dim})")

        return self

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if self._use_llm and self._llm_embedder:
            vec = self._llm_embedder.embed(text)
            if vec is not None:
                return vec
        return self._tfidf_embedder.embed(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts; returns (N, output_dim) array."""
        if self._use_llm and self._llm_embedder:
            batch = self._llm_embedder.embed_batch(texts)
            if batch is not None:
                return batch
        return self._tfidf_embedder.embed_batch(texts)
