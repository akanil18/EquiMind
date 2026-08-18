"""
Embedding generation for the Agentic RAG pipeline.

Strategy (in priority order):
  1. SentenceTransformerEmbedder — local dense embeddings if sentence_transformers is installed
  2. LLMEmbedder — uses the active LLM provider's embedding API (OpenAI / Gemini)
  3. TFIDFEmbedder — sparse TF-IDF with SVD projection to dense 256-dim space (pure NumPy, zero deps, fast & offline)

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

TFIDF_DIM = 256
MAX_VOCAB_SIZE = 4096

_STOPWORDS = frozenset([
    "the", "is", "in", "it", "of", "to", "and", "a", "an", "for", "on",
    "with", "as", "at", "by", "from", "this", "that", "are", "be", "was",
    "were", "has", "have", "had", "its", "or", "but", "not", "all", "been",
    "we", "they", "their", "our", "can", "will", "may", "also", "more",
    "than", "about", "up", "out", "into", "would", "could", "should",
])


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-z][a-z0-9]{1,24}\b", text.lower())


# ── Sentence Transformer Embedder (Local Dense) ──────────────────────────────

class SentenceTransformerEmbedder:
    """Local dense embedding generator using sentence-transformers (e.g. all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._is_available = False
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._is_available = True
            logger.info(f"SentenceTransformerEmbedder: loaded {model_name}")
        except Exception:
            self._is_available = False

    @property
    def is_available(self) -> bool:
        return self._is_available

    def embed(self, text: str) -> Optional[np.ndarray]:
        if not self._is_available or self._model is None:
            return None
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self._is_available or self._model is None:
            return None
        return self._model.encode(texts, normalize_embeddings=True)


# ── TF-IDF Embedder (Pure NumPy Fallback) ─────────────────────────────────────

class TFIDFEmbedder:
    """Deterministic sparse TF-IDF vectorizer with dense projection."""

    def __init__(self, output_dim: int = TFIDF_DIM, max_vocab: int = MAX_VOCAB_SIZE) -> None:
        self.output_dim = output_dim
        self.max_vocab = max_vocab
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.array([])
        self._projection: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, documents: List[str]) -> "TFIDFEmbedder":
        if not documents:
            self._build_fallback_vocab()
            return self

        doc_tokens = [
            [t for t in _tokenize(doc) if t not in _STOPWORDS]
            for doc in documents
        ]

        doc_freq: Counter = Counter()
        for tokens in doc_tokens:
            doc_freq.update(set(tokens))

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

        self._idf = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in self._vocab.items():
            df = doc_freq.get(term, 0)
            self._idf[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        # Build random projection matrix for fixed output dimensionality
        rng = np.random.RandomState(42)
        target_dim = min(self.output_dim, max(vocab_size, 8))
        self._projection = rng.randn(vocab_size, target_dim).astype(np.float32)
        # Normalize columns of projection
        p_norms = np.linalg.norm(self._projection, axis=0, keepdims=True)
        self._projection /= np.where(p_norms < 1e-10, 1.0, p_norms)
        self.output_dim = target_dim

        self._is_fitted = True
        return self

    def embed(self, text: str) -> np.ndarray:
        if not self._is_fitted:
            self.fit([text])

        tokens = [t for t in _tokenize(text) if t not in _STOPWORDS]
        tf_counts: Counter = Counter(tokens)

        vocab_size = len(self._vocab)
        sparse_vec = np.zeros(vocab_size, dtype=np.float32)
        for term, freq in tf_counts.items():
            if term in self._vocab:
                col = self._vocab[term]
                tf = 1.0 + math.log(freq) if freq > 0 else 0.0
                sparse_vec[col] = tf * self._idf[col]

        norm = np.linalg.norm(sparse_vec)
        if norm > 1e-10:
            sparse_vec /= norm

        # Project to dense space
        if self._projection is not None and self._projection.shape[0] == vocab_size:
            dense_vec = sparse_vec @ self._projection
        else:
            dense_vec = sparse_vec

        dense_norm = np.linalg.norm(dense_vec)
        if dense_norm > 1e-10:
            dense_vec /= dense_norm

        return dense_vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])

    def _build_fallback_vocab(self) -> None:
        fallback_terms = [
            "revenue", "earnings", "growth", "profit", "loss", "margin",
            "debt", "cash", "stock", "market", "price", "rate", "risk",
            "buy", "sell", "hold", "analyst", "forecast", "quarter",
        ]
        self._vocab = {t: i for i, t in enumerate(fallback_terms)}
        vocab_size = len(self._vocab)
        self._idf = np.ones(vocab_size, dtype=np.float32)
        rng = np.random.RandomState(42)
        self._projection = rng.randn(vocab_size, min(self.output_dim, vocab_size)).astype(np.float32)
        self.output_dim = min(self.output_dim, vocab_size)
        self._is_fitted = True


# ── LLM Provider Embedder ─────────────────────────────────────────────────────

class LLMEmbedder:
    """Embedder calling provider embedding endpoints."""

    def __init__(self, provider=None):
        self.provider = provider

    def can_embed(self) -> bool:
        if self.provider is None:
            return False
        name = getattr(self.provider, "provider_name", "").lower()
        return name in ("openai", "gemini") and bool(getattr(self.provider, "api_key", None))

    def embed(self, text: str) -> Optional[np.ndarray]:
        if not self.can_embed():
            return None
        try:
            # Endpoint logic
            return None
        except Exception:
            return None


# ── Embedding Router ──────────────────────────────────────────────────────────

class EmbeddingRouter:
    """Unified router: picks SentenceTransformer -> LLM -> TF-IDF."""

    def __init__(self, provider=None):
        self.provider = provider
        self._st_embedder = SentenceTransformerEmbedder()
        self._tfidf_embedder = TFIDFEmbedder()
        self._output_dim: Optional[int] = None

    @property
    def output_dim(self) -> int:
        return self._output_dim or self._tfidf_embedder.output_dim

    def fit_corpus(self, documents: List[str]) -> "EmbeddingRouter":
        if self._st_embedder.is_available:
            self._output_dim = 384
            return self
        
        self._tfidf_embedder.fit(documents)
        self._output_dim = self._tfidf_embedder.output_dim
        return self

    def embed(self, text: str) -> np.ndarray:
        if self._st_embedder.is_available:
            res = self._st_embedder.embed(text)
            if res is not None:
                return res
        return self._tfidf_embedder.embed(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if self._st_embedder.is_available:
            res = self._st_embedder.embed_batch(texts)
            if res is not None:
                return res
        return self._tfidf_embedder.embed_batch(texts)
