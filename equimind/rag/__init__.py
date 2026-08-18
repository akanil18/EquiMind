"""
equimind.rag — Production Agentic RAG Orchestration System

Exposes:
  - Vector Store: VectorStore, HNSWVectorStore, MetadataFilter
  - Hybrid Retrieval: HybridRetriever, BM25Index
  - Reranker: CrossEncoderReranker
  - Chunker: FinancialChunker, FinancialChunk
  - Embedder: SentenceTransformerEmbedder, TFIDFEmbedder, EmbeddingRouter
  - Evaluation: RAGEvaluator, RAGMetricsResult, GoldenQueryEntry
  - Agentic Orchestration: AgenticRAGOrchestrator, RAGCriticAgent, RAGQueryRewriter
  - Schemas: RAGCriticResult, RAGIterationLog, AgenticRAGResult
"""

from equimind.rag.schema import RAGCriticResult, RAGIterationLog, AgenticRAGResult
from equimind.rag.hnsw_index import HNSWIndex
from equimind.rag.embedder import SentenceTransformerEmbedder, TFIDFEmbedder, EmbeddingRouter
from equimind.rag.vector_store import VectorStore, HNSWVectorStore, MetadataFilter
from equimind.rag.hybrid_retriever import HybridRetriever, BM25Index
from equimind.rag.reranker import CrossEncoderReranker
from equimind.rag.chunker import FinancialChunker, FinancialChunk
from equimind.rag.evaluator import RAGEvaluator, RAGMetricsResult, GoldenQueryEntry
from equimind.rag.retriever import HNSWRetriever
from equimind.rag.critic_agent import RAGCriticAgent
from equimind.rag.query_rewriter import RAGQueryRewriter
from equimind.rag.orchestrator import AgenticRAGOrchestrator

__all__ = [
    "VectorStore",
    "HNSWVectorStore",
    "MetadataFilter",
    "HybridRetriever",
    "BM25Index",
    "CrossEncoderReranker",
    "FinancialChunker",
    "FinancialChunk",
    "SentenceTransformerEmbedder",
    "TFIDFEmbedder",
    "EmbeddingRouter",
    "RAGEvaluator",
    "RAGMetricsResult",
    "GoldenQueryEntry",
    "HNSWIndex",
    "HNSWRetriever",
    "RAGCriticAgent",
    "RAGQueryRewriter",
    "AgenticRAGOrchestrator",
    "RAGCriticResult",
    "RAGIterationLog",
    "AgenticRAGResult",
]
