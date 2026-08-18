"""
equimind.rag — Agentic RAG Orchestration with HNSW Vector Search

This module implements intelligent, iterative Retrieval-Augmented Generation
using Hierarchical Navigable Small World (HNSW) graphs as the vector index.

The agentic loop:
  1. Embed query → HNSW semantic search → retrieve top-k EvidenceNodes
  2. RAGCriticAgent evaluates evidence sufficiency (source diversity, sentiment
     coverage, age, query-relevance)
  3. If insufficient → RAGQueryRewriter generates a refined sub-query → go to 1
  4. Repeat up to max_iterations, then hand curated evidence to Bull/Bear/Judge debate
"""

from equimind.rag.schema import RAGCriticResult, RAGIterationLog, AgenticRAGResult
from equimind.rag.hnsw_index import HNSWIndex
from equimind.rag.embedder import EmbeddingRouter
from equimind.rag.retriever import HNSWRetriever
from equimind.rag.critic_agent import RAGCriticAgent
from equimind.rag.query_rewriter import RAGQueryRewriter
from equimind.rag.orchestrator import AgenticRAGOrchestrator

__all__ = [
    "RAGCriticResult",
    "RAGIterationLog",
    "AgenticRAGResult",
    "HNSWIndex",
    "EmbeddingRouter",
    "HNSWRetriever",
    "RAGCriticAgent",
    "RAGQueryRewriter",
    "AgenticRAGOrchestrator",
]
