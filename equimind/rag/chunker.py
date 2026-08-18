"""
FinancialChunker — Document-Aware Financial Chunking & Parent-Child Management.

Features:
  - Document-aware chunking for different financial source types:
      * SEC Filings (Item-aware / section-aware chunking)
      * Earnings Transcripts (Speaker / Q&A turn chunking)
      * Financial News (Paragraph & semantic boundary chunking)
      * Market Data & Macro (Structured indicator chunking)
  - Parent-Child Chunk Splitter:
      * Small child chunks (e.g. 100-200 tokens) for high precision vector retrieval
      * Full parent chunk (e.g. 800-1500 tokens) returned to LLM for rich context
  - Lineage & Metadata Tracking:
      * chunk_id, parent_id, ticker, source_type, timestamp, section_name
"""

import uuid
import re
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from equimind.evidence.schema import EvidenceNode, EvidenceSource, AuthorCredibility, SentimentPolarity

logger = logging.getLogger(__name__)


class FinancialChunk(BaseModel):
    """Chunk representing a segment of a financial document with full lineage."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    is_parent: bool = False
    content: str
    ticker: str
    source_type: EvidenceSource
    section_name: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1
    token_count_est: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    publication_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_evidence_node(self) -> EvidenceNode:
        """Convert chunk to an EvidenceNode for pipeline compatibility."""
        return EvidenceNode(
            id=self.chunk_id,
            source_type=self.source_type,
            title=f"[{self.source_type.value.upper()}] {self.section_name or self.ticker} (Chunk {self.chunk_index+1}/{self.total_chunks})",
            content=self.content,
            affected_ticker=self.ticker,
            publication_timestamp=self.publication_timestamp,
            metadata={
  
                **self.metadata,
                "parent_id": self.parent_id,
                "is_parent": self.is_parent,
                "chunk_index": self.chunk_index,
                "total_chunks": self.total_chunks,
                "section_name": self.section_name,
            },
        )


class FinancialChunker:
    """Document-aware financial chunker with parent-child hierarchical splitting."""

    @classmethod
    def chunk_document(
        cls,
        content: str,
        ticker: str,
        source_type: EvidenceSource,
        parent_id: Optional[str] = None,
        max_chunk_chars: int = 600,
        overlap_chars: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[FinancialChunk]:
        """Splits raw text into document-aware chunks."""
        meta = metadata or {}
        ticker_upper = ticker.upper()

        if source_type in (EvidenceSource.SEC_FILING, EvidenceSource.FINANCIAL_STATEMENTS):
            return cls._chunk_sec_filing(content, ticker_upper, source_type, parent_id, max_chunk_chars, overlap_chars, meta)
        elif source_type == EvidenceSource.EARNINGS_TRANSCRIPT:
            return cls._chunk_earnings_transcript(content, ticker_upper, parent_id, max_chunk_chars, meta)
        elif source_type in (EvidenceSource.FINANCIAL_NEWS, EvidenceSource.COMPANY_BLOG):
            return cls._chunk_news(content, ticker_upper, source_type, parent_id, max_chunk_chars, overlap_chars, meta)
        else:
            return cls._chunk_generic(content, ticker_upper, source_type, parent_id, max_chunk_chars, overlap_chars, meta)

    @classmethod
    def create_parent_child_chunks(
        cls,
        content: str,
        ticker: str,
        source_type: EvidenceSource,
        parent_chunk_chars: int = 1500,
        child_chunk_chars: int = 400,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[FinancialChunk], List[FinancialChunk]]:
        """Creates hierarchical parent (large) and child (small) chunks.
        
        Returns:
            (parent_chunks, child_chunks)
        """
        parent_chunks = cls.chunk_document(
            content=content,
            ticker=ticker,
            source_type=source_type,
            max_chunk_chars=parent_chunk_chars,
            overlap_chars=150,
            metadata=metadata,
        )

        all_children = []
        for p in parent_chunks:
            p.is_parent = True
            children = cls.chunk_document(
                content=p.content,
                ticker=ticker,
                source_type=source_type,
                parent_id=p.chunk_id,
                max_chunk_chars=child_chunk_chars,
                overlap_chars=80,
                metadata={**(metadata or {}), "parent_chunk_id": p.chunk_id},
            )
            all_children.extend(children)

        return parent_chunks, all_children

    @classmethod
    def _chunk_sec_filing(
        cls, content: str, ticker: str, source_type: EvidenceSource, parent_id: Optional[str],
        max_chars: int, overlap: int, meta: Dict[str, Any]
    ) -> List[FinancialChunk]:
        """Item/section aware chunking for SEC filings."""
        # Detect Item headers (e.g. Item 1A. Risk Factors, Item 7. MD&A)
        section_pattern = r"(Item\s+[0-9A-Z]+[\.\:\s]+[^\n]+)"
        sections = re.split(section_pattern, content, flags=re.IGNORECASE)

        chunks: List[FinancialChunk] = []
        current_section = "General SEC Disclosure"

        if len(sections) > 1:
            for i in range(1, len(sections), 2):
                header = sections[i].strip()
                body = sections[i+1].strip() if i+1 < len(sections) else ""
                current_section = header[:80]
                sec_chunks = cls._sliding_window(body, max_chars, overlap)
                for idx, text in enumerate(sec_chunks):
                    if text.strip():
                        chunks.append(FinancialChunk(
                            parent_id=parent_id,
                            content=f"[{header}]\n{text}",
                            ticker=ticker,
                            source_type=source_type,
                            section_name=current_section,
                            chunk_index=len(chunks),
                            token_count_est=len(text) // 4,
                            metadata=meta,
                        ))
        else:
            sub_texts = cls._sliding_window(content, max_chars, overlap)
            for idx, text in enumerate(sub_texts):
                chunks.append(FinancialChunk(
                    parent_id=parent_id,
                    content=text,
                    ticker=ticker,
                    source_type=source_type,
                    section_name="SEC Filing",
                    chunk_index=idx,
                    total_chunks=len(sub_texts),
                    token_count_est=len(text) // 4,
                    metadata=meta,
                ))

        for c in chunks:
            c.total_chunks = len(chunks)
        return chunks

    @classmethod
    def _chunk_earnings_transcript(
        cls, content: str, ticker: str, parent_id: Optional[str],
        max_chars: int, meta: Dict[str, Any]
    ) -> List[FinancialChunk]:
        """Speaker / Question & Answer aware chunking for transcripts."""
        # Split by speaker patterns (e.g., "CEO:", "Analyst:", "Tim Cook - Apple:")
        speaker_pattern = r"([A-Z][A-Za-z\s\.\,\-]+(?:\s*\(.*?\))?\s*\:)"
        parts = re.split(speaker_pattern, content)

        chunks: List[FinancialChunk] = []
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                speaker = parts[i].strip().rstrip(":")
                speech = parts[i+1].strip() if i+1 < len(parts) else ""
                if speech:
                    sub_chunks = cls._sliding_window(speech, max_chars, 80)
                    for text in sub_chunks:
                        chunks.append(FinancialChunk(
                            parent_id=parent_id,
                            content=f"Speaker [{speaker}]: {text}",
                            ticker=ticker,
                            source_type=EvidenceSource.EARNINGS_TRANSCRIPT,
                            section_name=f"Transcript: {speaker[:40]}",
                            chunk_index=len(chunks),
                            token_count_est=len(text) // 4,
                            metadata={**meta, "speaker": speaker},
                        ))
        else:
            sub_texts = cls._sliding_window(content, max_chars, 80)
            for idx, text in enumerate(sub_texts):
                chunks.append(FinancialChunk(
                    parent_id=parent_id,
                    content=text,
                    ticker=ticker,
                    source_type=EvidenceSource.EARNINGS_TRANSCRIPT,
                    section_name="Earnings Call",
                    chunk_index=idx,
                    total_chunks=len(sub_texts),
                    token_count_est=len(text) // 4,
                    metadata=meta,
                ))

        for c in chunks:
            c.total_chunks = len(chunks)
        return chunks

    @classmethod
    def _chunk_news(
        cls, content: str, ticker: str, source_type: EvidenceSource, parent_id: Optional[str],
        max_chars: int, overlap: int, meta: Dict[str, Any]
    ) -> List[FinancialChunk]:
        """Paragraph & semantic boundary aware chunking for news."""
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        chunks: List[FinancialChunk] = []
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            if current_len + len(p) <= max_chars:
                current_chunk.append(p)
                current_len += len(p) + 2
            else:
                if current_chunk:
                    text = "\n\n".join(current_chunk)
                    chunks.append(FinancialChunk(
                        parent_id=parent_id,
                        content=text,
                        ticker=ticker,
                        source_type=source_type,
                        chunk_index=len(chunks),
                        token_count_est=len(text) // 4,
                        metadata=meta,
                    ))
                current_chunk = [p]
                current_len = len(p)

        if current_chunk:
            text = "\n\n".join(current_chunk)
            chunks.append(FinancialChunk(
                parent_id=parent_id,
                content=text,
                ticker=ticker,
                source_type=source_type,
                chunk_index=len(chunks),
                token_count_est=len(text) // 4,
                metadata=meta,
            ))

        for c in chunks:
            c.total_chunks = len(chunks)
        return chunks

    @classmethod
    def _chunk_generic(
        cls, content: str, ticker: str, source_type: EvidenceSource, parent_id: Optional[str],
        max_chars: int, overlap: int, meta: Dict[str, Any]
    ) -> List[FinancialChunk]:
        sub_texts = cls._sliding_window(content, max_chars, overlap)
        chunks = []
        for idx, text in enumerate(sub_texts):
            chunks.append(FinancialChunk(
                parent_id=parent_id,
                content=text,
                ticker=ticker,
                source_type=source_type,
                chunk_index=idx,
                total_chunks=len(sub_texts),
                token_count_est=len(text) // 4,
                metadata=meta,
            ))
        return chunks

    @staticmethod
    def _sliding_window(text: str, max_chars: int, overlap: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        result = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunk = text[start:end]
            result.append(chunk)
            start += max_chars - overlap
        return result
