import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class EvidenceSource(str, Enum):
    REDDIT = "reddit"
    TWITTER_X = "twitter_x"
    STOCKTWITS = "stocktwits"
    SEC_FILING = "sec_filing"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    FINANCIAL_NEWS = "financial_news"
    GOVT_ANNOUNCEMENT = "govt_announcement"
    COMPANY_BLOG = "company_blog"
    GITHUB_COMMITS = "github_commits"
    JOB_POSTINGS = "job_postings"
    MARKET_PRICES = "market_prices"
    FINANCIAL_STATEMENTS = "financial_statements"
    MACRO_DATA = "macro_data"
    OTHER = "other"


class AuthorCredibility(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED_OFFICIAL = "verified_official"


class SentimentPolarity(str, Enum):
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class EdgeType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CORROBORATES = "corroborates"
    DERIVES_FROM = "derives_from"
    RELATED_TO = "related_to"


class EvidenceEdge(BaseModel):
    """Directed connection between two evidence nodes in the Evidence Graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    description: Optional[str] = None


class EvidenceNode(BaseModel):
    """Structured evidence unit with complete provenance metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: EvidenceSource
    title: str
    content: str
    url: Optional[str] = None
    publication_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    author: Optional[str] = None
    author_credibility: AuthorCredibility = AuthorCredibility.MEDIUM
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    sentiment: SentimentPolarity = SentimentPolarity.NEUTRAL
    affected_ticker: str
    affected_sector: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    vector_embedding: Optional[List[float]] = None
    supporting_references: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
