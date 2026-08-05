import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from equimind.evidence.schema import EvidenceNode


class MemoryTier(str, Enum):
    TIER_1_RAW = "tier1_raw_observations"
    TIER_2_DAILY = "tier2_daily_summaries"
    TIER_3_WEEKLY = "tier3_weekly_syntheses"
    TIER_4_MONTHLY = "tier4_monthly_theses"
    TIER_5_QUARTERLY_PERSISTENT = "tier5_quarterly_persistent_knowledge"


class ResearchReportRecord(BaseModel):
    """Container storing a historical research report output."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_query: str
    rating: str
    conviction_score: float
    summary: str
    evidence_count: int
    raw_evidence_ids: List[str] = Field(default_factory=list)


class EntityKnowledgeEntry(BaseModel):
    """Persistent long-term knowledge repository for a single company/ticker."""
    ticker: str
    company_name: str
    sector: str
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    persistent_thesis: str = ""
    historical_reports: List[ResearchReportRecord] = Field(default_factory=list)
    key_milestones: List[Dict[str, Any]] = Field(default_factory=list)
    cumulative_evidence_nodes: Dict[str, EvidenceNode] = Field(default_factory=dict)
