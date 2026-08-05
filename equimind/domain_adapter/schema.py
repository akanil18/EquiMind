from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DomainType(str, Enum):
    FINANCIAL_RESEARCH = "financial_research"
    LEGAL_CASE_RESEARCH = "legal_case_research"
    HEALTHCARE_MEDICAL_REVIEW = "healthcare_medical_review"
    CYBERSECURITY_THREAT_INTEL = "cybersecurity_threat_intel"


class DomainQueryContext(BaseModel):
    """Generic multi-domain research query context."""
    domain: DomainType
    entity_name: str  # Ticker, Legal Case Name, Drug Name, CVE ID
    user_query: str
    target_scope: str = "comprehensive"


class DomainResearchResult(BaseModel):
    """Generic structured output produced by domain-adapted orchestration engine."""
    domain: DomainType
    entity_name: str
    summary_verdict: str
    confidence_score: float
    key_evidence_count: int
    supporting_arguments: List[str] = Field(default_factory=list)
    counter_arguments: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
