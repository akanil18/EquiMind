from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class InvestmentRating(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class BullCase(BaseModel):
    """Evidence-backed thesis supporting investment."""
    thesis: str
    key_catalysts: List[str] = Field(default_factory=list)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    upside_price_target: Optional[float] = None


class BearCase(BaseModel):
    """Evidence-backed thesis opposing investment."""
    thesis: str
    key_headwinds: List[str] = Field(default_factory=list)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    downside_price_target: Optional[float] = None


class DebateSynthesis(BaseModel):
    """Judicial evaluation of Bull vs Bear evidence."""
    winning_thesis_summary: str
    discarded_unbacked_claims: List[str] = Field(default_factory=list)
    resolved_contradictions: List[str] = Field(default_factory=list)
    evidence_strength_ratio: float = 1.0  # Bull Evidence Score / Bear Evidence Score


class InvestmentRecommendation(BaseModel):
    """Comprehensive, explainable institutional investment recommendation."""
    ticker: str
    rating: InvestmentRating
    conviction_score: float = Field(ge=0.0, le=1.0)
    current_price: float
    target_entry_range: Tuple[float, float]
    expected_risk_reward_ratio: float
    recommended_portfolio_allocation: str
    bull_case: BullCase
    bear_case: BearCase
    debate_synthesis: DebateSynthesis
    provenance_citations: List[Dict[str, Any]] = Field(default_factory=list)
    assumptions_and_risks: List[str] = Field(default_factory=list)
