from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class OutcomeEvaluation(BaseModel):
    """Evaluation of a historical research recommendation against actual market outcome."""
    report_id: str
    ticker: str
    recommendation_date: datetime
    recommended_rating: str
    initial_price: float
    evaluation_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actual_price: float
    price_change_pct: float
    was_successful: bool
    bias_detected: Optional[str] = None
    reflection_notes: str


class ReflectionSummary(BaseModel):
    """Aggregate self-reflection and calibration metrics across historical recommendations."""
    total_evaluated: int
    successful_count: int
    accuracy_rate_pct: float
    detected_biases: List[str] = Field(default_factory=list)
    recommended_conviction_calibration_factor: float = 1.0  # >1.0 boost, <1.0 dampener
