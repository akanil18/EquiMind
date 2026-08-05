from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    """Standardized numerical feature vector for a specific entity/ticker at a point in time."""
    vector_id: str
    ticker: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_features: Dict[str, float]
    normalized_features: Dict[str, float] = Field(default_factory=dict)
    feature_count: int = 0
    lineage_sources: List[str] = Field(default_factory=list)


class FeatureSet(BaseModel):
    """Collection of feature vectors across tickers or time windows."""
    set_name: str
    vectors: List[FeatureVector] = Field(default_factory=list)
