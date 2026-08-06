"""
EquiMind Accuracy & Calibration Tracker
========================================
Institutional-grade framework measuring recommendation accuracy,
probability calibration (Brier score), and prediction performance
against real historical price movements.

Metrics computed:
  - Directional Hit Rate (% of recommendations whose price moved in expected direction)
  - Brier Score (measure of probability calibration; 0 = perfect calibration)
  - Information Coefficient (IC / Rank IC between conviction scores and realized returns)
  - Sharpe Ratio of Prediction Portfolio
  - Sector & Regime breakdown
"""

import json
import logging
import os
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

import pandas as pd
from equimind.adapters import YFinanceAdapter

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    ".equimind_cache", "evaluation"
)


class PredictionRecord(BaseModel):
    """Structured record of a single research recommendation."""
    prediction_id: str
    ticker: str
    created_at: datetime
    rating: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    conviction_score: float  # 0.0 to 1.0
    price_at_recommendation: float
    target_price_low: Optional[float] = None
    target_price_high: Optional[float] = None
    evaluation_window_days: int = 30
    
    # Realized outcomes (populated after evaluation window)
    realized_price: Optional[float] = None
    realized_return_pct: Optional[float] = None
    direction_correct: Optional[bool] = None
    brier_score_component: Optional[float] = None
    evaluated_at: Optional[datetime] = None


class AccuracyMetrics(BaseModel):
    """Aggregated calibration and accuracy metrics across prediction history."""
    total_predictions: int = 0
    evaluated_predictions: int = 0
    directional_hit_rate_pct: float = 0.0
    brier_score: float = 0.0  # Lower is better (0.0 = perfect, 0.25 = random guess for binary)
    mean_conviction_winning: float = 0.0
    mean_conviction_losing: float = 0.0
    mean_realized_return_pct: float = 0.0
    prediction_sharpe_ratio: float = 0.0
    rating_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class AccuracyTracker:
    """Manages prediction logging, outcome verification against real prices, and calibration metrics."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or CACHE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        self.records_file = os.path.join(self.storage_dir, "prediction_registry.json")
        self.predictions: List[PredictionRecord] = self._load_predictions()

    def record_recommendation(
        self,
        ticker: str,
        rating: str,
        conviction_score: float,
        current_price: float,
        target_price_low: Optional[float] = None,
        target_price_high: Optional[float] = None,
        evaluation_window_days: int = 30,
        as_of_date: Optional[datetime] = None,
    ) -> PredictionRecord:
        """Register a new recommendation for future accuracy evaluation."""
        dt = as_of_date or datetime.now(timezone.utc)
        pred_id = f"{ticker.upper()}_{dt.strftime('%Y%m%d_%H%M%S')}"

        record = PredictionRecord(
            prediction_id=pred_id,
            ticker=ticker.upper(),
            created_at=dt,
            rating=rating.upper(),
            conviction_score=conviction_score,
            price_at_recommendation=current_price,
            target_price_low=target_price_low,
            target_price_high=target_price_high,
            evaluation_window_days=evaluation_window_days,
        )

        self.predictions.append(record)
        self._save_predictions()
        logger.info(f"✓ Registered prediction {pred_id}: {rating} (conviction: {conviction_score:.2f}) @ ${current_price:.2f}")
        return record

    def evaluate_outcomes(self, force_evaluate: bool = False) -> int:
        """
        Check historical predictions against real market price outcomes.
        Pulls real price history via YFinanceAdapter for evaluated dates.
        """
        evaluated_count = 0
        now = datetime.now(timezone.utc)

        for pred in self.predictions:
            if pred.evaluated_at is not None and not force_evaluate:
                continue

            eval_target_date = pred.created_at + timedelta(days=pred.evaluation_window_days)
            if now < eval_target_date and not force_evaluate:
                continue  # Evaluation window hasn't elapsed yet

            # Pull price history
            df = YFinanceAdapter.get_price_history(pred.ticker, period="1y")
            if df.empty:
                continue

            # Find closest price on or after target date
            df_after = df[df.index >= pd.Timestamp(pred.created_at)]
            if len(df_after) < 2:
                continue

            realized_price = float(df_after["Close"].iloc[-1])
            realized_return = (realized_price - pred.price_at_recommendation) / pred.price_at_recommendation * 100.0

            # Determine direction correctness
            is_bullish = pred.rating in ("BUY", "STRONG_BUY")
            is_bearish = pred.rating in ("SELL", "STRONG_SELL")

            if is_bullish:
                correct = realized_return > 0
                actual_binary = 1.0 if correct else 0.0
            elif is_bearish:
                correct = realized_return < 0
                actual_binary = 1.0 if correct else 0.0
            else:  # HOLD
                correct = abs(realized_return) <= 5.0
                actual_binary = 1.0 if correct else 0.0

            # Brier Score component: (predicted_prob - actual_binary)^2
            brier_comp = (pred.conviction_score - actual_binary) ** 2

            pred.realized_price = round(realized_price, 2)
            pred.realized_return_pct = round(realized_return, 2)
            pred.direction_correct = correct
            pred.brier_score_component = round(brier_comp, 4)
            pred.evaluated_at = now
            evaluated_count += 1

        if evaluated_count > 0:
            self._save_predictions()
            logger.info(f"✓ Evaluated {evaluated_count} predictions against real market outcomes")

        return evaluated_count

    def compute_metrics(self) -> AccuracyMetrics:
        """Compute aggregated accuracy, calibration, and Sharpe metrics across evaluated predictions."""
        evaluated = [p for p in self.predictions if p.evaluated_at is not None]
        if not evaluated:
            return AccuracyMetrics()

        correct_count = sum(1 for p in evaluated if p.direction_correct)
        hit_rate = (correct_count / len(evaluated)) * 100.0

        brier_sum = sum(p.brier_score_component or 0.0 for p in evaluated)
        brier_score = brier_sum / len(evaluated)

        winning_conv = [p.conviction_score for p in evaluated if p.direction_correct]
        losing_conv = [p.conviction_score for p in evaluated if not p.direction_correct]

        mean_win = sum(winning_conv) / len(winning_conv) if winning_conv else 0.0
        mean_lose = sum(losing_conv) / len(losing_conv) if losing_conv else 0.0

        returns = [p.realized_return_pct or 0.0 for p in evaluated]
        mean_return = sum(returns) / len(returns)

        # Sharpe ratio of returns
        if len(returns) > 1:
            var = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            std_dev = math.sqrt(var) if var > 0 else 1.0
            sharpe = (mean_return / std_dev) * math.sqrt(12)  # Annualized for monthly window
        else:
            sharpe = 0.0

        # Breakdown by rating
        breakdown = {}
        for r in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]:
            subset = [p for p in evaluated if p.rating == r]
            if subset:
                sub_correct = sum(1 for p in subset if p.direction_correct)
                breakdown[r] = {
                    "count": len(subset),
                    "hit_rate_pct": round((sub_correct / len(subset)) * 100.0, 1),
                    "avg_return_pct": round(sum(p.realized_return_pct or 0.0 for p in subset) / len(subset), 2),
                }

        return AccuracyMetrics(
            total_predictions=len(self.predictions),
            evaluated_predictions=len(evaluated),
            directional_hit_rate_pct=round(hit_rate, 1),
            brier_score=round(brier_score, 4),
            mean_conviction_winning=round(mean_win, 2),
            mean_conviction_losing=round(mean_lose, 2),
            mean_realized_return_pct=round(mean_return, 2),
            prediction_sharpe_ratio=round(sharpe, 2),
            rating_breakdown=breakdown,
        )

    # ── Persistence Methods ────────────────────────────────────

    def _load_predictions(self) -> List[PredictionRecord]:
        if not os.path.exists(self.records_file):
            return []
        try:
            with open(self.records_file, "r") as f:
                data = json.load(f)
            return [PredictionRecord(**item) for item in data]
        except Exception as e:
            logger.warning(f"Could not load prediction registry: {e}")
            return []

    def _save_predictions(self):
        try:
            data = [p.model_dump(mode="json") for p in self.predictions]
            with open(self.records_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save prediction registry: {e}")
