"""
EquiMind Historical Walk-Forward Backtester
============================================
Simulates historical equity research execution across rolling time windows
using strict temporal isolation (TemporalGuard) to eliminate look-ahead bias.

Walk-Forward Mechanics:
  1. For each historical evaluation date (t_0):
     - Instantiate EquiMindEngine with temporal cutoff (`as_of_date=t_0`)
     - Execute complete research pipeline using only data published <= t_0
     - Record recommendation, conviction, and price @ t_0
  2. For evaluation window (e.g. 30 days):
     - Fetch actual market outcome at t_0 + 30 days via YFinanceAdapter
     - Evaluate prediction accuracy (direction, return, Brier score)
  3. Generate institutional backtest performance report
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from equimind.orchestrator.engine import EquiMindEngine
from equimind.evaluation.accuracy_tracker import AccuracyTracker, PredictionRecord, AccuracyMetrics

logger = logging.getLogger(__name__)


class BacktestSummary(BaseModel):
    """Structured report for historical walk-forward backtest."""
    ticker: str
    start_date: str
    end_date: str
    total_evaluations: int
    hit_rate_pct: float
    brier_score: float
    sharpe_ratio: float
    total_return_pct: float
    buy_hold_return_pct: float
    outperformance_pct: float
    predictions: List[PredictionRecord]


class WalkForwardBacktester:
    """Historical simulation engine executing EquiMind under temporal isolation."""

    def __init__(self, engine: Optional[EquiMindEngine] = None):
        self.engine = engine or EquiMindEngine()
        self.tracker = AccuracyTracker()

    def run_backtest(
        self,
        ticker: str,
        start_date_str: str,
        end_date_str: str,
        step_days: int = 30,
        eval_window_days: int = 30,
        query: str = "Should I invest in this stock for long-term growth?",
    ) -> BacktestSummary:
        """
        Execute walk-forward backtest for a ticker across historical dates.
        
        Args:
            ticker: Asset ticker symbol
            start_date_str: Start date ISO format ("2023-01-01")
            end_date_str: End date ISO format ("2024-01-01")
            step_days: Re-evaluation frequency in days (e.g. 30 = monthly)
            eval_window_days: Horizon for outcome checking
            query: Research query
        """
        ticker_upper = ticker.upper()
        dt_start = datetime.fromisoformat(start_date_str).replace(tzinfo=timezone.utc)
        dt_end = datetime.fromisoformat(end_date_str).replace(tzinfo=timezone.utc)

        current_dt = dt_start
        backtest_predictions: List[PredictionRecord] = []

        logger.info(f"Starting Walk-Forward Backtest for {ticker_upper} from {start_date_str} to {end_date_str} (step: {step_days}d)")

        while current_dt <= dt_end:
            as_of_str = current_dt.strftime("%Y-%m-%d")
            
            try:
                # 1. Run research pipeline under temporal guard cutoff
                res = self.engine.analyze_equity(
                    ticker=ticker_upper,
                    query=query,
                    as_of_date_str=as_of_str,
                )

                rec = res.get("recommendation", {})
                rating = rec.get("rating", "HOLD")
                conviction = rec.get("conviction_score", 0.5)

                # Get price at t_0 from market data node or quant summary
                price = 100.0
                if "compressed_evidence_count" in res:
                    # Look up price in evidence
                    price = rec.get("quant_summary", {}).get("last_price", 100.0)

                # 2. Record prediction
                pred = self.tracker.record_recommendation(
                    ticker=ticker_upper,
                    rating=rating,
                    conviction_score=conviction,
                    current_price=price,
                    evaluation_window_days=eval_window_days,
                    as_of_date=current_dt,
                )
                backtest_predictions.append(pred)

            except Exception as e:
                logger.warning(f"Backtest step failed for {ticker_upper} on {as_of_str}: {e}")

            current_dt += timedelta(days=step_days)

        # 3. Evaluate historical outcomes against real market data
        self.tracker.evaluate_outcomes(force_evaluate=True)
        metrics: AccuracyMetrics = self.tracker.compute_metrics()

        # Compute strategy vs buy & hold performance
        strategy_returns = [p.realized_return_pct or 0.0 for p in backtest_predictions if p.direction_correct is not None]
        total_strat_return = sum(strategy_returns)
        
        buy_hold_return = 0.0
        if backtest_predictions and backtest_predictions[-1].realized_price and backtest_predictions[0].price_at_recommendation:
            p0 = backtest_predictions[0].price_at_recommendation
            pN = backtest_predictions[-1].realized_price
            buy_hold_return = ((pN - p0) / p0) * 100.0

        return BacktestSummary(
            ticker=ticker_upper,
            start_date=start_date_str,
            end_date=end_date_str,
            total_evaluations=len(backtest_predictions),
            hit_rate_pct=metrics.directional_hit_rate_pct,
            brier_score=metrics.brier_score,
            sharpe_ratio=metrics.prediction_sharpe_ratio,
            total_return_pct=round(total_strat_return, 2),
            buy_hold_return_pct=round(buy_hold_return, 2),
            outperformance_pct=round(total_strat_return - buy_hold_return, 2),
            predictions=backtest_predictions,
        )
