import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from equimind.memory.schema import ResearchReportRecord, EntityKnowledgeEntry
from equimind.memory.hierarchical_store import HierarchicalMemoryStore
from equimind.reflection.schema import OutcomeEvaluation, ReflectionSummary
from equimind.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class SelfReflectionAgent:
    """Agent performing post-hoc analysis on past recommendations to detect bias and calibrate future conviction scores."""

    @classmethod
    def evaluate_past_report(
        cls,
        report: ResearchReportRecord,
        actual_current_price: float,
        initial_price: float,
        provider: Optional[LLMProvider] = None,
    ) -> OutcomeEvaluation:
        """Evaluates a single past report against actual price outcome."""
        if initial_price <= 0:
            price_change = 0.0
        else:
            price_change = round(((actual_current_price - initial_price) / initial_price) * 100.0, 2)

        is_buy_rating = report.rating in ("STRONG_BUY", "BUY")
        is_sell_rating = report.rating in ("STRONG_SELL", "SELL")

        was_successful = False
        bias = None

        if is_buy_rating:
            if price_change > 2.0:
                was_successful = True
            else:
                was_successful = False
                bias = "Over-optimism / Excess Bullish Bias"
        elif is_sell_rating:
            if price_change < -2.0:
                was_successful = True
            else:
                was_successful = False
                bias = "Over-pessimism / Excess Bearish Bias"
        else:
            # Hold rating success if price remained within +/- 5% range
            was_successful = abs(price_change) <= 5.0
            if not was_successful:
                bias = "Underestimated Volatility Breakout"

        notes = (
            f"Evaluated report for {report.ticker} generated on {report.timestamp.strftime('%Y-%m-%d')}. "
            f"Initial price: ${initial_price:.2f}, Actual price: ${actual_current_price:.2f} ({price_change:+.2f}%). "
            f"Rating '{report.rating}' was {'successful' if was_successful else 'unsuccessful'}. "
            f"{f'Detected Bias: {bias}.' if bias else 'No bias detected.'}"
        )

        return OutcomeEvaluation(
            report_id=report.id,
            ticker=report.ticker,
            recommendation_date=report.timestamp,
            recommended_rating=report.rating,
            initial_price=initial_price,
            actual_price=actual_current_price,
            price_change_pct=price_change,
            was_successful=was_successful,
            bias_detected=bias,
            reflection_notes=notes,
        )

    @classmethod
    def generate_reflection_summary(
        cls,
        evaluations: List[OutcomeEvaluation],
    ) -> ReflectionSummary:
        """Aggregates evaluations and computes calibration factor for future recommendations."""
        if not evaluations:
            return ReflectionSummary(
                total_evaluated=0,
                successful_count=0,
                accuracy_rate_pct=100.0,
                detected_biases=[],
                recommended_conviction_calibration_factor=1.0,
            )

        successful = sum(1 for e in evaluations if e.was_successful)
        total = len(evaluations)
        accuracy_pct = round((successful / total) * 100.0, 2)

        biases = list(dict.fromkeys([e.bias_detected for e in evaluations if e.bias_detected]))

        # Compute calibration factor: if accuracy < 70%, dampen conviction score
        if accuracy_pct >= 80.0:
            calibration_factor = 1.05
        elif accuracy_pct >= 60.0:
            calibration_factor = 1.00
        else:
            calibration_factor = 0.85

        return ReflectionSummary(
            total_evaluated=total,
            successful_count=successful,
            accuracy_rate_pct=accuracy_pct,
            detected_biases=biases,
            recommended_conviction_calibration_factor=calibration_factor,
        )
