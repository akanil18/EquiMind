import unittest
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone

from equimind.evaluation import AccuracyTracker, WalkForwardBacktester, PredictionRecord, AccuracyMetrics


class TestAccuracyAndBacktest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = AccuracyTracker(storage_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_record_and_evaluate_prediction(self):
        """Test recording a prediction and evaluating its outcome."""
        pred = self.tracker.record_recommendation(
            ticker="AAPL",
            rating="BUY",
            conviction_score=0.85,
            current_price=150.0,
            evaluation_window_days=30,
            as_of_date=datetime.now(timezone.utc) - timedelta(days=40)
        )
        self.assertIsNotNone(pred.prediction_id)
        self.assertEqual(pred.ticker, "AAPL")
        self.assertEqual(pred.rating, "BUY")

        # Evaluate outcome against real market price
        eval_count = self.tracker.evaluate_outcomes(force_evaluate=True)
        self.assertEqual(eval_count, 1)

        metrics = self.tracker.compute_metrics()
        self.assertEqual(metrics.total_predictions, 1)
        self.assertEqual(metrics.evaluated_predictions, 1)
        self.assertGreaterEqual(metrics.brier_score, 0.0)

    def test_accuracy_metrics_calculation(self):
        """Test accuracy metrics calculations with winning and losing predictions."""
        now = datetime.now(timezone.utc)
        
        # Add a winning prediction manually
        p1 = PredictionRecord(
            prediction_id="AAPL_WIN",
            ticker="AAPL",
            created_at=now - timedelta(days=40),
            rating="BUY",
            conviction_score=0.90,
            price_at_recommendation=100.0,
            realized_price=120.0,
            realized_return_pct=20.0,
            direction_correct=True,
            brier_score_component=0.01,
            evaluated_at=now,
        )

        # Add a losing prediction manually
        p2 = PredictionRecord(
            prediction_id="TSLA_LOSS",
            ticker="TSLA",
            created_at=now - timedelta(days=40),
            rating="BUY",
            conviction_score=0.80,
            price_at_recommendation=200.0,
            realized_price=180.0,
            realized_return_pct=-10.0,
            direction_correct=False,
            brier_score_component=0.64,
            evaluated_at=now,
        )

        self.tracker.predictions.extend([p1, p2])
        metrics = self.tracker.compute_metrics()

        self.assertEqual(metrics.evaluated_predictions, 2)
        self.assertEqual(metrics.directional_hit_rate_pct, 50.0)
        self.assertEqual(metrics.mean_conviction_winning, 0.90)
        self.assertEqual(metrics.mean_conviction_losing, 0.80)

    def test_walk_forward_backtester_initialization(self):
        """Test walk-forward backtester structure."""
        backtester = WalkForwardBacktester()
        self.assertIsNotNone(backtester.engine)
        self.assertIsNotNone(backtester.tracker)


if __name__ == "__main__":
    unittest.main()
