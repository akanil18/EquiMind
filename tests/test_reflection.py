import unittest
from datetime import datetime, timezone

from equimind.memory.schema import ResearchReportRecord
from equimind.reflection.schema import OutcomeEvaluation, ReflectionSummary
from equimind.reflection.reflection_agent import SelfReflectionAgent


class TestSelfReflection(unittest.TestCase):

    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.report_buy = ResearchReportRecord(
            ticker="NVDA",
            timestamp=self.now,
            user_query="Buy NVDA?",
            rating="BUY",
            conviction_score=0.85,
            summary="Bullish GPU growth",
            evidence_count=5,
        )

    def test_successful_recommendation_evaluation(self):
        eval_res = SelfReflectionAgent.evaluate_past_report(
            report=self.report_buy,
            actual_current_price=135.0,
            initial_price=120.0,
        )

        self.assertTrue(eval_res.was_successful)
        self.assertEqual(eval_res.price_change_pct, 12.5)
        self.assertIsNone(eval_res.bias_detected)

    def test_unsuccessful_recommendation_and_bias_detection(self):
        eval_res = SelfReflectionAgent.evaluate_past_report(
            report=self.report_buy,
            actual_current_price=105.0,
            initial_price=120.0,
        )

        self.assertFalse(eval_res.was_successful)
        self.assertEqual(eval_res.price_change_pct, -12.5)
        self.assertIn("Over-optimism", eval_res.bias_detected)

    def test_reflection_summary_calibration(self):
        e1 = SelfReflectionAgent.evaluate_past_report(self.report_buy, 135.0, 120.0)
        e2 = SelfReflectionAgent.evaluate_past_report(self.report_buy, 105.0, 120.0)

        summary = SelfReflectionAgent.generate_reflection_summary([e1, e2])
        self.assertEqual(summary.total_evaluated, 2)
        self.assertEqual(summary.successful_count, 1)
        self.assertEqual(summary.accuracy_rate_pct, 50.0)
        self.assertEqual(summary.recommended_conviction_calibration_factor, 0.85)


if __name__ == "__main__":
    unittest.main()
