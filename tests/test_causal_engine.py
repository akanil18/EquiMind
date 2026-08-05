import unittest
from equimind.quantitative.causal_engine import (
    CausalReasoningEngine,
    CausalAnalysisResult,
)


class TestCausalReasoningEngine(unittest.TestCase):

    def setUp(self):
        # Direct causal setup: Y = 2*X
        self.x_direct = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.y_direct = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

        # Spurious setup: X = Z + noise_X, Y = Z + noise_Y, where noise_X and noise_Y are orthogonal (zero correlation) given Z
        self.z_confounder = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        self.x_spurious   = [11.0, 21.0, 29.0, 39.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        self.y_spurious   = [11.0, 19.0, 31.0, 39.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    def test_direct_causal_effect_estimation(self):
        raw_corr, causal_eff, is_spurious = CausalReasoningEngine.estimate_causal_effect(
            self.x_direct, self.y_direct
        )

        self.assertEqual(raw_corr, 1.0)
        self.assertEqual(causal_eff, 1.0)
        self.assertFalse(is_spurious)

    def test_spurious_correlation_detection(self):
        result = CausalReasoningEngine.analyze_causal_mechanism(
            cause_name="IceCreamSales",
            target_name="DrowningIncidents",
            x_values=self.x_spurious,
            y_values=self.y_spurious,
            confounder_name="SummerTemperature",
            z_values=self.z_confounder,
        )

        self.assertIsInstance(result, CausalAnalysisResult)
        self.assertTrue(result.is_spurious_correlation)
        self.assertIn("SPURIOUS", result.explanation)


if __name__ == "__main__":
    unittest.main()
