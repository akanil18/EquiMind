import unittest
from equimind.quantitative.portfolio_optimizer import (
    PortfolioOptimizer,
    PortfolioOptimizationResult,
    OptimizationMethod,
)


class TestPortfolioOptimizer(unittest.TestCase):

    def setUp(self):
        self.returns_matrix = {
            "NVDA": [0.02, 0.015, -0.01, 0.03, 0.025, 0.01, -0.005, 0.02],
            "AAPL": [0.005, 0.01, 0.002, 0.008, 0.005, -0.002, 0.006, 0.004],
            "MSFT": [0.01, 0.012, -0.004, 0.015, 0.008, 0.002, 0.001, 0.009],
        }

    def test_mean_variance_optimization(self):
        result = PortfolioOptimizer.optimize_portfolio(
            asset_returns_matrix=self.returns_matrix,
            method=OptimizationMethod.MEAN_VARIANCE,
        )

        self.assertIsInstance(result, PortfolioOptimizationResult)
        self.assertEqual(result.method, OptimizationMethod.MEAN_VARIANCE)
        self.assertEqual(len(result.weights), 3)
        weight_sum = sum(result.weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=2)
        self.assertGreater(result.expected_portfolio_return, 0.0)

    def test_risk_parity_optimization(self):
        result = PortfolioOptimizer.optimize_portfolio(
            asset_returns_matrix=self.returns_matrix,
            method=OptimizationMethod.RISK_PARITY,
        )

        self.assertEqual(result.method, OptimizationMethod.RISK_PARITY)
        self.assertGreater(result.diversification_score, 0.5)

    def test_black_litterman_optimization(self):
        views = {"NVDA": 0.35, "AAPL": 0.10}
        result = PortfolioOptimizer.optimize_portfolio(
            asset_returns_matrix=self.returns_matrix,
            method=OptimizationMethod.BLACK_LITTERMAN,
            investor_views=views,
        )

        self.assertEqual(result.method, OptimizationMethod.BLACK_LITTERMAN)
        self.assertIn("NVDA", result.weights)

    def test_kelly_criterion_fraction(self):
        kelly_frac = PortfolioOptimizer.calculate_kelly_fraction(win_probability=0.60, win_loss_ratio=1.5)
        self.assertGreater(kelly_frac, 0.0)
        self.assertLessEqual(kelly_frac, 0.5)


if __name__ == "__main__":
    unittest.main()
