import unittest
from equimind.quantitative.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloSimulationResult,
)


class TestMonteCarloSimulator(unittest.TestCase):

    def setUp(self):
        self.prices = [100.0, 102.0, 104.0, 103.5, 106.0, 108.0, 110.0]

    def test_monte_carlo_simulation_execution(self):
        result = MonteCarloSimulator.run_simulation(
            prices=self.prices,
            ticker="NVDA",
            num_simulations=1000,
            horizon_days=30,
        )

        self.assertIsInstance(result, MonteCarloSimulationResult)
        self.assertEqual(result.ticker, "NVDA")
        self.assertEqual(result.num_simulations, 1000)
        self.assertEqual(result.horizon_days, 30)
        self.assertEqual(result.initial_price, 110.0)
        self.assertTrue(result.percentile_5 < result.median_final_price < result.percentile_95)
        self.assertTrue(0.0 <= result.probability_of_profit <= 1.0)
        self.assertTrue(result.simulated_max_drawdown_avg <= 0.0)


if __name__ == "__main__":
    unittest.main()
