import unittest
from equimind.quantitative.time_series import (
    TimeSeriesResearchEngine,
    TimeSeriesForecastResult,
    MarketRegime,
)


class TestTimeSeriesResearchEngine(unittest.TestCase):

    def setUp(self):
        self.bull_prices = [100.0, 102.0, 104.5, 103.8, 106.2, 108.0, 110.5, 112.0, 115.0, 118.0]
        self.volatile_prices = [100.0, 115.0, 88.0, 120.0, 85.0, 125.0, 80.0, 130.0, 75.0, 135.0]

    def test_kalman_filter_noise_reduction(self):
        filtered = TimeSeriesResearchEngine.apply_kalman_filter(self.bull_prices)
        self.assertEqual(len(filtered), len(self.bull_prices))
        self.assertIsInstance(filtered[-1], float)

    def test_market_regime_classification(self):
        bull_regime = TimeSeriesResearchEngine.detect_market_regime(self.bull_prices)
        self.assertEqual(bull_regime, MarketRegime.BULL_TREND)

        volatile_regime = TimeSeriesResearchEngine.detect_market_regime(self.volatile_prices)
        self.assertEqual(volatile_regime, MarketRegime.HIGH_VOLATILITY_SIDEWAYS)

    def test_garch_volatility_estimation(self):
        returns = [0.01, -0.012, 0.015, -0.008, 0.02, -0.01]
        garch_vol = TimeSeriesResearchEngine.estimate_garch_volatility(returns)
        self.assertGreater(garch_vol, 0.0)

    def test_ensemble_forecast_generation(self):
        result = TimeSeriesResearchEngine.generate_ensemble_forecast(self.bull_prices, ticker="NVDA", days=30)
        self.assertIsInstance(result, TimeSeriesForecastResult)
        self.assertEqual(result.ticker, "NVDA")
        self.assertGreater(result.projected_mean_price, 0.0)
        self.assertTrue(result.confidence_interval_95[0] < result.confidence_interval_95[1])
        self.assertIn("KalmanFilter", result.model_weights)


if __name__ == "__main__":
    unittest.main()
