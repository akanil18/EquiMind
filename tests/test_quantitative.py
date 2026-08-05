import unittest
import numpy as np
import pandas as pd

from equimind.quantitative.technical import TechnicalEngine
from equimind.quantitative.fundamental import FundamentalEngine
from equimind.quantitative.risk import RiskEngine


class TestQuantitativeEngine(unittest.TestCase):

    def setUp(self):
        # Generate 100 days of synthetic price data
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        base_price = 100.0
        returns = np.random.normal(loc=0.001, scale=0.015, size=100)
        price_series = base_price * np.exp(np.cumsum(returns))
        
        self.df = pd.DataFrame({
            "date": dates,
            "open": price_series * 0.99,
            "high": price_series * 1.02,
            "low": price_series * 0.98,
            "close": price_series,
            "volume": np.random.randint(1000000, 5000000, size=100),
        })
        self.prices = self.df["close"]
        self.returns = self.prices.pct_change().dropna()

    def test_technical_engine_metrics(self):
        rsi = TechnicalEngine.calculate_rsi(self.prices)
        self.assertTrue(0.0 <= rsi <= 100.0)

        macd = TechnicalEngine.calculate_macd(self.prices)
        self.assertIn("macd", macd)
        self.assertIn("signal", macd)
        self.assertIn("histogram", macd)

        bb = TechnicalEngine.calculate_bollinger_bands(self.prices)
        self.assertTrue(bb["upper"] >= bb["middle"] >= bb["lower"])

        ma = TechnicalEngine.calculate_moving_averages(self.prices)
        self.assertIn("sma_20", ma)
        self.assertIn("ema_50", ma)

        sr = TechnicalEngine.calculate_support_resistance(self.df["high"], self.df["low"], self.df["close"])
        self.assertTrue(len(sr["support"]) == 2)
        self.assertTrue(len(sr["resistance"]) == 2)

        summary = TechnicalEngine.analyze_dataframe(self.df)
        self.assertIn("rsi_14", summary)
        self.assertIn("bollinger_bands", summary)

    def test_fundamental_engine_ratios(self):
        val = FundamentalEngine.calculate_valuation_ratios(
            market_cap=3_000_000_000_000,
            price=120.0,
            eps=4.0,
            book_value_per_share=20.0,
            free_cash_flow=60_000_000_000,
            earnings_growth_rate=20.0,
        )
        self.assertEqual(val["pe_ratio"], 30.0)
        self.assertEqual(val["pb_ratio"], 6.0)
        self.assertEqual(val["peg_ratio"], 1.5)

        prof = FundamentalEngine.calculate_profitability_metrics(
            net_income=30_000_000,
            revenue=100_000_000,
            total_assets=200_000_000,
            shareholder_equity=150_000_000,
            operating_income=35_000_000,
        )
        self.assertEqual(prof["roe_pct"], 20.0)
        self.assertEqual(prof["operating_margin_pct"], 35.0)

        # Piotroski F-score check
        piotroski = FundamentalEngine.calculate_piotroski_f_score({
            "net_income": 100,
            "roa": 0.1,
            "operating_cash_flow": 120,
            "long_term_debt_current": 50,
            "long_term_debt_prior": 60,
            "current_ratio_current": 1.5,
            "current_ratio_prior": 1.2,
            "shares_outstanding_current": 1000,
            "shares_outstanding_prior": 1000,
            "gross_margin_current": 0.4,
            "gross_margin_prior": 0.35,
            "asset_turnover_current": 0.8,
            "asset_turnover_prior": 0.7,
        })
        self.assertEqual(piotroski["piotroski_f_score"], 9)
        self.assertEqual(piotroski["rating"], "Strong Financial Health")

        # Altman Z-score check
        z_res = FundamentalEngine.calculate_altman_z_score(
            working_capital=50,
            retained_earnings=100,
            ebit=80,
            market_cap=500,
            revenue=600,
            total_assets=400,
            total_liabilities=150,
        )
        self.assertGreater(z_res["z_score"], 2.99)
        self.assertIn("Safe Zone", z_res["zone"])

    def test_risk_engine_metrics(self):
        risk = RiskEngine.calculate_risk_metrics(self.returns)
        self.assertIn("annualized_volatility_pct", risk)
        self.assertIn("sharpe_ratio", risk)
        self.assertIn("max_drawdown_pct", risk)
        self.assertIn("var_95_daily_pct", risk)

        dist = RiskEngine.calculate_expected_return_distribution(self.returns, horizon_days=30)
        self.assertEqual(dist["horizon_days"], 30)
        self.assertTrue(dist["ci_95_upper_pct"] >= dist["ci_95_lower_pct"])


if __name__ == "__main__":
    unittest.main()
