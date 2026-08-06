import unittest
from datetime import datetime, timezone

from equimind.adapters import YFinanceAdapter
from equimind.adapters.sec_edgar_adapter import SECEdgarAdapter, ticker_to_cik
from equimind.adapters.news_adapter import NewsRSSAdapter
from equimind.teams.market_data_team import MarketDataTeam
from equimind.teams.fundamental_team import FundamentalTeam
from equimind.teams.web_intelligence_team import WebIntelligenceTeam
from equimind.native import is_native_available, fast_technical, fast_montecarlo, fast_dedup


class TestRealAdaptersAndNative(unittest.TestCase):

    def test_cpp_native_module(self):
        """Test that C++ native module compiles and executes correctly."""
        self.assertTrue(is_native_available(), "C++ native module should be loaded")
        
        # Test RSI calculation
        prices = [100 + i * 0.5 for i in range(50)]
        rsi = fast_technical.rsi(prices, 14)
        self.assertIsInstance(rsi, float)
        self.assertGreaterEqual(rsi, 0.0)
        self.assertLessEqual(rsi, 100.0)

        # Test Monte Carlo simulation
        mc = fast_montecarlo.simulate(100.0, 0.05, 0.20, days=252, num_paths=1000)
        self.assertIn("expected_price", mc)
        self.assertIn("p5", mc)
        self.assertIn("p95", mc)
        self.assertGreater(mc["expected_price"], 0)

        # Test deduplication
        texts = ["apple stock rising", "apple stock rising!", "banana price falling"]
        unique_idx = fast_dedup.deduplicate(texts, threshold=0.7)
        self.assertEqual(len(unique_idx), 2)

    def test_yfinance_adapter(self):
        """Test fetching real market data via yfinance adapter."""
        df = YFinanceAdapter.get_price_history("AAPL", period="1mo")
        self.assertFalse(df.empty, "yfinance should return data for AAPL")
        self.assertIn("Close", df.columns)

        info = YFinanceAdapter.get_company_info("AAPL")
        self.assertEqual(info.get("symbol"), "AAPL")
        self.assertIn("Apple", info.get("name", ""))

    def test_sec_edgar_adapter(self):
        """Test SEC EDGAR CIK lookup."""
        cik = ticker_to_cik("AAPL")
        self.assertEqual(cik, 320193)

        cik_nvda = ticker_to_cik("NVDA")
        self.assertEqual(cik_nvda, 1045810)

    def test_news_rss_adapter(self):
        """Test news RSS adapter."""
        articles = NewsRSSAdapter.fetch_news(ticker="AAPL", max_articles=5)
        self.assertIsInstance(articles, list)

    def test_market_data_team_real_data(self):
        """Test MarketDataTeam using real data + C++ technical engine."""
        team = MarketDataTeam()
        nodes = team.research("AAPL", "Analyze AAPL market trend")
        self.assertEqual(len(nodes), 1)
        self.assertIn("AAPL", nodes[0].content)
        self.assertIn("rsi", nodes[0].tags)

    def test_fundamental_team_real_data(self):
        """Test FundamentalTeam using real yfinance & SEC data."""
        team = FundamentalTeam()
        nodes = team.research("AAPL", "Analyze AAPL fundamentals")
        self.assertEqual(len(nodes), 1)
        self.assertIn("valuation", nodes[0].metadata)

    def test_web_intelligence_team_real_data(self):
        """Test WebIntelligenceTeam using SEC EDGAR and news RSS."""
        team = WebIntelligenceTeam()
        nodes = team.research("AAPL", "Check SEC filings and news for AAPL")
        self.assertGreaterEqual(len(nodes), 2)


if __name__ == "__main__":
    unittest.main()
