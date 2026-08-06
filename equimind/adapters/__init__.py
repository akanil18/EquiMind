"""
EquiMind yFinance Adapter — Real Market Data Integration
=========================================================
Wraps yfinance with caching, rate limiting, error handling,
and graceful fallback to synthetic data if API fails.

Data available:
  - OHLCV price history (1m to max)
  - Financial statements (annual + quarterly)
  - Company info (sector, industry, market cap, PE, EPS)
  - Analyst recommendations & price targets
  - Options chain
"""

import os
import json
import hashlib
import logging
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed — market data adapter will use synthetic fallback")


# ═══════════════════════════════════════════════════════════════
# CACHE LAYER
# ═══════════════════════════════════════════════════════════════

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".equimind_cache")

# TTLs in hours (from DATA_INTEGRATION_REFERENCE.md)
TTL_HISTORICAL = 24   # Daily OHLCV doesn't change after close
TTL_QUOTE = 0.25       # 15 minutes for delayed quotes
TTL_FINANCIALS = 168   # 7 days for financial statements
TTL_INFO = 168         # 7 days for company profile


def _cache_key(prefix: str, ticker: str, *args) -> str:
    raw = f"{prefix}:{ticker}:{':'.join(str(a) for a in args)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str, ttl_hours: float) -> Optional[Any]:
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        cached_at = datetime.fromisoformat(data["_cached_at"])
        if datetime.now() - cached_at < timedelta(hours=ttl_hours):
            return data["payload"]
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _cache_set(key: str, payload: Any):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w") as f:
            json.dump({"_cached_at": datetime.now().isoformat(), "payload": payload}, f, default=str)
    except (TypeError, OSError) as e:
        logger.debug(f"Cache write failed: {e}")


# ═══════════════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════════════

class _RateLimiter:
    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: list = []
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0]) + 0.1
                logger.debug(f"Rate limit hit, sleeping {sleep_time:.1f}s")
                time.sleep(max(0, sleep_time))
            self.calls.append(time.time())


_yf_limiter = _RateLimiter(max_calls=50, period_seconds=60.0)  # Conservative


# ═══════════════════════════════════════════════════════════════
# YFINANCE ADAPTER
# ═══════════════════════════════════════════════════════════════

class YFinanceAdapter:
    """Real market data adapter wrapping yfinance with caching and fallback."""

    @classmethod
    def get_price_history(cls, ticker: str, period: str = "2y",
                          interval: str = "1d") -> pd.DataFrame:
        """
        Fetch OHLCV price history.
        
        Args:
            ticker: Stock ticker symbol (e.g., "NVDA")
            period: History period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        cache_key = _cache_key("hist", ticker, period, interval)
        cached = _cache_get(cache_key, TTL_HISTORICAL)
        
        if cached is not None:
            logger.debug(f"Cache hit for {ticker} price history")
            df = pd.DataFrame(cached)
            df.index = pd.to_datetime(df.index)
            return df

        if not HAS_YFINANCE:
            return cls._synthetic_fallback(ticker, 500)

        try:
            _yf_limiter.wait()
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker} — using synthetic fallback")
                return cls._synthetic_fallback(ticker, 500)
            
            # Cache the result
            cache_data = df.to_dict()
            _cache_set(cache_key, cache_data)
            
            logger.info(f"✓ Fetched {len(df)} bars for {ticker} ({period}/{interval})")
            return df
            
        except Exception as e:
            logger.warning(f"yfinance error for {ticker}: {e} — using synthetic fallback")
            return cls._synthetic_fallback(ticker, 500)

    @classmethod
    def get_company_info(cls, ticker: str) -> Dict[str, Any]:
        """
        Fetch company profile: sector, industry, market cap, PE, EPS, etc.
        """
        cache_key = _cache_key("info", ticker)
        cached = _cache_get(cache_key, TTL_INFO)
        if cached is not None:
            return cached

        if not HAS_YFINANCE:
            return cls._synthetic_info(ticker)

        try:
            _yf_limiter.wait()
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or "symbol" not in info:
                return cls._synthetic_info(ticker)

            # Extract key fields
            result = {
                "symbol": info.get("symbol", ticker),
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "forward_eps": info.get("forwardEps"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "Unknown"),
                "description": info.get("longBusinessSummary", ""),
            }
            
            _cache_set(cache_key, result)
            logger.info(f"✓ Fetched company info for {ticker}: {result.get('name')}")
            return result
            
        except Exception as e:
            logger.warning(f"yfinance info error for {ticker}: {e}")
            return cls._synthetic_info(ticker)

    @classmethod
    def get_financials(cls, ticker: str) -> Dict[str, Any]:
        """
        Fetch financial statements: income statement, balance sheet, cash flow.
        Returns both annual and quarterly data.
        """
        cache_key = _cache_key("fin", ticker)
        cached = _cache_get(cache_key, TTL_FINANCIALS)
        if cached is not None:
            return cached

        if not HAS_YFINANCE:
            return cls._synthetic_financials(ticker)

        try:
            _yf_limiter.wait()
            stock = yf.Ticker(ticker)
            
            result = {
                "income_stmt": {},
                "balance_sheet": {},
                "cashflow": {},
                "quarterly_income": {},
            }
            
            # Annual income statement
            income = stock.income_stmt
            if income is not None and not income.empty:
                result["income_stmt"] = cls._df_to_serializable(income)
            
            # Annual balance sheet
            bs = stock.balance_sheet
            if bs is not None and not bs.empty:
                result["balance_sheet"] = cls._df_to_serializable(bs)
            
            # Annual cash flow
            cf = stock.cashflow
            if cf is not None and not cf.empty:
                result["cashflow"] = cls._df_to_serializable(cf)
            
            # Quarterly income
            qi = stock.quarterly_income_stmt
            if qi is not None and not qi.empty:
                result["quarterly_income"] = cls._df_to_serializable(qi)
            
            _cache_set(cache_key, result)
            logger.info(f"✓ Fetched financial statements for {ticker}")
            return result
            
        except Exception as e:
            logger.warning(f"yfinance financials error for {ticker}: {e}")
            return cls._synthetic_financials(ticker)

    @classmethod
    def get_analyst_data(cls, ticker: str) -> Dict[str, Any]:
        """Fetch analyst recommendations and price targets."""
        if not HAS_YFINANCE:
            return {"recommendations": [], "target_price": None}

        try:
            _yf_limiter.wait()
            stock = yf.Ticker(ticker)
            
            result = {"recommendations": [], "target_price": None}
            
            recs = stock.recommendations
            if recs is not None and not recs.empty:
                # Get last 10 recommendations
                recent = recs.tail(10)
                result["recommendations"] = recent.to_dict(orient="records")
            
            info = stock.info
            result["target_price"] = info.get("targetMeanPrice")
            result["target_high"] = info.get("targetHighPrice")
            result["target_low"] = info.get("targetLowPrice")
            result["recommendation_key"] = info.get("recommendationKey", "none")
            
            return result
            
        except Exception as e:
            logger.debug(f"Analyst data error for {ticker}: {e}")
            return {"recommendations": [], "target_price": None}

    @classmethod
    def validate_ticker(cls, ticker: str) -> bool:
        """Check if a ticker symbol is valid."""
        if not HAS_YFINANCE:
            return True  # Can't validate without yfinance
        try:
            _yf_limiter.wait()
            stock = yf.Ticker(ticker)
            info = stock.info
            return info is not None and "symbol" in info
        except Exception:
            return False

    # ── Utility Methods ────────────────────────────────────────

    @staticmethod
    def _df_to_serializable(df: pd.DataFrame) -> Dict:
        """Convert DataFrame to JSON-serializable dict."""
        result = {}
        for col in df.columns:
            col_key = str(col.date()) if hasattr(col, 'date') else str(col)
            result[col_key] = {}
            for idx in df.index:
                val = df.loc[idx, col]
                if pd.notna(val):
                    result[col_key][str(idx)] = float(val) if isinstance(val, (int, float, np.number)) else str(val)
        return result

    @staticmethod
    def _synthetic_fallback(ticker: str, num_days: int = 500) -> pd.DataFrame:
        """Generate synthetic price data when yfinance is unavailable."""
        np.random.seed(abs(hash(ticker)) % 10000)
        dates = pd.date_range(end=datetime.now(), periods=num_days, freq="D")
        base = {"NVDA": 800, "AAPL": 220, "TSLA": 250, "MSFT": 450, "AMZN": 200}.get(ticker, 150)
        returns = np.random.normal(0.0004, 0.018, num_days)
        prices = base * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            "Open": prices * 0.995,
            "High": prices * 1.015,
            "Low": prices * 0.985,
            "Close": prices,
            "Volume": np.random.randint(5_000_000, 50_000_000, num_days),
        }, index=dates)

    @staticmethod
    def _synthetic_info(ticker: str) -> Dict[str, Any]:
        return {
            "symbol": ticker, "name": f"{ticker} Inc.",
            "sector": "Technology", "industry": "Software",
            "market_cap": 1_000_000_000_000, "pe_ratio": 30.0,
            "eps": 5.0, "beta": 1.2, "price": 150.0,
            "currency": "USD", "exchange": "NASDAQ",
            "description": f"Synthetic profile for {ticker}",
        }

    @staticmethod
    def _synthetic_financials(ticker: str) -> Dict[str, Any]:
        return {
            "income_stmt": {}, "balance_sheet": {},
            "cashflow": {}, "quarterly_income": {},
        }
