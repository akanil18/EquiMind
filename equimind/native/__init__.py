"""
EquiMind Native Performance Bridge
===================================
Transparently uses C++ optimized implementations when available,
falls back to pure Python otherwise. This ensures the system works
everywhere while providing institutional-grade performance when compiled.

Usage:
    from equimind.native import fast_technical, fast_montecarlo, fast_dedup

    # These automatically use C++ if compiled, Python otherwise:
    rsi = fast_technical.rsi(prices, period=14)
    mc  = fast_montecarlo.simulate(s0=150, mu=0.08, sigma=0.25)
    idx = fast_dedup.deduplicate(texts, threshold=0.8)
"""

import logging
import os

logger = logging.getLogger(__name__)

# ── Try to import C++ native module ────────────────────────────
_USE_NATIVE = False
try:
    import equimind_native
    _USE_NATIVE = True
    logger.info("✓ C++ native module loaded — using optimized implementations")
except ImportError:
    logger.info("C++ native module not found — using pure Python fallback")


def is_native_available() -> bool:
    """Check if C++ optimized module is compiled and available."""
    return _USE_NATIVE


# ══════════════════════════════════════════════════════════════
# FAST TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════

class _NativeTechnical:
    """C++ accelerated technical indicators."""

    @staticmethod
    def rsi(prices: list, period: int = 14) -> float:
        return equimind_native.technical.rsi(prices, period)

    @staticmethod
    def macd(prices: list, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        result = equimind_native.technical.macd(prices, fast, slow, signal)
        return {"macd": result.macd_line, "signal": result.signal_line, "histogram": result.histogram}

    @staticmethod
    def bollinger(prices: list, period: int = 20, num_std: float = 2.0) -> dict:
        result = equimind_native.technical.bollinger(prices, period, num_std)
        return {"upper": result.upper, "middle": result.middle, "lower": result.lower,
                "bandwidth": result.bandwidth, "percent_b": result.percent_b}

    @staticmethod
    def sma(prices: list, period: int) -> list:
        return equimind_native.technical.sma(prices, period)

    @staticmethod
    def ema(prices: list, period: int) -> list:
        return equimind_native.technical.ema(prices, period)

    @staticmethod
    def atr(high: list, low: list, close: list, period: int = 14) -> float:
        return equimind_native.technical.atr(high, low, close, period)

    @staticmethod
    def annualized_volatility(prices: list, trading_days: int = 252) -> float:
        return equimind_native.technical.annualized_volatility(prices, trading_days)

    @staticmethod
    def full_analysis(close: list, high: list, low: list) -> dict:
        r = equimind_native.technical.full_analysis(close, high, low)
        return {
            "rsi_14": r.rsi_14,
            "last_price": r.last_price,
            "macd": {"macd": r.macd.macd_line, "signal": r.macd.signal_line, "histogram": r.macd.histogram},
            "bollinger_bands": {"upper": r.bollinger.upper, "middle": r.bollinger.middle, "lower": r.bollinger.lower},
            "atr_14": r.atr_14,
            "sma_20": r.sma_20, "sma_50": r.sma_50, "sma_200": r.sma_200,
            "ema_12": r.ema_12, "ema_26": r.ema_26,
            "annualized_volatility": r.annualized_volatility,
            "support_resistance": {"support": r.support_levels, "resistance": r.resistance_levels},
        }


class _PythonTechnical:
    """Pure Python fallback — delegates to existing equimind.quantitative.technical module."""

    @staticmethod
    def rsi(prices: list, period: int = 14) -> float:
        from equimind.quantitative.technical import TechnicalEngine as PyTech
        return PyTech.compute_rsi(prices, period)

    @staticmethod
    def macd(prices: list, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        from equimind.quantitative.technical import TechnicalEngine as PyTech
        return PyTech.compute_macd(prices, fast, slow, signal)

    @staticmethod
    def bollinger(prices: list, period: int = 20, num_std: float = 2.0) -> dict:
        from equimind.quantitative.technical import TechnicalEngine as PyTech
        return PyTech.compute_bollinger_bands(prices, period, num_std)

    @staticmethod
    def sma(prices: list, period: int) -> list:
        from equimind.quantitative.technical import TechnicalEngine as PyTech
        return PyTech.compute_sma(prices, period)

    @staticmethod
    def ema(prices: list, period: int) -> list:
        from equimind.quantitative.technical import TechnicalEngine as PyTech
        return PyTech.compute_ema(prices, period)

    @staticmethod
    def atr(high: list, low: list, close: list, period: int = 14) -> float:
        from equimind.quantitative.technical import TechnicalEngine as PyTech
        return PyTech.compute_atr(high, low, close, period)

    @staticmethod
    def annualized_volatility(prices: list, trading_days: int = 252) -> float:
        from equimind.quantitative.technical import TechnicalEngine as PyTech
        return PyTech.compute_volatility(prices, trading_days)

    @staticmethod
    def full_analysis(close: list, high: list, low: list) -> dict:
        import pandas as pd
        from equimind.quantitative.technical import TechnicalEngine as PyTech
        df = pd.DataFrame({"close": close, "high": high, "low": low,
                           "open": close, "volume": [0]*len(close)})
        return PyTech.analyze_dataframe(df)


# ══════════════════════════════════════════════════════════════
# FAST MONTE CARLO
# ══════════════════════════════════════════════════════════════

class _NativeMonteCarlo:
    """C++ accelerated Monte Carlo simulation."""

    @staticmethod
    def simulate(s0: float, mu: float, sigma: float, days: int = 252,
                 num_paths: int = 10000, jump_intensity: float = 0.0,
                 jump_mean: float = 0.0, jump_vol: float = 0.0, seed: int = 42) -> dict:
        r = equimind_native.montecarlo.simulate(
            s0, mu, sigma, days, num_paths, jump_intensity, jump_mean, jump_vol, seed
        )
        return {
            "expected_price": r.expected_price,
            "p5": r.p5, "p25": r.p25, "median": r.median,
            "p75": r.p75, "p95": r.p95,
            "prob_above_current": r.prob_above_current,
            "final_prices": r.final_prices,
        }


class _PythonMonteCarlo:
    """Pure Python fallback for Monte Carlo."""

    @staticmethod
    def simulate(s0: float, mu: float, sigma: float, days: int = 252,
                 num_paths: int = 10000, jump_intensity: float = 0.0,
                 jump_mean: float = 0.0, jump_vol: float = 0.0, seed: int = 42) -> dict:
        from equimind.quantitative.monte_carlo import MonteCarloSimulator
        return MonteCarloSimulator.simulate_gbm(
            s0=s0, mu=mu, sigma=sigma, days=days, num_paths=num_paths, seed=seed
        )


# ══════════════════════════════════════════════════════════════
# FAST TEXT DEDUPLICATION
# ══════════════════════════════════════════════════════════════

class _NativeDedup:
    """C++ accelerated text deduplication."""

    @staticmethod
    def deduplicate(texts: list, threshold: float = 0.8, shingle_size: int = 5) -> list:
        return equimind_native.dedup.deduplicate(texts, threshold, shingle_size)

    @staticmethod
    def fnv1a_hash(text: str) -> int:
        return equimind_native.dedup.fnv1a_hash(text)


class _PythonDedup:
    """Pure Python fallback for text deduplication."""

    @staticmethod
    def deduplicate(texts: list, threshold: float = 0.8, shingle_size: int = 5) -> list:
        # Simple exact-match deduplication
        seen = set()
        unique_indices = []
        for i, text in enumerate(texts):
            normalized = text.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_indices.append(i)
        return unique_indices

    @staticmethod
    def fnv1a_hash(text: str) -> int:
        h = 14695981039346656037
        for c in text:
            h ^= ord(c)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h


# ══════════════════════════════════════════════════════════════
# PUBLIC API — Auto-selects C++ or Python
# ══════════════════════════════════════════════════════════════

fast_technical = _NativeTechnical if _USE_NATIVE else _PythonTechnical
fast_montecarlo = _NativeMonteCarlo if _USE_NATIVE else _PythonMonteCarlo
fast_dedup = _NativeDedup if _USE_NATIVE else _PythonDedup
