from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class TechnicalEngine:
    """Pure mathematical Technical Analysis engine."""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculates Relative Strength Index (RSI)."""
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        last_gain = gain.iloc[-1]
        last_loss = loss.iloc[-1]

        if last_loss == 0:
            return 100.0 if last_gain > 0 else 50.0

        rs = last_gain / last_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return round(float(rsi), 2)

    @staticmethod
    def calculate_macd(
        prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, float]:
        """Calculates Moving Average Convergence Divergence (MACD)."""
        if len(prices) < slow + signal:
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_series = ema_fast - ema_slow
        signal_series = macd_series.ewm(span=signal, adjust=False).mean()
        hist_series = macd_series - signal_series

        return {
            "macd": round(float(macd_series.iloc[-1]), 4),
            "signal": round(float(signal_series.iloc[-1]), 4),
            "histogram": round(float(hist_series.iloc[-1]), 4),
        }

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Dict[str, float]:
        """Calculates Bollinger Bands (Upper, Middle, Lower, Bandwidth)."""
        if len(prices) < period:
            last = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
            return {"upper": last, "middle": last, "lower": last, "bandwidth": 0.0}

        sma = prices.rolling(window=period).mean()
        rstd = prices.rolling(window=period).std()

        middle = float(sma.iloc[-1])
        sd = float(rstd.iloc[-1])
        upper = middle + (std_dev * sd)
        lower = middle - (std_dev * sd)
        bandwidth = ((upper - lower) / middle) * 100.0 if middle > 0 else 0.0

        return {
            "upper": round(upper, 2),
            "middle": round(middle, 2),
            "lower": round(lower, 2),
            "bandwidth": round(bandwidth, 2),
        }

    @staticmethod
    def calculate_moving_averages(prices: pd.Series) -> Dict[str, float]:
        """Calculates SMA and EMA for standard periods (20, 50, 200)."""
        res = {}
        for p in (20, 50, 200):
            if len(prices) >= p:
                res[f"sma_{p}"] = round(float(prices.rolling(window=p).mean().iloc[-1]), 2)
                res[f"ema_{p}"] = round(float(prices.ewm(span=p, adjust=False).mean().iloc[-1]), 2)
            else:
                last_p = round(float(prices.iloc[-1]), 2) if len(prices) > 0 else 0.0
                res[f"sma_{p}"] = last_p
                res[f"ema_{p}"] = last_p

        return res

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> float:
        """Calculates Average True Range (ATR)."""
        if len(close) < period + 1:
            return 0.0

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return round(float(atr), 2)

    @staticmethod
    def calculate_support_resistance(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Dict[str, List[float]]:
        """Identifies pivot support and resistance price levels."""
        if len(close) < 1:
            return {"support": [], "resistance": []}

        recent_high = float(high.iloc[-20:].max()) if len(high) >= 20 else float(high.max())
        recent_low = float(low.iloc[-20:].min()) if len(low) >= 20 else float(low.min())
        last_close = float(close.iloc[-1])

        pivot = (recent_high + recent_low + last_close) / 3.0
        r1 = (2 * pivot) - recent_low
        s1 = (2 * pivot) - recent_high
        r2 = pivot + (recent_high - recent_low)
        s2 = pivot - (recent_high - recent_low)

        return {
            "pivot": round(pivot, 2),
            "support": [round(s1, 2), round(s2, 2)],
            "resistance": [round(r1, 2), round(r2, 2)],
        }

    @classmethod
    def analyze_dataframe(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs full technical suite on OHLCV DataFrame."""
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column.")

        close = df["close"]
        high = df.get("high", close)
        low = df.get("low", close)
        volume = df.get("volume", pd.Series([0] * len(close)))

        rsi = cls.calculate_rsi(close)
        macd = cls.calculate_macd(close)
        bb = cls.calculate_bollinger_bands(close)
        ma = cls.calculate_moving_averages(close)
        atr = cls.calculate_atr(high, low, close)
        sr = cls.calculate_support_resistance(high, low, close)

        last_price = round(float(close.iloc[-1]), 2)
        avg_volume = round(float(volume.iloc[-20:].mean()), 0) if len(volume) >= 20 else float(volume.mean())

        return {
            "last_price": last_price,
            "rsi_14": rsi,
            "macd": macd,
            "bollinger_bands": bb,
            "moving_averages": ma,
            "atr_14": atr,
            "support_resistance": sr,
            "average_volume_20d": avg_volume,
        }
