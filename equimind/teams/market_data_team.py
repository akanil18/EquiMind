from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import logging

import numpy as np
import pandas as pd

from equimind.evidence.schema import (
    EvidenceNode,
    EvidenceSource,
    AuthorCredibility,
    SentimentPolarity,
)
from equimind.teams.base_team import ResearchTeam
from equimind.providers.base import LLMProvider
from equimind.adapters import YFinanceAdapter

logger = logging.getLogger(__name__)


class MarketDataTeam(ResearchTeam):
    """Specialized team collecting real price data, liquidity, moving averages, and technical indicators.
    
    Data Sources:
      - yfinance (real market data) with synthetic fallback
      - Pure Python/Pandas technical analysis engine
    """

    @property
    def team_name(self) -> str:
        return "market_data"

    def research(
        self,
        ticker: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        provider: Optional[LLMProvider] = None,
        as_of_date: Optional[datetime] = None,
    ) -> List[EvidenceNode]:
        ref_date = as_of_date or datetime.now(timezone.utc)
        ticker_upper = ticker.upper()

        # ── Fetch REAL market data via yfinance ────────────────
        df = YFinanceAdapter.get_price_history(ticker_upper, period="2y", interval="1d")
        
        # Filter by as_of_date if provided (Time Machine support)
        if as_of_date and not df.empty:
            try:
                target_ts = pd.Timestamp(as_of_date)
                if df.index.tz is not None and target_ts.tzinfo is None:
                    target_ts = target_ts.tz_localize("UTC")
                elif df.index.tz is None and target_ts.tzinfo is not None:
                    target_ts = target_ts.tz_convert(None)
                df = df[df.index <= target_ts]
            except Exception as ex:
                logger.debug(f"Timestamp filter warning: {ex}")
        
        if df.empty:
            logger.warning(f"No price data for {ticker_upper}")
            return []

        data_source = "yfinance (Real)" if len(df) > 100 else "synthetic fallback"

        # ── Compute technical indicators ──────
        close_series = pd.Series(df["Close"].astype(float).values).dropna()
        last_price = float(close_series.iloc[-1]) if not close_series.empty and (close_series.iloc[-1] == close_series.iloc[-1]) else 100.0

        # RSI calculation
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss.replace(0, 1e-9))
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

        # Moving Averages
        sma_20 = float(close_series.rolling(20, min_periods=1).mean().iloc[-1])
        sma_50 = float(close_series.rolling(50, min_periods=1).mean().iloc[-1])
        sma_200 = float(close_series.rolling(200, min_periods=1).mean().iloc[-1])

        # MACD (12, 26, 9)
        exp1 = close_series.ewm(span=12, adjust=False).mean()
        exp2 = close_series.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line
        macd_val = float(macd_line.iloc[-1])
        signal_val = float(signal_line.iloc[-1])
        hist_val = float(hist.iloc[-1])

        # Bollinger Bands (20, 2)
        std_20 = float(close_series.rolling(20, min_periods=1).std().iloc[-1])
        bb_upper = sma_20 + (2.0 * std_20)
        bb_lower = sma_20 - (2.0 * std_20)

        # Volatility
        returns = close_series.pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(252) * 100.0) if len(returns) > 1 else 20.0

        engine_label = "Python Engine"

        # ── Determine sentiment from RSI ──────────────────────
        sentiment = SentimentPolarity.NEUTRAL
        if rsi > 70:
            sentiment = SentimentPolarity.BEARISH   # Overbought
        elif rsi < 30:
            sentiment = SentimentPolarity.BULLISH    # Oversold
        elif rsi >= 55:
            sentiment = SentimentPolarity.BULLISH
        elif rsi <= 45:
            sentiment = SentimentPolarity.BEARISH

        # ── Fetch company info for enrichment ────────────────
        company_info = YFinanceAdapter.get_company_info(ticker_upper)
        company_name = company_info.get("name", ticker_upper)
        sector = company_info.get("sector", "Unknown")
        market_cap = company_info.get("market_cap", 0)
        pe_ratio = company_info.get("pe_ratio")

        # ── Build evidence content ───────────────────────────
        content_str = (
            f"Market data for {company_name} ({ticker_upper}) — Source: {data_source}, Engine: {engine_label}\n"
            f"Last Price: ${last_price:.2f} | Sector: {sector}\n"
            f"Market Cap: ${market_cap/1e9:.1f}B" + (f" | P/E: {pe_ratio:.1f}" if pe_ratio else "") + "\n"
            f"RSI(14): {rsi:.1f} | "
            f"MACD: {macd_val:.2f} (Signal: {signal_val:.2f}, Hist: {hist_val:.2f})\n"
            f"Bollinger Bands: Upper=${bb_upper:.2f}, Mid=${sma_20:.2f}, Lower=${bb_lower:.2f}\n"
            f"SMA(20): ${sma_20:.2f} | SMA(50): ${sma_50:.2f} | SMA(200): ${sma_200:.2f}\n"
            f"Annualized Volatility: {ann_vol:.1f}%\n"
            f"Data: {len(df)} trading days analyzed"
        )

        # Add volume analysis
        if "Volume" in df.columns:
            avg_vol = float(df["Volume"].mean())
            recent_vol = float(df["Volume"].tail(5).mean())
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
            content_str += f"\nVolume: Avg={avg_vol/1e6:.1f}M, Recent={recent_vol/1e6:.1f}M (Ratio: {vol_ratio:.2f}x)"

        # ── Build enriched metadata ──────────────────────────
        metadata = {
            "last_price": last_price,
            "rsi_14": rsi,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "macd": {"macd": macd_val, "signal": signal_val, "histogram": hist_val},
            "bollinger_bands": {"upper": bb_upper, "middle": sma_20, "lower": bb_lower},
            "annualized_volatility": ann_vol,
            "data_source": data_source,
            "engine": engine_label,
            "company_name": company_name,
            "sector": sector,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "trading_days_analyzed": len(df),
            "data_start_date": str(df.index[0].date()) if not df.empty else None,
            "data_end_date": str(df.index[-1].date()) if not df.empty else None,
        }

        node = EvidenceNode(
            source_type=EvidenceSource.MARKET_PRICES,
            title=f"{company_name} ({ticker_upper}) — Real Market Data & Technical Analysis",
            content=content_str,
            publication_timestamp=ref_date,
            author=f"EquiMind Market Engine ({engine_label})",
            author_credibility=AuthorCredibility.VERIFIED_OFFICIAL,
            confidence_score=0.95,
            sentiment=sentiment,
            affected_ticker=ticker_upper,
            tags=["technical_analysis", "prices", "rsi", "macd", "bollinger_bands",
                  "real_data" if "Real" in data_source else "synthetic"],
            metadata=metadata,
        )

        return [node]
