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

# Try C++ optimized engine first, fall back to Python
try:
    from equimind.native import fast_technical, is_native_available
    _USING_CPP = is_native_available()
except ImportError:
    _USING_CPP = False


class MarketDataTeam(ResearchTeam):
    """Specialized team collecting real price data, liquidity, moving averages, and technical indicators.
    
    Data Sources:
      - yfinance (real market data) with synthetic fallback
      - C++ optimized technical indicator engine (with Python fallback)
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

        # ── Compute technical indicators (C++ or Python) ──────
        close_list = df["Close"].tolist()
        high_list = df["High"].tolist()
        low_list = df["Low"].tolist()

        if _USING_CPP:
            tech_summary = fast_technical.full_analysis(close_list, high_list, low_list)
            engine_label = "C++ Native"
        else:
            from equimind.quantitative.technical import TechnicalEngine
            # Build DataFrame in expected format
            analysis_df = pd.DataFrame({
                "close": close_list,
                "high": high_list,
                "low": low_list,
                "open": df["Open"].tolist(),
                "volume": df["Volume"].tolist(),
            })
            tech_summary = TechnicalEngine.analyze_dataframe(analysis_df)
            engine_label = "Python"

        rsi = tech_summary["rsi_14"]
        last_price = tech_summary["last_price"]
        macd_data = tech_summary.get("macd", {})
        bb_data = tech_summary.get("bollinger_bands", {})
        sr_data = tech_summary.get("support_resistance", {})

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
            f"MACD: {macd_data.get('macd', 0):.2f} (Signal: {macd_data.get('signal', 0):.2f}, "
            f"Hist: {macd_data.get('histogram', 0):.2f})\n"
            f"Bollinger Bands: Upper=${bb_data.get('upper', 0):.2f}, "
            f"Mid=${bb_data.get('middle', 0):.2f}, Lower=${bb_data.get('lower', 0):.2f}\n"
            f"SMA(20): ${tech_summary.get('sma_20', 0):.2f} | "
            f"SMA(50): ${tech_summary.get('sma_50', 0):.2f} | "
            f"SMA(200): ${tech_summary.get('sma_200', 0):.2f}\n"
            f"ATR(14): {tech_summary.get('atr_14', 0):.2f} | "
            f"Annualized Vol: {tech_summary.get('annualized_volatility', 0):.1f}%\n"
            f"Support: {sr_data.get('support', [])} | Resistance: {sr_data.get('resistance', [])}\n"
            f"Data: {len(df)} trading days analyzed"
        )

        # Add volume analysis
        if "Volume" in df.columns:
            avg_vol = df["Volume"].mean()
            recent_vol = df["Volume"].tail(5).mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
            content_str += f"\nVolume: Avg={avg_vol/1e6:.1f}M, Recent={recent_vol/1e6:.1f}M (Ratio: {vol_ratio:.2f}x)"

        # ── Build enriched metadata ──────────────────────────
        metadata = {
            **tech_summary,
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
