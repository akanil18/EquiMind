"""
EquiMind SEC EDGAR Adapter — Real Financial Filing Integration
================================================================
Connects to SEC EDGAR REST API (data.sec.gov) — completely free, no API key.
Fetches real 10-K, 10-Q filings and parses XBRL financial data.

API Reference:
  - Company Facts: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  - Submissions:   https://data.sec.gov/submissions/CIK{cik}.json
  - Ticker→CIK:    https://www.sec.gov/files/company_tickers.json
  - Rate Limit:    10 requests/second
  - Required:      User-Agent header
"""

import json
import logging
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import requests

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

EDGAR_BASE = "https://data.sec.gov"
EDGAR_HEADERS = {
    "User-Agent": "EquiMind/1.0 research@equimind.ai",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    ".equimind_cache", "edgar"
)

# Rate limiter: SEC allows 10 req/sec, we use 8 for safety
class _EdgarLimiter:
    def __init__(self):
        self.calls: list = []
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 1.0]
            if len(self.calls) >= 8:
                time.sleep(1.0 - (now - self.calls[0]) + 0.05)
            self.calls.append(time.time())

_limiter = _EdgarLimiter()

# ═══════════════════════════════════════════════════════════════
# CIK LOOKUP TABLE
# ═══════════════════════════════════════════════════════════════

_CIK_MAP: Dict[str, int] = {}  # ticker → CIK


def _load_cik_map() -> Dict[str, int]:
    """Load ticker-to-CIK mapping from SEC (cached for 30 days)."""
    global _CIK_MAP
    if _CIK_MAP:
        return _CIK_MAP

    cache_path = os.path.join(CACHE_DIR, "cik_map.json")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                data = json.load(f)
            if datetime.now() - datetime.fromisoformat(data["_cached_at"]) < timedelta(days=30):
                _CIK_MAP = data["map"]
                logger.debug(f"Loaded {len(_CIK_MAP)} tickers from CIK cache")
                return _CIK_MAP
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fetch from SEC
    try:
        _limiter.wait()
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS, timeout=15
        )
        resp.raise_for_status()
        raw = resp.json()
        
        _CIK_MAP = {
            v["ticker"].upper(): int(v["cik_str"])
            for v in raw.values()
        }
        
        # Cache
        with open(cache_path, "w") as f:
            json.dump({"_cached_at": datetime.now().isoformat(), "map": _CIK_MAP}, f)
        
        logger.info(f"✓ Loaded {len(_CIK_MAP)} ticker→CIK mappings from SEC")
        return _CIK_MAP
        
    except Exception as e:
        logger.warning(f"Failed to load CIK map: {e}")
        return {}


def ticker_to_cik(ticker: str) -> Optional[int]:
    """Convert ticker symbol to SEC CIK number."""
    cik_map = _load_cik_map()
    return cik_map.get(ticker.upper())


# ═══════════════════════════════════════════════════════════════
# SEC EDGAR ADAPTER
# ═══════════════════════════════════════════════════════════════

# Key XBRL concepts we want to extract
XBRL_INCOME_CONCEPTS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cost_of_revenue": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "rd_expense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
}

XBRL_BALANCE_CONCEPTS = {
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "total_liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "stockholders_equity": ["StockholdersEquity"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
}

XBRL_CASHFLOW_CONCEPTS = {
    "operating_cashflow": ["NetCashProvidedByUsedInOperatingActivities"],
    "investing_cashflow": ["NetCashProvidedByUsedInInvestingActivities"],
    "financing_cashflow": ["NetCashProvidedByUsedInFinancingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditures",
    ],
}


class SECEdgarAdapter:
    """Fetches and parses real financial data from SEC EDGAR XBRL API."""

    @classmethod
    def get_company_facts(cls, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch all XBRL facts for a company from SEC EDGAR.
        Returns parsed financial data organized by concept.
        """
        cik = ticker_to_cik(ticker)
        if cik is None:
            logger.warning(f"No CIK found for ticker {ticker}")
            return None

        # Check cache (24hr TTL)
        cache_path = os.path.join(CACHE_DIR, f"facts_{ticker.upper()}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                if datetime.now() - datetime.fromisoformat(data["_cached_at"]) < timedelta(hours=24):
                    return data["facts"]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Fetch from SEC
        try:
            _limiter.wait()
            url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik:010d}.json"
            resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
            resp.raise_for_status()
            raw = resp.json()
            
            facts = cls._parse_company_facts(raw)
            
            # Cache
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"_cached_at": datetime.now().isoformat(), "facts": facts}, f)
            
            logger.info(f"✓ Fetched SEC EDGAR facts for {ticker} (CIK: {cik})")
            return facts
            
        except requests.HTTPError as e:
            logger.warning(f"SEC EDGAR HTTP error for {ticker}: {e}")
            return None
        except Exception as e:
            logger.warning(f"SEC EDGAR error for {ticker}: {e}")
            return None

    @classmethod
    def get_recent_filings(cls, ticker: str, filing_types: List[str] = None,
                            max_filings: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch recent filing metadata (10-K, 10-Q, 8-K, etc.)
        """
        cik = ticker_to_cik(ticker)
        if cik is None:
            return []

        if filing_types is None:
            filing_types = ["10-K", "10-Q", "8-K"]

        try:
            _limiter.wait()
            url = f"{EDGAR_BASE}/submissions/CIK{cik:010d}.json"
            resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            filings = []
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            for i in range(min(len(forms), 100)):
                if forms[i] in filing_types:
                    filings.append({
                        "form": forms[i],
                        "filing_date": dates[i],
                        "accession": accessions[i],
                        "document": primary_docs[i] if i < len(primary_docs) else "",
                        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accessions[i].replace('-', '')}/{primary_docs[i] if i < len(primary_docs) else ''}",
                    })
                    if len(filings) >= max_filings:
                        break

            logger.info(f"✓ Found {len(filings)} recent filings for {ticker}")
            return filings

        except Exception as e:
            logger.warning(f"SEC submissions error for {ticker}: {e}")
            return []

    @classmethod
    def get_financial_summary(cls, ticker: str) -> Dict[str, Any]:
        """
        Get a structured financial summary with real SEC data.
        Returns most recent annual and quarterly figures.
        """
        facts = cls.get_company_facts(ticker)
        if not facts:
            return cls._synthetic_financial_summary(ticker)

        summary = {
            "ticker": ticker.upper(),
            "source": "SEC EDGAR XBRL",
            "income_statement": facts.get("income", {}),
            "balance_sheet": facts.get("balance", {}),
            "cash_flow": facts.get("cashflow", {}),
            "computed_ratios": {},
        }

        # Compute key ratios from real data
        income = facts.get("income", {})
        balance = facts.get("balance", {})

        revenue = cls._latest_value(income.get("revenue", []))
        net_income = cls._latest_value(income.get("net_income", []))
        equity = cls._latest_value(balance.get("stockholders_equity", []))
        total_assets = cls._latest_value(balance.get("total_assets", []))
        total_liabilities = cls._latest_value(balance.get("total_liabilities", []))

        if revenue and net_income:
            summary["computed_ratios"]["net_margin"] = round(net_income / revenue * 100, 2)
        if equity and net_income:
            summary["computed_ratios"]["roe"] = round(net_income / equity * 100, 2)
        if total_assets and net_income:
            summary["computed_ratios"]["roa"] = round(net_income / total_assets * 100, 2)
        if equity and total_liabilities:
            summary["computed_ratios"]["debt_to_equity"] = round(total_liabilities / equity, 2)

        return summary

    # ── Internal Parsing Methods ───────────────────────────────

    @classmethod
    def _parse_company_facts(cls, raw: Dict) -> Dict[str, Any]:
        """Parse raw XBRL company facts into structured financial data."""
        us_gaap = raw.get("facts", {}).get("us-gaap", {})
        
        result = {"income": {}, "balance": {}, "cashflow": {}}

        # Parse income statement concepts
        for key, concept_names in XBRL_INCOME_CONCEPTS.items():
            values = cls._extract_concept(us_gaap, concept_names)
            if values:
                result["income"][key] = values

        # Parse balance sheet concepts
        for key, concept_names in XBRL_BALANCE_CONCEPTS.items():
            values = cls._extract_concept(us_gaap, concept_names)
            if values:
                result["balance"][key] = values

        # Parse cash flow concepts
        for key, concept_names in XBRL_CASHFLOW_CONCEPTS.items():
            values = cls._extract_concept(us_gaap, concept_names)
            if values:
                result["cashflow"][key] = values

        return result

    @classmethod
    def _extract_concept(cls, us_gaap: Dict, concept_names: List[str]) -> List[Dict[str, Any]]:
        """Extract values for a financial concept from XBRL data."""
        for concept_name in concept_names:
            concept = us_gaap.get(concept_name)
            if concept is None:
                continue
            
            units = concept.get("units", {})
            # Try USD first, then shares, then pure
            for unit_key in ["USD", "shares", "USD/shares", "pure"]:
                entries = units.get(unit_key, [])
                if not entries:
                    continue
                
                # Filter to 10-K and 10-Q filings only, with fiscal period focus
                relevant = []
                for entry in entries:
                    form = entry.get("form", "")
                    if form in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                        # Prefer entries with 'frame' (fiscal period indicator)
                        relevant.append({
                            "value": entry.get("val"),
                            "end_date": entry.get("end"),
                            "start_date": entry.get("start"),
                            "form": form,
                            "fiscal_year": entry.get("fy"),
                            "fiscal_period": entry.get("fp"),
                            "filed": entry.get("filed"),
                            "frame": entry.get("frame"),
                        })
                
                if relevant:
                    # Sort by end date descending (most recent first)
                    relevant.sort(key=lambda x: x.get("end_date", ""), reverse=True)
                    # Return most recent 12 entries
                    return relevant[:12]
        
        return []

    @staticmethod
    def _latest_value(entries: List[Dict]) -> Optional[float]:
        """Get the most recent value from parsed XBRL entries."""
        if not entries:
            return None
        # Prefer annual (10-K) entries
        for entry in entries:
            if entry.get("form") == "10-K" and entry.get("value") is not None:
                return entry["value"]
        # Fall back to any entry
        for entry in entries:
            if entry.get("value") is not None:
                return entry["value"]
        return None

    @staticmethod
    def _synthetic_financial_summary(ticker: str) -> Dict[str, Any]:
        """Fallback synthetic data when SEC API is unavailable."""
        return {
            "ticker": ticker.upper(),
            "source": "synthetic_fallback",
            "income_statement": {},
            "balance_sheet": {},
            "cash_flow": {},
            "computed_ratios": {},
        }
