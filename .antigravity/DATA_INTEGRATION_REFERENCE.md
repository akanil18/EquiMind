# EquiMind Data Integration Reference Guide
> **Purpose:** Technical reference for connecting EquiMind to real data APIs  
> **Updated:** 2026-08-06

---

## Free APIs — No Key Required

### 1. Yahoo Finance (via yfinance)

**Install:** `pip install yfinance`  
**Rate Limit:** ~2000 requests/hour (unofficial)  
**Coverage:** US + international equities, ETFs, mutual funds, crypto, forex

#### Available Data Points
```
Price History:       OHLCV (1m to max), adjusted close, dividends, splits
Financials:          Income Statement, Balance Sheet, Cash Flow (annual + quarterly)
Key Stats:           Market cap, PE, EPS, beta, 52w range, avg volume
Analyst Data:        Recommendations, price targets, earnings estimates
Options:             Full options chain (calls + puts)
Institutional:       Major holders, institutional holders
Corporate Actions:   Dividends, splits, mergers
Sustainability:      ESG scores (when available)
```

#### Ticker-to-Sector Mapping (via yfinance)
```python
info = yf.Ticker("NVDA").info
sector = info.get("sector")        # "Technology"
industry = info.get("industry")    # "Semiconductors"
market_cap = info.get("marketCap") # 3200000000000
```

#### Key Limitations
- Unofficial API — Yahoo may change/block access
- No order book / Level 2 data
- Delayed quotes (~15 min for US equities)
- Historical intraday limited to ~60 days for 1m intervals

---

### 2. SEC EDGAR REST API

**Base URL:** `https://data.sec.gov`  
**Rate Limit:** 10 requests/second  
**Requirement:** Must set `User-Agent` header  
**Coverage:** All US public companies

#### Endpoints

| Endpoint | URL Pattern | Returns |
|:---|:---|:---|
| Company Facts | `/api/xbrl/companyfacts/CIK{cik}.json` | All XBRL financial data |
| Company Submissions | `/submissions/CIK{cik}.json` | Filing history & metadata |
| Full-Text Search | `efts.sec.gov/LATEST/search-index?q={query}` | Filing search results |
| Ticker-to-CIK Mapping | `sec.gov/files/company_tickers.json` | Complete CIK lookup table |

#### Key XBRL Concepts (us-gaap namespace)

**Income Statement:**
- `Revenues` / `RevenueFromContractWithCustomerExcludingAssessedTax`
- `CostOfGoodsAndServicesSold` / `CostOfRevenue`
- `GrossProfit`
- `OperatingIncomeLoss`
- `NetIncomeLoss`
- `EarningsPerShareBasic` / `EarningsPerShareDiluted`

**Balance Sheet:**
- `Assets` / `AssetsCurrent`
- `Liabilities` / `LiabilitiesCurrent`
- `StockholdersEquity`
- `CashAndCashEquivalentsAtCarryingValue`
- `LongTermDebt` / `LongTermDebtNoncurrent`
- `CommonStockSharesOutstanding`

**Cash Flow:**
- `NetCashProvidedByUsedInOperatingActivities`
- `NetCashProvidedByUsedInInvestingActivities`
- `NetCashProvidedByUsedInFinancingActivities`
- `CapitalExpenditure` (often as `PaymentsToAcquirePropertyPlantAndEquipment`)

#### CIK Lookup Table
```python
import requests
HEADERS = {"User-Agent": "EquiMind research@equimind.ai"}
cik_map = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS).json()
# Returns: {"0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."}, ...}
# Build reverse lookup: ticker -> CIK
ticker_to_cik = {v["ticker"]: int(v["cik_str"]) for v in cik_map.values()}
```

---

## Free APIs — API Key Required (Free Tier)

### 3. Reddit (via PRAW)

**Install:** `pip install praw`  
**Setup:** Create app at https://www.reddit.com/prefs/apps/ (free)  
**Rate Limit:** 60 requests/minute  

#### Useful Subreddits for Financial Sentiment
- `r/stocks` — General stock discussion (moderate quality)
- `r/wallstreetbets` — Retail sentiment (high volume, low reliability)
- `r/investing` — Long-term investing discussions
- `r/SecurityAnalysis` — Fundamental analysis (high quality, low volume)
- `r/options` — Options sentiment & strategy
- `r/StockMarket` — General market news

#### Sentiment Extraction Pattern
```python
import praw

reddit = praw.Reddit(
    client_id="...", client_secret="...", user_agent="EquiMind/1.0"
)

# Search for ticker mentions
posts = reddit.subreddit("stocks+wallstreetbets+investing").search(
    "NVDA", sort="relevance", time_filter="week", limit=50
)
for post in posts:
    title = post.title
    score = post.score           # Reddit upvotes (proxy for agreement)
    comments = post.num_comments # Engagement level
    created = post.created_utc   # Timestamp
    # Apply sentiment analysis (FinBERT or simple heuristic)
```

### 4. NewsAPI

**Install:** `pip install newsapi-python`  
**Setup:** Register at https://newsapi.org/ (free: 100 req/day)  
**Coverage:** 80,000+ news sources

#### Usage
```python
from newsapi import NewsApiClient
api = NewsApiClient(api_key="...")
articles = api.get_everything(
    q="NVIDIA stock",
    language="en",
    sort_by="relevancy",
    page_size=20
)
```

### 5. Financial Modeling Prep (FMP)

**Setup:** Register at https://financialmodelingprep.com/ (free: 250 req/day)  
**Coverage:** US + international; financials, analyst estimates, SEC filings

#### Key Endpoints
```
/api/v3/income-statement/{ticker}?apikey=...
/api/v3/balance-sheet-statement/{ticker}?apikey=...
/api/v3/cash-flow-statement/{ticker}?apikey=...
/api/v3/profile/{ticker}?apikey=...
/api/v3/analyst-estimates/{ticker}?apikey=...
/api/v3/rating/{ticker}?apikey=...
```

---

## Free Data — No API, Direct Download

### 6. FRED (Federal Reserve Economic Data)

**Install:** `pip install fredapi`  
**Key Required:** Yes (free at https://fred.stlouisfed.org/docs/api/api_key.html)  
**Coverage:** 800,000+ economic time series

#### Key Series for Macroeconomic Analysis
```
GDP:           GDPC1 (Real GDP)
Inflation:     CPIAUCSL (CPI), PCEPILFE (Core PCE)
Interest:      DFF (Fed Funds Rate), DGS10 (10Y Treasury)
Unemployment:  UNRATE
VIX:           VIXCLS
Oil:           DCOILWTICO (WTI Crude)
Gold:          GOLDPMGBD228NLBM
USD Index:     DTWEXBGS
Yield Curve:   T10Y2Y (10Y-2Y spread)
```

### 7. RSS Feeds (No API Key)

```
Reuters:        https://www.reuters.com/rssFeed/businessNews
Seeking Alpha:  https://seekingalpha.com/feed.xml
Yahoo Finance:  https://finance.yahoo.com/rss/
MarketWatch:    https://feeds.content.dowjones.io/public/rss/mw_topstories
CNBC:           https://www.cnbc.com/id/100003114/device/rss/rss.html
```

---

## Data Caching Strategy

### Recommended Cache TTLs

| Data Type | TTL | Rationale |
|:---|:---|:---|
| OHLCV Historical | 24 hours | Daily data doesn't change after market close |
| Real-time Quote | 15 minutes | Delayed quote window |
| Financial Statements | 7 days | Quarterly filings |
| SEC Filings List | 24 hours | New filings are daily |
| Reddit Posts | 1 hour | Sentiment changes rapidly |
| News Articles | 2 hours | News cycle |
| Macroeconomic Data | 24 hours | Updated daily/monthly |
| Company Profile/Info | 7 days | Rarely changes |

### File-Based Cache Implementation
```python
import json, hashlib, os
from datetime import datetime, timedelta

CACHE_DIR = ".equimind_cache"

def cache_get(key: str, ttl_hours: int = 24):
    path = os.path.join(CACHE_DIR, f"{hashlib.md5(key.encode()).hexdigest()}.json")
    if os.path.exists(path):
        data = json.load(open(path))
        cached_at = datetime.fromisoformat(data["_cached_at"])
        if datetime.now() - cached_at < timedelta(hours=ttl_hours):
            return data["payload"]
    return None

def cache_set(key: str, payload):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{hashlib.md5(key.encode()).hexdigest()}.json")
    json.dump({"_cached_at": datetime.now().isoformat(), "payload": payload}, open(path, "w"))
```

---

## Rate Limiting Strategy

```python
import time, threading

class RateLimiter:
    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls = []
        self.lock = threading.Lock()
    
    def wait(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                time.sleep(sleep_time)
            self.calls.append(time.time())

# Usage:
sec_limiter = RateLimiter(max_calls=10, period_seconds=1.0)    # SEC: 10/sec
yf_limiter = RateLimiter(max_calls=2000, period_seconds=3600.0) # yfinance: 2000/hr
reddit_limiter = RateLimiter(max_calls=60, period_seconds=60.0)  # PRAW: 60/min
```

---

*This reference should be consulted when implementing real data adapters for EquiMind.*
