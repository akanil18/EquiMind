"""
EquiMind News RSS Adapter — Free Financial News Integration
=============================================================
Fetches real financial news from free RSS feeds — no API key required.
Parses headlines and summaries from Reuters, MarketWatch, CNBC, Yahoo Finance.
"""

import logging
import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    logger.warning("feedparser not installed — news adapter will use fallback")

# ═══════════════════════════════════════════════════════════════
# RSS FEED SOURCES
# ═══════════════════════════════════════════════════════════════

NEWS_FEEDS = {
    "yahoo_finance": {
        "url": "https://finance.yahoo.com/rss/",
        "name": "Yahoo Finance",
        "credibility": "HIGH",
    },
    "marketwatch": {
        "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "name": "MarketWatch",
        "credibility": "HIGH",
    },
    "cnbc": {
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "name": "CNBC",
        "credibility": "HIGH",
    },
    "seeking_alpha": {
        "url": "https://seekingalpha.com/feed.xml",
        "name": "Seeking Alpha",
        "credibility": "MEDIUM",
    },
}

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    ".equimind_cache", "news"
)


class NewsRSSAdapter:
    """Fetches and filters financial news from free RSS feeds."""

    @classmethod
    def fetch_news(cls, ticker: str = None, keywords: List[str] = None,
                   max_articles: int = 20, max_age_hours: int = 72) -> List[Dict[str, Any]]:
        """
        Fetch recent financial news, optionally filtered by ticker/keywords.
        
        Args:
            ticker: Stock ticker to filter for (e.g., "NVDA")
            keywords: Additional search keywords
            max_articles: Maximum articles to return
            max_age_hours: Maximum age of articles in hours
        
        Returns:
            List of article dicts with: title, summary, link, published, source, relevance_score
        """
        if not HAS_FEEDPARSER:
            return cls._synthetic_news(ticker, max_articles)

        # Check cache (2hr TTL)
        cache_key = hashlib.md5(f"news:{ticker}:{keywords}".encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                if datetime.now() - datetime.fromisoformat(data["_cached_at"]) < timedelta(hours=2):
                    return data["articles"][:max_articles]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Build search terms
        search_terms = []
        if ticker:
            search_terms.append(ticker.upper())
            # Add common company name variations
            name_map = {
                "NVDA": ["NVIDIA", "nvidia"],
                "AAPL": ["Apple", "apple"],
                "TSLA": ["Tesla", "tesla"],
                "MSFT": ["Microsoft", "microsoft"],
                "AMZN": ["Amazon", "amazon"],
                "GOOGL": ["Google", "Alphabet", "google"],
                "META": ["Meta", "Facebook", "meta"],
            }
            search_terms.extend(name_map.get(ticker.upper(), []))
        if keywords:
            search_terms.extend(keywords)

        all_articles = []
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        for feed_id, feed_config in NEWS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_config["url"])
                
                for entry in feed.entries[:50]:  # Check first 50 entries per feed
                    # Parse publication date
                    pub_date = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    
                    # Skip old articles
                    if pub_date and pub_date < cutoff:
                        continue

                    title = getattr(entry, "title", "")
                    summary = getattr(entry, "summary", getattr(entry, "description", ""))
                    link = getattr(entry, "link", "")
                    
                    # Calculate relevance score
                    relevance = cls._calculate_relevance(title, summary, search_terms)
                    
                    if relevance > 0 or not search_terms:
                        all_articles.append({
                            "title": title,
                            "summary": summary[:500],  # Truncate long summaries
                            "link": link,
                            "published": pub_date.isoformat() if pub_date else None,
                            "source": feed_config["name"],
                            "source_credibility": feed_config["credibility"],
                            "relevance_score": relevance,
                            "ticker_mentioned": ticker.upper() if ticker else None,
                        })
                        
            except Exception as e:
                logger.debug(f"RSS feed error ({feed_id}): {e}")
                continue

        # Sort by relevance, then recency
        all_articles.sort(key=lambda x: (x["relevance_score"], x.get("published", "")), reverse=True)
        result = all_articles[:max_articles]

        # Cache
        try:
            with open(cache_path, "w") as f:
                json.dump({"_cached_at": datetime.now().isoformat(), "articles": result}, f)
        except OSError:
            pass

        if result:
            logger.info(f"✓ Fetched {len(result)} news articles for {ticker or 'general'}")
        else:
            logger.debug(f"No relevant news found for {ticker}")
            result = cls._synthetic_news(ticker, max_articles)

        return result

    @staticmethod
    def _calculate_relevance(title: str, summary: str, search_terms: List[str]) -> float:
        """Calculate relevance score based on search term matches."""
        if not search_terms:
            return 0.5
        
        text = f"{title} {summary}".lower()
        score = 0.0
        for term in search_terms:
            term_lower = term.lower()
            # Title match is worth more
            if term_lower in title.lower():
                score += 2.0
            elif term_lower in text:
                score += 1.0
        
        return min(score / len(search_terms), 1.0)

    @staticmethod
    def _synthetic_news(ticker: str = None, max_articles: int = 5) -> List[Dict[str, Any]]:
        """Fallback synthetic news when RSS feeds unavailable."""
        t = ticker or "MARKET"
        return [
            {
                "title": f"{t} — Market Update: Trading activity remains elevated",
                "summary": f"Recent trading in {t} has shown increased institutional interest with volume trending above 30-day averages.",
                "link": f"https://finance.yahoo.com/quote/{t}",
                "published": datetime.now().isoformat(),
                "source": "Synthetic Feed",
                "source_credibility": "LOW",
                "relevance_score": 0.5,
                "ticker_mentioned": t,
            }
        ]
