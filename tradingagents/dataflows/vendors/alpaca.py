"""Alpaca market data: Benzinga news, ticker-tagged, with history back to 2015.

Only the news endpoint is used. Alpaca's free plan serves IEX-only prices, a
small slice of market volume, so prices come from other vendors. Nothing here
reads an account or places an order.

Articles are dated, so a window in the past returns only what was published
by its end. A ticker with no Benzinga coverage in the window is "no data", so
a configured next vendor (Yahoo) can still serve its news.
"""

import html
import logging
import os
import re
from datetime import UTC, datetime, timedelta

import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)

logger = logging.getLogger(__name__)

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
REQUEST_TIMEOUT = 30
PAGE_SIZE = 50  # Alpaca's maximum per page
MAX_PAGES = 4

# Auto-generated filler: peer-comparison and options-activity write-ups with
# no reporting behind them.
FILLER_AUTHORS = {"benzinga insights"}

# Roundups ("95 Stocks Moving...") tag dozens of tickers and say a line about
# each. Past this many tags an article is dropped from a ticker's news.
MAX_TAGS = 10
# The "Tagged:" note lists at most this many other tickers.
TAG_NOTE_LIMIT = 5

# Market-wide news: articles tagged with the main index and Treasury ETFs.
MACRO_SYMBOLS = "SPY,QQQ,DIA,IWM,TLT"


class AlpacaNotConfiguredError(VendorNotConfiguredError):
    """Raised when Alpaca is selected but its key pair is not set."""


def _keys() -> tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise AlpacaNotConfiguredError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must both be set (paper-account keys work)."
        )
    return key, secret


def _request(params: dict) -> dict:
    key, secret = _keys()
    try:
        response = requests.get(
            NEWS_URL,
            params=params,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "Accept": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise VendorUnavailableError(f"Alpaca news request failed: {exc}") from None
    if response.status_code in (401, 403):
        raise AlpacaNotConfiguredError(
            f"Alpaca rejected the key pair ({response.status_code}). Check ALPACA_API_KEY and ALPACA_SECRET_KEY."
        )
    if response.status_code == 429 or response.status_code >= 500:
        raise VendorUnavailableError(f"Alpaca news returned {response.status_code}.")
    response.raise_for_status()
    return response.json() or {}


def _fetch(start: datetime, end: datetime, symbols: str | None, limit: int | None) -> list:
    """Articles created in ``[start, end)``, newest first, filler removed.

    Stops once ``limit`` articles are in hand; ``None`` reads every page up to
    ``MAX_PAGES``.
    """
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sort": "desc",
        "limit": PAGE_SIZE,
    }
    if symbols:
        params["symbols"] = symbols
    articles = []
    for _ in range(MAX_PAGES):
        body = _request(params)
        for article in body.get("news") or []:
            if (article.get("author") or "").strip().lower() in FILLER_AUTHORS:
                continue
            created = _parse(article.get("created_at"))
            # Alpaca's end is inclusive; the window here is half-open.
            if created is None or not start <= created < end:
                continue
            articles.append(article)
        token = body.get("next_page_token")
        if (limit is not None and len(articles) >= limit) or not token:
            break
        params["page_token"] = token
    return articles if limit is None else articles[:limit]


def _about(article: dict, ticker: str) -> bool:
    """True when the article is mainly about ``ticker``: few tags, or named in the headline."""
    symbols = article.get("symbols") or []
    headline = _text(article.get("headline")).upper()
    return len(symbols) <= 2 or re.search(rf"\b{re.escape(ticker)}\b", headline) is not None


def _select(articles: list, ticker: str, limit: int) -> list:
    """Up to ``limit`` articles, ones mainly about ``ticker`` first, shown newest first.

    Articles come newest first, so a plain cut-off lets a few busy days of
    multi-ticker coverage crowd out the rest of the window.
    """
    kept = [a for a in articles if len(a.get("symbols") or []) <= MAX_TAGS]
    focused = [a for a in kept if _about(a, ticker)]
    chosen = focused[:limit]
    chosen += [a for a in kept if not _about(a, ticker)][: limit - len(chosen)]
    return sorted(chosen, key=lambda a: _parse(a.get("created_at")), reverse=True)


def _parse(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _text(value) -> str:
    """Plain text: HTML tags dropped, entities (``&amp;amp;``) unescaped twice over."""
    text = re.sub(r"<[^>]+>", " ", html.unescape(html.unescape(str(value or ""))))
    return re.sub(r"\s+", " ", text).strip()


def _format(articles: list, ticker: str | None = None) -> str:
    blocks = []
    for article in articles:
        created = _parse(article.get("created_at"))
        symbols = [s for s in article.get("symbols") or [] if s]
        lines = [f"### {_text(article.get('headline'))} (Benzinga, {created:%Y-%m-%d %H:%M} UTC)"]
        if ticker and len(symbols) > 1:
            # Benzinga tags every ticker an article mentions, so a multi-ticker
            # piece may be mostly about something else.
            others = [s for s in symbols if s != ticker]
            tagged = ", ".join([ticker, *others[:TAG_NOTE_LIMIT]])
            if len(others) > TAG_NOTE_LIMIT:
                tagged += f" and {len(others) - TAG_NOTE_LIMIT} others"
            lines.append(f"Tagged: {tagged}. Judge how much of it concerns {ticker}.")
        summary = _text(article.get("summary"))
        if summary:
            lines.append(summary)
        if article.get("url"):
            lines.append(f"Link: {article['url']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Benzinga articles tagged with ``ticker`` published from ``start_date`` through ``end_date``."""
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
    symbol = ticker.upper()
    articles = _select(_fetch(start, end, symbol, None), symbol, get_config()["news_article_limit"])
    if not articles:
        raise NoMarketDataError(symbol, detail=f"no Benzinga news between {start_date} and {end_date}")
    return f"## {symbol} news (Benzinga), from {start_date} to {end_date}:\n\n{_format(articles, symbol)}"


def get_global_news(as_of_date: str, look_back_days: int | None = None, limit: int | None = None) -> str:
    """Market-wide Benzinga news: articles tagged with the main index and bond ETFs."""
    config = get_config()
    look_back_days = config["global_news_lookback_days"] if look_back_days is None else look_back_days
    limit = config["global_news_article_limit"] if limit is None else limit
    end = datetime.strptime(as_of_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
    start = end - timedelta(days=look_back_days + 1)
    articles = _fetch(start, end, MACRO_SYMBOLS, limit)
    start_date = start.strftime("%Y-%m-%d")
    if not articles:
        return f"No market-wide Benzinga news between {start_date} and {as_of_date}."
    return f"## Market news (Benzinga), from {start_date} to {as_of_date}:\n\n{_format(articles)}"
