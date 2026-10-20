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
import time
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
RETRIES = 5          # a 429 or 5xx is retried, waiting for the rate-limit reset or doubling from 2s
MAX_WAIT = 60
_sleep = time.sleep
_now = time.time
# A universe screen asks for many symbols at once: this many per request, and
# up to BULK_MAX_PAGES pages for each batch.
BULK_BATCH = 50
BULK_MAX_PAGES = 30

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


def _wait(response, attempt: int) -> float:
    """Seconds until Alpaca's rate-limit window resets (X-RateLimit-Reset is a Unix time), else Retry-After,
    else a doubling backoff."""
    headers = response.headers or {}
    try:
        return min(max(float(headers["X-RateLimit-Reset"]) - _now(), 0) + 1, MAX_WAIT)
    except (KeyError, TypeError, ValueError):
        pass
    try:
        return min(float(headers["Retry-After"]), MAX_WAIT)
    except (KeyError, TypeError, ValueError):
        return min(2 ** (attempt + 1), MAX_WAIT)


def _request(params: dict) -> dict:
    key, secret = _keys()
    for attempt in range(RETRIES + 1):
        try:
            response = requests.get(
                NEWS_URL,
                params=params,
                headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "Accept": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            if attempt < RETRIES:
                _sleep(min(2 ** (attempt + 1), MAX_WAIT))
                continue
            raise VendorUnavailableError(f"Alpaca news request failed: {exc}") from None
        if response.status_code in (401, 403):
            raise AlpacaNotConfiguredError(
                f"Alpaca rejected the key pair ({response.status_code}). Check ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )
        if response.status_code == 429 or response.status_code >= 500:
            # The news API allows about 200 calls a minute; a bulk read or a replay of
            # several mornings at once can pass that, so wait for the window to reset.
            if attempt < RETRIES:
                _sleep(_wait(response, attempt))
                continue
            raise VendorUnavailableError(f"Alpaca news returned {response.status_code}.")
        response.raise_for_status()
        return response.json() or {}
    raise AssertionError("unreachable")


def _fetch(start: datetime, end: datetime, symbols: str | None, limit: int | None,
           max_pages: int = MAX_PAGES) -> list:
    """Articles created in ``[start, end)``, newest first, filler removed.

    Stops once ``limit`` articles are in hand; ``None`` reads every page up to
    ``max_pages``.
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
    for _ in range(max_pages):
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


def _window(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    return start, datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)


def get_news_by_symbol(symbols: list[str], start_date: str, end_date: str, per_symbol: int,
                       cutoff: datetime | None = None, fresh_since: datetime | None = None) -> dict[str, str]:
    """Benzinga news for many tickers at once, as each ticker's formatted article list.

    Reads the window in batches of ``BULK_BATCH`` symbols, then picks each
    ticker's articles as ``get_news`` does. Tickers with no article in the
    window are left out. ``cutoff`` ends the window earlier than midnight after
    ``end_date``, so a replayed morning sees only the news out before the open.
    With ``fresh_since``, a ticker whose chosen articles are all older is left
    out, so a story from days ago doesn't keep coming back as a fresh call.
    """
    start, end = _window(start_date, end_date)
    if cutoff:
        end = min(end, cutoff)
    wanted = list(dict.fromkeys(s.upper() for s in symbols))
    by_symbol: dict[str, dict] = {s: {} for s in wanted}
    # Benzinga files class shares under the root (BRK) and tags them BRK/B, never BRK.B,
    # so a dotted ticker is queried by its root and matched on any of those tags.
    owners: dict[str, list[str]] = {}
    for symbol in wanted:
        root, _, share = symbol.partition(".")
        for tag in {symbol, root, f"{root}/{share}"} if share else {symbol}:
            owners.setdefault(tag, []).append(symbol)
    queries = list(dict.fromkeys(s.partition(".")[0] for s in wanted))
    for i in range(0, len(queries), BULK_BATCH):
        batch = queries[i:i + BULK_BATCH]
        for article in _fetch(start, end, ",".join(batch), None, BULK_MAX_PAGES):
            for tag in article.get("symbols") or []:
                for symbol in owners.get(tag, []):
                    by_symbol[symbol][article.get("id")] = article
    news = {}
    for symbol, found in by_symbol.items():
        chosen = _select(list(found.values()), symbol, per_symbol)
        if fresh_since and not any((_parse(a.get("created_at")) or fresh_since) > fresh_since for a in chosen):
            continue
        if chosen:
            news[symbol] = _format(chosen, symbol)
    return news
