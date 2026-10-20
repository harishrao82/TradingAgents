"""Alpaca vendor: Benzinga news, filler removal, window filtering and routing.

Article bodies follow the shape Alpaca served on a live read (AAPL,
2026-10-13). All API access is mocked.
"""
import copy
from unittest import mock

import pytest
import requests

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import router
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from tradingagents.dataflows.vendors import alpaca


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setattr(config_module, "_config", copy.deepcopy(default_config.DEFAULT_CONFIG))


def _article(headline, created_at, symbols=("AAPL",), author="Jane Doe", summary="", url="https://x.test/a"):
    return {"id": hash(headline), "headline": headline, "author": author, "created_at": created_at,
            "updated_at": created_at, "summary": summary, "content": "", "url": url, "images": [],
            "symbols": list(symbols), "source": "benzinga"}


def _stub(pages, status=200):
    """Serve ``pages`` in order, recording each request's params."""
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append(dict(params or {}))
        response = mock.Mock(spec=requests.Response)
        response.status_code = status
        response.headers = {}
        response.raise_for_status.return_value = None
        response.json.return_value = pages[len(calls) - 1] if pages else {}
        return response

    return get, calls


def test_formats_articles_and_drops_filler(monkeypatch):
    get, calls = _stub([{"news": [
        _article("Apple &amp;amp; the &lt;b&gt;iPhone&lt;/b&gt;", "2026-10-12T14:30:00Z",
                 summary="<p>Demand is strong.</p>"),
        _article("AAPL options activity", "2026-10-12T15:00:00Z", author="Benzinga Insights"),
    ], "next_page_token": None}])
    monkeypatch.setattr(alpaca.requests, "get", get)

    out = alpaca.get_news("aapl", "2026-10-10", "2026-10-12")

    assert "Apple & the iPhone (Benzinga, 2026-10-12 14:30 UTC)" in out
    assert "Demand is strong." in out
    assert "options activity" not in out
    assert "Tagged:" not in out
    assert calls[0]["symbols"] == "AAPL"
    # The end day is included: the request runs to midnight after it.
    assert calls[0]["end"] == "2026-10-13T00:00:00Z"


def test_multi_ticker_article_is_flagged(monkeypatch):
    get, _ = _stub([{"news": [_article("Chip stocks rally", "2026-10-12T14:30:00Z",
                                        symbols=("NVDA", "AMD", "AAPL"))]}])
    monkeypatch.setattr(alpaca.requests, "get", get)

    out = alpaca.get_news("AAPL", "2026-10-10", "2026-10-12")

    assert "Tagged: AAPL, NVDA, AMD. Judge how much of it concerns AAPL." in out


def test_articles_outside_the_window_are_dropped(monkeypatch):
    get, _ = _stub([{"news": [
        _article("Published after the window", "2026-10-13T00:00:00Z"),
        _article("Inside the window", "2026-10-12T23:59:00Z"),
    ]}])
    monkeypatch.setattr(alpaca.requests, "get", get)

    out = alpaca.get_news("AAPL", "2026-10-10", "2026-10-12")

    assert "Inside the window" in out
    assert "after the window" not in out


def test_reads_every_page_then_keeps_the_newest(monkeypatch):
    config_module._config["news_article_limit"] = 3
    get, calls = _stub([
        {"news": [_article("One", "2026-10-12T10:00:00Z"), _article("Two", "2026-10-12T09:00:00Z")],
         "next_page_token": "abc"},
        {"news": [_article("Three", "2026-10-12T08:00:00Z"), _article("Four", "2026-10-12T07:00:00Z")],
         "next_page_token": None},
    ])
    monkeypatch.setattr(alpaca.requests, "get", get)

    out = alpaca.get_news("AAPL", "2026-10-10", "2026-10-12")

    assert len(calls) == 2
    assert calls[1]["page_token"] == "abc"
    assert "Three" in out and "Four" not in out


def test_roundups_are_dropped_and_tag_note_is_capped(monkeypatch):
    many = [f"T{i}" for i in range(30)]
    get, _ = _stub([{"news": [
        _article("95 Stocks Moving In Monday's Session", "2026-10-12T15:00:00Z", symbols=["AAPL", *many]),
        _article("Big tech slides", "2026-10-12T14:00:00Z", symbols=["MSFT", "GOOG", "AAPL", "AMZN", "META",
                                                                     "NVDA", "TSLA", "NFLX"]),
    ]}])
    monkeypatch.setattr(alpaca.requests, "get", get)

    out = alpaca.get_news("AAPL", "2026-10-10", "2026-10-12")

    assert "Stocks Moving" not in out
    assert "Tagged: AAPL, MSFT, GOOG, AMZN, META, NVDA and 2 others." in out


def test_articles_about_the_ticker_outrank_newer_shared_coverage(monkeypatch):
    config_module._config["news_article_limit"] = 2
    get, _ = _stub([{"news": [
        _article("Chip stocks rally", "2026-10-12T15:00:00Z", symbols=("NVDA", "AMD", "AAPL")),
        _article("Megacaps mixed", "2026-10-12T14:00:00Z", symbols=("MSFT", "AAPL", "GOOG")),
        _article("AAPL and MSFT lead", "2026-10-11T14:00:00Z", symbols=("MSFT", "AAPL", "GOOG")),
        _article("Apple raises buyback", "2026-10-07T14:00:00Z"),
    ]}])
    monkeypatch.setattr(alpaca.requests, "get", get)

    out = alpaca.get_news("AAPL", "2026-10-06", "2026-10-12")

    assert "Chip stocks" not in out and "Megacaps" not in out
    # Shown newest first.
    assert out.index("AAPL and MSFT lead") < out.index("Apple raises buyback")


def test_no_articles_is_no_data(monkeypatch):
    get, _ = _stub([{"news": [], "next_page_token": None}])
    monkeypatch.setattr(alpaca.requests, "get", get)

    with pytest.raises(NoMarketDataError):
        alpaca.get_news("ZZZZ", "2026-10-10", "2026-10-12")


def test_missing_keys_is_not_configured(monkeypatch):
    monkeypatch.delenv("ALPACA_SECRET_KEY")
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    with pytest.raises(VendorNotConfiguredError):
        alpaca.get_news("AAPL", "2026-10-10", "2026-10-12")


@pytest.mark.parametrize("status, error", [(401, VendorNotConfiguredError), (429, VendorUnavailableError),
                                           (503, VendorUnavailableError)])
def test_http_errors_map_to_vendor_errors(monkeypatch, status, error):
    get, calls = _stub([], status=status)
    monkeypatch.setattr(alpaca.requests, "get", get)
    monkeypatch.setattr(alpaca, "_sleep", lambda seconds: None)

    with pytest.raises(error):
        alpaca.get_news("AAPL", "2026-10-10", "2026-10-12")
    # Throttling and server errors are retried before giving up; a rejected key is not.
    assert len(calls) == (1 if status == 401 else alpaca.RETRIES + 1)


def test_global_news_uses_index_etfs(monkeypatch):
    get, calls = _stub([{"news": [_article("Fed holds rates", "2026-10-12T18:00:00Z", symbols=("SPY", "TLT"))]}])
    monkeypatch.setattr(alpaca.requests, "get", get)

    out = alpaca.get_global_news("2026-10-12", look_back_days=3, limit=5)

    assert calls[0]["symbols"] == alpaca.MACRO_SYMBOLS
    assert "Fed holds rates" in out
    assert "Tagged:" not in out


@pytest.mark.parametrize("status", [401, 503])
def test_router_falls_back_to_yahoo(monkeypatch, status):
    get, _ = _stub([], status=status)
    monkeypatch.setattr(alpaca.requests, "get", get)
    monkeypatch.setattr(alpaca, "_sleep", lambda seconds: None)
    monkeypatch.setitem(router.VENDOR_METHODS["get_news"], "yfinance", lambda *a, **k: "yahoo news")

    assert router.route_to_vendor("get_news", "AAPL", "2026-10-10", "2026-10-12") == "yahoo news"


def test_router_falls_back_to_yahoo_when_benzinga_has_nothing(monkeypatch):
    get, _ = _stub([{"news": []}])
    monkeypatch.setattr(alpaca.requests, "get", get)
    monkeypatch.setitem(router.VENDOR_METHODS["get_news"], "yfinance", lambda *a, **k: "yahoo news")

    assert router.route_to_vendor("get_news", "AAPL", "2026-10-10", "2026-10-12") == "yahoo news"
