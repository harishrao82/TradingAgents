"""Tradier vendor: live-only option chains, one-item collapsing, IV hygiene and routing.

Response bodies follow the shapes Tradier served on a live read-only run
(AAPL, 2026-10-12). All API access is mocked.
"""
import copy
from unittest import mock

import pytest
import requests

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import router
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import tradier

TODAY = "2026-10-12"


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("TRADIER_ACCESS_TOKEN", "test-token")
    monkeypatch.delenv("TRADIER_SANDBOX", raising=False)
    monkeypatch.setattr(tradier, "is_historical", lambda date: date < TODAY)


def _option(kind, strike, bid, ask, delta, mid_iv, oi, expiration="2026-11-20"):
    return {"symbol": f"AAPL{kind}{strike}", "option_type": kind, "strike": strike, "bid": bid, "ask": ask,
            "open_interest": oi, "expiration_date": expiration, "expiration_type": "standard",
            "greeks": {"delta": delta, "mid_iv": mid_iv, "smv_vol": 0.25}}


CHAIN = [
    _option("call", 330.0, 15.0, 15.4, 0.70, 0.24, 500),
    _option("put", 330.0, 4.0, 4.2, -0.30, 0.26, 800),
    _option("call", 340.0, 8.0, 8.4, 0.52, 0.22, 900),
    _option("put", 340.0, 7.0, 7.4, -0.48, 0.23, 1000),
    _option("call", 350.0, 3.9, 4.1, 0.25, 0.20, 700),
    _option("put", 350.0, 13.0, 13.4, -0.75, 0.21, 300),
    # Deep out of the money with no bid: its 0.998 mid IV is junk.
    _option("put", 260.0, 0.0, 2.16, -0.001, 0.998, 0),
]


def _stub(chains, expirations=("2026-10-14", "2026-11-20", "2027-01-15"), quote=None):
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        path = url.split("/v1/", 1)[1]
        calls.append((url, path, params, headers))
        response = mock.Mock(spec=requests.Response)
        response.status_code = 200
        response.raise_for_status.return_value = None
        if path == "markets/quotes":
            body = {"quotes": {"quote": quote or {"symbol": "AAPL", "last": 341.0}}}
        elif path == "markets/options/expirations":
            # One date comes back as a bare string, not a list.
            dates = list(expirations)
            body = {"expirations": {"date": dates[0] if len(dates) == 1 else dates}}
        else:
            option = chains.get(params["expiration"], [])
            body = {"options": {"option": option[0] if len(option) == 1 else option} if option else None}
        response.json.return_value = body
        return response
    return get, calls


@pytest.mark.unit
class TestOptionChain:
    def test_summarises_the_expiries_in_the_window(self):
        get, calls = _stub({"2026-11-20": CHAIN})
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            out = tradier.get_option_chain("aapl", TODAY)
        # Only 2026-11-20 is 14-60 days out; the weekly and the January are not.
        assert [p["expiration"] for _, path, p, _ in calls if path == "markets/options/chains"] == ["2026-11-20"]
        assert "### 2026-11-20 (39 days, standard)" in out
        assert "ATM strike 340.0; ATM IV 22.5%" in out
        assert "ATM straddle 15.40, an implied move of about ±4.5% by expiry" in out
        assert "25-delta skew (put IV minus call IV) +6.0 points" in out
        assert "| 340.0 (ATM) | 22.5% | 8.00/8.40 | 0.52 | 900 | 7.00/7.40 | -0.48 | 1000 |" in out
        # Below the money the put's IV is read, above it the call's.
        assert "| 330.0 | 26.0% |" in out and "| 350.0 | 20.0% |" in out

    def test_a_contract_with_no_bid_shows_no_iv(self):
        get, _ = _stub({"2026-11-20": CHAIN})
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            out = tradier.get_option_chain("AAPL", TODAY)
        assert "99.8%" not in out

    def test_single_items_and_empty_chains_are_handled(self):
        get, _ = _stub({"2026-11-20": [CHAIN[2]]}, expirations=("2026-11-20",))
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            out = tradier.get_option_chain("AAPL", TODAY)
        assert "ATM strike 340.0" in out
        get, _ = _stub({}, expirations=("2026-11-20",))
        with mock.patch.object(tradier.requests, "get", side_effect=get), pytest.raises(NoMarketDataError):
            tradier.get_option_chain("AAPL", TODAY)

    def test_an_expiry_with_no_open_interest_is_skipped(self):
        fresh = [dict(o, open_interest=0, expiration_type="weeklys") for o in CHAIN]
        get, _ = _stub({"2026-10-26": fresh, "2026-11-20": CHAIN},
                       expirations=("2026-10-26", "2026-11-20"))
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            out = tradier.get_option_chain("AAPL", TODAY)
        assert "Skipped for having no open interest yet: 2026-10-26." in out
        assert "### 2026-10-26" not in out and "### 2026-11-20" in out
        get, _ = _stub({"2026-11-20": fresh}, expirations=("2026-11-20",))
        with mock.patch.object(tradier.requests, "get", side_effect=get), pytest.raises(NoMarketDataError):
            tradier.get_option_chain("AAPL", TODAY)

    def test_expiries_spread_across_the_window_with_the_monthly(self):
        # Eight dailies 7-30 days out: show the nearest, the month-end and the furthest.
        dates = ("2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23",
                 "2026-10-26", "2026-10-30", "2026-11-06")
        monthly = ("2026-10-16",)  # third Friday of October falls before the window
        get, calls = _stub(dict.fromkeys(dates, CHAIN), expirations=monthly + dates)
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            out = tradier.get_option_chain("SPY", TODAY, 7, 30)
        fetched = [p["expiration"] for _, path, p, _ in calls if path == "markets/options/chains"]
        assert fetched == ["2026-10-19", "2026-10-30", "2026-11-06"]
        assert out.index("### 2026-10-19") < out.index("### 2026-10-30") < out.index("### 2026-11-06")

    def test_a_skipped_expiry_is_replaced_by_the_next_closest_to_the_middle(self):
        dates = ("2026-10-19", "2026-10-23", "2026-10-26", "2026-10-30", "2026-11-06")
        chains = dict.fromkeys(dates, CHAIN)
        chains["2026-10-30"] = [dict(o, open_interest=0) for o in CHAIN]
        get, calls = _stub(chains, expirations=dates)
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            tradier.get_option_chain("SPY", TODAY, 7, 30)
        fetched = [p["expiration"] for _, path, p, _ in calls if path == "markets/options/chains"]
        assert fetched == ["2026-10-19", "2026-10-30", "2026-11-06", "2026-10-26"]

    def test_a_monthly_in_the_window_is_preferred_over_adjacent_dates(self):
        dates = ("2026-10-26", "2026-10-30", "2026-11-06", "2026-11-20", "2026-12-04")
        get, calls = _stub(dict.fromkeys(dates, CHAIN), expirations=dates)
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            tradier.get_option_chain("AAPL", TODAY)
        fetched = [p["expiration"] for _, path, p, _ in calls if path == "markets/options/chains"]
        assert fetched == ["2026-10-26", "2026-11-20", "2026-12-04"]

    def test_expiries_after_the_earnings_report_are_tagged(self):
        dates = ("2026-10-30", "2026-11-06", "2026-11-20")
        get, _ = _stub(dict.fromkeys(dates, CHAIN), expirations=dates)
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            out = tradier.get_option_chain("AAPL", TODAY, earnings_date="2026-10-30", earnings_time="postmarket")
        # A report after the close on expiry day comes too late for that expiry.
        assert "### 2026-10-30 (18 days, standard)" in out
        assert "### 2026-11-06 (25 days, standard, spans earnings 2026-10-30)" in out
        assert "### 2026-11-20 (39 days, standard, spans earnings 2026-10-30)" in out
        assert "Next earnings 2026-10-30 (postmarket)" in out and "2026-11-06, 2026-11-20." in out
        assert "No earnings date was available" not in out

    def test_a_premarket_report_on_expiry_day_is_spanned(self):
        assert tradier._spans_earnings("2026-10-30", "2026-10-30", "premarket")
        assert not tradier._spans_earnings("2026-10-30", "2026-10-30", "postmarket")
        assert not tradier._spans_earnings("2026-10-23", "2026-10-30", "premarket")

    def test_without_an_earnings_date_the_footer_says_so(self):
        get, _ = _stub({"2026-11-20": CHAIN})
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            out = tradier.get_option_chain("AAPL", TODAY)
        assert "spans earnings" not in out and "No earnings date was available" in out

    def test_wide_quotes_are_marked(self):
        wide = CHAIN + [_option("call", 360.0, 0.80, 1.46, 0.17, 0.199, 50)]
        get, _ = _stub({"2026-11-20": wide})
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            out = tradier.get_option_chain("AAPL", TODAY)
        assert "0.80/1.46 (wide)" in out and "8.00/8.40 |" in out

    def test_a_historical_date_is_not_served_todays_chain(self):
        with mock.patch.object(tradier.requests, "get") as get:
            out = tradier.get_option_chain("AAPL", "2026-01-09")
        get.assert_not_called()
        assert "DATA_UNAVAILABLE" in out

    def test_unknown_symbol_is_no_data(self):
        def get(url, params=None, headers=None, timeout=None):
            response = mock.Mock(spec=requests.Response, status_code=200)
            response.json.return_value = {"quotes": {"unmatched_symbols": {"symbol": "ZZZZ"}}}
            return response
        with mock.patch.object(tradier.requests, "get", side_effect=get), pytest.raises(NoMarketDataError):
            tradier.get_option_chain("ZZZZ", TODAY)


@pytest.mark.unit
class TestRequest:
    def test_token_in_header_and_sandbox_switch(self, monkeypatch):
        get, calls = _stub({"2026-11-20": CHAIN})
        monkeypatch.setenv("TRADIER_SANDBOX", "true")
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            tradier.get_option_chain("AAPL", TODAY)
        url, _, params, headers = calls[0]
        assert url.startswith("https://sandbox.tradier.com/v1/")
        assert headers["Authorization"] == "Bearer test-token"
        assert "test-token" not in str(params)

    def test_only_market_data_paths_are_called(self):
        get, calls = _stub({"2026-11-20": CHAIN})
        with mock.patch.object(tradier.requests, "get", side_effect=get):
            tradier.get_option_chain("AAPL", TODAY)
        assert all(path.startswith("markets/") for _, path, _, _ in calls)

    @pytest.mark.parametrize("status", [429, 500, 503])
    def test_throttle_and_server_errors_are_unavailable(self, status):
        response = mock.Mock(spec=requests.Response, status_code=status)
        with mock.patch.object(tradier.requests, "get", return_value=response), pytest.raises(VendorUnavailableError):
            tradier.get_option_chain("AAPL", TODAY)

    def test_missing_or_rejected_token_is_not_configured(self, monkeypatch):
        response = mock.Mock(spec=requests.Response, status_code=401)
        with mock.patch.object(tradier.requests, "get", return_value=response), \
                pytest.raises(tradier.TradierNotConfiguredError):
            tradier.get_option_chain("AAPL", TODAY)
        monkeypatch.delenv("TRADIER_ACCESS_TOKEN")
        with pytest.raises(tradier.TradierNotConfiguredError):
            tradier.get_option_chain("AAPL", TODAY)


@pytest.mark.unit
def test_routing_sends_chains_to_tradier_and_degrades_without_a_token(monkeypatch):
    monkeypatch.setattr(config_module, "_config", copy.deepcopy(default_config.DEFAULT_CONFIG))
    get, _ = _stub({"2026-11-20": CHAIN})
    with mock.patch.object(tradier.requests, "get", side_effect=get):
        assert "ATM strike" in router.route_to_vendor("get_option_chain", "AAPL", TODAY, None, None)
    monkeypatch.delenv("TRADIER_ACCESS_TOKEN")
    out = router.route_to_vendor("get_option_chain", "AAPL", TODAY, None, None)
    assert out.startswith("DATA_UNAVAILABLE: optional options_data")


@pytest.mark.unit
def test_the_chain_tool_passes_the_next_earnings_date(monkeypatch):
    from tradingagents.agents import tools
    calls = []

    def route(method, *args):
        calls.append((method, args))
        return ("2026-10-30", "postmarket") if method == "get_next_earnings" else "chain"
    monkeypatch.setattr(tools, "route_to_vendor", route)
    tools.get_option_chain.func("AAPL", TODAY, None, None, TODAY)
    assert calls[-1] == ("get_option_chain", ("AAPL", TODAY, None, None, "2026-10-30", "postmarket"))

    # A calendar vendor that is unavailable answers with a string: no date is passed.
    monkeypatch.setattr(tools, "route_to_vendor",
                        lambda method, *args: calls.append((method, args)) or "DATA_UNAVAILABLE: x")
    tools.get_option_chain.func("AAPL", TODAY, None, None, TODAY)
    assert calls[-1] == ("get_option_chain", ("AAPL", TODAY, None, None, None, None))


def test_read_timeout_is_retried_once(monkeypatch):
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        if len(calls) == 1:
            raise requests.Timeout("read timed out")
        response = mock.Mock(spec=requests.Response)
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {"ok": True}
        return response

    monkeypatch.setattr(tradier.requests, "get", get)

    assert tradier._request("markets/quotes", {}) == {"ok": True}
    assert len(calls) == 2


def test_second_timeout_is_unavailable(monkeypatch):
    def get(*a, **k):
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(tradier.requests, "get", get)

    with pytest.raises(VendorUnavailableError):
        tradier._request("markets/quotes", {})
