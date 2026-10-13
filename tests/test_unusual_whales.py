"""Unusual Whales vendor: auth and error mapping, point-in-time trimming,
output formatting, and router integration.

All API access is mocked, so these run without a network connection or a key.
"""
import copy
from unittest import mock

import pytest
import requests

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.agents.context import options_desk_reports
from tradingagents.dataflows import router
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import unusual_whales as uw
from tradingagents.graph.analyst_execution import build_analyst_execution_plan


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("UNUSUAL_WHALES_API_KEY", "test-token")
    monkeypatch.delenv("UW_API_KEY", raising=False)


@pytest.fixture
def _isolated_config(monkeypatch):
    monkeypatch.setattr(config_module, "_config", copy.deepcopy(default_config.DEFAULT_CONFIG))


def _response(status=200, body=None):
    response = mock.Mock(spec=requests.Response)
    response.status_code = status
    response.json.return_value = body if body is not None else {"data": []}
    response.text = "error"
    response.raise_for_status.return_value = None
    return response


def _stub(routes):
    """A requests.get replacement answering by the path after /api/."""
    calls = []

    def _get(url, params=None, headers=None, timeout=None):
        path = url.split("/api/", 1)[1]
        calls.append((path, params, headers))
        return _response(body={"data": routes[path]})
    return _get, calls


ALERTS = [
    {"created_at": "2026-10-08T15:01:00Z", "type": "call", "strike": "200", "expiry": "2026-11-20T00:00:00Z",
     "total_premium": "1500000", "total_ask_side_prem": "1400000", "total_bid_side_prem": "100000",
     "total_size": 900, "volume_oi_ratio": "2.5", "underlying_price": "190.1", "alert_rule": "RepeatedHits",
     "has_sweep": True, "has_floor": False, "has_multileg": False},
    {"created_at": "2026-10-09T14:00:00Z", "type": "put", "strike": "180", "expiry": "2026-10-31T00:00:00Z",
     "total_premium": "500000", "total_ask_side_prem": "300000", "total_bid_side_prem": "200000",
     "total_size": 400, "volume_oi_ratio": "0.8", "underlying_price": "191.0", "alert_rule": "FloorTradeLargeCap",
     "has_sweep": False, "has_floor": True, "has_multileg": False},
    # After the as-of date: must never be shown.
    {"created_at": "2026-10-12T14:00:00Z", "type": "call", "strike": "999", "expiry": "2026-12-18T00:00:00Z",
     "total_premium": "9000000", "total_ask_side_prem": "9000000", "total_bid_side_prem": "0",
     "total_size": 1, "volume_oi_ratio": "1", "underlying_price": "200", "alert_rule": "RepeatedHits"},
]


@pytest.mark.unit
class TestRequest:
    def test_missing_key_is_not_configured(self, monkeypatch):
        monkeypatch.delenv("UNUSUAL_WHALES_API_KEY")
        with pytest.raises(uw.UnusualWhalesNotConfiguredError):
            uw.get_options_flow("AAPL", "2026-10-10")

    def test_uw_api_key_alias_is_accepted(self, monkeypatch):
        monkeypatch.delenv("UNUSUAL_WHALES_API_KEY")
        monkeypatch.setenv("UW_API_KEY", "alias-token")
        assert uw.get_api_key() == "alias-token"

    def test_token_goes_in_a_bearer_header_not_the_url(self):
        get, calls = _stub({"option-trades/flow-alerts": []})
        with mock.patch.object(uw.requests, "get", side_effect=get):
            uw.get_options_flow("aapl", "2026-10-10")
        path, params, headers = calls[0]
        assert headers["Authorization"] == "Bearer test-token"
        assert "test-token" not in str(params)
        assert params["ticker_symbol"] == "AAPL"
        assert params["older_than"] == "2026-10-10T23:59:59Z"

    @pytest.mark.parametrize("status", [403, 429, 500, 503])
    def test_throttle_plan_and_server_errors_are_unavailable(self, status):
        with mock.patch.object(uw.requests, "get", return_value=_response(status)), pytest.raises(VendorUnavailableError):
            uw.get_market_tide("2026-10-10")

    def test_rejected_token_is_not_configured(self):
        with mock.patch.object(uw.requests, "get", return_value=_response(401)), pytest.raises(uw.UnusualWhalesNotConfiguredError):
            uw.get_market_tide("2026-10-10")

    def test_network_failure_is_unavailable(self):
        with mock.patch.object(uw.requests, "get", side_effect=requests.ConnectionError("down")), pytest.raises(VendorUnavailableError):
            uw.get_market_tide("2026-10-10")


@pytest.mark.unit
class TestOptionsFlow:
    def test_summarises_and_drops_alerts_after_the_as_of_date(self):
        get, _ = _stub({"option-trades/flow-alerts": ALERTS})
        with mock.patch.object(uw.requests, "get", side_effect=get):
            out = uw.get_options_flow("AAPL", "2026-10-10")
        assert "Alerts: 2 (1 call, 1 put)" in out
        assert "Call premium $1.50M (ask side $1.40M" in out
        assert "Call/put premium ratio: 3.00" in out
        assert "999" not in out and "$9.00M" not in out

    def test_a_full_page_is_followed_back_to_the_window_start(self):
        def alert(i, day):
            return {"id": f"a{i}", "created_at": f"{day}T15:{i % 60:02d}:00Z", "type": "call",
                    "total_premium": "1000", "all_opening_trades": i % 2 == 0}
        pages = [[alert(i, "2026-10-10") for i in range(uw.FLOW_PAGE_SIZE)],
                 [alert(i, "2026-10-08") for i in range(1000, 1010)]]
        calls = []

        def get(url, params=None, headers=None, timeout=None):
            calls.append(dict(params))
            return _response(body={"data": pages[len(calls) - 1]})
        with mock.patch.object(uw.requests, "get", side_effect=get):
            out = uw.get_options_flow("AAPL", "2026-10-10")
        assert len(calls) == 2
        assert calls[1]["older_than"] == "2026-10-10T15:00:00Z"
        assert f"Alerts: {uw.FLOW_PAGE_SIZE + 10} " in out
        assert "only of opening trades: 105 of 210" in out
        assert "cover only" not in out

    def test_no_alerts_is_not_read_as_no_activity(self):
        get, _ = _stub({"option-trades/flow-alerts": []})
        with mock.patch.object(uw.requests, "get", side_effect=get):
            out = uw.get_options_flow("AAPL", "2026-10-10")
        assert "No flow alerts" in out and "not no options activity" in out


@pytest.mark.unit
class TestOptionsActivity:
    def test_reports_latest_session_on_or_before_the_date_and_iv_rank(self):
        volume = [
            {"date": "2026-10-13", "call_volume": 1, "put_volume": 1},  # after as-of
            {"date": "2026-10-09", "call_volume": 200000, "put_volume": 100000,
             "avg_30_day_call_volume": "100000", "avg_30_day_put_volume": "100000",
             "bullish_premium": "5000000", "bearish_premium": "2000000",
             "net_call_premium": "1000000", "net_put_premium": "-500000",
             "call_open_interest": 10, "put_open_interest": 20},
        ]
        # As served live: the rank on a 0-100 scale, volatility as a fraction, a year of rows.
        iv = [{"date": "2025-10-10", "volatility": "0.50", "iv_rank_1y": "70.1"},
              {"date": "2026-07-09", "volatility": "0.30", "iv_rank_1y": "20.0"},
              {"date": "2026-10-09", "volatility": "0.45", "iv_rank_1y": "0.85"}]
        get, _ = _stub({"stock/AAPL/options-volume": volume, "stock/AAPL/iv-rank": iv})
        with mock.patch.object(uw.requests, "get", side_effect=get):
            out = uw.get_options_activity("AAPL", "2026-10-10")
        assert "Latest session (2026-10-09)" in out
        assert "2.0x its 30-day average" in out
        assert "put/call volume ratio 0.50" in out
        assert "IV 45.0%, 1-year IV rank 1 (" in out   # 0.85 on a 0-100 scale, not 85
        assert "on 2026-07-09 IV was 30.0%" in out     # the row nearest three months back
        assert "2026-10-13" not in out

    def test_nothing_at_all_is_no_data(self):
        get, _ = _stub({"stock/AAPL/options-volume": [], "stock/AAPL/iv-rank": []})
        with mock.patch.object(uw.requests, "get", side_effect=get), pytest.raises(NoMarketDataError):
            uw.get_options_activity("AAPL", "2026-10-10")


@pytest.mark.unit
def test_market_tide_reads_its_eastern_timestamps_as_eastern_running_totals():
    ticks = [
        {"timestamp": "2026-10-12T09:30:00-04:00", "net_call_premium": "-17790938.0", "net_put_premium": "9489911.0", "net_volume": -42218},
        {"timestamp": "2026-10-12T16:10:00-04:00", "net_call_premium": "-372698747.0", "net_put_premium": "-36382757.0", "net_volume": -249397},
    ]
    get, _ = _stub({"market/market-tide": ticks})
    with mock.patch.object(uw.requests, "get", side_effect=get):
        out = uw.get_market_tide("2026-10-12")
    assert "Session total through 16:10 ET:** net call premium -$372.70M" in out
    assert "| 09:30 | -$17.79M |" in out and "UTC" not in out


@pytest.mark.unit
class TestCalendars:
    def test_economic_calendar_lists_events_from_the_as_of_date(self):
        events = [
            {"event": "CPI", "time": "2026-10-14T12:30:00.000Z", "type": "report",
             "reported_period": "September", "forecast": "0.3%", "prev": "0.4%"},
            {"event": "Old event", "time": "2026-10-06T12:30:00.000Z", "type": "report"},
        ]
        get, _ = _stub({"market/economic-calendar": events})
        with mock.patch.object(uw.requests, "get", side_effect=get):
            out = uw.get_economic_calendar("2026-10-10", today="2026-10-10")
        assert "| CPI | September | 0.3% | 0.4% |" in out
        assert "Old event" not in out

    def test_economic_calendar_drops_the_duplicate_of_each_event(self):
        event = {"event": "Existing Home Sales", "time": "2026-10-13T14:00:00Z", "type": "report",
                 "reported_period": "September", "forecast": None, "prev": "4000000"}
        get, _ = _stub({"market/economic-calendar": [event, dict(event)]})
        with mock.patch.object(uw.requests, "get", side_effect=get):
            out = uw.get_economic_calendar("2026-10-12", today="2026-10-12")
        assert out.count("Existing Home Sales") == 1
        assert "| September |  | 4.00M |" in out

    def test_economic_calendar_is_unavailable_for_a_historical_date(self):
        with mock.patch.object(uw.requests, "get") as get:
            out = uw.get_economic_calendar("2025-01-10", today="2026-10-10")
        get.assert_not_called()
        assert "DATA_UNAVAILABLE" in out

    def test_earnings_hides_results_of_reports_after_the_as_of_date(self):
        reports = [
            {"report_date": "2026-07-30", "report_time": "postmarket", "actual_eps": "1.50",
             "street_mean_est": "1.40", "expected_move_perc": "0.04", "post_earnings_move_1d": "0.06",
             "long_straddle_1d": "0.30", "short_straddle_1d": "-0.30"},
            {"report_date": "2026-04-30", "report_time": "postmarket", "actual_eps": "1.20",
             "street_mean_est": "1.25", "expected_move_perc": "0.05", "post_earnings_move_1d": "-0.02",
             "long_straddle_1d": "-0.40", "short_straddle_1d": "0.40"},
            {"report_date": "2026-10-29", "report_time": "postmarket", "actual_eps": "9.99",
             "source": "company", "post_earnings_move_1d": "0.50"},
        ]
        get, _ = _stub({"earnings/AAPL": reports})
        with mock.patch.object(uw.requests, "get", side_effect=get):
            out = uw.get_earnings_calendar("AAPL", "2026-10-10")
        assert "**Next report:** 2026-10-29, postmarket (confirmed by the company), 19 calendar days" in out
        assert "9.99" not in out and "50.0%" not in out
        assert "average absolute 1-day move 4.0% versus average implied move 4.5%" in out


@pytest.mark.unit
class TestRouting:
    def test_options_categories_are_optional_and_degrade_without_a_key(self, monkeypatch, _isolated_config):
        monkeypatch.delenv("UNUSUAL_WHALES_API_KEY")
        out = router.route_to_vendor("get_options_flow", "AAPL", "2026-10-10", None, None)
        assert out.startswith("DATA_UNAVAILABLE: optional options_data")
        out = router.route_to_vendor("get_earnings_calendar", "AAPL", "2026-10-10")
        assert out.startswith("DATA_UNAVAILABLE: optional event_calendar")

    def test_default_config_routes_to_unusual_whales(self, _isolated_config):
        assert router.get_vendor("options_data") == "unusual_whales,tradier"
        assert router.get_vendor("event_calendar") == "unusual_whales"


@pytest.mark.unit
class TestDeskWiring:
    def test_new_analysts_are_selectable(self):
        plan = build_analyst_execution_plan(["options", "calendar"])
        assert [s.agent_node for s in plan.specs] == ["Options Flow Analyst", "Event Calendar Analyst"]
        assert [s.report_key for s in plan.specs] == ["options_report", "calendar_report"]

    def test_desk_reports_are_absent_when_those_analysts_did_not_run(self):
        assert options_desk_reports({"options_report": "", "calendar_report": ""}) == ""
        assert options_desk_reports({}) == ""

    def test_desk_reports_are_labelled_when_present(self):
        out = options_desk_reports({"options_report": "calls bid", "calendar_report": "CPI Tue"})
        assert out == "Options flow report: calls bid\nEarnings and economic calendar report: CPI Tue"

    def test_crypto_runs_drop_the_stock_only_analysts(self):
        from cli.models import AnalystType, AssetType
        from cli.prompts import filter_analysts_for_asset_type
        kept = filter_analysts_for_asset_type(list(AnalystType), AssetType.CRYPTO)
        assert AnalystType.OPTIONS not in kept and AnalystType.CALENDAR not in kept
