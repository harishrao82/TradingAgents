"""Conviction screen: ETF holdings, bulk news, the cheap screen, the strategist and the report.

Every vendor and model is faked; nothing reaches a network.
"""
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

import tradingagents.default_config as default_config
from tradingagents.conviction import screen as screen_mod, strategist as strategist_mod, workflow
from tradingagents.conviction.screen import NewsVerdict, screen_news, shortlist
from tradingagents.conviction.strategist import OptionLeg, OptionsIdea, contracts_for, max_gain
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors import alpaca, unusual_whales


def _response(body, status=200):
    response = mock.Mock(spec=requests.Response)
    response.status_code = status
    response.raise_for_status.return_value = None
    response.json.return_value = body
    return response


def test_etf_holdings_keep_stocks_heaviest_first(monkeypatch):
    monkeypatch.setenv("UNUSUAL_WHALES_API_KEY", "k")
    seen = {}
    rows = [{"ticker": "MSFT", "weight": "6.5"}, {"ticker": "AAPL", "weight": "7.1"},
            {"ticker": "BRK.B", "weight": "1.6"}, {"ticker": "", "weight": "0.2"},
            {"ticker": "USD CASH", "weight": "0.1"}]
    monkeypatch.setattr(unusual_whales.requests, "get",
                        lambda url, params=None, **k: seen.update(params) or _response({"data": rows}))

    assert unusual_whales.get_etf_holdings("spy") == [("AAPL", 7.1), ("MSFT", 6.5), ("BRK.B", 1.6)]
    assert seen["limit"] >= 600  # the endpoint returns only 250 rows by default


def test_etf_without_holdings_is_no_data(monkeypatch):
    monkeypatch.setenv("UNUSUAL_WHALES_API_KEY", "k")
    monkeypatch.setattr(unusual_whales.requests, "get", lambda *a, **k: _response({"data": []}))

    with pytest.raises(NoMarketDataError):
        unusual_whales.get_etf_holdings("SPY")


def _article(i, headline, symbols, created="2026-10-12T14:00:00Z"):
    return {"id": i, "headline": headline, "author": "Jane", "created_at": created, "summary": "",
            "url": "", "symbols": list(symbols)}


def test_bulk_news_batches_symbols_and_splits_by_ticker(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr(alpaca, "BULK_BATCH", 2)
    requested = []

    def get(url, params=None, headers=None, timeout=None):
        requested.append(params["symbols"])
        if params["symbols"] == "AAPL,MSFT":
            return _response({"news": [_article(1, "Apple and Microsoft sign cloud deal", ["AAPL", "MSFT"]),
                                       _article(2, "Apple recalls chargers", ["AAPL"])]})
        return _response({"news": [_article(3, "Nvidia guides higher", ["NVDA"])]})

    monkeypatch.setattr(alpaca.requests, "get", get)

    news = alpaca.get_news_by_symbol(["aapl", "MSFT", "NVDA"], "2026-10-10", "2026-10-12", per_symbol=5)

    assert requested == ["AAPL,MSFT", "NVDA"]
    assert set(news) == {"AAPL", "MSFT", "NVDA"}
    assert "Apple recalls chargers" in news["AAPL"] and "Apple recalls" not in news["MSFT"]
    assert "Tagged: MSFT, AAPL" in news["MSFT"]


class _ScreenLLM:
    """Answers the screen prompt from a ticker -> reply table."""

    def __init__(self, replies):
        self.replies = replies

    def invoke(self, prompt):
        ticker = prompt.split("Below are the Benzinga articles about ", 1)[1].split(" ", 1)[0]
        reply = self.replies[ticker]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply)


def test_screen_reads_json_and_records_failures():
    llm = _ScreenLLM({
        "AAPL": '```json\n{"direction": "up", "conviction": 5, "catalyst": "guidance raise", "reason": "r"}\n```',
        "MSFT": "I think it goes up",
        "NVDA": '{"direction": "sideways", "conviction": 3, "catalyst": "c", "reason": "r"}',
        "TSLA": RuntimeError("429 from provider"),
    })
    news = dict.fromkeys(("AAPL", "MSFT", "NVDA", "TSLA"), "news")

    pauses = []

    verdicts, failures = screen_news(llm, news, "2026-10-10", "2026-10-12", workers=2, sleep=pauses.append)

    assert [(v.ticker, v.direction, v.conviction) for v in verdicts] == [("AAPL", "up", 5)]
    assert set(failures) == {"MSFT", "NVDA", "TSLA"}
    assert failures["TSLA"] == "rate limited (429)"
    assert pauses == [4, 8, 16, 32]  # four retries with a growing pause, then it gives up


def _verdict(ticker, direction, conviction):
    return NewsVerdict(ticker=ticker, direction=direction, conviction=conviction, catalyst="c", reason="r")


def test_shortlist_keeps_strong_directional_calls_strongest_then_heaviest():
    verdicts = [_verdict("A", "up", 4), _verdict("B", "down", 5), _verdict("C", "none", 5),
                _verdict("D", "up", 3), _verdict("E", "down", 4)]

    picked = shortlist(verdicts, min_conviction=4, max_names=3, weights={"A": 1.0, "E": 2.0}, direction="up")

    assert [v.ticker for v in picked] == ["A"]

    picked = shortlist(verdicts, min_conviction=4, max_names=3, weights={"A": 1.0, "E": 2.0}, direction="down")

    assert [v.ticker for v in picked] == ["B", "E"]


def test_shortlist_saves_half_the_slots_for_each_side():
    ups = [_verdict(f"U{i}", "up", 5) for i in range(6)]
    downs = [_verdict(f"D{i}", "down", 4) for i in range(3)]

    picked = shortlist(ups + downs, min_conviction=4, max_names=4)

    # The weaker bearish calls still get two of the four slots, listed after the bullish ones.
    assert [v.ticker for v in picked] == ["U0", "U1", "D0", "D1"]


def test_shortlist_gives_unused_slots_to_the_other_side():
    ups = [_verdict(f"U{i}", "up", 5) for i in range(6)]

    picked = shortlist(ups + [_verdict("D0", "down", 4)], min_conviction=4, max_names=4)

    assert [v.ticker for v in picked] == ["U0", "U1", "U2", "D0"]


def _idea(max_loss=250.0, trade=True):
    return OptionsIdea(trade=trade, structure="bull call spread", net_premium=2.5, max_loss=max_loss, max_gain=250.0,
                       breakeven="102.50", rationale="why", exit_plan="exit", invalidation="wrong",
                       legs=[OptionLeg(action="buy", option_type="call", strike=100, expiration="2026-11-20"),
                             OptionLeg(action="sell", option_type="call", strike=105, expiration="2026-11-20")])


def test_contracts_fit_the_per_trade_risk():
    assert contracts_for(_idea(250.0), 100_000, 0.03) == 12
    assert contracts_for(_idea(250.0), None, 0.03) is None
    assert contracts_for(_idea(trade=False), 100_000, 0.03) is None
    assert contracts_for(_idea(5_000.0), 100_000, 0.03) == 0


class _StrategistLLM:
    def __init__(self, idea):
        self.idea, self.prompts = idea, []

    def with_structured_output(self, schema):
        outer = self

        class Bound:
            def invoke(self, prompt):
                outer.prompts.append(prompt)
                return outer.idea

        return Bound()

    def invoke(self, prompt):
        return SimpleNamespace(content="free text idea")


def test_full_screen_report(monkeypatch, tmp_path):
    monkeypatch.setattr(unusual_whales, "get_etf_holdings", lambda etf: [("AAPL", 7.0), ("MSFT", 6.0), ("XOM", 1.0)])
    monkeypatch.setattr(alpaca, "get_news_by_symbol",
                        lambda symbols, start, end, per: {"AAPL": "Apple news", "MSFT": "Microsoft news"})
    monkeypatch.setattr(strategist_mod, "gather",
                        lambda ticker, date: {"chain": "CHAIN", "activity": "IV rank 40", "flow": "FLOW"})
    screener = _ScreenLLM({
        "AAPL": '{"direction": "up", "conviction": 5, "catalyst": "guidance raise", "reason": "Raised guide."}',
        "MSFT": '{"direction": "none", "conviction": 2, "catalyst": "recap", "reason": "Routine."}',
    })
    strategist = _StrategistLLM(_idea())
    config = {**default_config.DEFAULT_CONFIG, "results_dir": str(tmp_path)}

    result = workflow.run_conviction_screen(config, "2026-10-12", equity=100_000, llms=(screener, strategist))
    text = workflow.render_report(result, 100_000, 0.03)
    path = workflow.save_report(result, text, str(tmp_path))

    assert result.universe == 3 and result.with_news == 2 and result.start == "2026-10-08"
    assert [v.ticker for v in result.shortlist] == ["AAPL"]
    assert "CHAIN" in strategist.prompts[0] and "Apple news" in strategist.prompts[0]
    assert "## AAPL: bullish (conviction 5/5)" in text
    assert "bull call spread, 2.50 debit" in text
    assert "| buy | 1 | call | 100 | 2026-11-20 |" in text
    assert "12 unit(s)" in text
    assert "no orders were placed" in text
    assert path.name.startswith("SPY_2026-10-12_") and path.read_text() == text


def test_screen_only_skips_the_strategist(monkeypatch):
    monkeypatch.setattr(alpaca, "get_news_by_symbol", lambda symbols, start, end, per: {"AAPL": "Apple news"})
    screener = _ScreenLLM({"AAPL": '{"direction": "down", "conviction": 4, "catalyst": "probe", "reason": "r"}'})
    strategist = mock.Mock()

    result = workflow.run_conviction_screen(dict(default_config.DEFAULT_CONFIG), "2026-10-12", tickers=["aapl"],
                                            screen_only=True, llms=(screener, strategist))
    text = workflow.render_report(result)

    strategist.assert_not_called()
    assert result.etf == "custom list"
    assert "Short list: AAPL (down 4)" in text


def test_screen_prompt_mentions_the_window():
    prompt = screen_mod.SCREEN_PROMPT.format(ticker="AAPL", start="2026-10-10", end="2026-10-12", news="x")
    assert "from 2026-10-10 to 2026-10-12" in prompt


def test_news_window_counts_weekdays():
    assert workflow._weekdays_back("2026-10-12", 3) == "2026-10-08"  # Monday reaches back to Thursday
    assert workflow._weekdays_back("2026-10-14", 3) == "2026-10-12"
    assert workflow._weekdays_back("2026-10-14", 1) == "2026-10-14"


class _Flaky:
    """Rate-limited on the first call, then answers."""

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self.calls == 1:
            error = RuntimeError('Error code: 429 - {"error": {"metadata": {"user_id": "user_secret"}}}')
            error.status_code = 429
            raise error
        return SimpleNamespace(content='{"direction": "up", "conviction": 4, "catalyst": "c", "reason": "r"}')


def test_rate_limited_call_is_retried():
    llm = _Flaky()
    verdicts, failures = screen_news(llm, {"AAPL": "news"}, "2026-10-10", "2026-10-12", workers=1, sleep=lambda s: None)

    assert llm.calls == 2 and not failures and verdicts[0].conviction == 4


def test_failure_reason_is_one_short_line_without_the_error_body():
    error = RuntimeError('Error code: 400 - {"error": {"metadata": {"user_id": "user_secret"}}}')
    error.status_code = 400

    reason = screen_mod._short(error)

    assert "user_secret" not in reason and "{" not in reason
    assert reason == "RuntimeError 400: Error code: 400 -"


def test_throttle_spaces_call_starts():
    now, slept = [100.0], []
    throttle = screen_mod.Throttle(30, clock=lambda: now[0], sleep=slept.append)

    for _ in range(3):
        throttle.wait()

    assert slept == [2.0, 4.0]  # 30 a minute: one start every 2 seconds
    assert screen_mod.Throttle(0, sleep=slept.append).wait() is None and len(slept) == 2


def _puts(*legs, premium):
    return OptionsIdea(trade=True, structure="s", net_premium=premium, max_loss=1.0, rationale="r",
                       legs=[OptionLeg(action=a, option_type="put", strike=k, expiration="2026-11-20") for a, k in legs])


def test_put_ideas_have_a_capped_max_gain():
    assert max_gain(_puts(("buy", 940), premium=55.90)) == 88_410.0  # a long put is worth the most at zero
    assert max_gain(_puts(("buy", 225), ("sell", 215), premium=3.45)) == 655.0
    long_call = OptionsIdea(trade=True, structure="long call", net_premium=5.0, max_loss=500.0, rationale="r",
                            legs=[OptionLeg(action="buy", option_type="call", strike=100, expiration="2026-11-20")])
    assert max_gain(long_call) is None
    assert max_gain(_idea()) == 250.0  # the model's own figure is kept
