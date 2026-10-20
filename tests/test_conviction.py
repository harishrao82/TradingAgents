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
            {"ticker": "BRK.B", "weight": "1.6", "sector": "Financial Services"}, {"ticker": "", "weight": "0.2"},
            {"ticker": "USD CASH", "weight": "0.1"}, {"ticker": "BFB", "weight": "0.01"}]
    monkeypatch.setattr(unusual_whales.requests, "get",
                        lambda url, params=None, **k: seen.update(params) or _response({"data": rows}))

    assert unusual_whales.get_etf_holdings("spy") == [("AAPL", 7.1), ("MSFT", 6.5), ("BRK.B", 1.6), ("BF.B", 0.01)]
    assert unusual_whales.get_etf_constituents("spy")[2] == ("BRK.B", 1.6, "Financial Services")
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


def test_class_shares_are_queried_by_root_and_matched_on_slash_tags(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    requested = []

    def get(url, params=None, headers=None, timeout=None):
        requested.append(params["symbols"])
        return _response({"news": [_article(1, "Berkshire buys OxyChem", ["BRK", "BRK/A", "BRK/B", "OXY"])]})

    monkeypatch.setattr(alpaca.requests, "get", get)

    news = alpaca.get_news_by_symbol(["BRK.B", "OXY"], "2026-10-10", "2026-10-12", per_symbol=5)

    assert requested == ["BRK,OXY"]
    assert "Berkshire buys OxyChem" in news["BRK.B"] and "Berkshire buys OxyChem" in news["OXY"]


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
    assert contracts_for(_idea(46.0), 200_000, 0.03, max_contracts=40) == 40  # not 130 one-dollar spreads


def test_total_risk_cap_cuts_the_ideas_that_come_last():
    ideas = [workflow.Idea(_verdict(t, "up", c), _idea(200.0), 30) for t, c in (("A", 4), ("B", 5), ("C", 4))]

    workflow.cap_total_risk(ideas, 15_000, lambda item: -item.verdict.conviction)

    # B is strongest and keeps its $6,000; A comes next and keeps $6,000; C gets the $3,000 left.
    assert [item.contracts for item in ideas] == [30, 30, 15]
    assert ideas[2].sized_down.startswith("cut from 30") and not ideas[0].sized_down


def test_market_open_only_after_half_past_nine_new_york_time_today():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    assert workflow._market_open("2026-10-20", datetime(2026, 10, 20, 9, 50, tzinfo=ny))
    assert not workflow._market_open("2026-10-20", datetime(2026, 10, 20, 9, 10, tzinfo=ny))
    assert not workflow._market_open("2026-10-19", datetime(2026, 10, 20, 9, 50, tzinfo=ny))


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
    monkeypatch.setattr(unusual_whales, "get_etf_constituents",
                        lambda etf: [("AAPL", 7.0, "Technology"), ("MSFT", 6.0, "Technology"), ("XOM", 1.0, "Energy")])
    monkeypatch.setattr(alpaca, "get_news_by_symbol",
                        lambda symbols, start, end, per: {"AAPL": "Apple news", "MSFT": "Microsoft news"})
    monkeypatch.setattr(strategist_mod, "gather",
                        lambda ticker, date, *dte: {"chain": "CHAIN", "activity": "IV rank 40", "flow": "FLOW"})
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
    assert "at most 40 units, so choose strikes whose max loss is at least $75 a unit" in strategist.prompts[0]
    assert "**Total at risk:** $3,000 across 1 idea(s), 3.0% of $100,000." in text
    assert "no orders were placed" in text
    assert "## SPY view: bullish (+7.00)" in text
    assert "MSFT 6.0% (news points neither way)" in text and "XOM 1.0% (no news)" in text
    assert "| Technology | 13.0% | +7.00 | 1 | 0 |" in text
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
    assert "view:" not in text  # a custom list has no weights


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


class _FakeBatches:
    """Stands in for client.messages.batches: ends after one poll, then serves canned results."""

    def __init__(self, answers):
        self.answers, self.created, self.polls = answers, None, 0

    def create(self, requests):
        self.created = requests
        return SimpleNamespace(id="b1", processing_status="in_progress")

    def retrieve(self, batch_id):
        self.polls += 1
        return SimpleNamespace(id=batch_id, processing_status="ended",
                               request_counts=SimpleNamespace(succeeded=len(self.created), errored=0))

    def results(self, batch_id):
        for request in self.created:
            ticker = request["params"]["messages"][0]["content"].split("articles about ", 1)[1].split(" ", 1)[0]
            kind, text, stop = self.answers[ticker]
            message = SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="thinking"),
                                                                 SimpleNamespace(type="text", text=text)])
            yield SimpleNamespace(custom_id=request["custom_id"], result=SimpleNamespace(type=kind, message=message))


def _batch_client(answers):
    return SimpleNamespace(messages=SimpleNamespace(batches=_FakeBatches(answers)))


def test_batch_screen_parses_results_and_records_failures():
    from tradingagents.conviction.batch import screen_news_batch

    ok = '{"direction": "down", "conviction": 5, "catalyst": "guidance cut", "reason": "r"}'
    client = _batch_client({"BRK.B": ("succeeded", ok, "end_turn"), "PEP": ("succeeded", ok, "end_turn"),
                            "T": ("succeeded", '{"direction": "up"', "max_tokens"), "KR": ("errored", "", None)})
    news = dict.fromkeys(("BRK.B", "PEP", "T", "KR"), "news")

    verdicts, failures = screen_news_batch(client, "claude-haiku-5-5", news, "2026-10-09", "2026-10-13",
                                           sleep=lambda s: None)

    batches = client.messages.batches
    assert [r["custom_id"] for r in batches.created] == ["n0", "n1", "n2", "n3"]  # BRK.B's dot isn't allowed in ids
    assert batches.created[0]["params"]["model"] == "claude-haiku-5-5" and batches.polls == 1
    assert sorted(v.ticker for v in verdicts) == ["BRK.B", "PEP"]
    assert failures == {"T": "answer cut off at the token limit", "KR": "batch request errored"}


def test_screen_with_batch_option(monkeypatch):
    monkeypatch.setattr(alpaca, "get_news_by_symbol", lambda symbols, start, end, per: {"PEP": "Pepsi news"})
    client = _batch_client({"PEP": ("succeeded", '{"direction": "down", "conviction": 5, "catalyst": "c", '
                                                 '"reason": "r"}', "end_turn")})
    monkeypatch.setattr(workflow.screen_news_batch.__globals__["time"], "sleep", lambda s: None)
    config = {**default_config.DEFAULT_CONFIG, "screen_batch": True, "screen_llm_provider": "anthropic"}

    result = workflow.run_conviction_screen(config, "2026-10-13", tickers=["PEP"], screen_only=True,
                                            llms=(client, None))

    assert [(v.ticker, v.direction) for v in result.shortlist] == [("PEP", "down")]


def test_batch_needs_an_anthropic_screener(monkeypatch):
    monkeypatch.setattr(alpaca, "get_news_by_symbol", lambda symbols, start, end, per: {"PEP": "news"})
    config = {**default_config.DEFAULT_CONFIG, "screen_batch": True, "screen_llm_provider": "openrouter"}

    with pytest.raises(ValueError, match="Batches API"):
        workflow.run_conviction_screen(config, "2026-10-13", tickers=["PEP"], screen_only=True, llms=(object(), None))


def test_index_view_weights_calls_and_flags_concentration():
    from tradingagents.conviction import index_view

    def v(t, d, c):
        return screen_mod.NewsVerdict(ticker=t, direction=d, conviction=c, catalyst=f"{t} news", reason="r")

    holdings = {"AAPL": (7.3, "Technology"), "MSFT": (5.9, "Technology"), "JPM": (1.32, "Financial Services"),
                "JNJ": (0.94, "Healthcare"), "NVDA": (8.2, "Technology"), "GS": (0.8, "Financial Services")}
    verdicts = [v("AAPL", "up", 3), v("MSFT", "up", 3), v("JPM", "down", 4), v("JNJ", "up", 5), v("GS", "none", 2)]
    text = index_view.render_index_view("SPY", verdicts, {}, holdings, {"AAPL", "MSFT", "JPM", "JNJ", "GS"})

    assert "## SPY view: neutral (-0.12)" in text  # 0.94 - 1.056
    assert "| All | +7.80 | bullish |" in text
    assert "**Concentrated:** AAPL (3/5) and MSFT (3/5) make up 101% of the all-calls tilt." in text
    assert "NVDA 8.2% (no news)" in text and "GS 0.8% (news points neither way)" in text
    assert "| JPM | 1.32% | down 4 | -1.06 | JPM news |" in text
    assert [index_view.label(t) for t in (4, 2, 0, -2, -5)] == [
        "bullish", "mildly bullish", "neutral", "mildly bearish", "bearish"]


def _opt(kind, strike, bid, ask, oi=0, gamma=0.0):
    return {"option_type": kind, "strike": strike, "bid": bid, "ask": ask, "open_interest": oi,
            "greeks": {"gamma": gamma}}


def test_levels_zones_from_bars_chains_max_pain_and_gamma():
    from tradingagents.conviction import levels as lv

    days = [{"date": "2026-10-12", "high": 778.75, "low": 770, "close": 775},
            {"date": "2026-10-16", "high": 772, "low": 760.64, "close": 765},
            {"date": "2026-10-19", "high": 770.46, "low": 765.82, "close": 766.48}]
    today = [_opt("call", 764, 1.8, 1.9, 1000), _opt("put", 764, 1.8, 1.9, 1000),
             _opt("call", 775, 0.1, 0.2, 50_000), _opt("put", 751, 0.1, 0.2, 40_000)]
    friday = [_opt("call", 764, 4.7, 4.8, 100), _opt("put", 764, 4.7, 4.8, 100),
              _opt("call", 780, 0.5, 0.6, 90_000), _opt("put", 745, 0.5, 0.6, 70_000)]
    gamma = {750: -5e9, 760: -4e9, 765: -1e9, 770: 6e9, 780: 8e9}
    got = lv.compute("SPY", "2026-10-20", {"bid": 764.1, "ask": 764.3, "last": 766.48, "prevclose": 766.48},
                     days, [{"high": 767, "low": 763}], {"2026-10-20": today, "2026-10-23": friday},
                     vix={"bid": 18.9, "ask": 19.1, "prevclose": 18.45}, max_pain={"2026-10-20": 770, "2026-10-23": 770},
                     gamma=gamma)
    rows = {label: price for price, label in got.rows}

    assert got.spot == pytest.approx(764.2) and got.spot_source == "bid/ask mid"
    assert rows["pivot"] == pytest.approx((770.46 + 765.82 + 766.48) / 3)
    assert rows["last week's high"] == 778.75 and rows["last week's low"] == 760.64  # the week of Oct 12-16
    assert rows["top of today's implied range"] == pytest.approx(764.2 + 3.7)
    assert rows["bottom of week's implied range"] == pytest.approx(764.2 - 9.5)
    assert rows["today's call wall"] == 775 and rows["week's put wall"] == 745
    assert rows["gamma flip"] == 770 and rows["biggest negative gamma"] == 750
    text = lv.render(got)
    assert "| 770.00-772.23 | R2, prior high, today's max pain, week's max pain, gamma flip |" in text
    assert text.index("today's call wall") < text.index("**now**") < text.index("week's put wall")
    assert "VIX 19.00 (+3.0%)." in text and "below the flip, in negative gamma" in text


def test_reaction_classifies_gap_against_the_news():
    from tradingagents.conviction import reaction

    assert reaction.classify("up", 0.018) == "agree"
    assert reaction.classify("up", -0.039) == "disagree"
    assert reaction.classify("down", 0.091) == "disagree"
    assert reaction.classify("down", -0.001) == "flat"
    assert reaction.classify("up", None) == "unknown"
    assert reaction.Reaction(gap=0.09, spread=0.03).unreliable


def test_day_mode_ranks_agreeing_names_first_and_asks_for_short_expiries(monkeypatch, tmp_path):
    from tradingagents.conviction import reaction

    monkeypatch.setattr(alpaca, "get_news_by_symbol", lambda symbols, start, end, per: {t: f"{t} news" for t in symbols})
    seen = {}
    monkeypatch.setattr(strategist_mod, "gather",
                        lambda ticker, date, lo, hi: seen.update({ticker: (lo, hi)}) or
                        {"chain": "CHAIN", "activity": "", "flow": ""})
    gaps = {"GE": -0.039, "RTX": 0.018, "NOC": 0.008}
    expiry = {"GE": "2026-10-16", "RTX": "2026-10-16", "NOC": "2026-11-20"}  # NOC lists no weekly
    monkeypatch.setattr(reaction, "check", lambda v, date: reaction.Reaction(
        gaps[v.ticker], 0.001, 0.05, reaction.classify(v.direction, gaps[v.ticker]), nearest_expiry=expiry[v.ticker]))
    screener = _ScreenLLM({
        "GE": '{"direction": "up", "conviction": 5, "catalyst": "Fresh: beat and raise", "reason": "r"}',
        "RTX": '{"direction": "up", "conviction": 4, "catalyst": "Fresh: beat and raise", "reason": "r"}',
        "NOC": '{"direction": "up", "conviction": 5, "catalyst": "Fresh: beat and raise", "reason": "r"}',
    })
    strategist = _StrategistLLM(_idea())
    config = {**default_config.DEFAULT_CONFIG, "screen_mode": "day", "screen_max_names": 1}

    result = workflow.run_conviction_screen(config, "2026-10-12", tickers=["GE", "RTX", "NOC"], llms=(screener, strategist))
    text = workflow.render_report(result)

    assert result.start == "2026-10-09"  # two weekdays back from Monday
    # RTX agrees with its gap, so it beats GE's 5/5; NOC agrees too but has no expiry within the week.
    assert [v.ticker for v in result.shortlist] == ["RTX"]
    assert "| NOC | up 5 | +0.8% | ±5.0% | agree; no options before 2026-11-20 |" in workflow.render_report(result)
    assert seen == {"RTX": (0, 7)} and "one- to five-day moves" in strategist.prompts[0]
    assert "# custom list pre-market day-trade screen" in text
    assert "| GE | up 5 | -3.9% | ±5.0% | disagree | Fresh: beat and raise |" in text
    assert result.skipped == ["GE"] and "No idea for names moving against their news: GE." in text
    result.market_open = True
    after_open = workflow.render_report(result)
    assert "## News against the move since the prior close" in after_open and "| Since close |" in after_open
    assert "Run after the open" in after_open
    assert workflow.save_report(result, text, str(tmp_path)).name.endswith("_day.md")


def test_unusual_whales_retries_a_dropped_connection_once(monkeypatch):
    monkeypatch.setenv("UNUSUAL_WHALES_API_KEY", "k")
    monkeypatch.setattr(unusual_whales, "_sleep", lambda s: None)
    calls = []

    def get(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise unusual_whales.requests.ConnectionError("Connection reset by peer")
        return _response({"data": [{"expiry": "2026-10-23", "max_pain": "770"}]})

    monkeypatch.setattr(unusual_whales.requests, "get", get)

    assert unusual_whales.get_max_pain("SPY") == {"2026-10-23": 770.0} and len(calls) == 2


def test_replay_scores_the_short_list_with_news_before_the_open(monkeypatch):
    from tradingagents.conviction import replay

    monkeypatch.setattr(unusual_whales, "get_etf_constituents",
                        lambda etf: [("GE", 1.0, "Industrials"), ("BAC", 1.2, "Financials"), ("XOM", 0.9, "Energy")])
    cutoffs = []
    monkeypatch.setattr(alpaca, "get_news_by_symbol",
                        lambda symbols, start, end, per, cutoff=None: cutoffs.append(cutoff) or
                        {t: f"{t} news" for t in symbols})
    calendar = ["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]
    bars = {
        "SPY": {d: {"open": 1, "close": 1} for d in calendar},
        # GE opens 100 on the 9th, closes 103 that day and 95 two days in.
        "GE": {"2026-10-08": {"close": 98}, "2026-10-09": {"open": 100, "close": 103},
               "2026-10-12": {"close": 95}, "2026-10-13": {"close": 101}, "2026-10-14": {"close": 104},
               "2026-10-15": {"close": 110}},
        "BAC": {"2026-10-09": {"open": 50, "close": 51}},
    }
    monkeypatch.setattr(replay, "_prior_close", lambda ticker, date: {"GE": 98, "BAC": 50}[ticker])
    screener = _ScreenLLM({
        "GE": '{"direction": "up", "conviction": 5, "catalyst": "Fresh: beat", "reason": "r"}',
        "BAC": '{"direction": "down", "conviction": 4, "catalyst": "fine", "reason": "r"}',
        "XOM": '{"direction": "none", "conviction": 2, "catalyst": "recap", "reason": "r"}',
    })
    config = {**default_config.DEFAULT_CONFIG, "screen_mode": "day"}

    result = replay.run_replay(config, ["2026-10-09"], llms=(screener,),
                               gaps=lambda ticker, date: {"GE": 100.0, "BAC": 50.5}[ticker],
                               bars=lambda ticker: bars[ticker])
    text = replay.render_replay(result)

    assert cutoffs[0].isoformat() == "2026-10-09T09:25:00-04:00"
    ge, bac = sorted(result.picks, key=lambda p: p.verdict.ticker != "GE")
    assert ge.picked and ge.status == "agree" and ge.gap == pytest.approx(100 / 98 - 1)
    assert ge.returns["day"] == pytest.approx(0.03) and ge.returns["2 days"] == pytest.approx(-0.05)
    assert ge.returns["5 days"] == pytest.approx(0.10)
    # BAC gapped up against a down call, so the day mode skips it; scored as a down call it lost 2%.
    assert not bac.picked and bac.status == "disagree" and bac.returns["day"] == pytest.approx(-0.02)
    assert "| Short list (what day mode would trade) | 1 | +3.0% (1/1 right) |" in text
    assert "| BAC | down 4 | +1.0% | disagree | no | -2.0% | ? | ? | fine |" in text
