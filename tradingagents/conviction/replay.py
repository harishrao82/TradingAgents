"""Replay the day screen on past mornings and score its picks against what each stock did.

For each trading day:

1. the ETF's holdings today (a name added or dropped since is a small bias),
2. Benzinga news from two weekdays back up to 9:25 ET that morning, screened
   with the day prompt,
3. each strong call's pre-market move: the last 5-minute bar before 9:30
   against the prior close (Tradier timesales, which reach back about 40 days),
4. the short list the live day mode would build: names moving against their
   news skipped, agreeing names first.

Each name is scored by the stock's move in the call's direction, from that
day's open to its close and to the close two and five trading days in. Option
P&L is not replayed: there are no historical chains.

Market data and news only; nothing reads an account or places an order.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean
from zoneinfo import ZoneInfo

import anthropic

from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.vendors import alpaca, unusual_whales
from tradingagents.dataflows.vendors.tradier import _as_list, _num, _request

from .batch import screen_news_batch
from .reaction import RANK, classify
from .screen import DAY_SCREEN_PROMPT, NewsVerdict, screen_news, shortlist

logger = logging.getLogger(__name__)

NEW_YORK = ZoneInfo("America/New_York")
HOLDS = {"day": 0, "2 days": 1, "5 days": 4}   # trading days after the entry day's open, to that close
DAY_WORKERS = 4


@dataclass
class Pick:
    date: str
    verdict: NewsVerdict
    gap: float | None            # pre-market move from the prior close
    status: str                  # agree, disagree, flat or unknown
    picked: bool                 # on the short list the day mode would trade
    returns: dict[str, float | None] = field(default_factory=dict)  # move in the call's direction, by hold


@dataclass
class ReplayResult:
    etf: str
    dates: list[str]
    picks: list[Pick]
    failures: dict[str, int]     # names the screen couldn't read, by date


def trading_days(end: str, count: int) -> list[str]:
    """The last ``count`` sessions before ``end``, from SPY's daily bars."""
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=count * 2 + 10)).strftime("%Y-%m-%d")
    days = [d["date"] for d in _daily("SPY", start, end) if d["date"] < end]
    return days[-count:]


def _daily(ticker: str, start: str, end: str) -> list[dict]:
    history = _request("markets/history", {"symbol": ticker, "interval": "daily", "start": start, "end": end})
    return _as_list((history.get("history") or {}).get("day"))


def premarket_price(ticker: str, date: str) -> float | None:
    """The close of the last 5-minute bar before 9:30 ET, or None when nothing traded."""
    series = _request("markets/timesales", {"symbol": ticker, "interval": "5min", "session_filter": "all",
                                            "start": f"{date} 04:00", "end": f"{date} 09:29"})
    bars = _as_list((series.get("series") or {}).get("data"))
    return _num(bars[-1].get("close")) if bars else None


def cutoff(date: str) -> datetime:
    """9:25 ET on ``date``, the last moment of news a pre-market run could have read."""
    return datetime.strptime(date, "%Y-%m-%d").replace(hour=9, minute=25, tzinfo=NEW_YORK)


def outcome(direction: str, bars: dict[str, dict], calendar: list[str], date: str) -> dict[str, float | None]:
    """The move in the call's direction from ``date``'s open to each hold's close; None past the data."""
    sign = 1 if direction == "up" else -1
    i = calendar.index(date)
    entry = _num((bars.get(date) or {}).get("open"))
    moves = {}
    for hold, ahead in HOLDS.items():
        exit_day = calendar[i + ahead] if i + ahead < len(calendar) else None
        close = _num((bars.get(exit_day) or {}).get("close")) if exit_day else None
        moves[hold] = sign * (close / entry - 1) if entry and close else None
    return moves


def _screen(config, news, start, date, screener):
    if config["screen_batch"]:
        return screen_news_batch(screener or anthropic.Anthropic(), config["screen_llm"], news, start, date,
                                 max_tokens=config["screen_max_tokens"], template=DAY_SCREEN_PROMPT)
    from .workflow import screen_llm
    return screen_news(screener or screen_llm(config), news, start, date, config["screen_workers"],
                       per_minute=config["screen_requests_per_minute"], template=DAY_SCREEN_PROMPT)


def replay_day(config: dict, date: str, weights: dict[str, float | None], screener=None,
               gaps=premarket_price) -> tuple[list[Pick], int]:
    """The strong calls for one morning, marked with the ones the day mode would have traded."""
    from .workflow import _weekdays_back, prior_close

    start = _weekdays_back(date, config["day_lookback_days"])
    news = alpaca.get_news_by_symbol(list(weights), start, date, config["screen_articles_per_name"],
                                     cutoff=cutoff(date),
                                     fresh_since=prior_close(date) if config["day_fresh_only"] else None)
    verdicts, failures = _screen(config, news, start, date, screener)
    strong = [v for v in verdicts if v.direction != "none" and v.conviction >= config["screen_min_conviction"]]
    status = {}
    gap_of = {}
    for v in strong:
        try:
            price = gaps(v.ticker, date)
            prev = _prior_close(v.ticker, date)
            gap_of[v.ticker] = price / prev - 1 if price and prev else None
        except Exception as exc:  # one name's missing bars must not stop the day
            logger.warning("No pre-market move for %s on %s: %s", v.ticker, date, exc)
            gap_of[v.ticker] = None
        status[v.ticker] = classify(v.direction, gap_of[v.ticker])
    candidates = [v for v in strong if not (config["day_skip_disagree"] and status[v.ticker] == "disagree")]
    picked = {v.ticker for v in shortlist(candidates, config["screen_min_conviction"], config["screen_max_names"],
                                          weights, config["screen_direction"],
                                          {t: RANK[s] for t, s in status.items()})}
    return [Pick(date, v, gap_of[v.ticker], status[v.ticker], v.ticker in picked) for v in strong], len(failures)


def _prior_close(ticker: str, date: str) -> float | None:
    start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
    before = [d for d in _daily(ticker, start, date) if d["date"] < date]
    return _num(before[-1]["close"]) if before else None


def run_replay(config: dict, dates: list[str], etf: str = "SPY", llms: tuple | None = None,
               progress=None, gaps=premarket_price, bars=None) -> ReplayResult:
    """Replay the day screen on each of ``dates`` and score every strong call."""
    say = progress or (lambda message: None)
    with run_config(config):
        weights = {t: w for t, w, _ in unusual_whales.get_etf_constituents(etf)}
        say(f"{len(weights)} names in {etf}; replaying {len(dates)} mornings")
        screener = (llms or (None,))[0]

        def one(date):
            found = replay_day(config, date, weights, screener, gaps)
            say(f"{date}: {len(found[0])} strong calls, {sum(p.picked for p in found[0])} on the short list")
            return date, found

        with ThreadPoolExecutor(DAY_WORKERS) as pool:
            days = dict(pool.map(one, dates))
        picks = [p for d in dates for p in days[d][0]]

        last = (datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=14)).strftime("%Y-%m-%d")
        first = (datetime.strptime(dates[0], "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        bars = bars or (lambda ticker: {d["date"]: d for d in _daily(ticker, first, last)})
        today = datetime.now(NEW_YORK).strftime("%Y-%m-%d")
        calendar = sorted(d for d in bars("SPY") if d < today)  # today's bar is still moving
        cache = {}
        for p in picks:
            ticker = p.verdict.ticker
            if ticker not in cache:
                try:
                    cache[ticker] = bars(ticker)
                except Exception as exc:
                    logger.warning("No daily bars for %s: %s", ticker, exc)
                    cache[ticker] = {}
            p.returns = outcome(p.verdict.direction, cache[ticker], calendar, p.date)
    return ReplayResult(etf, dates, picks, {d: days[d][1] for d in dates if days[d][1]})


def _pct(value: float | None) -> str:
    return "?" if value is None else f"{value:+.1%}"


def _row(name: str, picks: list[Pick]) -> str:
    cells = []
    for hold in HOLDS:
        moves = [p.returns[hold] for p in picks if p.returns.get(hold) is not None]
        cells.append(f"{_pct(mean(moves)) if moves else '?'} ({sum(m > 0 for m in moves)}/{len(moves)} right)")
    return f"| {name} | {len(picks)} | " + " | ".join(cells) + " |"


def render_replay(result: ReplayResult) -> str:
    picks = result.picks
    groups = [
        ("Short list (what day mode would trade)", [p for p in picks if p.picked]),
        ("Agreeing, not on the short list", [p for p in picks if not p.picked and p.status == "agree"]),
        ("Moving against the news", [p for p in picks if p.status == "disagree"]),
        ("Every strong call", picks),
    ]
    lines = [
        f"# {result.etf} day-screen replay, {result.dates[0]} to {result.dates[-1]}", "",
        f"{len(result.dates)} mornings. News up to 9:25 ET each day, the day prompt, and the pre-market move from "
        "the last 5-minute bar before the open. Each move is the stock's, from that day's open, in the call's "
        "direction: positive means the call was right. Option P&L is not replayed. Analysis only.", "",
        "| Group | Calls | Open to close | Open to day-2 close | Open to day-5 close |", "|---|---|---|---|---|",
        *[_row(name, group) for name, group in groups if group], "",
    ]
    if result.failures:
        lines += ["Names the screen couldn't read: " + ", ".join(f"{d} {n}" for d, n in result.failures.items()), ""]
    for date in result.dates:
        day = sorted((p for p in picks if p.date == date and (p.picked or p.status == "disagree")),
                     key=lambda p: (not p.picked, RANK[p.status], -p.verdict.conviction, p.verdict.ticker))
        if not day:
            continue
        lines += [f"## {date}", "", "| Ticker | Call | Pre-market | Reaction | Traded | Day | 2 days | 5 days | Catalyst |",
                  "|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {p.verdict.ticker} | {p.verdict.direction} {p.verdict.conviction} | {_pct(p.gap)} | {p.status} | "
                  f"{'yes' if p.picked else 'no'} | {_pct(p.returns.get('day'))} | {_pct(p.returns.get('2 days'))} | "
                  f"{_pct(p.returns.get('5 days'))} | {p.verdict.catalyst} |" for p in day]
        lines.append("")
    return "\n".join(lines) + "\n"
