"""Price levels to watch in an index ETF for the day and the week.

From Tradier:
- the prior session's high, low and close, last week's high and low, and the
  classic floor pivots built from the prior session
- the overnight range: after hours since the prior close plus today's pre-market
- the expected move for the nearest expiry and for this week's, from the
  at-the-money straddle
- the call wall (most call open interest above the price) and put wall (most
  put open interest below it) for the nearest expiry and for this week's
- VIX and its change

From Unusual Whales, when the plan allows:
- max pain for those two expiries
- net open-interest gamma by strike: the flip (where it turns from negative
  below to positive above) and the biggest negative and positive strikes.
  Negative gamma usually means dealer hedging adds to moves; positive damps them.

Levels within ``ZONE`` of each other are merged into one zone. Live only; market
data endpoints only, nothing reads the account or places orders.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.vendors import unusual_whales
from tradingagents.dataflows.vendors.tradier import _as_list, _num, _quote, _request

logger = logging.getLogger(__name__)

ZONE = 0.003         # levels within 0.3% of each other form one zone
GAMMA_BAND = 0.04    # strikes within 4% of the price count toward the gamma read
WALL_BAND = 0.07     # and within 7% toward the open-interest walls


@dataclass
class Levels:
    ticker: str
    date: str
    spot: float
    spot_source: str
    rows: list[tuple[float, str]] = field(default_factory=list)  # (price, label)
    notes: list[str] = field(default_factory=list)


def _mid(option: dict) -> float | None:
    bid, ask = _num(option.get("bid")), _num(option.get("ask"))
    if bid and ask:
        return (bid + ask) / 2
    return _num(option.get("last"))


def quote_price(quote: dict) -> tuple[float | None, str]:
    """The quote's bid/ask mid, which moves before the open when ``last`` doesn't; else the last trade."""
    bid, ask = _num(quote.get("bid")), _num(quote.get("ask"))
    if bid and ask:
        return (bid + ask) / 2, "bid/ask mid"
    return _num(quote.get("last")) or _num(quote.get("prevclose")), "last trade"


def straddle(options: list, spot: float) -> tuple[float, float] | None:
    """The at-the-money strike and its straddle price, or None if either side has no price."""
    calls = {_num(o.get("strike")): o for o in options if o.get("option_type") == "call"}
    puts = {_num(o.get("strike")): o for o in options if o.get("option_type") == "put"}
    for strike in sorted(set(calls) & set(puts) - {None}, key=lambda s: abs(s - spot)):
        call, put = _mid(calls[strike]), _mid(puts[strike])
        if call and put:
            return strike, call + put
    return None


def walls(options: list, spot: float) -> tuple[float | None, float | None]:
    """The strike with the most call open interest above the price and the most put open interest below."""
    calls, puts = {}, {}
    for o in options:
        strike, oi = _num(o.get("strike")), _num(o.get("open_interest")) or 0
        if strike is None or not oi or abs(strike / spot - 1) > WALL_BAND:
            continue
        if o.get("option_type") == "call" and strike > spot:
            calls[strike] = calls.get(strike, 0) + oi
        elif o.get("option_type") == "put" and strike < spot:
            puts[strike] = puts.get(strike, 0) + oi
    return (max(calls, key=calls.get) if calls else None, max(puts, key=puts.get) if puts else None)


def gamma_read(gamma: dict[float, float], spot: float) -> dict:
    """The flip (the strike nearest the price where net gamma turns from negative below to positive) and the extremes."""
    near = {s: g for s, g in gamma.items() if abs(s / spot - 1) <= GAMMA_BAND and g}
    if not near:
        return {}
    strikes = sorted(near)
    flips = [hi for lo, hi in itertools.pairwise(strikes) if near[lo] < 0 < near[hi]]
    return {"flip": min(flips, key=lambda s: abs(s - spot)) if flips else None,
            "most_negative": min(near, key=near.get), "most_positive": max(near, key=near.get),
            "net": sum(near.values())}


def pivots(high: float, low: float, close: float) -> dict[str, float]:
    p = (high + low + close) / 3
    return {"R2": p + (high - low), "R1": 2 * p - low, "Pivot": p, "S1": 2 * p - high, "S2": p - (high - low)}


def week_end(date: str) -> str:
    day = datetime.strptime(date, "%Y-%m-%d")
    return (day + timedelta(days=(4 - day.weekday()) % 7)).strftime("%Y-%m-%d")


def _week(date: str) -> tuple[int, int]:
    return tuple(datetime.strptime(date, "%Y-%m-%d").isocalendar()[:2])


def compute(ticker: str, date: str, quote: dict, days: list[dict], overnight: list[dict],
            chains: dict[str, list], vix: dict | None = None, max_pain: dict | None = None,
            gamma: dict | None = None) -> Levels:
    """Levels from already-fetched data: daily bars, overnight bars, chains by expiry and the extras."""
    spot, source = quote_price(quote)
    levels = Levels(ticker, date, spot, source)
    add = levels.rows.append

    before = [d for d in days if d.get("date") and d["date"] < date]
    if before:
        prior = before[-1]
        high, low, close = (_num(prior[k]) for k in ("high", "low", "close"))
        add((high, "prior high"))
        add((low, "prior low"))
        add((close, "prior close"))
        for name, price in pivots(high, low, close).items():
            add((price, "pivot" if name == "Pivot" else name))
        last_week = [d for d in before if _week(d["date"]) < _week(date)]
        if last_week:
            bars = [d for d in last_week if _week(d["date"]) == _week(last_week[-1]["date"])]
            add((max(_num(d["high"]) for d in bars), "last week's high"))
            add((min(_num(d["low"]) for d in bars), "last week's low"))
        levels.notes.append(f"{ticker} {spot:.2f} ({source}), {spot / close - 1:+.2%} from the prior close "
                            f"of {close:.2f}.")
    highs = [_num(b.get("high")) for b in overnight if _num(b.get("high"))]
    lows = [_num(b.get("low")) for b in overnight if _num(b.get("low"))]
    if highs and lows:
        add((max(highs), "overnight high"))
        add((min(lows), "overnight low"))

    expiries = sorted(chains)
    if expiries:
        weekly = next((e for e in expiries if e >= week_end(date)), expiries[-1])
        for label, expiry in [("today's", expiries[0])] + ([("week's", weekly)] if weekly != expiries[0] else []):
            found = straddle(chains[expiry], spot)
            if found:
                move = found[1]
                add((spot + move, f"top of {label} implied range"))
                add((spot - move, f"bottom of {label} implied range"))
                levels.notes.append(f"Implied move to {expiry}: ±{move:.2f} ({move / spot:.2%}), "
                                    f"{spot - move:.2f} to {spot + move:.2f}.")
            call_wall, put_wall = walls(chains[expiry], spot)
            if call_wall:
                add((call_wall, f"{label} call wall"))
            if put_wall:
                add((put_wall, f"{label} put wall"))
            if max_pain and max_pain.get(expiry):
                add((max_pain[expiry], f"{label} max pain"))
    read = gamma_read(gamma or {}, spot)
    if read:
        if read["flip"]:
            add((read["flip"], "gamma flip"))
            levels.notes.append("Gamma: " + ("below the flip, in negative gamma, so moves can speed up."
                                             if spot < read["flip"] else
                                             "above the flip, in positive gamma, so moves tend to be damped."))
        else:
            levels.notes.append(f"Gamma: net {'positive' if read['net'] > 0 else 'negative'} near the price, no flip.")
        add((read["most_negative"], "biggest negative gamma"))
        add((read["most_positive"], "biggest positive gamma"))
    if vix:
        level, prev = quote_price(vix)[0], _num(vix.get("prevclose"))
        if level:
            levels.notes.append(f"VIX {level:.2f}" + (f" ({level / prev - 1:+.1%})." if prev else "."))
    return levels


def zones(rows: list[tuple[float, str]], spot: float) -> list[tuple[float, float, list[str]]]:
    """Rows merged into zones no wider than ``ZONE``, never straddling the price, highest first, as (low, high, labels)."""
    merged: list[tuple[float, float, list[str]]] = []
    for price, label in sorted(rows, key=lambda r: -r[0]):
        if merged and merged[-1][1] - price <= spot * ZONE and not merged[-1][0] >= spot > price:
            _, high, labels = merged[-1]
            merged[-1] = (price, high, labels + [label])
        else:
            merged.append((price, price, [label]))
    return merged


def render(levels: Levels) -> str:
    lines = [f"## {levels.ticker} levels, {levels.date}", "", *levels.notes, "", "| Zone | What's there |", "|---|---|"]
    now = f"| **{levels.spot:.2f}** | **now** |"
    for low, high, labels in zones(levels.rows, levels.spot):
        if now and high < levels.spot:
            lines.append(now)
            now = ""
        span = f"{low:.2f}" if high - low < 0.005 else f"{low:.2f}-{high:.2f}"
        lines.append(f"| {span} | {', '.join(dict.fromkeys(labels))} |")
    if now:
        lines.append(now)
    lines += ["", "Before the open, option quotes are the last session's, so implied ranges, walls and gamma "
              "reflect the prior close.", ""]
    return "\n".join(lines) + "\n"


def _extra(label, call, *args):
    try:
        return call(*args)
    except Exception as exc:  # each extra is optional; the rest of the levels still stand
        logger.warning("No %s: %s", label, exc)
        return None


def near_expiries(ticker: str, date: str) -> list[str]:
    """The nearest expiry on or after ``date`` and this week's (the first on or after Friday)."""
    listed = sorted(_as_list((_request("markets/options/expirations", {"symbol": ticker})
                              .get("expirations") or {}).get("date")))
    upcoming = [e for e in listed if e >= date]
    if not upcoming:
        return []
    weekly = next((e for e in upcoming if e >= week_end(date)), upcoming[-1])
    return list(dict.fromkeys([upcoming[0], weekly]))


def fetch(ticker: str, date: str) -> dict:
    """Everything ``compute`` needs, from Tradier and Unusual Whales."""
    quote = _quote(ticker)
    start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=21)).strftime("%Y-%m-%d")
    history = _request("markets/history", {"symbol": ticker, "interval": "daily", "start": start, "end": date})
    days = _as_list((history.get("history") or {}).get("day"))
    prior = [d for d in days if d.get("date", "") < date]
    overnight = []
    if prior:
        series = _extra("overnight range", _request, "markets/timesales", {
            "symbol": ticker, "interval": "5min", "session_filter": "all",
            "start": f"{prior[-1]['date']} 16:00", "end": f"{date} 09:29"})
        overnight = _as_list(((series or {}).get("series") or {}).get("data"))
    chains = {}
    for expiry in near_expiries(ticker, date):
        chain = _request("markets/options/chains", {"symbol": ticker, "expiration": expiry, "greeks": "true"})
        options = _as_list((chain.get("options") or {}).get("option"))
        if options:
            chains[expiry] = options
    return {"quote": quote, "days": days, "overnight": overnight, "chains": chains,
            "vix": _extra("VIX", _quote, "VIX"),
            "max_pain": _extra("max pain", unusual_whales.get_max_pain, ticker),
            "gamma": _extra("gamma by strike", unusual_whales.get_gamma_by_strike, ticker)}


def get_levels(ticker: str, date: str) -> str:
    """The levels section for ``ticker`` today; a past date has no live data."""
    if is_historical(date):
        return f"## {ticker} levels\n\nLive data only; none for {date}.\n"
    return render(compute(ticker, date, **fetch(ticker, date)))
