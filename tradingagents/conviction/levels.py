"""Price levels to watch in an index ETF for the day and the week, from Tradier market data.

- The prior session's high, low and close, last week's high and low, and the
  classic floor pivots built from the prior session.
- The overnight range: after-hours since the prior close plus today's pre-market.
- The expected move for the nearest expiry and for this week's, from the
  at-the-money straddle.
- Where options open interest sits on the expiries out to this week's: the
  call wall (most call open interest above the price), the put wall (most put
  open interest below it) and the strike with the most gamma, using the usual
  convention that calls add gamma and puts take it away. That convention is a
  rough read of dealer positioning, not a measurement of it.

Live only, like the rest of Tradier: a past date gets nothing. Market data
endpoints only; nothing reads the account or places orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.vendors.tradier import _as_list, _num, _quote, _request

MAX_CHAINS = 6       # chains fetched for open interest, nearest expiries first
STRIKE_BAND = 0.07   # strikes within 7% of the price count toward walls and gamma


@dataclass
class Levels:
    ticker: str
    date: str
    spot: float
    spot_source: str
    rows: list[tuple[float, str, str]] = field(default_factory=list)  # (price, level, note)
    notes: list[str] = field(default_factory=list)


def _mid(option: dict) -> float | None:
    bid, ask = _num(option.get("bid")), _num(option.get("ask"))
    if bid and ask:
        return (bid + ask) / 2
    return _num(option.get("last"))


def straddle(options: list, spot: float) -> tuple[float, float] | None:
    """The at-the-money strike and its straddle price, or None if either side has no price."""
    calls = {_num(o.get("strike")): o for o in options if o.get("option_type") == "call"}
    puts = {_num(o.get("strike")): o for o in options if o.get("option_type") == "put"}
    for strike in sorted(set(calls) & set(puts) - {None}, key=lambda s: abs(s - spot)):
        call, put = _mid(calls[strike]), _mid(puts[strike])
        if call and put:
            return strike, call + put
    return None


def positioning(options: list, spot: float) -> dict:
    """Call wall, put wall, biggest-gamma strike and net gamma ($ per 1% move) near the price."""
    call_oi, put_oi, gamma = {}, {}, {}
    for o in options:
        strike, oi = _num(o.get("strike")), _num(o.get("open_interest")) or 0
        if strike is None or abs(strike / spot - 1) > STRIKE_BAND:
            continue
        g = _num((o.get("greeks") or {}).get("gamma")) or 0
        sign = 1 if o.get("option_type") == "call" else -1
        gamma[strike] = gamma.get(strike, 0) + sign * g * oi * 100 * spot * spot * 0.01
        side = call_oi if sign > 0 else put_oi
        side[strike] = side.get(strike, 0) + oi
    above = {s: oi for s, oi in call_oi.items() if s > spot and oi}
    below = {s: oi for s, oi in put_oi.items() if s < spot and oi}
    return {
        "call_wall": max(above, key=above.get) if above else None,
        "call_wall_oi": max(above.values()) if above else 0,
        "put_wall": max(below, key=below.get) if below else None,
        "put_wall_oi": max(below.values()) if below else 0,
        "gamma_strike": max(gamma, key=lambda s: abs(gamma[s])) if gamma else None,
        "net_gamma": sum(gamma.values()),
    }


def pivots(high: float, low: float, close: float) -> dict[str, float]:
    p = (high + low + close) / 3
    return {"R2": p + (high - low), "R1": 2 * p - low, "Pivot": p, "S1": 2 * p - high, "S2": p - (high - low)}


def _week_end(date: str) -> str:
    day = datetime.strptime(date, "%Y-%m-%d")
    return (day + timedelta(days=(4 - day.weekday()) % 7)).strftime("%Y-%m-%d")


def compute(ticker: str, date: str, quote: dict, days: list[dict], overnight: list[dict],
            chains: dict[str, list]) -> Levels:
    """Levels from already-fetched data: daily bars, overnight bars and chains by expiry."""
    before = [d for d in days if d.get("date") and d["date"] < date]
    prior = before[-1] if before else None
    # The quote's last trade can be yesterday's close before the open, so a newer
    # extended-hours bar wins.
    spot, source = _num(quote.get("last")), "last trade"
    traded = (_num(quote.get("trade_date")) or 0) / 1000
    if overnight and _num(overnight[-1].get("close")) and (_num(overnight[-1].get("timestamp")) or 0) > traded:
        spot, source = _num(overnight[-1]["close"]), f"extended hours, {overnight[-1].get('time', '')}".rstrip(", ")
    spot = spot or _num(quote.get("prevclose"))
    levels = Levels(ticker, date, spot, source)
    add = levels.rows.append

    if prior:
        high, low, close = (_num(prior[k]) for k in ("high", "low", "close"))
        add((high, "Prior high", prior["date"]))
        add((low, "Prior low", prior["date"]))
        add((close, "Prior close", prior["date"]))
        for name, price in pivots(high, low, close).items():
            if name != "Pivot" or abs(price - close) > 0.01:
                add((price, name if name == "Pivot" else f"{name} (floor pivot)", "from the prior session"))
        week = datetime.strptime(date, "%Y-%m-%d").isocalendar()[:2]
        last_week = [d for d in before if datetime.strptime(d["date"], "%Y-%m-%d").isocalendar()[:2] < week]
        if last_week:
            wk = datetime.strptime(last_week[-1]["date"], "%Y-%m-%d").isocalendar()[:2]
            bars = [d for d in last_week if datetime.strptime(d["date"], "%Y-%m-%d").isocalendar()[:2] == wk]
            add((max(_num(d["high"]) for d in bars), "Last week's high", f"week of {bars[0]['date']}"))
            add((min(_num(d["low"]) for d in bars), "Last week's low", f"week of {bars[0]['date']}"))
    if overnight:
        highs = [_num(b.get("high")) for b in overnight if _num(b.get("high"))]
        lows = [_num(b.get("low")) for b in overnight if _num(b.get("low"))]
        if highs and lows:
            add((max(highs), "Overnight high", "after hours + pre-market"))
            add((min(lows), "Overnight low", "after hours + pre-market"))

    expiries = sorted(chains)
    if expiries:
        week_end = _week_end(date)
        weekly = next((e for e in expiries if e >= week_end), expiries[-1])
        moves = [("Day", expiries[0])] + ([("Week", weekly)] if weekly != expiries[0] else [])
        for label, expiry in moves:
            found = straddle(chains[expiry], spot)
            if not found:
                levels.notes.append(f"No priced at-the-money straddle for {expiry}.")
                continue
            strike, move = found
            add((spot + move, f"{label} expected move, high", f"{expiry} straddle ±{move:.2f}"))
            add((spot - move, f"{label} expected move, low", f"{expiry} straddle ±{move:.2f}"))
        pos = positioning([o for e in expiries if e <= weekly for o in chains[e]], spot)
        through = f"expiries {expiries[0]} to {weekly}"
        if pos["call_wall"]:
            add((pos["call_wall"], "Call wall", f"{pos['call_wall_oi']:,.0f} calls open, {through}"))
        if pos["put_wall"]:
            add((pos["put_wall"], "Put wall", f"{pos['put_wall_oi']:,.0f} puts open, {through}"))
        if pos["gamma_strike"]:
            add((pos["gamma_strike"], "Most gamma", through))
        net = pos["net_gamma"]
        levels.notes.append(
            f"Net gamma through {weekly}: {net / 1e9:+.2f}B per 1% move. "
            + ("Positive usually means dealer hedging damps moves, so levels tend to hold."
               if net > 0 else "Negative usually means dealer hedging adds to moves, so breaks can run.")
        )
    return levels


def fetch(ticker: str, date: str) -> tuple[dict, list, list, dict]:
    """Quote, daily bars, overnight bars and the chains out to this week's expiry, from Tradier."""
    quote = _quote(ticker)
    start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=21)).strftime("%Y-%m-%d")
    history = _request("markets/history", {"symbol": ticker, "interval": "daily", "start": start, "end": date})
    days = _as_list((history.get("history") or {}).get("day"))
    prior = [d for d in days if d.get("date", "") < date]
    overnight = []
    if prior:
        try:
            series = _request("markets/timesales", {
                "symbol": ticker, "interval": "5min", "session_filter": "all",
                "start": f"{prior[-1]['date']} 16:00", "end": f"{date} 09:29"})
            overnight = _as_list((series.get("series") or {}).get("data"))
        except Exception:  # the overnight range is a nicety; the rest still stands
            overnight = []
    expirations = _as_list((_request("markets/options/expirations", {"symbol": ticker})
                            .get("expirations") or {}).get("date"))
    week_end = _week_end(date)
    wanted = [e for e in sorted(expirations) if e >= date]
    wanted = [e for e in wanted if e <= week_end] or wanted[:1]
    if wanted and wanted[-1] < week_end:
        later = [e for e in sorted(expirations) if e > week_end]
        wanted += later[:1]
    chains = {}
    for expiry in wanted[:MAX_CHAINS]:
        chain = _request("markets/options/chains", {"symbol": ticker, "expiration": expiry, "greeks": "true"})
        options = _as_list((chain.get("options") or {}).get("option"))
        if options:
            chains[expiry] = options
    return quote, days, overnight, chains


def render(levels: Levels) -> str:
    lines = [f"## {levels.ticker} levels, {levels.date}", "",
             f"Price {levels.spot:.2f} ({levels.spot_source}).", "",
             "| Price | Level | From |", "|---|---|---|"]
    rows = sorted(levels.rows + [(levels.spot, "**Now**", levels.spot_source)], key=lambda r: -r[0])
    lines += [f"| {price:.2f} | {name} | {note} |" for price, name, note in rows]
    lines += [""] + [f"{note}" for note in levels.notes]
    lines += ["", "Before the open, option quotes are the last session's, so the expected moves and gamma "
              "reflect yesterday's close.", ""]
    return "\n".join(lines) + "\n"


def get_levels(ticker: str, date: str) -> str:
    """The levels section for ``ticker`` today; a past date has no live data."""
    if is_historical(date):
        return f"## {ticker} levels\n\nLive data only; none for {date}.\n"
    return render(compute(ticker, date, *fetch(ticker, date)))
