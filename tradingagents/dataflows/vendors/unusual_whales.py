"""Unusual Whales vendor: options flow, options activity, market tide and event calendars.

Reads the Unusual Whales REST API (https://api.unusualwhales.com/docs) with a
bearer token from ``UNUSUAL_WHALES_API_KEY`` (``UW_API_KEY`` is accepted too).
Without a key every call raises ``UnusualWhalesNotConfiguredError``, which the
routing layer treats as "vendor unavailable", so a run without a subscription
degrades to a sentinel instead of failing.

Every function takes the run's as-of date and never shows data from after it:
flow alerts are bounded with ``older_than``, daily rows past the date are
dropped, and an earnings report after the date shows only its scheduled date.
The economic calendar endpoint serves only the current and next week, so a
historical date gets an explicit "not available" rather than today's calendar.
"""
import logging
import os
from datetime import UTC, datetime, timedelta

import requests

from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)

logger = logging.getLogger(__name__)

UW_API_BASE = "https://api.unusualwhales.com/api"

# Network timeout (seconds) so a stalled request can't hang the agents,
# mirroring the FRED and Alpha Vantage clients.
REQUEST_TIMEOUT = 30

# Trailing window for flow alerts when the caller does not give one.
DEFAULT_FLOW_LOOKBACK_DAYS = 5

# Rows cap for rendered tables, so a busy ticker can't flood the context.
MAX_ROWS = 25

# The economic calendar endpoint covers "the current & next week" only.
CALENDAR_HISTORY_DAYS = 7


class UnusualWhalesNotConfiguredError(VendorNotConfiguredError):
    """Raised when Unusual Whales is selected but no API key is configured."""


def get_api_key() -> str:
    """The Unusual Whales API token from the environment."""
    api_key = os.getenv("UNUSUAL_WHALES_API_KEY") or os.getenv("UW_API_KEY")
    if not api_key:
        raise UnusualWhalesNotConfiguredError(
            "UNUSUAL_WHALES_API_KEY environment variable is not set. Create a token "
            "at https://unusualwhales.com/dashboard/api."
        )
    return api_key


def _request(path: str, params: dict | None = None) -> list:
    """GET an endpoint and return its ``data`` rows.

    The token travels in a header, never the URL, so request errors can be
    logged as they are. A rejected token is "not configured"; throttling, a
    plan without access to the endpoint, a server error or a network failure
    is "unavailable", which says nothing about the instrument.
    """
    api_key = get_api_key()
    try:
        response = requests.get(
            f"{UW_API_BASE}/{path}",
            params=params or {},
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise VendorUnavailableError(f"Unusual Whales request failed: {exc}") from None
    if response.status_code == 401:
        raise UnusualWhalesNotConfiguredError(
            "Unusual Whales rejected the API token (401). Check UNUSUAL_WHALES_API_KEY."
        )
    if response.status_code == 403:
        raise VendorUnavailableError(
            f"Unusual Whales plan has no access to /{path} (403)."
        )
    if response.status_code == 422:
        raise ValueError(f"Unusual Whales rejected the request to /{path}: {response.text[:300]}")
    if response.status_code == 429 or response.status_code >= 500:
        raise VendorUnavailableError(
            f"Unusual Whales returned {response.status_code} for /{path}."
        )
    response.raise_for_status()
    body = response.json()
    rows = body.get("data", []) if isinstance(body, dict) else body
    if isinstance(rows, dict):  # some endpoints return a single object
        rows = [rows]
    return rows or []


def _day(value) -> str:
    """The ``yyyy-mm-dd`` part of an ISO date or timestamp."""
    return str(value or "")[:10]


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _money(value) -> str:
    """A premium as $1.2M / $350K, or '?' when missing."""
    number = _num(value)
    if number is None:
        return "?"
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number >= 1e9:
        return f"{sign}${number / 1e9:.2f}B"
    if number >= 1e6:
        return f"{sign}${number / 1e6:.2f}M"
    if number >= 1e3:
        return f"{sign}${number / 1e3:.0f}K"
    return f"{sign}${number:.0f}"


def _pct(value, digits: int = 1) -> str:
    number = _num(value)
    return "?" if number is None else f"{number * 100:.{digits}f}%"


def _end_of_day(as_of_date: str) -> str:
    """The last moment of the as-of date, as an ISO datetime for ``older_than``."""
    return f"{as_of_date}T23:59:59Z"


def get_options_flow(
    symbol: str,
    as_of_date: str,
    look_back_days: int | None = None,
    min_premium: int | None = None,
) -> str:
    """Options flow alerts (sweeps, floor trades, repeated hits) for ``symbol``.

    Returns a summary of call versus put premium and ask- versus bid-side
    premium, followed by the largest alerts in the window ending on
    ``as_of_date``.
    """
    if look_back_days is None:
        look_back_days = DEFAULT_FLOW_LOOKBACK_DAYS
    start = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    params = {
        "ticker_symbol": symbol.upper(),
        "newer_than": start,
        "older_than": _end_of_day(as_of_date),
        "limit": 200,
    }
    if min_premium:
        params["min_premium"] = int(min_premium)
    alerts = _request("option-trades/flow-alerts", params)
    # The window is enforced here too, so a server that ignores older_than
    # can't leak alerts from after the as-of date.
    alerts = [a for a in alerts if start <= _day(a.get("created_at")) <= as_of_date]

    header = f"## Unusual Whales options flow alerts: {symbol.upper()}, {start} to {as_of_date}\n"
    if not alerts:
        return header + (
            "\nNo flow alerts fired for this ticker in the window. Alerts flag "
            "unusually large or aggressive trades only, so this means no standout "
            "flow, not no options activity.\n"
        )

    def total(kind: str, field: str) -> float:
        return sum(_num(a.get(field)) or 0.0 for a in alerts if a.get("type") == kind)

    call_prem, put_prem = total("call", "total_premium"), total("put", "total_premium")
    call_ask, call_bid = total("call", "total_ask_side_prem"), total("call", "total_bid_side_prem")
    put_ask, put_bid = total("put", "total_ask_side_prem"), total("put", "total_bid_side_prem")
    sweeps = sum(1 for a in alerts if a.get("has_sweep"))
    floors = sum(1 for a in alerts if a.get("has_floor"))
    ratio = f"{call_prem / put_prem:.2f}" if put_prem else "n/a (no put alerts)"

    summary = (
        f"\n- Alerts: {len(alerts)} ({sum(1 for a in alerts if a.get('type') == 'call')} call, "
        f"{sum(1 for a in alerts if a.get('type') == 'put')} put); {sweeps} with sweeps, {floors} with floor trades\n"
        f"- Call premium {_money(call_prem)} (ask side {_money(call_ask)}, bid side {_money(call_bid)})\n"
        f"- Put premium {_money(put_prem)} (ask side {_money(put_ask)}, bid side {_money(put_bid)})\n"
        f"- Call/put premium ratio: {ratio}\n"
        "- Ask-side premium is buyer-initiated: ask-side calls and bid-side puts lean bullish, "
        "ask-side puts and bid-side calls lean bearish. Opening versus closing is not known per alert.\n"
    )

    largest = sorted(alerts, key=lambda a: _num(a.get("total_premium")) or 0.0, reverse=True)[:MAX_ROWS]
    rows = [
        "| Time (UTC) | Type | Strike | Expiry | Premium | Ask-side | Bid-side | Size | Vol/OI | Underlying | Rule |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for a in largest:
        flags = "+".join(f for f, on in (("sweep", a.get("has_sweep")), ("floor", a.get("has_floor")),
                                          ("multileg", a.get("has_multileg"))) if on)
        vol_oi = _num(a.get("volume_oi_ratio"))
        rows.append(
            f"| {str(a.get('created_at', ''))[:16].replace('T', ' ')} | {a.get('type', '?')} | "
            f"{a.get('strike', '?')} | {_day(a.get('expiry'))} | {_money(a.get('total_premium'))} | "
            f"{_money(a.get('total_ask_side_prem'))} | {_money(a.get('total_bid_side_prem'))} | "
            f"{a.get('total_size', '?')} | {'?' if vol_oi is None else f'{vol_oi:.2f}'} | "
            f"{a.get('underlying_price', '?')} | {a.get('alert_rule', '?')}{f' ({flags})' if flags else ''} |"
        )
    note = f"\n_(largest {len(largest)} of {len(alerts)} alerts by premium)_\n" if len(alerts) > MAX_ROWS else "\n"
    return header + summary + note + "\n".join(rows) + "\n"


def get_options_activity(symbol: str, as_of_date: str) -> str:
    """Daily options volume, premium and open interest, plus implied volatility rank.

    Volume rows come from the most recent sessions on or before ``as_of_date``;
    IV rank is read for ``as_of_date`` itself over a three-month span.
    """
    ticker = symbol.upper()
    header = f"## Unusual Whales options activity: {ticker}, as of {as_of_date}\n"
    parts = [header]

    volume = _request(f"stock/{ticker}/options-volume", {"limit": 30})
    volume = sorted((r for r in volume if _day(r.get("date")) <= as_of_date),
                    key=lambda r: _day(r.get("date")), reverse=True)[:10]
    iv = _request(f"stock/{ticker}/iv-rank", {"date": as_of_date, "timespan": "3m"})
    iv = sorted((r for r in iv if _day(r.get("date")) <= as_of_date), key=lambda r: _day(r.get("date")))

    if not volume and not iv:
        raise NoMarketDataError(ticker, detail=f"no options volume or IV rank on or before {as_of_date}")

    if volume:
        latest = volume[0]
        call_v, put_v = _num(latest.get("call_volume")), _num(latest.get("put_volume"))
        avg30_c, avg30_p = _num(latest.get("avg_30_day_call_volume")), _num(latest.get("avg_30_day_put_volume"))
        pcr = f"{put_v / call_v:.2f}" if call_v else "?"
        rel_c = f"{call_v / avg30_c:.1f}x" if call_v is not None and avg30_c else "?"
        rel_p = f"{put_v / avg30_p:.1f}x" if put_v is not None and avg30_p else "?"
        parts.append(
            f"\n**Latest session ({_day(latest.get('date'))}):** call volume {latest.get('call_volume', '?')} "
            f"({rel_c} its 30-day average), put volume {latest.get('put_volume', '?')} ({rel_p}), "
            f"put/call volume ratio {pcr}; bullish premium {_money(latest.get('bullish_premium'))}, "
            f"bearish premium {_money(latest.get('bearish_premium'))}; open interest calls "
            f"{latest.get('call_open_interest', '?')}, puts {latest.get('put_open_interest', '?')}.\n"
        )
        parts.append(
            "\n| Date | Call vol | Put vol | Net call prem | Net put prem | Bullish prem | Bearish prem |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            + "\n".join(
                f"| {_day(r.get('date'))} | {r.get('call_volume', '?')} | {r.get('put_volume', '?')} | "
                f"{_money(r.get('net_call_premium'))} | {_money(r.get('net_put_premium'))} | "
                f"{_money(r.get('bullish_premium'))} | {_money(r.get('bearish_premium'))} |"
                for r in volume
            )
            + "\n"
        )
    else:
        parts.append("\nOptions volume: no rows on or before the as-of date.\n")

    if iv:
        last = iv[-1]
        rank = _num(last.get("iv_rank_1y"))
        rank_text = "?" if rank is None else f"{rank:.0f}" if rank > 1 else f"{rank * 100:.0f}"
        first = iv[0]
        parts.append(
            f"\n**Implied volatility ({_day(last.get('date'))}):** IV {_pct(last.get('volatility'))}, "
            f"1-year IV rank {rank_text} (0 = cheapest, 100 = richest of the past year); "
            f"three months earlier ({_day(first.get('date'))}) IV was {_pct(first.get('volatility'))}.\n"
        )
    else:
        parts.append("\nIV rank: no rows on or before the as-of date.\n")
    return "".join(parts)


def get_market_tide(as_of_date: str) -> str:
    """Market-wide net call and put premium through the session of ``as_of_date``.

    Market Tide nets ask-side against bid-side premium across all US-listed
    options; rising net call premium with falling net put premium is risk-on.
    """
    ticks = _request("market/market-tide", {"date": as_of_date, "interval_5m": True})
    ticks = [t for t in ticks if _day(t.get("timestamp") or t.get("date")) <= as_of_date]
    header = f"## Unusual Whales Market Tide, {as_of_date}\n"
    if not ticks:
        raise NoMarketDataError("MARKET", detail=f"no Market Tide ticks for {as_of_date}")
    ticks.sort(key=lambda t: str(t.get("timestamp", "")))
    last = ticks[-1]
    step = max(1, len(ticks) // 12)
    sampled = ticks[::step]
    if sampled[-1] is not last:
        sampled.append(last)
    table = (
        "\n| Time (UTC) | Net call premium | Net put premium | Net volume |\n| --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {str(t.get('timestamp', ''))[11:16]} | {_money(t.get('net_call_premium'))} | "
            f"{_money(t.get('net_put_premium'))} | {t.get('net_volume', '?')} |"
            for t in sampled
        )
        + "\n"
    )
    return header + (
        f"\n**Latest tick ({str(last.get('timestamp', ''))[:16].replace('T', ' ')} UTC):** net call premium "
        f"{_money(last.get('net_call_premium'))}, net put premium {_money(last.get('net_put_premium'))}, "
        f"net volume {last.get('net_volume', '?')}. Net premium is ask side minus bid side; positive "
        "net calls and negative net puts lean bullish.\n"
    ) + table


def get_economic_calendar(as_of_date: str, today: str | None = None) -> str:
    """Scheduled economic reports, FOMC meetings and Fed speakers from ``as_of_date`` on.

    Unusual Whales serves the current and next week only, so an as-of date more
    than a week in the past gets a "not available" note rather than events that
    were not yet known then.
    """
    today = today or datetime.now(UTC).strftime("%Y-%m-%d")
    header = f"## Unusual Whales economic calendar from {as_of_date}\n"
    oldest = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=CALENDAR_HISTORY_DAYS)).strftime("%Y-%m-%d")
    if as_of_date < oldest:
        return header + (
            f"\nDATA_UNAVAILABLE: the economic calendar covers only the current and next "
            f"week, so it has nothing for {as_of_date}. Use get_macro_indicators for the "
            f"data released by then, and do not invent scheduled events.\n"
        )
    events = _request("market/economic-calendar")
    upcoming = sorted((e for e in events if _day(e.get("time")) >= as_of_date), key=lambda e: str(e.get("time", "")))
    if not upcoming:
        return header + "\nNo scheduled events on or after the as-of date in the calendar's window.\n"
    rows = ["| Time (UTC) | Type | Event | Period | Forecast | Previous |", "| --- | --- | --- | --- | --- | --- |"]
    for e in upcoming[:MAX_ROWS * 2]:
        rows.append(
            f"| {str(e.get('time', ''))[:16].replace('T', ' ')} | {e.get('type', '?')} | {e.get('event', '?')} | "
            f"{e.get('reported_period') or ''} | {e.get('forecast') or ''} | {e.get('prev') or ''} |"
        )
    return header + "\n" + "\n".join(rows) + "\n"


def get_earnings_calendar(symbol: str, as_of_date: str) -> str:
    """The ticker's next scheduled earnings report and how past reports moved the stock.

    A report dated after ``as_of_date`` shows only its date and time, so a
    backtest never sees results or moves that had not happened yet.
    """
    ticker = symbol.upper()
    reports = _request(f"earnings/{ticker}")
    if not reports:
        raise NoMarketDataError(ticker, detail="no earnings history at Unusual Whales")
    reports.sort(key=lambda r: _day(r.get("report_date")))
    past = [r for r in reports if _day(r.get("report_date")) and _day(r.get("report_date")) <= as_of_date]
    future = [r for r in reports if _day(r.get("report_date")) > as_of_date]

    header = f"## Earnings for {ticker}, as of {as_of_date}\n"
    if future:
        nxt = future[0]
        days = (datetime.strptime(_day(nxt.get("report_date")), "%Y-%m-%d")
                - datetime.strptime(as_of_date, "%Y-%m-%d")).days
        source = nxt.get("source")
        source_text = " (estimated date)" if source == "estimation" else " (confirmed by the company)" if source == "company" else ""
        next_line = (
            f"\n**Next report:** {_day(nxt.get('report_date'))}, {nxt.get('report_time', 'time unknown')}"
            f"{source_text}, {days} calendar days after the as-of date. Options expiring after this "
            "date carry the earnings move in their price.\n"
        )
    else:
        next_line = "\n**Next report:** no scheduled date in the data.\n"

    recent = past[-8:][::-1]
    if not recent:
        return header + next_line + "\nNo past reports on or before the as-of date.\n"
    moves = [abs(_num(r.get("post_earnings_move_1d")) or 0.0) for r in recent if _num(r.get("post_earnings_move_1d")) is not None]
    expected = [abs(_num(r.get("expected_move_perc")) or 0.0) for r in recent if _num(r.get("expected_move_perc")) is not None]
    stats = ""
    if moves and expected:
        stats = (
            f"\n**Last {len(recent)} reports:** average absolute 1-day move {_pct(sum(moves) / len(moves))} "
            f"versus average implied move {_pct(sum(expected) / len(expected))}. A realised move above the "
            "implied one favours owning options into the report; below it favours selling premium.\n"
        )
    rows = [
        "| Report date | Time | EPS | Estimate | Implied move | 1-day move | Long straddle 1d | Short straddle 1d |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in recent:
        rows.append(
            f"| {_day(r.get('report_date'))} | {r.get('report_time', '?')} | {r.get('actual_eps', '?')} | "
            f"{r.get('street_mean_est', '?')} | {_pct(r.get('expected_move_perc'))} | "
            f"{_pct(r.get('post_earnings_move_1d'))} | {_pct(r.get('long_straddle_1d'))} | "
            f"{_pct(r.get('short_straddle_1d'))} |"
        )
    return header + next_line + stats + "\n" + "\n".join(rows) + "\n"
