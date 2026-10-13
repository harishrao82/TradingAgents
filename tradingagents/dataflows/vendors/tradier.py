"""Tradier vendor: live option chains with greeks, for strike and expiry selection.

Reads Tradier's market data API (https://docs.tradier.com) with the bearer token
in ``TRADIER_ACCESS_TOKEN``. ``TRADIER_SANDBOX=true`` points it at the sandbox
host instead. Without a token every call raises ``TradierNotConfiguredError``,
which the routing layer treats as "vendor unavailable".

Option chains and quotes are live only: Tradier serves no historical chain, so a
run dated before today gets an explicit "not available" rather than today's
prices. Only market data endpoints are called here; nothing reads the account
or places orders.
"""
import logging
import os
import time
from datetime import datetime, timedelta

import requests

from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)

logger = logging.getLogger(__name__)

TRADIER_API_BASE = "https://api.tradier.com/v1"
TRADIER_SANDBOX_BASE = "https://sandbox.tradier.com/v1"

# Network timeout (seconds) so a stalled request can't hang the agents.
REQUEST_TIMEOUT = 30
RETRIES = 2
RETRY_PAUSE = 2  # seconds, growing with each retry
_sleep = time.sleep

# Swing trades on this desk hold days to weeks, so the default chain window is
# the expiries a 14-60 day trade would use.
DEFAULT_MIN_DTE = 14
DEFAULT_MAX_DTE = 60

# Expirations shown per call, chains fetched at most to find them, and strikes
# shown each side of the money.
MAX_EXPIRATIONS = 3
MAX_CHAIN_FETCHES = 6
STRIKES_EACH_SIDE = 5

# A quote whose spread is wider than this share of its mid is marked as wide.
WIDE_SPREAD = 0.25


class TradierNotConfiguredError(VendorNotConfiguredError):
    """Raised when Tradier is selected but no access token is configured."""


def get_api_key() -> str:
    """The Tradier access token from the environment."""
    token = os.getenv("TRADIER_ACCESS_TOKEN")
    if not token:
        raise TradierNotConfiguredError(
            "TRADIER_ACCESS_TOKEN environment variable is not set. Create a token "
            "at https://dash.tradier.com/settings/api."
        )
    return token


def _base_url() -> str:
    sandbox = os.getenv("TRADIER_SANDBOX", "").strip().lower() in ("1", "true", "yes", "on")
    return TRADIER_SANDBOX_BASE if sandbox else TRADIER_API_BASE


def _request(path: str, params: dict) -> dict:
    """GET a market data endpoint and return its JSON body.

    A rejected token is "not configured"; throttling, a server error or a
    network failure is "unavailable", which says nothing about the instrument.
    """
    token = get_api_key()
    # Tradier now and then takes just over the timeout, or answers 502-504 from
    # its gateway, on a request that usually comes back in a fraction of a
    # second, so those get retried after a short pause.
    for attempt in range(RETRIES + 1):
        last = attempt == RETRIES
        try:
            response = requests.get(
                f"{_base_url()}/{path}",
                params=params,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.Timeout as exc:
            if last:
                raise VendorUnavailableError(f"Tradier request timed out {attempt + 1} times: {exc}") from None
            logger.info("Tradier /%s timed out; retrying.", path)
            _sleep(RETRY_PAUSE * (attempt + 1))
            continue
        except requests.RequestException as exc:
            raise VendorUnavailableError(f"Tradier request failed: {exc}") from None
        if response.status_code in (502, 503, 504) and not last:
            logger.info("Tradier /%s returned %s; retrying.", path, response.status_code)
            _sleep(RETRY_PAUSE * (attempt + 1))
            continue
        break
    if response.status_code == 401:
        raise TradierNotConfiguredError("Tradier rejected the access token (401). Check TRADIER_ACCESS_TOKEN.")
    if response.status_code == 400:
        raise ValueError(f"Tradier rejected the request to /{path}: {response.text[:300]}")
    if response.status_code == 429 or response.status_code >= 500:
        raise VendorUnavailableError(f"Tradier returned {response.status_code} for /{path}.")
    response.raise_for_status()
    return response.json() or {}


def _as_list(value) -> list:
    """Tradier turns a one-item list into the item itself, and an empty one into null."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quote(symbol: str) -> dict:
    quotes = _request("markets/quotes", {"symbols": symbol}).get("quotes") or {}
    rows = _as_list(quotes.get("quote"))
    if not rows:
        raise NoMarketDataError(symbol, detail="Tradier has no quote for it")
    return rows[0]


def _iv(option: dict) -> float | None:
    """The option's implied volatility, or None where the quote can't support one.

    A contract with no bid prints a meaningless mid IV (0.998 on a deep
    out-of-the-money put), so it gets none; ORATS' smoothed ``smv_vol`` is the
    fallback when the mid is missing.
    """
    greeks = option.get("greeks") or {}
    if not _num(option.get("bid")):
        return None
    mid = _num(greeks.get("mid_iv"))
    return mid if mid else _num(greeks.get("smv_vol"))


def _quote_text(option: dict) -> str:
    """``bid/ask``, marked ``(wide)`` when the spread is a large share of the mid."""
    bid, ask = _num(option.get("bid")), _num(option.get("ask"))
    text = f"{_price(bid)}/{_price(ask)}"
    if bid and ask and (ask - bid) / ((ask + bid) / 2) > WIDE_SPREAD:
        text += " (wide)"
    return text


def _pct(value: float | None) -> str:
    return "?" if value is None else f"{value * 100:.1f}%"


def _price(value) -> str:
    number = _num(value)
    return "?" if number is None else f"{number:.2f}"


def _summarise_expiration(expiration: str, days: int, options: list, spot: float, earnings: str | None = None) -> str:
    calls = {o["strike"]: o for o in options if o.get("option_type") == "call" and _num(o.get("strike")) is not None}
    puts = {o["strike"]: o for o in options if o.get("option_type") == "put" and _num(o.get("strike")) is not None}
    strikes = sorted(set(calls) | set(puts), key=float)
    if not strikes:
        return f"\n### {expiration} ({days} days): no contracts listed.\n"

    atm = min(strikes, key=lambda k: abs(float(k) - spot))
    call, put = calls.get(atm, {}), puts.get(atm, {})
    mids = [(_num(o.get("bid")) or 0.0, _num(o.get("ask"))) for o in (call, put)]
    straddle = sum((b + a) / 2 for b, a in mids if a) if all(a for _, a in mids) else None
    atm_ivs = [iv for iv in (_iv(call), _iv(put)) if iv]
    atm_iv = sum(atm_ivs) / len(atm_ivs) if atm_ivs else None

    def near_delta(book: dict, target: float, otm):
        # Out-of-the-money contracts only: in-the-money quotes are wide and
        # their implied volatilities unreliable.
        quoted = [o for o in book.values()
                  if otm(float(o["strike"])) and _iv(o) and _num((o.get("greeks") or {}).get("delta")) is not None]
        return min(quoted, key=lambda o: abs(_num(o["greeks"]["delta"]) - target), default=None)

    put25 = near_delta(puts, -0.25, lambda k: k < spot)
    call25 = near_delta(calls, 0.25, lambda k: k > spot)
    skew = (_iv(put25) - _iv(call25)) if put25 and call25 else None
    call_oi = sum(int(_num(o.get("open_interest")) or 0) for o in calls.values())
    put_oi = sum(int(_num(o.get("open_interest")) or 0) for o in puts.values())
    kind = options[0].get("expiration_type") or ""

    tags = [f"{days} days", *([kind] if kind else []), *([f"spans earnings {earnings}"] if earnings else [])]
    lines = [f"\n### {expiration} ({', '.join(tags)})"]
    facts = [f"ATM strike {atm}"]
    if atm_iv is not None:
        facts.append(f"ATM IV {_pct(atm_iv)}")
    if straddle is not None:
        facts.append(f"ATM straddle {straddle:.2f}, an implied move of about ±{straddle / spot * 100:.1f}% by expiry")
    if skew is not None:
        facts.append(f"25-delta skew (put IV minus call IV) {skew * 100:+.1f} points")
    facts.append(
        f"open interest calls {call_oi}, puts {put_oi}"
        + (f", put/call {put_oi / call_oi:.2f}" if call_oi else "")
    )
    lines.append("- " + "; ".join(facts))

    index = strikes.index(atm)
    window = strikes[max(0, index - STRIKES_EACH_SIDE): index + STRIKES_EACH_SIDE + 1]
    lines.append(
        "\n| Strike | IV (OTM side) | Call bid/ask | Call delta | Call OI | Put bid/ask | Put delta | Put OI |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for strike in window:
        c, p = calls.get(strike, {}), puts.get(strike, {})
        cd, pd_ = (c.get("greeks") or {}).get("delta"), (p.get("greeks") or {}).get("delta")
        # Each strike's IV comes from its out-of-the-money side; the ATM row averages both.
        iv = atm_iv if strike == atm else _iv(p) if float(strike) < spot else _iv(c)
        lines.append(
            f"| {strike}{' (ATM)' if strike == atm else ''} | {_pct(iv)} | {_quote_text(c)} | "
            f"{'?' if _num(cd) is None else f'{_num(cd):.2f}'} | {c.get('open_interest', '?')} | "
            f"{_quote_text(p)} | {'?' if _num(pd_) is None else f'{_num(pd_):.2f}'} | {p.get('open_interest', '?')} |"
        )
    return "\n".join(lines) + "\n"


def _open_interest(options: list) -> int:
    return sum(int(_num(o.get("open_interest")) or 0) for o in options)


def _is_monthly(date: str) -> bool:
    """Whether an expiry is the standard monthly: the third Friday of its month."""
    day = datetime.strptime(date, "%Y-%m-%d")
    return day.weekday() == 4 and 15 <= day.day <= 21


def _is_month_end(date: str) -> bool:
    """Whether an expiry is the last weekday of its month, where index end-of-month options expire."""
    day = datetime.strptime(date, "%Y-%m-%d")
    following = day + timedelta(days=1 if day.weekday() < 4 else 7 - day.weekday())
    return day.weekday() < 5 and following.month != day.month


def _expiry_order(in_window: list) -> list:
    """The order to try expiries in: the nearest, a middle one, the furthest, then the rest.

    The middle pick is the standard monthly or month-end expiry closest to the
    middle of the window, where open interest is deepest. Showing the first few
    dates instead would give adjacent dailies on an index.
    """
    dates = [d for d, _ in in_window]
    ordinal = {d: datetime.strptime(d, "%Y-%m-%d").toordinal() for d in dates}
    middle = (ordinal[dates[0]] + ordinal[dates[-1]]) / 2

    def distance(d):
        return abs(ordinal[d] - middle)

    anchors = sorted((d for d in dates[1:-1] if _is_monthly(d) or _is_month_end(d)), key=distance)
    rest = sorted(dates, key=distance)
    order = list(dict.fromkeys([dates[0], *anchors[:1], dates[-1], *anchors[1:], *rest]))
    by_date = dict(in_window)
    return [(d, by_date[d]) for d in order]


def _spans_earnings(expiration: str, report_date: str, report_time: str | None) -> bool:
    """Whether an expiry is still open after the earnings report moves the stock.

    A report before the open on expiry day moves the stock while the option
    still trades; one after the close comes too late.
    """
    if expiration == report_date:
        return (report_time or "").lower() in ("premarket", "bmo", "pre")
    return expiration > report_date


def get_option_chain(
    symbol: str,
    as_of_date: str,
    min_dte: int | None = None,
    max_dte: int | None = None,
    earnings_date: str | None = None,
    earnings_time: str | None = None,
) -> str:
    """Live option chains for the expiries ``min_dte``-``max_dte`` days out.

    For each expiry: the at-the-money IV, the straddle's implied move, the
    25-delta skew, open interest, and the strikes around the money with bid,
    ask, delta, IV and open interest. Given the next earnings date, each expiry
    still open after the report is tagged as spanning it.
    """
    ticker = symbol.upper()
    header = f"## Tradier option chains: {ticker}, as of {as_of_date}\n"
    if is_historical(as_of_date):
        return header + (
            f"\nDATA_UNAVAILABLE: Tradier serves live chains only, so it has no chain as of "
            f"{as_of_date}. Do not estimate strikes, premiums or greeks for that date.\n"
        )
    min_dte = DEFAULT_MIN_DTE if min_dte is None else int(min_dte)
    max_dte = DEFAULT_MAX_DTE if max_dte is None else int(max_dte)

    quote = _quote(ticker)
    spot = _num(quote.get("last")) or _num(quote.get("close")) or _num(quote.get("prevclose"))
    if not spot:
        raise NoMarketDataError(ticker, detail="Tradier quote has no price")

    expirations = _as_list(
        (_request("markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
         .get("expirations") or {}).get("date")
    )
    if not expirations:
        raise NoMarketDataError(ticker, detail="no listed options at Tradier")
    today = datetime.strptime(as_of_date, "%Y-%m-%d")
    in_window = [
        (date, (datetime.strptime(date, "%Y-%m-%d") - today).days)
        for date in expirations
        if min_dte <= (datetime.strptime(date, "%Y-%m-%d") - today).days <= max_dte
    ]
    if not in_window:
        return header + (
            f"\nNo expiration {min_dte}-{max_dte} days out. Listed: "
            f"{', '.join(expirations[:12])}{' ...' if len(expirations) > 12 else ''}\n"
        )

    parts = [header, f"\nUnderlying {ticker} last {spot:.2f}. Expirations {min_dte}-{max_dte} days out: "
             f"{', '.join(d for d, _ in in_window)}.\n"]
    spanning = set()
    if earnings_date and earnings_date >= as_of_date:
        spanning = {d for d, _ in in_window if _spans_earnings(d, earnings_date, earnings_time)}
        listed = ", ".join(sorted(spanning)) or "none of them"
        parts.append(f"Next earnings {earnings_date} ({earnings_time or 'time unknown'}). Expirations in the "
                     f"window that span it and carry the earnings move in their IV: {listed}.\n")
    shown, skipped, fetched = [], [], 0
    for date, days in _expiry_order(in_window):
        if len(shown) == MAX_EXPIRATIONS or fetched == MAX_CHAIN_FETCHES:
            break
        chain = _request("markets/options/chains", {"symbol": ticker, "expiration": date, "greeks": "true"})
        fetched += 1
        options = _as_list((chain.get("options") or {}).get("option"))
        if not options:
            continue
        # A newly listed expiry has no open interest yet and quotes too thin to read.
        if _open_interest(options) == 0:
            skipped.append(date)
            continue
        shown.append((date, _summarise_expiration(date, days, options, spot,
                                                  earnings_date if date in spanning else None)))
    if not shown:
        detail = f"no chain with open interest for expirations {min_dte}-{max_dte} days out"
        raise NoMarketDataError(ticker, detail=detail)
    if skipped:
        parts.append(f"Skipped for having no open interest yet: {', '.join(skipped)}.\n")
    parts.extend(text for _, text in sorted(shown))
    parts.append(
        "\nIVs are annualised, read from the out-of-the-money side of each strike (the ATM row "
        "averages both); a contract with no bid shows no IV. Delta is per share; bid/ask are per "
        "share (x100 per contract); (wide) marks a spread over 25% of the mid.\n"
        + ("" if earnings_date else "No earnings date was available: an expiry after the next report "
           "carries the earnings move in its IV, so check the earnings calendar.\n")
    )
    return "".join(parts)
