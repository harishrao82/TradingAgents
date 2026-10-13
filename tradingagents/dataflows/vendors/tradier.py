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
from datetime import datetime

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

# Swing trades on this desk hold days to weeks, so the default chain window is
# the expiries a 14-60 day trade would use.
DEFAULT_MIN_DTE = 14
DEFAULT_MAX_DTE = 60

# Expirations shown per call, and strikes shown each side of the money.
MAX_EXPIRATIONS = 3
STRIKES_EACH_SIDE = 5


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
    try:
        response = requests.get(
            f"{_base_url()}/{path}",
            params=params,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise VendorUnavailableError(f"Tradier request failed: {exc}") from None
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


def _pct(value: float | None) -> str:
    return "?" if value is None else f"{value * 100:.1f}%"


def _price(value) -> str:
    number = _num(value)
    return "?" if number is None else f"{number:.2f}"


def _summarise_expiration(expiration: str, days: int, options: list, spot: float) -> str:
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

    def near_delta(book: dict, target: float):
        quoted = [o for o in book.values() if _iv(o) and _num((o.get("greeks") or {}).get("delta")) is not None]
        return min(quoted, key=lambda o: abs(_num(o["greeks"]["delta"]) - target), default=None)

    put25, call25 = near_delta(puts, -0.25), near_delta(calls, 0.25)
    skew = (_iv(put25) - _iv(call25)) if put25 and call25 else None
    call_oi = sum(int(_num(o.get("open_interest")) or 0) for o in calls.values())
    put_oi = sum(int(_num(o.get("open_interest")) or 0) for o in puts.values())
    kind = options[0].get("expiration_type") or ""

    lines = [f"\n### {expiration} ({days} days{', ' + kind if kind else ''})"]
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
    if call_oi + put_oi == 0:
        lines.append("- No open interest yet: a newly listed expiry, likely thin; prefer a listed monthly.")

    index = strikes.index(atm)
    window = strikes[max(0, index - STRIKES_EACH_SIDE): index + STRIKES_EACH_SIDE + 1]
    lines.append(
        "\n| Strike | Call bid/ask | Call delta | Call IV | Call OI | Put bid/ask | Put delta | Put IV | Put OI |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for strike in window:
        c, p = calls.get(strike, {}), puts.get(strike, {})
        cd, pd_ = (c.get("greeks") or {}).get("delta"), (p.get("greeks") or {}).get("delta")
        lines.append(
            f"| {strike}{' (ATM)' if strike == atm else ''} | {_price(c.get('bid'))}/{_price(c.get('ask'))} | "
            f"{'?' if _num(cd) is None else f'{_num(cd):.2f}'} | {_pct(_iv(c))} | {c.get('open_interest', '?')} | "
            f"{_price(p.get('bid'))}/{_price(p.get('ask'))} | {'?' if _num(pd_) is None else f'{_num(pd_):.2f}'} | "
            f"{_pct(_iv(p))} | {p.get('open_interest', '?')} |"
        )
    return "\n".join(lines) + "\n"


def get_option_chain(
    symbol: str,
    as_of_date: str,
    min_dte: int | None = None,
    max_dte: int | None = None,
) -> str:
    """Live option chains for the expiries ``min_dte``-``max_dte`` days out.

    For each expiry: the at-the-money IV, the straddle's implied move, the
    25-delta skew, open interest, and the strikes around the money with bid,
    ask, delta, IV and open interest.
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
    shown = 0
    for date, days in in_window:
        if shown == MAX_EXPIRATIONS:
            break
        chain = _request("markets/options/chains", {"symbol": ticker, "expiration": date, "greeks": "true"})
        options = _as_list((chain.get("options") or {}).get("option"))
        if not options:
            continue
        parts.append(_summarise_expiration(date, days, options, spot))
        shown += 1
    if shown == 0:
        raise NoMarketDataError(ticker, detail=f"no chain data for expirations {min_dte}-{max_dte} days out")
    parts.append(
        "\nIVs are annualised; delta is per share; a contract with no bid shows no IV. "
        "Bid/ask are per share (x100 per contract).\n"
    )
    return "".join(parts)
