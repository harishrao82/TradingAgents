"""Does the options flow before the open agree with a news call?

Unusual Whales flow alerts between the prior session's open and 9:25 ET are
labelled by who started each trade: calls bought at the ask or puts sold at
the bid lean bullish, puts bought at the ask or calls sold at the bid lean
bearish. Only trades that look like bets count: all opening, single-leg, and
expiring within ``MAX_DTE`` days. Big long-dated, multi-leg or closing trades
are mostly hedges and spreads. The net (bullish minus bearish premium) agrees
with the call, disagrees, or is too small to say.

Market data only; nothing reads an account or places an order.
"""
from __future__ import annotations

from datetime import datetime

from tradingagents.dataflows.vendors import unusual_whales
from tradingagents.dataflows.vendors.unusual_whales import _num

LEAN = 0.6             # share of sided premium on one side to call a trade bought or sold
MAX_DTE = 60
MIN_NET = 250_000      # net premium below this says nothing either way
MIN_PREMIUM = 100_000  # alerts under this are not read
PAGES = 3


def lean(alert: dict) -> int:
    """+1 bullish, -1 bearish, 0 when the premium sits near the mid or isn't classified."""
    ask, bid = _num(alert.get("total_ask_side_prem")) or 0, _num(alert.get("total_bid_side_prem")) or 0
    if ask + bid <= 0:
        return 0
    share = ask / (ask + bid)
    side = 1 if share >= LEAN else -1 if share <= 1 - LEAN else 0
    return side if alert.get("type") == "call" else -side


def bet_like(alert: dict) -> bool:
    try:
        dte = (datetime.strptime(str(alert.get("expiry"))[:10], "%Y-%m-%d")
               - datetime.strptime(str(alert.get("created_at"))[:10], "%Y-%m-%d")).days
    except ValueError:
        return False
    return bool(alert.get("all_opening_trades")) and not alert.get("has_multileg") and 0 <= dte <= MAX_DTE


def net_flow(ticker: str, since: str, until: str) -> float:
    """Bullish minus bearish premium of the bet-like alerts on ``ticker`` from ``since`` to ``until`` (ISO times)."""
    params = {"ticker_symbol": ticker, "newer_than": since, "older_than": until, "limit": 200,
              "min_premium": MIN_PREMIUM}
    alerts = []
    for _ in range(PAGES):
        page = unusual_whales._request("option-trades/flow-alerts", params)
        alerts += page
        if len(page) < params["limit"]:
            break
        params = {**params, "older_than": min(str(a.get("created_at", "")) for a in page)}
    seen, net = set(), 0.0
    for a in alerts:
        key = a.get("id") or (a.get("created_at"), a.get("option_chain"), a.get("total_premium"))
        if key in seen or not bet_like(a):
            continue
        seen.add(key)
        net += lean(a) * (_num(a.get("total_premium")) or 0)
    return net


def agreement(direction: str, net: float | None) -> str:
    if net is None:
        return "unknown"
    if abs(net) < MIN_NET:
        return "neutral"
    return "agree" if (net > 0) == (direction == "up") else "disagree"
