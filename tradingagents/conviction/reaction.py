"""Pre-market reaction check: does the market's move this morning agree with the news call?

For each name with a strong call, the pre-market move is the Tradier bid/ask mid
against the prior close (``last`` doesn't move before the open; the quotes do).
The week's implied move is the at-the-money straddle on the first expiry on or
after Friday. A name whose gap agrees with its news is a cleaner one-day setup
than one the market is already reading the other way.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from tradingagents.dataflows.vendors.tradier import _as_list, _num, _quote, _request

from .levels import near_expiries, quote_price, straddle
from .screen import NewsVerdict

logger = logging.getLogger(__name__)

FLAT = 0.003         # a gap under 0.3% says nothing either way
WIDE_QUOTE = 0.01    # a spread over 1% of the price makes the gap unreliable


@dataclass
class Reaction:
    gap: float | None = None          # pre-market move from the prior close
    spread: float | None = None       # bid/ask spread as a share of the price
    implied: float | None = None      # week's implied move as a share of the price
    status: str = "unknown"           # agree, disagree, flat or unknown
    error: str = ""
    nearest_expiry: str | None = None  # first listed option expiry on or after the date

    @property
    def unreliable(self) -> bool:
        return self.spread is None or self.spread > WIDE_QUOTE


RANK = {"agree": 0, "flat": 1, "unknown": 1, "disagree": 2}


def classify(direction: str, gap: float | None) -> str:
    if gap is None:
        return "unknown"
    if abs(gap) < FLAT:
        return "flat"
    return "agree" if (gap > 0) == (direction == "up") else "disagree"


def check(verdict: NewsVerdict, date: str) -> Reaction:
    quote = _quote(verdict.ticker)
    price, _ = quote_price(quote)
    prev = _num(quote.get("prevclose"))
    bid, ask = _num(quote.get("bid")), _num(quote.get("ask"))
    gap = price / prev - 1 if price and prev else None
    spread = (ask - bid) / ((ask + bid) / 2) if bid and ask else None
    implied, expiries = None, []
    try:
        expiries = near_expiries(verdict.ticker, date)
        if expiries and price:
            chain = _request("markets/options/chains", {"symbol": verdict.ticker, "expiration": expiries[-1]})
            found = straddle(_as_list((chain.get("options") or {}).get("option")), prev or price)
            implied = found[1] / (prev or price) if found else None
    except Exception as exc:  # the implied move is context; the gap still stands without it
        logger.warning("No implied move for %s: %s", verdict.ticker, exc)
    return Reaction(gap, spread, implied, classify(verdict.direction, gap),
                    nearest_expiry=expiries[0] if expiries else None)


def check_all(verdicts: list[NewsVerdict], min_conviction: int, date: str) -> dict[str, Reaction]:
    reactions = {}
    for v in verdicts:
        if v.direction == "none" or v.conviction < min_conviction:
            continue
        try:
            reactions[v.ticker] = check(v, date)
        except Exception as exc:  # one name's missing quote must not stop the check
            logger.warning("No pre-market quote for %s: %s", v.ticker, exc)
            reactions[v.ticker] = Reaction(error=str(exc)[:120])
    return reactions


def _pct(value: float | None, signed: bool = True) -> str:
    return "?" if value is None else (f"{value:+.1%}" if signed else f"±{value:.1%}")


def tradable(reaction: Reaction, last_expiry: str) -> bool:
    """Whether the name lists an expiry by ``last_expiry``; unknown counts as tradable."""
    return reaction.nearest_expiry is None or reaction.nearest_expiry <= last_expiry


def render(verdicts: list[NewsVerdict], reactions: dict[str, Reaction], horizon: str | None = None) -> str:
    by_ticker = {v.ticker: v for v in verdicts}
    order = sorted(reactions, key=lambda t: (RANK[reactions[t].status], -by_ticker[t].conviction, t))
    lines = ["## News against the pre-market move", "",
             "Agree: the gap goes the news's way. Disagree: the market is reading it the other way; "
             "wait for the first 15-30 minutes after the open before trusting either. "
             "Pre-market quotes are thin.", "",
             "| Ticker | Call | Pre-market | Week implied | Reaction | Catalyst |", "|---|---|---|---|---|---|"]
    for t in order:
        r, v = reactions[t], by_ticker[t]
        status = r.status + (" (wide quote)" if r.status != "unknown" and r.unreliable else "")
        if r.nearest_expiry and r.nearest_expiry > (horizon or r.nearest_expiry):
            status += f"; no options before {r.nearest_expiry}"
        lines.append(f"| {t} | {v.direction} {v.conviction} | {_pct(r.gap)} | {_pct(r.implied, False)} | "
                     f"{status} | {v.catalyst} |")
    return "\n".join(lines) + "\n"
