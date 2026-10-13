"""Basket tilt: the desk's checks nudge the weights of an equal-weight stock basket.

Built for the REL-HYBRID basket (10 momentum + 10 pullback names picked by
SPY-relative Sortino). The basket itself is never changed here; each name
gets a multiplier on its equal weight from two checks:

1. The news screen (the same cheap-model screen as ``tradingagents screen``):
   a fresh catalyst pointing up raises the weight, one pointing down cuts it.
2. Earnings before the next rebalance are noted on each name. Setting
   ``tilt_earnings_hold_days`` above 0 caps an up tilt at equal weight when the
   report lands within that many weekdays; it is off by default.

Multipliers are clipped to [low, high] (0.5x-1.5x by default) and rescaled so
they average 1, which keeps the whole book invested. A strong down call
(conviction 5) also flags the name as a drop candidate; dropping it stays the
human's call (the rebalance cell's EXCLUDE set).

Options flow, GEX, IV, short interest, insider and dark-pool data are left out
on purpose: they were backtested on this basket (2022-2026) as filters and
re-rankers and rejected. The news tilt table below is a starting point, not a
backtested rule.

Analysis only: nothing here reads an account or places an order.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date as _date, datetime, timedelta
from pathlib import Path

from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.vendors import unusual_whales

from .screen import NewsVerdict
from .workflow import run_conviction_screen

logger = logging.getLogger(__name__)

# Weight multiplier for each (direction, conviction) news call; anything missing is 1.0.
TILT_BY_CALL = {
    ("up", 5): 1.5, ("up", 4): 1.3, ("up", 3): 1.1,
    ("down", 3): 0.85, ("down", 4): 0.6, ("down", 5): 0.5,
}
DROP_FLAG = ("down", 5)


@dataclass
class NameTilt:
    ticker: str
    bucket: str                      # "momentum" or "pullback"
    raw: float                       # multiplier before clipping and rescaling
    multiplier: float                # final weight relative to equal weight (1.0 = equal)
    weight: float                    # share of the book
    call: str = "no news"            # e.g. "up 4"
    catalyst: str = ""
    earnings: str | None = None      # next report date when it falls before the next rebalance
    drop: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class BasketTilt:
    date: str
    until: str
    names: list[NameTilt]
    screen_model: str = ""
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def multipliers(self) -> dict[str, float]:
        return {n.ticker: n.multiplier for n in self.names}


def next_month_end(day: str) -> str:
    """The last weekday of the month after ``day``'s month: the next REL-HYBRID rebalance."""
    d = datetime.strptime(day, "%Y-%m-%d").date()
    first_after_next = _date(d.year + (d.month + 1) // 12, (d.month + 1) % 12 + 1, 1)
    last = first_after_next - timedelta(days=1)
    while last.weekday() >= 5:
        last -= timedelta(days=1)
    return last.isoformat()


def bounded_rescale(raw: dict[str, float], low: float, high: float, rounds: int = 50) -> dict[str, float]:
    """Rescale multipliers to average 1 while keeping each within [low, high].

    Names pinned at a bound keep it; the rest are scaled to absorb the
    difference, repeated until nothing moves past a bound.
    """
    if not raw:
        return {}
    n = len(raw)
    if not low <= 1.0 <= high:
        raise ValueError("the bounds must include equal weight (1.0)")
    pinned: dict[str, float] = {}
    for _ in range(rounds):
        free = {t: v for t, v in raw.items() if t not in pinned}
        left = n - sum(pinned.values())
        total = sum(free.values())
        if not free or total <= 0:
            break
        scaled = {t: v * left / total for t, v in free.items()}
        over = {t: high for t, v in scaled.items() if v > high + 1e-12}
        under = {t: low for t, v in scaled.items() if v < low - 1e-12}
        if not over and not under:
            return {**pinned, **scaled}
        pinned.update(over or under)  # pin one side per round so the other can still move
    return {t: pinned.get(t, 1.0) for t in raw}


def _weekdays_ahead(day: str, count: int) -> str:
    d = datetime.strptime(day, "%Y-%m-%d").date()
    for _ in range(count):
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
    return d.isoformat()


def tilt_names(momentum: list[str], pullback: list[str], verdicts: list[NewsVerdict],
               earnings: dict[str, str | None], low: float = 0.5, high: float = 1.5,
               hold_until: str | None = None) -> list[NameTilt]:
    """Turn news calls and earnings dates into one weight per basket name.

    An up tilt is held to equal weight when the report falls on or before
    ``hold_until`` (every listed report when it is None).
    """
    by_ticker = {v.ticker: v for v in verdicts}
    names = []
    for bucket, tickers in (("momentum", momentum), ("pullback", pullback)):
        for t in tickers:
            v = by_ticker.get(t)
            raw, notes = 1.0, []
            call, catalyst, drop = "no news", "", False
            if v is not None:
                call = f"{v.direction} {v.conviction}" if v.direction != "none" else "neutral"
                catalyst = v.catalyst
                raw = TILT_BY_CALL.get((v.direction, v.conviction), 1.0)
                drop = (v.direction, v.conviction) == DROP_FLAG
            report = earnings.get(t)
            if report and raw > 1.0 and (hold_until is None or report <= hold_until):
                notes.append(f"up tilt held to equal weight: earnings {report}")
                raw = 1.0
            elif report:
                notes.append(f"earnings {report}")
            if drop:
                notes.append("drop candidate: strong negative news")
            names.append(NameTilt(t, bucket, raw, 1.0, 0.0, call, catalyst, report, drop, notes))

    final = bounded_rescale({n.ticker: min(max(n.raw, low), high) for n in names}, low, high)
    count = len(names)
    for n in names:
        n.multiplier = round(final[n.ticker], 4)
        n.weight = n.multiplier / count if count else 0.0
    return names


def weighted_water_fill(cost: dict[str, float], mult: dict[str, float], avail: float) -> float:
    """The cost level T where filling each name up to ``mult * T`` spends exactly ``avail``.

    The weighted form of the rebalance cell's equal-cost top-up: a name whose
    cost basis already sits at or above its own level gets nothing (never
    trims); a new name (cost 0) gets ``mult * T``. With every multiplier at
    1.0 this is the cell's plain equal-cost T.
    """
    names = [s for s in cost if mult.get(s, 1.0) > 0]
    if not names or avail <= 0:
        return min((cost[s] / mult.get(s, 1.0) for s in names), default=0.0)

    def spend(level):
        return sum(max(mult.get(s, 1.0) * level - cost[s], 0.0) for s in names)

    lo, hi = 0.0, (avail + sum(cost[s] for s in names)) / min(mult.get(s, 1.0) for s in names)
    for _ in range(200):
        mid = (lo + hi) / 2
        if spend(mid) < avail:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _earnings_in_window(tickers: list[str], day: str, until: str) -> dict[str, str | None]:
    found = {}
    for t in tickers:
        try:
            nxt = unusual_whales.get_next_earnings(t, day)
        except Exception as exc:  # a missing date must not stop the tilt
            logger.warning("No earnings date for %s: %s", t, exc)
            nxt = None
        found[t] = nxt[0] if nxt and nxt[0] <= until else None
    return found


def run_basket_tilt(config: dict, day: str, momentum: list[str], pullback: list[str],
                    until: str | None = None, progress=None, llm=None) -> BasketTilt:
    """Screen the basket's news, look up earnings before ``until``, and tilt the weights."""
    until = until or next_month_end(day)
    tickers = momentum + pullback
    screen = run_conviction_screen(config, day, tickers=tickers, screen_only=True, progress=progress,
                                   llms=(llm, None) if llm else None)
    with run_config(config):
        earnings = _earnings_in_window(tickers, day, until)
    names = tilt_names(momentum, pullback, screen.verdicts, earnings,
                       config["tilt_min_multiplier"], config["tilt_max_multiplier"],
                       _weekdays_ahead(day, config["tilt_earnings_hold_days"]))
    return BasketTilt(day, until, names, screen.screen_model, screen.failures)


def render_tilt(result: BasketTilt) -> str:
    lines = [
        f"# Basket tilt, {result.date}",
        "",
        f"News screened by {result.screen_model}; earnings checked through {result.until} (next rebalance). "
        "Weights relative to equal weight; analysis only, no orders were placed.",
        "",
        "| Name | Bucket | News call | Weight x | Book % | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for n in result.names:
        why = "; ".join(([n.catalyst] if n.catalyst else []) + n.notes)
        lines.append(f"| {n.ticker} | {n.bucket} | {n.call} | {n.multiplier:.2f} | {n.weight:.1%} | {why} |")
    drops = [n.ticker for n in result.names if n.drop]
    if drops:
        lines += ["", "Drop candidates (add to EXCLUDE to act on them): " + ", ".join(drops)]
    if result.failures:
        lines += ["", "Not screened, left at equal weight: " + ", ".join(sorted(result.failures))]
    return "\n".join(lines) + "\n"


def save_tilt(result: BasketTilt, text: str, results_dir: str) -> tuple[Path, Path]:
    folder = Path(results_dir) / "tilts"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"basket_{result.date}_{datetime.now():%H%M}"
    md, js = folder / f"{stem}.md", folder / f"{stem}.json"
    md.write_text(text)
    js.write_text(json.dumps({"date": result.date, "until": result.until, "screen_model": result.screen_model,
                              "multipliers": result.multipliers, "failures": result.failures,
                              "names": [asdict(n) for n in result.names]}, indent=2))
    (folder / "latest.json").write_text(js.read_text())
    return md, js
