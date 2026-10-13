"""A SPY view from a conviction-screen report: each directional call weighted by the name's SPY weight.

    python scripts/spy_view.py ~/.tradingagents/logs/screens/SPY_2026-10-13_1111.md

Reads the report's "Every directional call" table, takes today's weights from
Unusual Whales, and adds up how much of SPY the news points up or down, both
by plain weight and by weight x conviction / 5.
"""
from __future__ import annotations

import sys
from pathlib import Path

from tradingagents.dataflows.vendors.unusual_whales import get_etf_holdings

TILT = 0.15  # net / (up + down) beyond this reads as a lean rather than neutral


def read_calls(path: Path) -> list[tuple[str, str, int, str]]:
    text = path.read_text()
    section = text.split("## Every directional call", 1)[1]
    calls = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[1] in ("up", "down") and cells[2].isdigit():
            calls.append((cells[0], cells[1], int(cells[2]), cells[3]))
    return calls


def view(calls, weights: dict[str, float | None]) -> str:
    rows, missing = [], []
    for ticker, direction, conviction, catalyst in calls:
        weight = weights.get(ticker)
        if not weight:
            missing.append(ticker)
            continue
        sign = 1 if direction == "up" else -1
        rows.append((ticker, direction, conviction, catalyst, weight, sign * weight * conviction / 5))

    out = []
    for label, subset in (("All calls", rows), ("Conviction 4-5", [r for r in rows if r[2] >= 4])):
        ups = [r for r in subset if r[1] == "up"]
        downs = [r for r in subset if r[1] == "down"]
        score = sum(r[5] for r in subset)
        gross = sum(abs(r[5]) for r in subset)
        tilt = score / gross if gross else 0.0
        lean = "bullish" if tilt > TILT else "bearish" if tilt < -TILT else "neutral"
        out += [f"## {label}: {lean} (tilt {tilt:+.2f})", "",
                f"- Up: {len(ups)} names, {sum(r[4] for r in ups):.1f}% of SPY",
                f"- Down: {len(downs)} names, {sum(r[4] for r in downs):.1f}% of SPY",
                f"- Conviction-weighted net: {score:+.2f}% of SPY (gross {gross:.2f}%)", ""]

    total_weight = sum(w for w in weights.values() if w)
    covered = sum(r[4] for r in rows)
    out += [f"Directional calls cover {covered:.1f}% of SPY's {total_weight:.1f}% listed weight; "
            "the rest had no news or news that didn't point either way.", ""]

    out += ["## Biggest contributors", "", "| Ticker | Direction | Conviction | SPY weight | Contribution | Catalyst |",
            "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -abs(r[5]))[:15]:
        out.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[4]:.2f}% | {r[5]:+.2f} | {r[3]} |")
    if missing:
        out += ["", "No SPY weight for: " + ", ".join(missing)]
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        sys.exit(__doc__)
    path = Path(argv[1]).expanduser()
    calls = read_calls(path)
    weights = dict(get_etf_holdings("SPY"))
    print(f"# SPY view from {path.name}\n")
    print(view(calls, weights))


if __name__ == "__main__":
    main(sys.argv)
