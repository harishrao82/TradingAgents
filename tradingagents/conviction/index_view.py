"""The ETF view: every directional news call weighted by the name's weight in the ETF.

Each name adds weight % x (+1 up, -1 down) x conviction / 5, and nothing when
its news points neither way or it had no news. The sum, the tilt, is in
index-weight points: +100 would be every holding a 5/5 up, and a 5/5 up on a
7% name adds +7. The headline uses conviction 4-5 calls only.
"""
from __future__ import annotations

from .screen import NewsVerdict

TOP_MOVERS = 8
TOP_HOLDINGS = 10
CONCENTRATED = 0.5  # warn when the two biggest contributors make up this share of the tilt


def label(tilt: float) -> str:
    """Provisional cut-offs, not yet calibrated against how the index moved afterwards."""
    if tilt >= 4:
        return "bullish"
    if tilt >= 1.5:
        return "mildly bullish"
    if tilt > -1.5:
        return "neutral"
    return "mildly bearish" if tilt > -4 else "bearish"


def _signal(v: NewsVerdict) -> float:
    return {"up": 1, "down": -1}.get(v.direction, 0) * v.conviction / 5


def _contributions(verdicts, holdings, min_conviction=1):
    return {v.ticker: (holdings[v.ticker][0] or 0) * _signal(v) for v in verdicts
            if v.direction != "none" and v.conviction >= min_conviction and v.ticker in holdings}


def render_index_view(etf: str, verdicts: list[NewsVerdict], failures: dict[str, str],
                      holdings: dict[str, tuple[float | None, str | None]], news_names) -> str:
    """The view section for the screen report, or "" when there are no weights to use."""
    if not any(weight for weight, _ in holdings.values()):
        return ""
    by_ticker = {v.ticker: v for v in verdicts}
    tilts = {floor: sum(_contributions(verdicts, holdings, floor).values()) for floor in (1, 3, 4)}
    contrib = _contributions(verdicts, holdings)

    def weight(ticker):
        return holdings[ticker][0] or 0

    ups = [t for t, c in contrib.items() if c > 0]
    downs = [t for t, c in contrib.items() if c < 0]
    total = sum(weight(t) for t in holdings)
    up_w, down_w = sum(weight(t) for t in ups), sum(weight(t) for t in downs)

    lines = [
        f"## {etf} view: {label(tilts[4])} ({tilts[4]:+.2f})", "",
        f"Each name's news call weighted by its {etf} weight, in index-weight points "
        "(a 5/5 up on a 7% name adds +7). The headline counts conviction 4-5 calls only.", "",
        "| Calls counted | Tilt | Reading |", "|---|---|---|",
        f"| Conviction 4-5 | {tilts[4]:+.2f} | {label(tilts[4])} |",
        f"| Conviction 3-5 | {tilts[3]:+.2f} | {label(tilts[3])} |",
        f"| All | {tilts[1]:+.2f} | {label(tilts[1])} |", "",
        f"**Breadth:** {len(ups)} up, {len(downs)} down. {up_w / total:.1%} of the index weight has an up call, "
        f"{down_w / total:.1%} a down call and {1 - (up_w + down_w) / total:.1%} no call.", "",
    ]

    top_two = sorted(contrib, key=lambda t: -abs(contrib[t]))[:2]
    if tilts[1] and len(top_two) == 2:
        share = sum(contrib[t] for t in top_two) / tilts[1]
        if share >= CONCENTRATED:
            calls = " and ".join(f"{t} ({by_ticker[t].conviction}/5)" for t in top_two)
            lines += [f"**Concentrated:** {calls} make up {share:.0%} of the all-calls tilt.", ""]

    silent = []
    for ticker, _ in sorted(holdings.items(), key=lambda item: -(item[1][0] or 0))[:TOP_HOLDINGS]:
        if ticker in contrib:
            continue
        why = ("no news" if ticker not in news_names else f"not screened: {failures[ticker]}"
               if ticker in failures else "news points neither way")
        silent.append(f"{ticker} {weight(ticker):.1f}% ({why})")
    if silent:
        lines += ["**Biggest holdings with no call:** " + ", ".join(silent) + ".", ""]

    lines += ["| Ticker | Weight | Call | Contribution | Catalyst |", "|---|---|---|---|---|"]
    movers = (sorted(ups, key=lambda t: -contrib[t])[:TOP_MOVERS]
              + sorted(downs, key=lambda t: contrib[t])[:TOP_MOVERS])
    lines += [f"| {t} | {weight(t):.2f}% | {by_ticker[t].direction} {by_ticker[t].conviction} | "
              f"{contrib[t]:+.2f} | {by_ticker[t].catalyst} |" for t in movers]

    sectors: dict[str, list] = {}
    for ticker, (w, sector) in holdings.items():
        row = sectors.setdefault(sector or "Unknown", [0.0, 0.0, 0, 0])
        row[0] += w or 0
        row[1] += contrib.get(ticker, 0)
        row[2] += ticker in ups
        row[3] += ticker in downs
    if set(sectors) != {"Unknown"}:
        lines += ["", "| Sector | Weight | Tilt | Up | Down |", "|---|---|---|---|---|"]
        lines += [f"| {s} | {r[0]:.1f}% | {r[1]:+.2f} | {r[2]} | {r[3]} |"
                  for s, r in sorted(sectors.items(), key=lambda item: -item[1][0])]

    lines += ["", f"News only: no prices, flows or macro. Weights are {etf}'s current holdings.", ""]
    return "\n".join(lines) + "\n"
