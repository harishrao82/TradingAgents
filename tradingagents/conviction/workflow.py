"""Conviction screen: ETF holdings, then a cheap news screen, then options ideas.

1. The ETF's holdings come from Unusual Whales.
2. Each holding's Benzinga news for the last few days is read in bulk through
   Alpaca, and a cheap model (Claude Haiku 5.5 by default, optionally through
   Anthropic's Batches API at half price) screens it for a move up or down
   with conviction.
3. The strongest calls go to the options strategist on the deep model tier,
   which suggests a defined-risk trade for each.

The day mode (``screen_mode = "day"``) reads one to two weekdays of news for a
one- to five-day move, checks each strong call against this morning's
pre-market move, ranks the names the market agrees with first, and asks for
trades on 0-7 day expiries. An ETF screen dated today opens with its levels.

Analysis only: nothing here reads an account or places an order.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic

from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.vendors import alpaca, unusual_whales
from tradingagents.llm_clients.factory import (
    build_llm_kwargs,
    create_llm_client,
    create_tier_client,
)

from .batch import screen_news_batch
from .index_view import render_index_view
from .levels import get_levels
from .reaction import RANK, Reaction, check_all, render as render_reactions, tradable
from .screen import DAY_SCREEN_PROMPT, SCREEN_PROMPT, NewsVerdict, screen_news, shortlist
from .strategist import DAY, SWING, OptionsIdea, contracts_for, max_gain, sizing_note, suggest

logger = logging.getLogger(__name__)


@dataclass
class Idea:
    verdict: NewsVerdict
    idea: OptionsIdea | str
    contracts: int | None = None
    sized_down: str = ""    # why it holds fewer units than the per-trade budget allows


@dataclass
class ScreenResult:
    etf: str
    date: str
    start: str
    universe: int
    with_news: int
    verdicts: list[NewsVerdict]
    failures: dict[str, str]
    shortlist: list[NewsVerdict]
    ideas: list[Idea] = field(default_factory=list)
    screen_model: str = ""
    holdings: dict[str, tuple[float | None, str | None]] = field(default_factory=dict)
    news_names: list[str] = field(default_factory=list)
    levels: str = ""
    mode: str = "swing"
    reactions: dict[str, Reaction] = field(default_factory=dict)
    last_expiry: str | None = None
    market_open: bool = False
    skipped: list[str] = field(default_factory=list)   # day mode: strong calls the market is moving against


def screen_llm(config: dict):
    provider = config["screen_llm_provider"]
    kwargs = build_llm_kwargs({**config, "llm_provider": provider})
    return create_llm_client(provider=provider, model=config["screen_llm"], **kwargs).get_llm()


def _weekdays_back(date: str, days: int) -> str:
    """The start of a window of ``days`` weekdays ending on ``date``, so Monday's reaches back past the weekend."""
    day = datetime.strptime(date, "%Y-%m-%d")
    for _ in range(days - 1):
        day -= timedelta(days=1)
        while day.weekday() >= 5:
            day -= timedelta(days=1)
    return day.strftime("%Y-%m-%d")


def _market_open(date: str, now: datetime | None = None) -> bool:
    """Whether ``date`` is today in New York and the regular session has started."""
    now = now or datetime.now(ZoneInfo("America/New_York"))
    return now.strftime("%Y-%m-%d") == date and (now.hour, now.minute) >= (9, 30)


def cap_total_risk(ideas: list[Idea], budget: float, order) -> None:
    """Cut units from the ideas that come last in ``order`` until their combined max loss fits ``budget``."""
    left = budget
    for item in sorted(ideas, key=order):
        if not item.contracts:
            continue
        loss = item.idea.max_loss
        fit = min(item.contracts, max(0, math.floor(left / loss)))
        if fit < item.contracts:
            item.sized_down = f"cut from {item.contracts} to keep the screen's total risk within the cap"
            item.contracts = fit
        left -= fit * loss


def _days_after(date: str, days: int) -> str:
    return (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def run_conviction_screen(config: dict, date: str, etf: str = "SPY", tickers: list[str] | None = None,
                          equity: float | None = None, screen_only: bool = False, progress=None,
                          llms: tuple | None = None) -> ScreenResult:
    """Screen ``etf``'s holdings (or ``tickers``) as of ``date`` and suggest options for the short list."""
    with run_config(config):
        return _run(config, date, etf, tickers, equity, screen_only, progress, llms)


def _run(config, date, etf, tickers, equity, screen_only, progress, llms) -> ScreenResult:
    say = progress or (lambda message: None)
    day = config["screen_mode"] == "day"
    template = DAY_SCREEN_PROMPT if day else SCREEN_PROMPT
    holdings = ({t: (w, sector) for t, w, sector in unusual_whales.get_etf_constituents(etf)} if not tickers
                else {t.upper(): (None, None) for t in tickers})
    weights = {t: w for t, (w, _) in holdings.items()}
    say(f"{len(weights)} names in {etf if not tickers else 'the list'}")

    start = _weekdays_back(date, config["day_lookback_days" if day else "screen_lookback_days"])
    news = alpaca.get_news_by_symbol(list(weights), start, date, config["screen_articles_per_name"])
    say(f"{len(news)} names have Benzinga news from {start} to {date}")

    screener, strategist = llms or (None, None)
    if config["screen_batch"]:
        if config["screen_llm_provider"].lower() != "anthropic":
            raise ValueError("Batch screening uses Anthropic's Batches API; set screen_llm_provider to anthropic.")
        verdicts, failures = screen_news_batch(
            screener or anthropic.Anthropic(), config["screen_llm"], news, start, date,
            max_tokens=config["screen_max_tokens"], progress=say, template=template,
        )
    else:
        verdicts, failures = screen_news(
            screener or screen_llm(config), news, start, date, config["screen_workers"],
            per_minute=config["screen_requests_per_minute"],
            progress=lambda done, total, ticker: say(f"screened {done}/{total}") if done % 50 == 0 or done == total else None,
            template=template,
        )
    reactions, rank = {}, {}
    if day:
        say("checking strong calls against the pre-market move")
        reactions = check_all(verdicts, config["screen_min_conviction"], date)
        rank = {t: RANK[r.status] for t, r in reactions.items()}
    # A day trade needs an expiry within the window, so names that list none give their slot to the next.
    last_expiry = _days_after(date, config["day_max_dte"]) if day else None
    candidates = [v for v in verdicts if not day or v.ticker not in reactions
                  or tradable(reactions[v.ticker], last_expiry)]
    skipped = []
    if day and config["day_skip_disagree"]:
        skipped = sorted(t for t, r in reactions.items() if r.status == "disagree")
        candidates = [v for v in candidates if v.ticker not in skipped]
    picked = shortlist(candidates, config["screen_min_conviction"], config["screen_max_names"], weights,
                       config["screen_direction"], rank)
    say(f"{len(picked)} names on the short list: {', '.join(v.ticker for v in picked) or 'none'}")

    result = ScreenResult(etf=etf if not tickers else "custom list", date=date, start=start, universe=len(weights),
                          with_news=len(news), verdicts=verdicts, failures=failures, shortlist=picked,
                          screen_model=config["screen_llm"], holdings=holdings, news_names=list(news),
                          mode=config["screen_mode"], reactions=reactions, last_expiry=last_expiry,
                          market_open=_market_open(date), skipped=skipped)
    if not tickers and not is_historical(date):
        try:
            result.levels = get_levels(etf, date)
        except Exception as exc:  # levels add to the report; their absence shouldn't stop it
            logger.warning("No levels for %s: %s", etf, exc)
            result.levels = f"## {etf} levels\n\nUnavailable: {exc}\n"
    if screen_only or not picked:
        return result

    strategist = strategist or create_tier_client(config, "deep").get_llm()
    for verdict in picked:
        say(f"strategist: {verdict.ticker}")
        note = sizing_note(equity, config["risk_per_trade"], config["max_contracts"])
        if day and verdict.ticker in reactions and reactions[verdict.ticker].status == "disagree":
            note += ("This morning's pre-market move goes against the news. The plan must say to wait for the first "
                    "15-30 minutes after the open and enter only if the price turns the news's way.\n")
        dte = (config["day_min_dte"], config["day_max_dte"]) if day else (14, 60)
        try:
            idea = suggest(strategist, verdict, news[verdict.ticker], date, min_dte=dte[0], max_dte=dte[1],
                           horizon=DAY if day else SWING, note=note)
        except Exception as exc:
            logger.warning("Strategist failed for %s: %s", verdict.ticker, exc)
            idea = f"The strategist failed: {exc}"
        contracts = (contracts_for(idea, equity, config["risk_per_trade"], config["max_contracts"])
                     if isinstance(idea, OptionsIdea) else None)
        item = Idea(verdict, idea, contracts)
        if contracts and contracts == config["max_contracts"] < contracts_for(idea, equity, config["risk_per_trade"]):
            item.sized_down = f"capped at {contracts} contracts"
        result.ideas.append(item)
    if equity and config["max_total_risk"]:
        # The names the market agrees with keep their size first, then the strongest calls.
        cap_total_risk(result.ideas, equity * config["max_total_risk"],
                       lambda item: (rank.get(item.verdict.ticker, 0), -item.verdict.conviction))
    return result


def _money(value: float | None) -> str:
    return "uncapped" if value is None else f"${value:,.0f}"


def _render_idea(item: Idea, equity: float | None, risk: float) -> str:
    v, idea = item.verdict, item.idea
    side = "bullish" if v.direction == "up" else "bearish"
    head = f"## {v.ticker}: {side} (conviction {v.conviction}/5)\n\n**Catalyst:** {v.catalyst}. {v.reason}\n"
    if isinstance(idea, str):
        return head + "\n" + idea + "\n"
    if not idea.trade:
        return head + f"\n**No trade.** {idea.rationale}\n"
    legs = "\n".join(f"| {leg.action} | {leg.ratio} | {leg.option_type} | {leg.strike:g} | {leg.expiration} |"
                     for leg in idea.legs)
    premium = f"{abs(idea.net_premium):.2f} {'debit' if idea.net_premium >= 0 else 'credit'}"
    size = ""
    if item.contracts is not None:
        size = (f"\n**Size:** {item.contracts} unit(s) keeps the max loss within {risk:.0%} of "
                f"${equity:,.0f} (${item.contracts * idea.max_loss:,.0f} at risk)"
                + (f"; {item.sized_down}" if item.sized_down else "") + ".\n")
        if item.contracts == 0:
            size += ("The screen's total risk cap is used up by the ideas ahead of this one.\n" if item.sized_down
                     else "One unit risks more than the budget, so this idea doesn't fit at this size.\n")
    return (head + f"\n**Trade:** {idea.structure}, {premium} per unit. "
            f"Max loss {_money(idea.max_loss)}, max gain {_money(max_gain(idea))} per unit. "
            f"Breakeven {idea.breakeven}.\n\n| Action | Ratio | Type | Strike | Expiry |\n|---|---|---|---|---|\n"
            f"{legs}\n{size}\n**Why:** {idea.rationale}\n\n**Exit:** {idea.exit_plan}\n\n"
            f"**Wrong if:** {idea.invalidation}\n")


def _total_risk(ideas: list[Idea], equity: float | None, cap: float | None) -> str:
    sized = [item for item in ideas if item.contracts]
    if not equity or not sized:
        return ""
    total = sum(item.contracts * item.idea.max_loss for item in sized)
    limit = f", within the {cap:.0%} cap" if cap else ""
    return (f"**Total at risk:** ${total:,.0f} across {len(sized)} idea(s), {total / equity:.1%} of "
            f"${equity:,.0f}{limit}.\n")


def render_report(result: ScreenResult, equity: float | None = None, risk: float = 0.03,
                  total_cap: float | None = None) -> str:
    lines = [
        f"# {result.etf} {'pre-market day-trade' if result.mode == 'day' else 'conviction'} screen, {result.date}",
        "",
        f"News from {result.start} to {result.date}, screened by {result.screen_model}. "
        f"{result.universe} names, {result.with_news} with news, {len(result.verdicts)} screened, "
        f"{len(result.shortlist)} on the short list. Analysis only; no orders were placed.",
        "",
    ]
    if result.mode == "day" and result.market_open:
        lines += ["Run after the open: each name's move is since the prior close, not pre-market.", ""]
    view = render_index_view(result.etf, result.verdicts, result.failures, result.holdings, set(result.news_names))
    sections = [result.levels, view] if result.mode == "day" else [view, result.levels]
    lines += [section for section in sections if section]
    if result.reactions:
        lines.append(render_reactions(result.verdicts, result.reactions, result.last_expiry, result.market_open))
    if result.skipped:
        lines.append("No idea for names moving against their news: " + ", ".join(result.skipped) + ".\n")
    total = _total_risk(result.ideas, equity, total_cap)
    if total:
        lines.append(total)
    lines += [_render_idea(item, equity, risk) for item in result.ideas]
    if result.shortlist and not result.ideas:
        lines.append("Short list: " + ", ".join(f"{v.ticker} ({v.direction} {v.conviction})" for v in result.shortlist) + "\n")
    lines += ["## Every directional call", "", "| Ticker | Direction | Conviction | Catalyst |", "|---|---|---|---|"]
    directional = sorted((v for v in result.verdicts if v.direction != "none"), key=lambda v: (-v.conviction, v.ticker))
    lines += [f"| {v.ticker} | {v.direction} | {v.conviction} | {v.catalyst} |" for v in directional]
    if result.failures:
        lines += ["", f"Not screened ({len(result.failures)}): " + ", ".join(sorted(result.failures))]
    return "\n".join(lines) + "\n"


def save_report(result: ScreenResult, text: str, results_dir: str) -> Path:
    name = result.etf.replace(" ", "_")
    # The run time is in the name, so a second screen the same day doesn't overwrite the first.
    suffix = "_day" if result.mode == "day" else ""
    path = Path(results_dir) / "screens" / f"{name}_{result.date}_{datetime.now():%H%M}{suffix}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
