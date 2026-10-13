"""Conviction screen: ETF holdings, then a cheap news screen, then options ideas.

1. The ETF's holdings come from Unusual Whales.
2. Each holding's Benzinga news for the last few days is read in bulk through
   Alpaca, and a cheap model (Claude Haiku 5.5 by default, optionally through
   Anthropic's Batches API at half price) screens it for a move up or down
   with conviction.
3. The strongest calls go to the options strategist on the deep model tier,
   which suggests a defined-risk trade for each.

Analysis only: nothing here reads an account or places an order.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import anthropic

from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.vendors import alpaca, unusual_whales
from tradingagents.llm_clients.factory import (
    build_llm_kwargs,
    create_llm_client,
    create_tier_client,
)

from .batch import screen_news_batch
from .screen import NewsVerdict, screen_news, shortlist
from .strategist import OptionsIdea, contracts_for, max_gain, suggest

logger = logging.getLogger(__name__)


@dataclass
class Idea:
    verdict: NewsVerdict
    idea: OptionsIdea | str
    contracts: int | None = None


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


def run_conviction_screen(config: dict, date: str, etf: str = "SPY", tickers: list[str] | None = None,
                          equity: float | None = None, screen_only: bool = False, progress=None,
                          llms: tuple | None = None) -> ScreenResult:
    """Screen ``etf``'s holdings (or ``tickers``) as of ``date`` and suggest options for the short list."""
    with run_config(config):
        return _run(config, date, etf, tickers, equity, screen_only, progress, llms)


def _run(config, date, etf, tickers, equity, screen_only, progress, llms) -> ScreenResult:
    say = progress or (lambda message: None)
    weights = dict(unusual_whales.get_etf_holdings(etf)) if not tickers else {t.upper(): None for t in tickers}
    say(f"{len(weights)} names in {etf if not tickers else 'the list'}")

    start = _weekdays_back(date, config["screen_lookback_days"])
    news = alpaca.get_news_by_symbol(list(weights), start, date, config["screen_articles_per_name"])
    say(f"{len(news)} names have Benzinga news from {start} to {date}")

    screener, strategist = llms or (None, None)
    if config["screen_batch"]:
        if config["screen_llm_provider"].lower() != "anthropic":
            raise ValueError("Batch screening uses Anthropic's Batches API; set screen_llm_provider to anthropic.")
        verdicts, failures = screen_news_batch(
            screener or anthropic.Anthropic(), config["screen_llm"], news, start, date,
            max_tokens=config["screen_max_tokens"], progress=say,
        )
    else:
        verdicts, failures = screen_news(
            screener or screen_llm(config), news, start, date, config["screen_workers"],
            per_minute=config["screen_requests_per_minute"],
            progress=lambda done, total, ticker: say(f"screened {done}/{total}") if done % 50 == 0 or done == total else None,
        )
    picked = shortlist(verdicts, config["screen_min_conviction"], config["screen_max_names"], weights,
                       config["screen_direction"])
    say(f"{len(picked)} names on the short list: {', '.join(v.ticker for v in picked) or 'none'}")

    result = ScreenResult(etf=etf if not tickers else "custom list", date=date, start=start, universe=len(weights),
                          with_news=len(news), verdicts=verdicts, failures=failures, shortlist=picked,
                          screen_model=config["screen_llm"])
    if screen_only or not picked:
        return result

    strategist = strategist or create_tier_client(config, "deep").get_llm()
    for verdict in picked:
        say(f"strategist: {verdict.ticker}")
        try:
            idea = suggest(strategist, verdict, news[verdict.ticker], date)
        except Exception as exc:
            logger.warning("Strategist failed for %s: %s", verdict.ticker, exc)
            idea = f"The strategist failed: {exc}"
        contracts = contracts_for(idea, equity, config["risk_per_trade"]) if isinstance(idea, OptionsIdea) else None
        result.ideas.append(Idea(verdict, idea, contracts))
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
                f"${equity:,.0f} (${item.contracts * idea.max_loss:,.0f} at risk).\n")
        if item.contracts == 0:
            size += "One unit risks more than the budget, so this idea doesn't fit at this size.\n"
    return (head + f"\n**Trade:** {idea.structure}, {premium} per unit. "
            f"Max loss {_money(idea.max_loss)}, max gain {_money(max_gain(idea))} per unit. "
            f"Breakeven {idea.breakeven}.\n\n| Action | Ratio | Type | Strike | Expiry |\n|---|---|---|---|---|\n"
            f"{legs}\n{size}\n**Why:** {idea.rationale}\n\n**Exit:** {idea.exit_plan}\n\n"
            f"**Wrong if:** {idea.invalidation}\n")


def render_report(result: ScreenResult, equity: float | None = None, risk: float = 0.03) -> str:
    lines = [
        f"# {result.etf} conviction screen, {result.date}",
        "",
        f"News from {result.start} to {result.date}, screened by {result.screen_model}. "
        f"{result.universe} names, {result.with_news} with news, {len(result.verdicts)} screened, "
        f"{len(result.shortlist)} on the short list. Analysis only; no orders were placed.",
        "",
    ]
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
    path = Path(results_dir) / "screens" / f"{name}_{result.date}_{datetime.now():%H%M}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
