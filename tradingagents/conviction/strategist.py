"""Options strategist: turns one conviction call into a defined-risk options idea.

The deep model gets the screen's verdict, the news behind it, the live option
chain (Tradier, with the expiries that span earnings marked), options activity
and IV rank, and recent flow (Unusual Whales). It picks a structure, strikes
and an expiry. Contract counts are worked out here from the idea's maximum
loss, never by the model. Nothing is sent to a broker.
"""
from __future__ import annotations

import logging
import math
from typing import Literal

from pydantic import BaseModel, Field

from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.dataflows.router import route_to_vendor

from .screen import NewsVerdict

logger = logging.getLogger(__name__)

MIN_DTE, MAX_DTE = 14, 60


class OptionLeg(BaseModel):
    action: Literal["buy", "sell"]
    option_type: Literal["call", "put"]
    strike: float
    expiration: str = Field(description="YYYY-MM-DD, an expiry listed in the chain")
    ratio: int = Field(default=1, ge=1, description="Contracts of this leg per unit of the structure")


class OptionsIdea(BaseModel):
    trade: bool = Field(description="False when no defined-risk structure fits; then explain in rationale")
    structure: str = Field(default="", description="e.g. bull call spread, bear put spread, long call")
    legs: list[OptionLeg] = Field(default_factory=list)
    net_premium: float = Field(default=0.0, description="Per unit, in option price points; positive is a debit")
    max_loss: float = Field(default=0.0, description="Per unit, in dollars (x100 multiplier)")
    max_gain: float | None = Field(default=None, description="Per unit, in dollars; null if uncapped")
    breakeven: str = ""
    rationale: str = Field(description="Why this structure, strikes and expiry fit the catalyst and the volatility")
    exit_plan: str = Field(default="", description="Profit target, stop, and when to close before expiry or earnings")
    invalidation: str = Field(default="", description="What would show the thesis is wrong")


STRATEGIST_PROMPT = """You are the options strategist on a swing-trading desk. Holding periods are days to weeks.

A news screen flagged {ticker}: it expects a move {direction} with conviction {conviction}/5.
Catalyst: {catalyst}
Screen's reason: {reason}

Propose one options trade that expresses this view, using only the data below.

Desk rules:
- Defined risk only. Never naked short options.
- For a move up use calls: a long call, a bull call spread, or a bull put credit spread.
- For a move down use puts: a long put, a bear put spread, or a bear call credit spread.
- Expiry {min_dte} to {max_dte} days out, chosen from the expiries in the chain.
- Use the strikes and bid/ask in the chain; price debits near the mid and say so.
- Pick by implied volatility: with low IV rank a single long option is fine; by default use a debit spread; when IV is rich (high IV rank, or an expiry that spans earnings) prefer a credit spread, which sells that rich premium.
- If the chain is unavailable or no structure fits, set trade to false and say why.
- Give max loss and max gain per one unit of the structure in dollars (one contract is 100 shares). Only a bought call that isn't sold back is uncapped; a long put's max gain is (strike - premium) x 100.

{no_tools}

## News
{news}

## Option chain
{chain}

## Options activity and IV rank
{activity}

## Recent options flow
{flow}
"""


def _data(method: str, *args) -> str:
    try:
        return str(route_to_vendor(method, *args))
    except Exception as exc:
        logger.warning("%s failed for %s: %s", method, args[0], exc)
        return f"Unavailable: {exc}"


def _earnings(ticker: str, date: str) -> tuple[str | None, str | None]:
    found = None
    try:
        found = route_to_vendor("get_next_earnings", ticker, date)
    except Exception as exc:
        logger.warning("No earnings date for %s: %s", ticker, exc)
    return found if isinstance(found, tuple) else (None, None)


def gather(ticker: str, date: str) -> dict[str, str]:
    """The market data the strategist reads, each as text (or why it is unavailable)."""
    earnings_date, earnings_time = _earnings(ticker, date)
    return {
        "chain": _data("get_option_chain", ticker, date, MIN_DTE, MAX_DTE, earnings_date, earnings_time),
        "activity": _data("get_options_activity", ticker, date),
        "flow": _data("get_options_flow", ticker, date, None, None),
    }


def max_gain(idea: OptionsIdea) -> float | None:
    """Per-unit max gain, worked out from the legs when the model left it blank.

    An idea made only of puts is worth the most if the stock goes to zero, so
    its gain is capped there: strikes bought minus strikes sold, less the net
    premium paid. Only a structure with a call bought and not sold is uncapped.
    """
    if idea.max_gain is not None or not idea.legs:
        return idea.max_gain
    if all(leg.option_type == "put" for leg in idea.legs):
        at_zero = sum((1 if leg.action == "buy" else -1) * leg.strike * leg.ratio for leg in idea.legs)
        return round((at_zero - idea.net_premium) * 100, 2)
    return None


def contracts_for(idea: OptionsIdea, equity: float | None, risk_per_trade: float) -> int | None:
    """Units of the structure whose combined max loss fits the per-trade risk budget."""
    if not equity or not idea.trade or idea.max_loss <= 0:
        return None
    return math.floor(equity * risk_per_trade / idea.max_loss)


def suggest(llm, verdict: NewsVerdict, news: str, date: str, data: dict[str, str] | None = None) -> OptionsIdea | str:
    """The strategist's idea for one name; free text when the model cannot give the structured form."""
    data = data or gather(verdict.ticker, date)
    prompt = STRATEGIST_PROMPT.format(
        ticker=verdict.ticker, direction=verdict.direction, conviction=verdict.conviction,
        catalyst=verdict.catalyst, reason=verdict.reason, min_dte=MIN_DTE, max_dte=MAX_DTE,
        no_tools=NO_EXTERNAL_TOOLS, news=news, **data,
    )
    idea = invoke_structured(bind_structured(llm, OptionsIdea, "Options Strategist"), prompt, "Options Strategist")
    if idea is not None:
        return idea
    content = llm.invoke(prompt).content
    return content if isinstance(content, str) else str(content)
