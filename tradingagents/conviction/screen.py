"""News screen: a cheap model reads each name's recent news and says whether it
points to a move up or down, and with how much conviction.

One call per name, run in parallel. The model answers in JSON, which is parsed
here rather than through a provider's structured-output mode, so any
OpenRouter model can do the screen.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class NewsVerdict(BaseModel):
    """What one name's news says about its next few weeks."""

    ticker: str = ""
    direction: Literal["up", "down", "none"]
    conviction: int = Field(ge=1, le=5)
    catalyst: str = Field(description="The one development driving the call, in a few words")
    reason: str = Field(description="One or two sentences on why the news points this way")


SCREEN_PROMPT = """You screen news for an options desk that holds swing trades for days to weeks.

Below are the Benzinga articles about {ticker} published from {start} to {end}.
Decide whether this news points to {ticker} moving up or down over the next one to six weeks, and how strongly.

Conviction:
5 - a fresh, material, company-specific catalyst that clearly points one way: earnings with a guidance change, a merger or buyout, a regulatory decision, a major contract won or lost, an accounting or legal shock.
4 - a clear catalyst pointing one way, but smaller or partly priced in.
3 - some signal, but mixed, stale or mostly about the sector rather than the company.
2 - routine items: price-target changes, roundups, recaps of moves that already happened.
1 - noise, or nothing about the company.

Use direction "none" when the news does not point either way. Judge only from these articles; do not use other knowledge of the stock.

Answer with JSON only, no other text:
{{"direction": "up" | "down" | "none", "conviction": 1-5, "catalyst": "...", "reason": "..."}}

Articles:
{news}"""


def _parse(text: str, ticker: str) -> NewsVerdict:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in the screen answer: {str(text)[:200]!r}")
    return NewsVerdict.model_validate({**json.loads(match.group(0)), "ticker": ticker})


def screen_one(llm, ticker: str, news: str, start: str, end: str) -> NewsVerdict:
    prompt = SCREEN_PROMPT.format(ticker=ticker, start=start, end=end, news=news)
    content = llm.invoke(prompt).content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return _parse(content, ticker)


def screen_news(llm, news_by_ticker: dict[str, str], start: str, end: str, workers: int = 8,
                progress=None) -> tuple[list[NewsVerdict], dict[str, str]]:
    """Screen every name's news; returns the verdicts and the names that failed, with why."""
    verdicts, failures = [], {}

    def run(ticker):
        try:
            return ticker, screen_one(llm, ticker, news_by_ticker[ticker], start, end), None
        except (ValidationError, ValueError) as exc:
            return ticker, None, f"unreadable answer: {exc}"
        except Exception as exc:  # one name's failed call must not stop the screen
            return ticker, None, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, (ticker, verdict, error) in enumerate(pool.map(run, news_by_ticker), 1):
            if verdict is not None:
                verdicts.append(verdict)
            else:
                logger.warning("Screen failed for %s: %s", ticker, error)
                failures[ticker] = error
            if progress:
                progress(done, len(news_by_ticker), ticker)
    return verdicts, failures


def shortlist(verdicts: list[NewsVerdict], min_conviction: int, max_names: int,
              weights: dict[str, float | None] | None = None) -> list[NewsVerdict]:
    """Names with a direction and enough conviction, strongest first, then heaviest in the ETF."""
    weights = weights or {}
    picked = [v for v in verdicts if v.direction != "none" and v.conviction >= min_conviction]
    picked.sort(key=lambda v: (-v.conviction, -(weights.get(v.ticker) or 0)))
    return picked[:max_names]
