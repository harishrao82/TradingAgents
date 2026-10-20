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
import threading
import time
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
5 - a fresh, material, company-specific catalyst that clearly points one way. Up: earnings with a guidance raise, a buyout offer, an approval, a major contract won. Down: a guidance cut or earnings miss, a failed trial or rejected approval, a probe, lawsuit or accounting problem, a major customer or contract lost, a dilutive share sale.
4 - a clear catalyst pointing one way, but smaller or partly priced in.
3 - some signal, but mixed, stale or mostly about the sector rather than the company.
2 - routine items: price-target changes, roundups, recaps of moves that already happened.
1 - noise, or nothing about the company.

Bad news counts as much as good news: score a clear negative catalyst "down" on the same scale. Use direction "none" when the news does not point either way. Judge only from these articles; do not use other knowledge of the stock.

Answer with JSON only, no other text:
{{"direction": "up" | "down" | "none", "conviction": 1-5, "catalyst": "...", "reason": "..."}}

Articles:
{news}"""


DAY_SCREEN_PROMPT = """You screen news for an options desk trading moves over the next one to five trading days.

Below are the Benzinga articles about {ticker} published from {start} to {end}.
Decide whether this news points to {ticker} moving up or down over the next one to five trading days, and how strongly.

Conviction:
5 - a fresh, material, company-specific catalyst released after the prior close or this morning that clearly points one way: earnings with a guidance raise or cut, a buyout offer, an FDA decision, a major contract won or lost, a probe or lawsuit, a dilutive share sale.
4 - a clear catalyst pointing one way that is a day or two old, or smaller.
3 - some signal, but mixed, stale or mostly about the sector rather than the company.
2 - routine items: price-target changes, roundups, recaps of moves that already happened.
1 - noise, or nothing about the company.

Bad news counts as much as good news: score a clear negative catalyst "down" on the same scale. Use direction "none" when the news does not point either way. Judge only from these articles; do not use other knowledge of the stock. Start the catalyst with "Fresh:" when it came out after the prior close or this morning.

Answer with JSON only, no other text:
{{"direction": "up" | "down" | "none", "conviction": 1-5, "catalyst": "...", "reason": "..."}}

Articles:
{news}"""


def _parse(text: str, ticker: str) -> NewsVerdict:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in the screen answer: {str(text)[:200]!r}")
    return NewsVerdict.model_validate({**json.loads(match.group(0)), "ticker": ticker})


def screen_one(llm, ticker: str, news: str, start: str, end: str, template: str = SCREEN_PROMPT) -> NewsVerdict:
    prompt = template.format(ticker=ticker, start=start, end=end, news=news)
    content = llm.invoke(prompt).content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return _parse(content, ticker)


RETRIES = 5
BACKOFF_SECONDS = 4


def _rate_limited(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status == 429 or "RateLimit" in type(exc).__name__ or "429" in str(exc)[:200]


def _short(exc: Exception) -> str:
    """One line on why a call failed; provider error bodies can carry account ids."""
    if _rate_limited(exc):
        return "rate limited (429)"
    status = getattr(exc, "status_code", None)
    first = str(exc).split("\n", 1)[0].split("{", 1)[0].strip()[:120]
    return f"{type(exc).__name__}{f' {status}' if status else ''}: {first}".rstrip(": ")


class Throttle:
    """Spaces call starts so all workers together stay under ``per_minute``; 0 means no cap."""

    def __init__(self, per_minute: int, clock=time.monotonic, sleep=time.sleep):
        self.gap = 60.0 / per_minute if per_minute else 0.0
        self.clock, self.sleep, self.lock, self.next = clock, sleep, threading.Lock(), 0.0

    def wait(self):
        if not self.gap:
            return
        with self.lock:
            now = self.clock()
            start = max(now, self.next)
            self.next = start + self.gap
        if start > now:
            self.sleep(start - now)


def screen_news(llm, news_by_ticker: dict[str, str], start: str, end: str, workers: int = 8,
                progress=None, per_minute: int = 0, sleep=time.sleep,
                template: str = SCREEN_PROMPT) -> tuple[list[NewsVerdict], dict[str, str]]:
    """Screen every name's news; returns the verdicts and the names that failed, with why.

    A rate-limited call is retried with a growing pause; ``per_minute`` caps how
    often calls start across all workers.
    """
    verdicts, failures = [], {}
    throttle = Throttle(per_minute, sleep=sleep)

    def run(ticker):
        for attempt in range(RETRIES):
            throttle.wait()
            try:
                return ticker, screen_one(llm, ticker, news_by_ticker[ticker], start, end, template), None
            except (ValidationError, ValueError) as exc:
                return ticker, None, f"unreadable answer: {str(exc)[:120]}"
            except Exception as exc:  # one name's failed call must not stop the screen
                if _rate_limited(exc) and attempt < RETRIES - 1:
                    sleep(BACKOFF_SECONDS * 2 ** attempt)
                    continue
                return ticker, None, _short(exc)

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
              weights: dict[str, float | None] | None = None, direction: str = "both",
              rank: dict[str, int] | None = None) -> list[NewsVerdict]:
    """Names with a direction and enough conviction, strongest first, then heaviest in the ETF.

    With ``direction="both"`` each side gets up to half the slots, so a run of
    good news can't crowd out the put ideas; slots one side leaves empty go to
    the other. ``rank`` puts lower-ranked names ahead of higher ones before
    conviction counts, as the day mode does for news the pre-market agrees with.
    """
    weights, rank = weights or {}, rank or {}
    sides = ("up", "down") if direction == "both" else (direction,)
    picked = [v for v in verdicts if v.direction in sides and v.conviction >= min_conviction]
    picked.sort(key=lambda v: (rank.get(v.ticker, 0), -v.conviction, -(weights.get(v.ticker) or 0)))
    if len(sides) == 1:
        return picked[:max_names]
    by_side = {side: [v for v in picked if v.direction == side] for side in sides}
    half = (max_names + 1) // 2
    chosen = by_side["up"][:half] + by_side["down"][:half]
    chosen += [v for v in picked if v not in chosen][: max(0, max_names - len(chosen))]
    chosen = chosen[:max_names]
    return sorted(chosen, key=lambda v: (v.direction != "up", rank.get(v.ticker, 0), -v.conviction,
                                         -(weights.get(v.ticker) or 0)))
