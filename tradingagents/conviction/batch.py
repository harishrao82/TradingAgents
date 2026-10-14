"""News screen through Anthropic's Message Batches API: half the price, no rate limits.

Every name's screen prompt goes out in one batch, which is polled until it
ends. A full SPY screen came back in under three minutes in testing; Anthropic
allows up to 24 hours, so this suits a pre-market or overnight run rather than
an intraday one. Answers are parsed exactly as the live screen parses them.
"""
from __future__ import annotations

import logging
import time

from .screen import SCREEN_PROMPT, NewsVerdict, _parse, _short

logger = logging.getLogger(__name__)

POLL_SECONDS = 15
MAX_WAIT_SECONDS = 24 * 3600


def _text(message) -> str:
    return "".join(block.text for block in message.content if getattr(block, "type", "") == "text")


def screen_news_batch(client, model: str, news_by_ticker: dict[str, str], start: str, end: str,
                      max_tokens: int = 1500, progress=None, sleep=time.sleep,
                      clock=time.monotonic) -> tuple[list[NewsVerdict], dict[str, str]]:
    """Screen every name's news in one batch; returns the verdicts and the names that failed, with why."""
    tickers = list(news_by_ticker)
    # Batch ids allow only letters, digits, - and _, so BRK.B travels by position.
    requests = [
        {"custom_id": f"n{i}",
         "params": {"model": model, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": SCREEN_PROMPT.format(
                        ticker=ticker, start=start, end=end, news=news_by_ticker[ticker])}]}}
        for i, ticker in enumerate(tickers)
    ]
    batch = client.messages.batches.create(requests=requests)
    if progress:
        progress(f"batch {batch.id} submitted with {len(requests)} names")

    began, reported = clock(), 0
    while batch.processing_status != "ended":
        if clock() - began > MAX_WAIT_SECONDS:
            raise TimeoutError(f"batch {batch.id} had not ended after 24 hours")
        sleep(POLL_SECONDS)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        done = counts.succeeded + counts.errored
        if progress and done != reported:
            progress(f"batch: {done} of {len(requests)} done")
            reported = done

    verdicts, failures = [], {}
    for entry in client.messages.batches.results(batch.id):
        ticker = tickers[int(entry.custom_id[1:])]
        result = entry.result
        if result.type != "succeeded":
            failures[ticker] = f"batch request {result.type}"
            continue
        if result.message.stop_reason == "max_tokens":
            failures[ticker] = "answer cut off at the token limit"
            continue
        try:
            verdicts.append(_parse(_text(result.message), ticker))
        except Exception as exc:
            failures[ticker] = f"unreadable answer: {_short(exc)}"
    for ticker in set(tickers) - {v.ticker for v in verdicts} - set(failures):
        failures[ticker] = "missing from the batch results"
    for ticker, why in failures.items():
        logger.warning("Screen failed for %s: %s", ticker, why)
    return verdicts, failures
