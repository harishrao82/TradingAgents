"""Anthropic structured output avoids forced tool use where the API rejects it."""
from unittest import mock

import pytest
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from tradingagents.llm_clients.anthropic_client import NormalizedChatAnthropic


class Verdict(BaseModel):
    rating: str


def _method(**llm_kwargs):
    llm = NormalizedChatAnthropic(api_key="test", **llm_kwargs)
    with mock.patch.object(ChatAnthropic, "with_structured_output") as parent:
        llm.with_structured_output(Verdict)
    return parent.call_args.kwargs["method"]


@pytest.mark.unit
@pytest.mark.parametrize(("llm_kwargs", "method"), [
    ({"model": "claude-opus-5-5"}, "json_schema"),
    ({"model": "claude-sonnet-5-5-20261001"}, "json_schema"),
    ({"model": "claude-sonnet-5", "effort": "high"}, "json_schema"),
    ({"model": "claude-sonnet-4-5", "thinking": {"type": "enabled", "budget_tokens": 2048}}, "json_schema"),
    ({"model": "claude-sonnet-4-5"}, "function_calling"),
])
def test_structured_output_method(llm_kwargs, method):
    assert _method(**llm_kwargs) == method


@pytest.mark.unit
def test_an_explicit_method_is_kept():
    llm = NormalizedChatAnthropic(api_key="test", model="claude-opus-5-5")
    with mock.patch.object(ChatAnthropic, "with_structured_output") as parent:
        llm.with_structured_output(Verdict, method="function_calling")
    assert parent.call_args.kwargs["method"] == "function_calling"
