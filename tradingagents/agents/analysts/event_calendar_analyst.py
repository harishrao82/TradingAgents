from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import (
    get_earnings_calendar,
    get_economic_calendar,
    get_macro_indicators,
)

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_earnings_calendar,
    get_economic_calendar,
    get_macro_indicators,
)


def create_event_calendar_analyst(llm):
    def event_calendar_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        system_message = (
            "You are the event calendar analyst on a trading desk. Your job is to map the scheduled events that can move the instrument under analysis over the next few weeks, so the desk sizes positions and picks option expiries around them. Use the available tools: get_earnings_calendar(curr_date) for the next earnings date and how past reports moved the stock against the options-implied move; get_economic_calendar(curr_date) for upcoming economic reports, FOMC meetings and Fed speakers with forecasts and previous values; and get_macro_indicators(indicator, curr_date, look_back_days) for the latest released data behind those reports (e.g. 'cpi', 'core_pce', 'unemployment', 'nonfarm_payrolls', 'fed_funds_rate', '10y_treasury')."
            " Cover: (1) the next earnings date, whether it is confirmed or estimated, and how many days away it is; (2) whether past realised earnings moves ran above or below the implied move, and what that says about owning versus selling options into the report; (3) the economic events in the next two weeks most likely to move this instrument or the market, with forecast against previous; and (4) a dated list of event risk windows the trader should avoid or target."
            " If a tool reports data unavailable, say so and do not invent dates or forecasts."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])

        return {
            "messages": [result],
            "calendar_report": report,
        }

    return event_calendar_analyst_node
