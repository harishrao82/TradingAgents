from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import get_market_tide, get_options_activity, get_options_flow

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_options_flow,
    get_options_activity,
    get_market_tide,
)


def create_options_flow_analyst(llm):
    def options_flow_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        system_message = (
            "You are an options flow analyst on a trading desk. Read how options traders are positioned in the instrument under analysis and in the market as a whole. Use the available tools: get_options_flow(curr_date, look_back_days, min_premium) for unusual flow alerts such as sweeps, floor trades and repeated hits; get_options_activity(curr_date) for daily call and put volume against the 30-day average, net premium, open interest, implied volatility and 1-year IV rank; and get_market_tide(curr_date) for market-wide net call and put premium."
            " Cover: (1) the directional lean of the flow, weighing ask-side against bid-side premium rather than raw call or put counts; (2) where the size is concentrated by strike and expiry, and whether it sits before or after the next likely catalyst; (3) whether volume is unusual against its average and against open interest, which suggests new positions; (4) whether implied volatility is cheap or rich on its IV rank, and what that means for buying versus selling premium; and (5) whether the stock's flow agrees with the market tide."
            " Flow alerts do not say whether a trade opens or closes a position, and a large trade may be a hedge; say how much weight the evidence bears. If a tool reports data unavailable, say so and do not estimate the missing values."
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
            "options_report": report,
        }

    return options_flow_analyst_node
