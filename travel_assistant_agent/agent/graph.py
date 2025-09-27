import json
import logging
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from pydantic import BaseModel

from travel_assistant_agent.agent.context import Context
from travel_assistant_agent.agent.state import InputState, State, UserQueryIntent, UserPreferences
from travel_assistant_agent.agent.tools import TOOLS, get_weather_forecast
from travel_assistant_agent.agent.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DESTINATION_RECOMMENDATION_SYSTEM_PROMPT,
    ACTIVITY_RECOMMENDATION_SYSTEM_PROMPT,
    PACKING_RECOMMENDATION_SYSTEM_PROMPT,
    FOOD_RECOMMENDATION_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    PREFERENCES_EXTRACTION_SYSTEM_PROMPT,
)


INTENT_TO_PROMPT = {
    UserQueryIntent.DESTINATION_RECOMMENDATION: DESTINATION_RECOMMENDATION_SYSTEM_PROMPT,
    UserQueryIntent.ACTIVITY_RECOMMENDATION: ACTIVITY_RECOMMENDATION_SYSTEM_PROMPT,
    UserQueryIntent.PACKING_RECOMMENDATION: PACKING_RECOMMENDATION_SYSTEM_PROMPT,
    UserQueryIntent.FOOD_RECOMMENDATION: FOOD_RECOMMENDATION_SYSTEM_PROMPT,
}

INTENT_TO_TOOLS = {
    UserQueryIntent.PACKING_RECOMMENDATION: [get_weather_forecast]
}


class UserIntentClassification(BaseModel):
    user_query_intent: UserQueryIntent
    confidence: float
    rationale: str


def classify_user_intent(
        state: State,
        runtime: Runtime[Context]
) -> Dict[Literal['user_query_intent'], UserQueryIntent]:
    llm = init_chat_model(
        model=runtime.context.model,
        model_provider=runtime.context.model_provider,
    )
    parser = PydanticOutputParser(pydantic_object=UserIntentClassification)
    chain = llm | parser

    system_message = INTENT_CLASSIFICATION_SYSTEM_PROMPT.format(
        previous_intent=state.user_query_intent.value,
        json_schema=UserIntentClassification.model_json_schema(),
    )

    try:
        classification = cast(
            UserIntentClassification,
            chain.invoke([SystemMessage(system_message), *state.messages[-3:]]),
        )
    except Exception as e:
        logging.debug(e)
        classification = UserIntentClassification(
            user_query_intent=UserQueryIntent.UNKNOWN,
            confidence=0.0,
            rationale='Failed to classify intent due to an error.'
        )

    return {
        'user_query_intent': classification.user_query_intent if classification.confidence > 0.5 else UserQueryIntent.UNKNOWN
    }


def extract_user_preferences(
        state: State,
        runtime: Runtime[Context]
) -> Dict[Literal['user_preferences'], UserPreferences]:
    llm = init_chat_model(
        model=runtime.context.model,
        model_provider=runtime.context.model_provider,
    )
    parser = PydanticOutputParser(pydantic_object=UserPreferences)
    chain = llm | parser

    system_message = PREFERENCES_EXTRACTION_SYSTEM_PROMPT.format(
        json_schema=json.dumps(UserPreferences.model_json_schema()),
        current_preferences=state.user_preferences.model_dump()
    )

    try:
        extracted_preferences = cast(
            UserPreferences,
            chain.invoke([SystemMessage(system_message), state.messages[-1]]),
        )
    except Exception as e:
        logging.debug(e)
        extracted_preferences = UserPreferences()

    return {
        'user_preferences': UserPreferences.merge(state.user_preferences, extracted_preferences)
    }


def chatbot(
        state: State,
        runtime: Runtime[Context]
) -> Dict[Literal['messages'], List[AIMessage]]:
    tools = INTENT_TO_TOOLS.get(state.user_query_intent, [])
    system_prompt = INTENT_TO_PROMPT.get(state.user_query_intent, DEFAULT_SYSTEM_PROMPT)

    model = init_chat_model(
        model=runtime.context.model,
        model_provider=runtime.context.model_provider,
    ).bind_tools(tools)

    system_message = system_prompt.format(
        current_date=datetime.now(tz=UTC).strftime('%Y-%m-%d'),
        user_preferences=state.user_preferences.model_dump(),
    )

    response = cast(
        AIMessage,
        model.invoke([SystemMessage(system_message), *state.messages[-10:]]),
    )

    return {
        'messages': [response]
    }


def route_model_output(state: State) -> str:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f'Expected AIMessage in output edges, but got {type(last_message).__name__}')
    if last_message.tool_calls:
        return 'tools'
    return END


builder = StateGraph(State, input_schema=InputState, context_schema=Context)
builder.add_node('classify_user_intent', classify_user_intent)
builder.add_node('extract_user_preferences', extract_user_preferences)
builder.add_node('chatbot', chatbot)
builder.add_node('tools', ToolNode(TOOLS))
builder.add_edge(START, 'classify_user_intent')
builder.add_edge(START, 'extract_user_preferences')
builder.add_edge('extract_user_preferences', 'chatbot')
builder.add_edge('classify_user_intent', 'chatbot')
builder.add_edge('tools', 'chatbot')
builder.add_conditional_edges('chatbot', route_model_output)
agent = builder.compile(name='Travel Assistant Agent', checkpointer=InMemorySaver())
context = Context(model='gpt-oss:20b')


def invoke_agent(user_message: str) -> str:
    state = agent.invoke(
        input={'messages': [HumanMessage(user_message)]},
        context=context,
        config={'configurable': {'thread_id': '1'}},
    )

    agent_response = state['messages'][-1]
    
    if not isinstance(agent_response, AIMessage) or agent_response.tool_calls:
        return 'Could not get a response from the AI agent. Please try again...'

    return agent_response.content
