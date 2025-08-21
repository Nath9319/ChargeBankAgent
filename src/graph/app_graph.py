from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from poml.parser import parse_poml_file
from llm.openrouter import get_chat_openrouter


class AppState(BaseModel):
    input: str
    prompt_path: str
    model: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    messages: List[Dict[str, str]] = Field(default_factory=list)
    response: str | None = None


def _to_lc_messages(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    result: List[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        else:
            result.append(HumanMessage(content=content))
    return result


def compose_prompt_node(state: AppState) -> Dict[str, Any]:
    variables = dict(state.variables or {})
    variables.setdefault("input", state.input)
    messages = parse_poml_file(state.prompt_path, variables)
    return {"messages": messages}


def llm_call_node(state: AppState) -> Dict[str, Any]:
    messages = state.messages or []
    lc_messages = _to_lc_messages(messages)
    chat = get_chat_openrouter(model=state.model)
    ai_message = chat.invoke(lc_messages)
    content = ai_message.content if isinstance(ai_message, AIMessage) else str(ai_message)
    messages_out = messages + [{"role": "assistant", "content": content}]
    return {"messages": messages_out, "response": content}


def build_app_graph():
    graph = StateGraph(AppState)
    graph.add_node("compose_prompt", compose_prompt_node)
    graph.add_node("llm_call", llm_call_node)
    graph.add_edge(START, "compose_prompt")
    graph.add_edge("compose_prompt", "llm_call")
    graph.add_edge("llm_call", END)
    return graph.compile()