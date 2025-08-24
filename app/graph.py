import os
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage

from .poml import PomlSpec, build_base_messages, load_poml_file, render_user_prompt


class AgentState(TypedDict):
	input: str
	variables: Dict[str, Any]
	messages: Annotated[List[BaseMessage], add_messages]


def _make_llm(model: str):
	from langchain_openai import ChatOpenAI

	api_key = os.environ.get("OPENROUTER_API_KEY")
	base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
	headers = {
		"HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "https://github.com/"),
		"X-Title": os.environ.get("OPENROUTER_APP_TITLE", "ChargeBankAgent"),
	}

	model_name = model or os.environ.get("OPENROUTER_MODEL", "openrouter/auto")

	return ChatOpenAI(
		model=model_name,
		openai_api_key=api_key,
		base_url=base_url,
		openai_api_base=base_url,
		default_headers=headers,
		temperature=0.2,
	)


def build_graph(poml_path: str, model: str = ""):
	spec: PomlSpec = load_poml_file(poml_path)

	def prepare_messages(state: AgentState) -> AgentState:
		base_messages = build_base_messages(spec, state.get("variables", {}))
		user_text = render_user_prompt(spec, {**(state.get("variables") or {}), "input": state.get("input", "")})
		return {
			"messages": base_messages + [HumanMessage(content=user_text)],
		}

	llm = _make_llm(model)

	def call_model(state: AgentState) -> AgentState:
		response = llm.invoke(state["messages"])  # returns an AIMessage
		return {"messages": [response]}

	graph = StateGraph(AgentState)
	graph.add_node("prepare", prepare_messages)
	graph.add_node("call", call_model)

	graph.add_edge(START, "prepare")
	graph.add_edge("prepare", "call")
	graph.add_edge("call", END)

	return graph.compile()