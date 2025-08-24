import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


POML_SECTION_PATTERN = re.compile(r"<(?P<tag>[a-zA-Z][a-zA-Z0-9_-]*)>\s*(?P<body>[\s\S]*?)\s*</(?P=tag)>", re.MULTILINE)
SHOT_PATTERN = re.compile(r"<shot>\s*(?P<body>[\s\S]*?)\s*</shot>", re.MULTILINE)
TAGGED_SUB_PATTERN = re.compile(r"<(?P<tag>user|assistant)>\s*(?P<body>[\s\S]*?)\s*</(?P=tag)>", re.MULTILINE)
VARIABLE_PATTERN = re.compile(r"\{\{\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_\.-]*)\s*}}")
COMMENT_PATTERN = re.compile(r"<!--([\s\S]*?)-->", re.MULTILINE)


@dataclass
class PomlSpec:
	system: str = ""
	persona: str = ""
	constraints: str = ""
	style: str = ""
	context: str = ""
	user_template: str = "{{input}}"
	shots: List[Tuple[str, str]] = None

	def __post_init__(self):
		if self.shots is None:
			self.shots = []


def _strip(text: Optional[str]) -> str:
	return (text or "").strip()


def _render_template(template: str, variables: Dict[str, Any]) -> str:
	def replace_match(match: re.Match) -> str:
		name = match.group("name")
		return "{" + name + "}"

	converted = VARIABLE_PATTERN.sub(replace_match, template)

	class SafeDict(dict):
		def __missing__(self, key):  # type: ignore[override]
			return "{" + key + "}"

	return converted.format_map(SafeDict(variables or {}))


def parse_poml_string(poml_text: str) -> PomlSpec:
	# strip HTML-style comments before extraction
	text = COMMENT_PATTERN.sub("", poml_text)
	sections: Dict[str, str] = {}
	for m in POML_SECTION_PATTERN.finditer(text):
		tag = m.group("tag").lower()
		body = m.group("body").strip()
		sections[tag] = body

	shots: List[Tuple[str, str]] = []
	for sm in SHOT_PATTERN.finditer(text):
		body = sm.group("body")
		user_text = ""
		assistant_text = ""
		for tm in TAGGED_SUB_PATTERN.finditer(body):
			sub_tag = tm.group("tag").lower()
			sub_body = tm.group("body").strip()
			if sub_tag == "user":
				user_text = sub_body
			elif sub_tag == "assistant":
				assistant_text = sub_body
		if user_text or assistant_text:
			shots.append((user_text, assistant_text))

	return PomlSpec(
		system=_strip(sections.get("system")),
		persona=_strip(sections.get("persona")),
		constraints=_strip(sections.get("constraints")),
		style=_strip(sections.get("style")),
		context=_strip(sections.get("context")),
		user_template=_strip(sections.get("user") or "{{input}}"),
		shots=shots,
	)


def load_poml_file(path: str) -> PomlSpec:
	with open(path, "r", encoding="utf-8") as f:
		return parse_poml_string(f.read())


def build_base_messages(spec: PomlSpec, variables: Optional[Dict[str, Any]] = None) -> List[BaseMessage]:
	variables = variables or {}

	system_chunks: List[str] = []
	if spec.system:
		system_chunks.append(spec.system)
	if spec.persona:
		system_chunks.append("Persona:\n" + spec.persona)
	if spec.constraints:
		system_chunks.append("Constraints:\n" + spec.constraints)
	if spec.style:
		system_chunks.append("Style:\n" + spec.style)
	if spec.context:
		system_chunks.append("Context:\n" + spec.context)

	system_text = "\n\n".join([_render_template(chunk, variables) for chunk in system_chunks if chunk])

	messages: List[BaseMessage] = []
	if system_text:
		messages.append(SystemMessage(content=system_text))

	for user_shot, assistant_shot in spec.shots:
		if user_shot:
			messages.append(HumanMessage(content=_render_template(user_shot, variables)))
		if assistant_shot:
			messages.append(AIMessage(content=_render_template(assistant_shot, variables)))

	return messages


def render_user_prompt(spec: PomlSpec, variables: Optional[Dict[str, Any]] = None) -> str:
	variables = variables or {}
	return _render_template(spec.user_template or "{{input}}", variables)


def compose_messages(spec: PomlSpec, variables: Optional[Dict[str, Any]] = None) -> List[BaseMessage]:
	variables = variables or {}
	messages = build_base_messages(spec, variables)
	user_text = render_user_prompt(spec, variables)
	messages.append(HumanMessage(content=user_text))
	return messages