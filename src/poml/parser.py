from __future__ import annotations

from typing import Any, Dict, List
import xml.etree.ElementTree as ET


class _SafeDict(dict):
    """dict that returns the placeholder if key missing during format_map"""
    def __missing__(self, key):  # type: ignore[override]
        return "{" + key + "}"


def _text(node: ET.Element) -> str:
    return (node.text or "").strip()


def _format_text(text: str, variables: Dict[str, Any]) -> str:
    return text.format_map(_SafeDict(variables))


def parse_poml_string(poml_text: str, variables: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    """
    Parse a simplified POML-like XML string into a list of role/content messages.

    Supported tags inside <prompt>:
    - <system>, <user>, <assistant>
    - <block role="system|user|assistant"> ... </block>
    - <vars><var name="foo">bar</var>...</vars>

    Variables are resolved with Python str.format placeholders, e.g., {name}.
    """
    variables = dict(variables or {})
    root = ET.fromstring(poml_text)
    if root.tag != "prompt":
        raise ValueError("Root element must be <prompt>")

    # Collect inline <vars>
    for vars_node in root.findall("vars"):
        for var_node in vars_node.findall("var"):
            name = var_node.attrib.get("name")
            if not name:
                continue
            variables[name] = _text(var_node)

    messages: List[Dict[str, str]] = []

    def add_message(role: str, content: str) -> None:
        if not content:
            return
        messages.append({"role": role, "content": _format_text(content, variables)})

    for child in root:
        if child.tag == "vars":
            continue
        if child.tag in ("system", "user", "assistant"):
            add_message(child.tag, _text(child))
        elif child.tag == "block":
            role = child.attrib.get("role", "user")
            add_message(role, _text(child))
        else:
            # Ignore unknown tags for forward-compat
            continue

    return messages


def parse_poml_file(path: str, variables: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_poml_string(text, variables)