import argparse
import json
import os
import sys
from typing import Dict, Any

# Ensure `src` is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from graph.app_graph import build_app_graph


def parse_vars_kv(pairs: list[str]) -> Dict[str, Any]:
    variables: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid var format: {pair}. Use key=value")
        key, value = pair.split("=", 1)
        # Try to parse JSON values, fallback to string
        try:
            variables[key] = json.loads(value)
        except json.JSONDecodeError:
            variables[key] = value
    return variables


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangGraph with POML-like prompting and OpenRouter")
    parser.add_argument("--prompt", required=True, help="Path to POML-like prompt file")
    parser.add_argument("--model", required=True, help="OpenRouter model id, e.g., anthropic/claude-3.5-sonnet")
    parser.add_argument("--input", required=True, help="User input text")
    parser.add_argument("--vars", default=None, help="JSON object string for variables to interpolate into POML")
    parser.add_argument("--var", action="append", default=[], help="Repeatable key=value variables")
    args = parser.parse_args()

    variables: Dict[str, Any] = {}
    if args.vars:
        variables = json.loads(args.vars)
    if args.var:
        variables.update(parse_vars_kv(args.var))

    # Build and run the app graph
    app = build_app_graph()
    initial_state = {
        "input": args.input,
        "prompt_path": os.path.abspath(args.prompt),
        "model": args.model,
        "variables": variables,
    }
    result = app.invoke(initial_state)
    response = result.get("response", "")
    print(response)


if __name__ == "__main__":
    main()