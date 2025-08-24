import json
import os
from pathlib import Path
from typing import Any, Dict, List

import typer
from rich import print

from .graph import build_graph
from .poml import compose_messages, load_poml_file

app = typer.Typer(add_completion=False)


def _parse_key_value(pairs: List[str]) -> Dict[str, Any]:
	result: Dict[str, Any] = {}
	for pair in pairs:
		if "=" not in pair:
			raise typer.BadParameter(f"Invalid --var '{pair}'. Use key=value")
		key, value = pair.split("=", 1)
		key = key.strip()
		value = value.strip()
		try:
			result[key] = json.loads(value)
		except json.JSONDecodeError:
			result[key] = value
	return result


@app.command()
def run(
	input: str = typer.Option(..., "--input", help="User input to fill into the POML user template ({{input}})."),
	poml: Path = typer.Option(Path("prompts/example.poml"), "--poml", help="Path to the POML-like prompt file."),
	model: str = typer.Option("", "--model", help="OpenRouter model id (overrides OPENROUTER_MODEL)."),
	var: List[str] = typer.Option([], "--var", help="Additional variables for template: key=value (value can be JSON).", metavar="key=value"),
	dry_run: bool = typer.Option(False, "--dry-run", help="Do not call the model; just render messages."),
) -> None:
	variables = _parse_key_value(var)
	variables.setdefault("input", input)

	spec = load_poml_file(str(poml))

	if dry_run or not os.environ.get("OPENROUTER_API_KEY"):
		messages = compose_messages(spec, variables)
		print("[bold]Rendered messages (dry-run):[/bold]")
		for msg in messages:
			role = msg.type
			print(f"[cyan]{role}[/cyan]: {msg.content}")
		if not os.environ.get("OPENROUTER_API_KEY") and not dry_run:
			raise typer.Exit(code=0)
		return

	graph = build_graph(str(poml), model=model)

	state = {
		"input": input,
		"variables": variables,
		"messages": [],
	}

	result = graph.invoke(state)
	messages = result.get("messages", [])
	ai_messages = [m for m in messages if getattr(m, "type", "") == "ai"]
	if ai_messages:
		print("[bold green]Response:[/bold green]", ai_messages[-1].content)
	else:
		print("[yellow]No AI response produced.[/yellow]")


if __name__ == "__main__":
	app()