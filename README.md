POML-like prompting with LangGraph and OpenRouter

Quickstart

1. Set environment variables:
   - `OPENROUTER_API_KEY`
   - Optional: `OPENAI_BASE_URL` (defaults to `https://openrouter.ai/api/v1`)
2. Install deps: `pip install -r requirements.txt`
3. Run: `python main.py --prompt examples/prompts/basic.poml --model anthropic/claude-3.5-sonnet --input "What is POML?"`

Files

- `src/poml/` — POML-like parser and loader
- `src/llm/openrouter.py` — OpenRouter client for LangChain and raw OpenAI SDK
- `src/graph/app_graph.py` — Minimal LangGraph app using the POML loader
- `examples/prompts/basic.poml` — Example POML-like prompt