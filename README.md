# ChargeBankAgent

POML-style prompting with LangGraph + OpenRouter

Quick start

1) Create and activate a venv
   - python -m venv .venv && source .venv/bin/activate

2) Install deps
   - pip install -r requirements.txt

3) Set env vars
   - export OPENROUTER_API_KEY=sk-or-...
   - optional: export OPENROUTER_MODEL="anthropic/claude-3.5-sonnet" (defaults to openrouter/auto)
   - optional: export OPENROUTER_REFERRER="https://your.app" and OPENROUTER_APP_TITLE="ChargeBankAgent"

4) Run
   - python -m app.main --input "Summarize why POML helps prompt robustness." --var audience=engineers

Files
- prompts/example.poml: sample POML-like prompt
- app/poml.py: POML parser and message composer
- app/graph.py: LangGraph workflow using OpenRouter via LangChain
- app/main.py: CLI entrypoint