# Makefile for ChargeBankAgent
# POML + LangGraph + OpenRouter integration

.PHONY: help install setup demo test clean run examples

help:
	@echo "🔋 ChargeBankAgent - Microsoft POML + LangGraph + OpenRouter"
	@echo "============================================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make setup      - Setup environment and validate configuration"
	@echo "  make demo       - Run demonstration of all features"
	@echo "  make test       - Run tests"
	@echo "  make run        - Start interactive CLI"
	@echo "  make examples   - Run usage examples"
	@echo "  make clean      - Clean up temporary files"
	@echo "  make validate   - Validate environment and configuration"
	@echo ""
	@echo "Quick start:"
	@echo "  1. cp .env.example .env"
	@echo "  2. Edit .env and add your OPENROUTER_API_KEY"
	@echo "  3. make install"
	@echo "  4. make demo"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

setup: install
	@echo "⚙️  Setting up environment..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "📝 Created .env file from template"; \
		echo "⚠️  Please edit .env and add your OPENROUTER_API_KEY"; \
	else \
		echo "✅ .env file already exists"; \
	fi
	@python config.py
	@echo "✅ Setup complete"

demo: validate
	@echo "🎭 Running feature demonstration..."
	python demo.py

test:
	@echo "🧪 Running tests..."
	python test_agent.py

run: validate
	@echo "🚀 Starting interactive CLI..."
	python run_agent.py

examples: validate
	@echo "📚 Running usage examples..."
	python examples.py

validate:
	@echo "🔍 Validating configuration..."
	@python -c "from config import ChargeBankConfig; print('✅ Configuration valid')"
	@python -c "import os; exit(0 if os.getenv('OPENROUTER_API_KEY') else 1)" || \
		(echo "⚠️  OPENROUTER_API_KEY not set. Some features may not work." && echo "Edit .env file to add your API key.")

clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	@echo "✅ Cleanup complete"

# Development commands
dev-install: install
	pip install pytest pytest-asyncio black flake8 mypy

lint:
	@echo "🔍 Running linting..."
	black --check .
	flake8 .
	mypy .

format:
	@echo "🎨 Formatting code..."
	black .

# Quick usage examples
quick-analysis:
	python -c "import asyncio; from charge_bank_agent import quick_analysis; print(asyncio.run(quick_analysis('Find Tesla charging in Seattle', 'Seattle, WA')))"

quick-route:
	python -c "import asyncio; from charge_bank_agent import plan_charging_route; print(asyncio.run(plan_charging_route('San Francisco', 'Los Angeles', 'Tesla Model 3')))"

quick-help:
	python -c "import asyncio; from charge_bank_agent import troubleshoot_charging; print(asyncio.run(troubleshoot_charging('Charging cable won\\'t connect')))"

# Information commands
info:
	@echo "🔋 ChargeBankAgent Information"
	@echo "=============================="
	@echo "Version: 1.0.0"
	@echo "Features: Microsoft POML + LangGraph + OpenRouter"
	@echo "Python: $(shell python --version)"
	@echo "Dependencies: $(shell pip list | grep -E 'langgraph|langchain|openai' | wc -l) key packages installed"
	@python config.py

models:
	@echo "🤖 Available Models:"
	@python -c "from config import ChargeBankConfig; [print(f'• {k}: {v.description}') for k, v in ChargeBankConfig.MODELS.items()]"