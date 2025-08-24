# POML-Enhanced LangGraph Agent

A sophisticated AI agent system that implements Microsoft's **POML (Parrot-Olympiad-Math-Logic)** prompting technique with LangGraph workflows and OpenRouter integration for accessing multiple LLM models.

## 🌟 Features

- **POML Reasoning Framework**: Structured 6-phase reasoning process
  - **Understand** (Parrot): Comprehend and restate problems
  - **Decompose** (Olympiad): Break down complex problems
  - **Analyze** (Math): Apply logical reasoning
  - **Synthesize** (Math-Logic): Combine solutions
  - **Verify** (Logic): Validate results
  - **Reflect**: Learn and improve

- **LangGraph Integration**: State-based workflow management with conditional branching
- **OpenRouter Support**: Access to 10+ LLM models including:
  - Claude 3.5 Sonnet
  - GPT-4 Turbo
  - Llama 3.1 (70B, 405B)
  - Gemini Pro 1.5
  - And more...

- **Flexible Architecture**: Both full POML workflow and simplified single-shot modes
- **Cost Optimization**: Model recommendations and cost estimation
- **Comprehensive Testing**: Full test suite with mocking support

## 📋 Prerequisites

- Python 3.8+
- OpenRouter API key ([Get one here](https://openrouter.ai/))

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ChargeBankAgent

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### 2. Configuration

Edit `.env` file with your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
DEFAULT_MODEL=anthropic/claude-3.5-sonnet
```

### 3. Basic Usage

```python
from langgraph_poml_agent import POMLAgent
import asyncio

# Initialize the agent
agent = POMLAgent(
    model_name="anthropic/claude-3.5-sonnet",
    temperature=0.7
)

# Process a task
async def solve_problem():
    result = await agent.process(
        "Design a rate limiter for an API that handles 1M requests/second"
    )
    print(result["final_output"])

# Run
asyncio.run(solve_problem())
```

### 4. Run Examples

```bash
# Interactive example menu
python examples.py

# Run specific example
python -c "from examples import example_math_problem; import asyncio; asyncio.run(example_math_problem())"
```

## 📖 Documentation

### POML Prompting Technique

The POML (Parrot-Olympiad-Math-Logic) technique enhances AI reasoning through structured problem-solving phases:

1. **Parrot Phase**: Accurately understand and restate the problem
2. **Olympiad Phase**: Apply systematic problem-solving strategies
3. **Math Phase**: Use logical, step-by-step reasoning
4. **Logic Phase**: Verify conclusions through formal validation

### Agent Types

#### Full POML Agent
```python
from langgraph_poml_agent import POMLAgent

agent = POMLAgent(
    model_name="anthropic/claude-3.5-sonnet",
    temperature=0.5,
    max_iterations=10  # Maximum reasoning iterations
)

result = await agent.process("Your complex problem here")
```

#### Simple POML Agent (Single-shot)
```python
from langgraph_poml_agent import SimplePOMLAgent

agent = SimplePOMLAgent(
    model_name="meta-llama/llama-3.1-70b-instruct",
    temperature=0.7
)

result = await agent.process("Your quick task here")
```

### Custom POML Prompts

```python
from poml_prompts import POMLPromptBuilder

builder = POMLPromptBuilder()

# Create custom system prompt
system_prompt = builder.build_system_prompt(
    role="code reviewer",
    domain="Python development"
)

# Create task-specific prompt
task_prompt = builder.build_task_prompt(
    task="Review this code for security vulnerabilities",
    context={"language": "Python", "framework": "Django"}
)

# Create verification prompt
verification_prompt = builder.build_verification_prompt(
    solution="Your solution here",
    criteria=["Security", "Performance", "Maintainability"]
)
```

### Model Selection

```python
from openrouter_config import get_recommended_model, OpenRouterModels

# Get recommended model for use case
model = get_recommended_model("reasoning")  # Best for complex reasoning
model = get_recommended_model("coding")     # Best for code generation
model = get_recommended_model("budget", budget_conscious=True)  # Cost-effective

# List available models
models = OpenRouterModels.list_models(
    supports_functions=True,
    supports_vision=True
)

# Get cheapest model meeting requirements
cheapest = OpenRouterModels.get_cheapest_model(
    min_context=50000,
    supports_functions=True
)
```

### Cost Estimation

```python
from openrouter_config import estimate_cost

cost = estimate_cost(
    model_name="anthropic/claude-3.5-sonnet",
    input_tokens=1000,
    output_tokens=500
)
print(f"Estimated cost: ${cost['total_cost']:.4f}")
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
python test_poml_agent.py

# Or use pytest directly
pytest test_poml_agent.py -v
```

## 📁 Project Structure

```
ChargeBankAgent/
├── poml_prompts.py           # POML prompting system implementation
├── langgraph_poml_agent.py   # Main agent with LangGraph workflow
├── openrouter_config.py      # OpenRouter configuration and utilities
├── examples.py               # Usage examples for different scenarios
├── test_poml_agent.py        # Comprehensive test suite
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## 🎯 Use Cases

The POML agent excels at:

- **Complex Problem Solving**: Mathematical problems, logic puzzles
- **Code Generation**: Creating well-structured, documented code
- **System Design**: Analyzing trade-offs and designing architectures
- **Debugging**: Identifying and fixing code issues
- **Analysis Tasks**: Deep analysis with structured reasoning
- **Creative Writing**: Structured creative content with logical flow

## 🔧 Advanced Configuration

### LangGraph Workflow Customization

```python
class CustomPOMLAgent(POMLAgent):
    def _build_graph(self):
        # Customize the workflow
        workflow = super()._build_graph()
        
        # Add custom nodes
        workflow.add_node("custom_step", self.custom_node)
        
        # Modify edges
        workflow.add_edge("verify", "custom_step")
        
        return workflow.compile()
```

### Custom Reasoning Steps

```python
from poml_prompts import ReasoningStep

# Define custom reasoning steps
custom_steps = [
    ReasoningStep.UNDERSTAND,
    ReasoningStep.ANALYZE,
    ReasoningStep.VERIFY
]

# Use in prompt generation
prompt = create_poml_enhanced_prompt(
    base_prompt="Your task",
    reasoning_steps=custom_steps
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Microsoft Research for the POML prompting technique
- LangChain team for LangGraph framework
- OpenRouter for unified LLM API access

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the examples in `examples.py`
- Review the test cases in `test_poml_agent.py`

## 🚦 Quick Command Reference

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API key

# Run examples
python examples.py

# Run tests
python test_poml_agent.py

# Quick test
python quick_start.py
```

---

**Note**: This implementation of POML is based on the conceptual framework and adapted for practical use with LangGraph and OpenRouter. The specific implementation details may vary from Microsoft's internal systems.