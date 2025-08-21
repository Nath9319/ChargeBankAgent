# ChargeBank Agent with Microsoft POML-like Prompting

A sophisticated AI agent system that implements Microsoft POML (Prompt Orchestration Markup Language) like prompting techniques, integrated with LangGraph for workflow orchestration and OpenRouter for multi-model LLM access.

## 🚀 Features

- **POML-like Prompting**: Structured prompt management with variables, functions, and output validation
- **LangGraph Integration**: Seamless workflow orchestration with state management
- **OpenRouter Support**: Access to multiple LLM providers (OpenAI, Anthropic, Google, Meta)
- **Template System**: Pre-built prompts for common use cases
- **Function Calling**: Native support for tool execution and function calling
- **Cost Optimization**: Intelligent model selection based on requirements and budget
- **Validation**: Output validation against expected schemas

## 📁 Project Structure

```
├── poml_prompts.py          # Core POML prompting system
├── langgraph_integration.py # LangGraph workflow integration
├── openrouter_config.py     # OpenRouter configuration and model selection
├── example_usage.py         # Comprehensive usage examples
├── sample_prompts.yaml      # Sample POML prompts in YAML format
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd chargebank-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**:
   ```bash
   export OPENROUTER_API_KEY="your_openrouter_api_key_here"
   ```

   Or create a `.env` file:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

## 🔑 Getting Started

### Basic POML Prompt Usage

```python
from poml_prompts import POMLPromptEngine, CommonPOMLPrompts

# Create engine
engine = POMLPromptEngine()

# Register a prompt
analysis_prompt = CommonPOMLPrompts.create_analysis_prompt()
engine.register_prompt(analysis_prompt)

# Render prompt with variables
variables = {
    "data": "Sales increased by 15% in Q3",
    "context": "E-commerce business"
}

rendered = engine.render_prompt("data_analysis", variables)
print(rendered["user_message"])
```

### LangGraph Integration

```python
from langgraph_integration import create_chargebank_agent

# Create agent
agent = create_chargebank_agent()

# Execute a prompt
result = agent.execute_prompt("data_analysis", {
    "data": "Monthly revenue: $50K, $55K, $60K, $65K",
    "context": "Q1 2024 financial data"
})

print(result["output"])
```

### OpenRouter Model Selection

```python
from openrouter_config import select_model_for_prompt

# Select best model for function calling
model = select_model_for_prompt(
    prompt_requirements=["function_calling"],
    budget_constraint=0.01
)
print(f"Selected model: {model}")
```

## 📝 POML Prompt Structure

POML prompts are structured with the following components:

### Core Elements

- **name**: Unique identifier for the prompt
- **version**: POML version
- **description**: What the prompt does
- **system_message**: Role definition and context
- **user_message_template**: Template with variables using Jinja2 syntax

### Variables

```yaml
variables:
  - name: "data"
    description: "The data to analyze"
    type: "string"
    required: true
  - name: "context"
    description: "Additional context"
    type: "string"
    required: false
    default: "No additional context"
```

### Output Format

```yaml
output_format:
  type: "object"
  properties:
    key_findings:
      type: "array"
      items:
        type: "string"
    recommendations:
      type: "array"
      items:
        type: "string"
```

### Functions

```yaml
functions:
  - name: "calculate_metrics"
    description: "Calculate financial metrics"
    parameters:
      - name: "data"
        description: "Financial data"
        type: "string"
        required: true
    returns: "Calculated metrics object"
```

## 🔄 LangGraph Workflow

The system uses LangGraph for orchestration with the following workflow:

1. **Process Prompt**: Parse and render POML prompt with variables
2. **Execute LLM**: Send to OpenRouter with appropriate model
3. **Validate Output**: Check output against expected schema
4. **Handle Errors**: Manage errors and edge cases

### State Management

```python
class AgentState(TypedDict):
    messages: List[Any]                    # Conversation history
    current_prompt: Optional[str]          # Active prompt name
    prompt_variables: Dict[str, Any]      # Input variables
    prompt_output: Optional[Dict[str, Any]] # LLM response
    error: Optional[str]                   # Error messages
    metadata: Dict[str, Any]              # Additional data
```

## 🌐 OpenRouter Integration

### Supported Models

- **OpenAI**: GPT-4, GPT-4 Turbo
- **Anthropic**: Claude 3 Opus, Claude 3 Sonnet
- **Google**: Gemini Pro
- **Meta**: Llama 3 70B

### Model Selection

The system automatically selects the best model based on:

- **Capabilities**: Function calling, JSON mode, vision
- **Context Length**: Required input/output length
- **Budget**: Cost constraints
- **Performance**: Model capabilities vs. requirements

### Cost Optimization

```python
# Estimate costs
config = OpenRouterConfig()
cost = config.get_cost_estimate("gpt-4", 1000, 500)
print(f"Estimated cost: ${cost:.6f}")

# Select budget-friendly model
model = select_model_for_prompt(
    ["function_calling"], 
    budget_constraint=0.01
)
```

## 📊 Sample Prompts

The system includes several pre-built prompts:

### Data Analysis
- Analyzes data and provides insights
- Identifies trends and patterns
- Offers actionable recommendations

### Code Review
- Reviews code for quality issues
- Identifies security concerns
- Suggests performance improvements

### Financial Planning
- Creates strategic financial plans
- Assesses risks and opportunities
- Provides timeline and resource planning

### Customer Service
- Handles customer inquiries
- Provides solutions and next steps
- Manages escalation requirements

### Fraud Detection
- Analyzes transaction patterns
- Identifies suspicious activities
- Recommends security actions

### Investment Advisory
- Provides investment recommendations
- Manages portfolio allocation
- Offers risk management strategies

## 🎯 Advanced Usage

### Custom Prompt Creation

```python
from poml_prompts import POMLPrompt, PromptVariable

custom_prompt = POMLPrompt(
    name="custom_analysis",
    description="Custom analysis prompt",
    system_message="You are a specialized analyst...",
    user_message_template="Analyze: {{data}}",
    variables=[
        PromptVariable("data", "Data to analyze", "string", True)
    ],
    output_format={
        "type": "object",
        "properties": {
            "analysis": {"type": "string"}
        }
    }
)
```

### Loading from YAML

```python
with open("prompts.yaml", "r") as f:
    yaml_content = f.read()

prompt = engine.load_prompt_from_yaml(yaml_content)
engine.register_prompt(prompt)
```

### Function Calling Integration

```python
# Define tools
@tool
def get_financial_data(symbol: str) -> str:
    """Get financial data for a symbol"""
    return f"Financial data for {symbol}"

# Use in prompts
prompt.functions = [
    PromptFunction(
        name="get_financial_data",
        description="Retrieve financial data",
        parameters=[PromptVariable("symbol", "Stock symbol", "string", True)],
        returns="Financial data string"
    )
]
```

## 🧪 Testing and Examples

Run the comprehensive example:

```bash
python example_usage.py
```

This will demonstrate:
- Basic POML functionality
- Engine operations
- OpenRouter configuration
- LangGraph integration
- Custom prompt creation
- Template usage

## 🔧 Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENROUTER_BASE_URL`: Custom OpenRouter endpoint (optional)

### Model Configuration

```python
# Custom model configuration
config = OpenRouterConfig()
config.MODELS["custom_model"] = OpenRouterModel(
    name="custom/provider/model",
    provider="Custom",
    context_length=16384,
    max_tokens=4096,
    pricing={"input": 5.0, "output": 15.0},
    capabilities=["function_calling"],
    description="Custom model description"
)
```

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Variable Validation**: Ensures required variables are provided
- **Template Rendering**: Handles Jinja2 template errors
- **LLM Execution**: Manages API failures and timeouts
- **Output Validation**: Validates responses against schemas
- **Workflow Errors**: Graceful error handling in LangGraph

## 📈 Performance Optimization

### Token Management

- Automatic context length optimization
- Cost-effective model selection
- Efficient prompt rendering

### Caching

- Prompt template caching
- Model configuration caching
- Response caching (configurable)

### Parallel Execution

- Concurrent prompt processing
- Batch LLM requests
- Asynchronous workflow execution

## 🔒 Security Features

- **API Key Management**: Secure environment variable handling
- **Input Validation**: Comprehensive variable validation
- **Output Sanitization**: Safe response handling
- **Rate Limiting**: Built-in API rate limiting
- **Error Masking**: Secure error message handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the examples in `example_usage.py`
- Review the sample prompts in `sample_prompts.yaml`

## 🔮 Future Enhancements

- **Multi-modal Support**: Image and audio processing
- **Advanced Caching**: Redis-based prompt caching
- **Prompt Versioning**: Version control for prompts
- **A/B Testing**: Prompt performance testing
- **Analytics**: Usage and performance metrics
- **Plugin System**: Extensible prompt plugins

---

**Built with ❤️ for the ChargeBank team**