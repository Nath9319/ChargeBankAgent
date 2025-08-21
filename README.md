# ChargeBankAgent

🔋 **Advanced Electric Vehicle Charging Assistant**

A sophisticated AI agent that combines **Microsoft POML** (Prompt Optimization Markup Language) techniques with **LangGraph** workflows and **OpenRouter** LLM access to provide intelligent electric vehicle charging assistance.

## ✨ Features

- **🎯 POML-Structured Prompting**: Uses Microsoft POML techniques for clear, maintainable, and effective prompts
- **🔄 LangGraph Workflows**: Intelligent routing and multi-step reasoning using LangGraph state machines
- **🌐 OpenRouter Integration**: Access to multiple LLM providers through a unified API
- **📍 Location-Aware**: Provides location-specific charging station recommendations
- **🗺️ Route Planning**: Optimizes charging stops for long-distance travel
- **🔧 Troubleshooting**: Diagnoses and resolves charging-related issues
- **💰 Cost Optimization**: Finds the most cost-effective charging solutions

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ChargeBankAgent

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### 2. Configuration

Create a `.env` file with your OpenRouter API key:

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 3. Basic Usage

```python
import asyncio
from charge_bank_agent import ChargeBankAgentInterface

async def main():
    agent = ChargeBankAgentInterface()
    
    response = await agent.process_query(
        "Find the best charging stations in San Francisco for a Tesla Model 3",
        location="San Francisco, CA",
        vehicle_type="Tesla Model 3"
    )
    
    print(response)

asyncio.run(main())
```

### 4. CLI Interface

```bash
# Interactive mode
python run_agent.py

# Single query mode
python run_agent.py "Find charging stations near LAX airport"
```

## 🏗️ Architecture

### POML Templates

The system uses Microsoft POML-inspired structured prompts:

```xml
<poml>
<role>
You are an expert Electric Vehicle Charging Infrastructure Analyst...
</role>

<task>
Analyze the user's charging needs and provide comprehensive recommendations.
</task>

<context>
<user-location>San Francisco, CA</user-location>
<vehicle-type>Tesla Model 3</vehicle-type>
<budget-constraints>moderate</budget-constraints>
</context>

<instructions>
1. Location Analysis: Assess charging infrastructure availability
2. Cost Optimization: Identify cost-effective options
3. Accessibility: Consider charging speed and connector types
</instructions>

<output-format>
Structure your response with clear sections and actionable advice.
</output-format>
</poml>
```

### LangGraph Workflow

The agent uses a state-based workflow:

1. **Query Classification**: Determines the type of request (analysis/planning/troubleshooting)
2. **Specialized Processing**: Routes to appropriate POML template
3. **Response Synthesis**: Combines results into a coherent response
4. **Error Handling**: Graceful error recovery and user guidance

### OpenRouter Integration

Supports multiple LLM providers:
- Anthropic Claude 3.5 Sonnet (recommended)
- OpenAI GPT-4 Turbo
- Meta Llama models
- Google Gemini Pro
- Mistral models

## 📚 Usage Examples

### Charging Station Analysis

```python
from charge_bank_agent import quick_analysis

response = await quick_analysis(
    "Compare charging networks in Texas",
    location="Austin, TX",
    vehicle_type="Ford F-150 Lightning"
)
```

### Route Planning

```python
from charge_bank_agent import plan_charging_route

route_plan = await plan_charging_route(
    start="Denver, CO",
    destination="Salt Lake City, UT", 
    vehicle_type="Rivian R1T"
)
```

### Troubleshooting

```python
from charge_bank_agent import troubleshoot_charging

help_response = await troubleshoot_charging(
    "Charging cable won't lock into my vehicle",
    vehicle_type="BMW i4",
    station_name="Electrify America"
)
```

### Session Management

```python
agent = ChargeBankAgentInterface()

# Create persistent session
session = agent.create_session(
    session_id="user_123",
    user_preferences={
        "preferred_networks": ["Tesla Supercharger"],
        "max_charging_time": "30 minutes",
        "budget_preference": "cost-effective"
    }
)

# Queries in session maintain context
response1 = await agent.process_query(
    "Find charging stations near me",
    session_id="user_123",
    location="Portland, OR"
)

response2 = await agent.process_query(
    "What about pricing at those stations?",
    session_id="user_123"  # Context from previous query preserved
)
```

### Specialized Agents

```python
from charge_bank_agent import ChargeBankAgentFactory

# Cost optimization specialist
cost_agent = ChargeBankAgentFactory.create_specialized_agent("cost_optimizer")

# Route planning specialist  
route_agent = ChargeBankAgentFactory.create_specialized_agent("route_planner")

# Technical support specialist
tech_agent = ChargeBankAgentFactory.create_specialized_agent("technical_support")
```

## 🔧 Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `OPENROUTER_SITE_URL`: Your site URL for OpenRouter rankings (optional)
- `OPENROUTER_SITE_NAME`: Your site name for OpenRouter rankings (optional)
- `MODEL_NAME`: Default model to use (optional, defaults to claude-3.5-sonnet)
- `TEMPERATURE`: Default temperature setting (optional, defaults to 0.7)

### Model Selection

Choose models based on your needs:

- **Fast responses**: `claude-3-haiku` or `gpt-3.5-turbo`
- **Balanced performance**: `gpt-3.5-turbo` or `claude-3.5-sonnet`
- **Premium quality**: `claude-3.5-sonnet` or `gpt-4-turbo`

## 🎯 POML Best Practices

The system implements Microsoft POML techniques:

1. **Structured Roles**: Clear role definitions for different agent personalities
2. **Task Specification**: Explicit task descriptions and objectives
3. **Context Management**: Organized context information in structured blocks
4. **Output Formatting**: Consistent response structures and formatting
5. **Example Integration**: Relevant examples embedded in prompts
6. **Constraint Definition**: Clear operational boundaries and limitations

## 🔗 Integration Guide

### With Existing LangChain Applications

```python
from langgraph_agent import ChargeBankAgent
from your_app import existing_chain

# Create agent
agent = ChargeBankAgent()

# Integrate with existing chain
combined_response = await existing_chain.arun(
    user_input,
    charging_analysis=await agent.process_query(user_input)
)
```

### With FastAPI

```python
from fastapi import FastAPI
from charge_bank_agent import ChargeBankAgentInterface

app = FastAPI()
agent = ChargeBankAgentInterface()

@app.post("/charging-help")
async def get_charging_help(query: str, location: str = None):
    return await agent.process_query(query, location=location)
```

### With Streamlit

```python
import streamlit as st
from charge_bank_agent import ChargeBankAgentInterface

agent = ChargeBankAgentInterface()

st.title("🔋 EV Charging Assistant")
user_query = st.text_input("Ask about charging stations, routes, or issues:")

if user_query:
    with st.spinner("Processing..."):
        response = agent.process_query_sync(user_query)
        st.write(response)
```

## 🧪 Testing

```bash
# Run examples
python examples.py

# Run configuration check
python config.py

# Interactive CLI
python run_agent.py
```

## 📖 API Reference

### ChargeBankAgentInterface

Main interface for the agent system.

#### Methods

- `process_query(user_query, session_id=None, **context)`: Process a query asynchronously
- `process_query_sync(user_query, session_id=None, **context)`: Process a query synchronously
- `create_session(session_id, user_preferences=None)`: Create a new user session
- `get_session_info(session_id)`: Get session information
- `update_user_preferences(session_id, preferences)`: Update user preferences

### POML Templates

- `ChargeBankAnalystPrompt`: For charging station analysis and recommendations
- `ChargeBankPlannerPrompt`: For route planning and optimization
- `ChargeBankTroubleshooterPrompt`: For issue diagnosis and resolution

### Convenience Functions

- `quick_analysis(query, location=None, **kwargs)`: Quick charging analysis
- `plan_charging_route(start, destination, vehicle_type, **kwargs)`: Route planning
- `troubleshoot_charging(issue_description, **kwargs)`: Issue troubleshooting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Microsoft POML**: For structured prompting techniques
- **LangGraph**: For workflow orchestration
- **OpenRouter**: For LLM API access
- **LangChain**: For LLM integration framework