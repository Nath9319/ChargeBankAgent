"""
Example Usage of POML Prompts with LangGraph and OpenRouter
This file demonstrates various use cases and integrations
"""

import json
import os
from dotenv import load_dotenv
from poml_prompts import (
    POMLPromptEngine, 
    POMLPrompt, 
    PromptVariable, 
    PromptFunction,
    CommonPOMLPrompts,
    create_poml_from_template,
    export_poml_to_yaml,
    export_poml_to_json
)
from langgraph_integration import POMLLangGraphAgent, create_chargebank_agent
from openrouter_config import OpenRouterConfig, select_model_for_prompt

load_dotenv()


def demo_basic_poml():
    """Demonstrate basic POML prompt creation and usage"""
    print("=== Basic POML Demo ===")
    
    # Create a simple prompt
    simple_prompt = POMLPrompt(
        name="greeting",
        description="Generate personalized greetings",
        system_message="You are a friendly AI assistant that creates personalized greetings.",
        user_message_template="Create a greeting for {{name}} who is a {{role}} in the {{company}} industry.",
        variables=[
            PromptVariable("name", "Person's name", "string", True),
            PromptVariable("role", "Person's job role", "string", True),
            PromptVariable("company", "Company or industry", "string", False, "technology")
        ],
        output_format={
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "personalized_message": {"type": "string"},
                "suggested_topics": {"type": "array", "items": {"type": "string"}}
            }
        }
    )
    
    # Export to different formats
    print("YAML Export:")
    print(export_poml_to_yaml(simple_prompt))
    
    print("\nJSON Export:")
    print(export_poml_to_json(simple_prompt))
    
    return simple_prompt


def demo_poml_engine():
    """Demonstrate POML prompt engine functionality"""
    print("\n=== POML Engine Demo ===")
    
    engine = POMLPromptEngine()
    
    # Register prompts
    engine.register_prompt(CommonPOMLPrompts.create_analysis_prompt())
    engine.register_prompt(CommonPOMLPrompts.create_code_review_prompt())
    
    # Render a prompt
    variables = {
        "data": "Sales increased by 15% in Q3",
        "context": "E-commerce business"
    }
    
    rendered = engine.render_prompt("data_analysis", variables)
    print("Rendered prompt:")
    print(f"System: {rendered['system_message']}")
    print(f"User: {rendered['user_message']}")
    
    # Get schema for function calling
    schema = engine.get_prompt_schema("data_analysis")
    print(f"\nFunction schema: {json.dumps(schema, indent=2)}")
    
    return engine


def demo_openrouter_config():
    """Demonstrate OpenRouter configuration and model selection"""
    print("\n=== OpenRouter Config Demo ===")
    
    try:
        config = OpenRouterConfig()
        
        print("Available models:")
        for name, model in config.MODELS.items():
            print(f"- {name}: ${model.pricing['input'] + model.pricing['output']}/1M tokens")
        
        # Model selection examples
        print("\nModel selection examples:")
        
        # For function calling
        function_model = select_model_for_prompt(["function_calling"])
        print(f"Best model for function calling: {function_model}")
        
        # For budget constraint
        budget_model = select_model_for_prompt(["function_calling"], budget_constraint=0.01)
        print(f"Best model under $0.01: {budget_model}")
        
        # Cost estimation
        cost = config.get_cost_estimate("gpt-4", 1000, 500)
        print(f"Estimated cost for GPT-4 (1K input, 500 output): ${cost:.6f}")
        
    except Exception as e:
        print(f"OpenRouter config error: {e}")


def demo_langgraph_integration():
    """Demonstrate LangGraph integration with POML prompts"""
    print("\n=== LangGraph Integration Demo ===")
    
    try:
        # Create agent
        agent = create_chargebank_agent()
        
        print(f"Available prompts: {agent.get_available_prompts()}")
        
        # Example execution
        result = agent.execute_prompt("data_analysis", {
            "data": "Monthly revenue: $50K, $55K, $60K, $65K",
            "context": "Q1 2024 financial data"
        })
        
        print(f"Execution result: {json.dumps(result, indent=2)}")
        
        return agent
        
    except Exception as e:
        print(f"LangGraph integration error: {e}")
        print("Make sure OPENROUTER_API_KEY is set")


def demo_custom_prompt_creation():
    """Demonstrate creating custom prompts for specific use cases"""
    print("\n=== Custom Prompt Creation Demo ===")
    
    # Create a financial advisor prompt
    financial_prompt = POMLPrompt(
        name="financial_advisor",
        description="Provide financial advice based on user profile and goals",
        system_message="""You are a certified financial advisor with expertise in personal finance, 
        investment strategies, and retirement planning. Provide personalized, actionable advice 
        based on the user's financial situation and goals.""",
        user_message_template="""
        Financial Profile:
        - Age: {{age}}
        - Income: {{income}}
        - Current Savings: {{savings}}
        - Financial Goals: {{goals}}
        - Risk Tolerance: {{risk_tolerance}}
        - Time Horizon: {{time_horizon}}
        
        Please provide comprehensive financial advice including:
        1. Budget recommendations
        2. Investment strategy
        3. Risk management
        4. Retirement planning
        5. Tax optimization strategies
        """,
        variables=[
            PromptVariable("age", "User's age", "integer", True),
            PromptVariable("income", "Annual income", "string", True),
            PromptVariable("savings", "Current savings amount", "string", True),
            PromptVariable("goals", "Financial goals", "string", True),
            PromptVariable("risk_tolerance", "Risk tolerance level", "string", False, "moderate"),
            PromptVariable("time_horizon", "Investment time horizon", "string", False, "10-20 years")
        ],
        output_format={
            "type": "object",
            "properties": {
                "budget_recommendations": {"type": "array", "items": {"type": "string"}},
                "investment_strategy": {"type": "object", "properties": {
                    "asset_allocation": {"type": "string"},
                    "recommended_products": {"type": "array", "items": {"type": "string"}},
                    "rebalancing_schedule": {"type": "string"}
                }},
                "risk_management": {"type": "array", "items": {"type": "string"}},
                "retirement_planning": {"type": "object", "properties": {
                    "target_amount": {"type": "string"},
                    "monthly_contribution": {"type": "string"},
                    "strategies": {"type": "array", "items": {"type": "string"}}
                }},
                "tax_optimization": {"type": "array", "items": {"type": "string"}},
                "next_steps": {"type": "array", "items": {"type": "string"}}
            }
        }
    )
    
    print("Created financial advisor prompt:")
    print(f"Name: {financial_prompt.name}")
    print(f"Variables: {[v.name for v in financial_prompt.variables]}")
    print(f"Output format keys: {list(financial_prompt.output_format.get('properties', {}).keys())}")
    
    return financial_prompt


def demo_prompt_templates():
    """Demonstrate using prompt templates"""
    print("\n=== Prompt Templates Demo ===")
    
    # Create prompt from template
    analysis_prompt = create_poml_from_template("analysis")
    print(f"Created analysis prompt from template: {analysis_prompt.name}")
    
    # Create code review prompt
    code_review_prompt = create_poml_from_template("code_review")
    print(f"Created code review prompt from template: {code_review_prompt.name}")
    
    # Create planning prompt
    planning_prompt = create_poml_from_template("planning")
    print(f"Created planning prompt from template: {planning_prompt.name}")


def main():
    """Run all demos"""
    print("🚀 POML Prompts with LangGraph and OpenRouter Demo\n")
    
    # Run demos
    demo_basic_poml()
    demo_poml_engine()
    demo_openrouter_config()
    demo_langgraph_integration()
    demo_custom_prompt_creation()
    demo_prompt_templates()
    
    print("\n✅ All demos completed!")
    print("\nTo use this system:")
    print("1. Set OPENROUTER_API_KEY environment variable")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Import and use the modules in your code")


if __name__ == "__main__":
    main()