#!/usr/bin/env python3
"""
Demo script for ChargeBankAgent
Shows POML + LangGraph + OpenRouter integration in action
"""

import asyncio
import os
from dotenv import load_dotenv

from charge_bank_agent import ChargeBankAgentInterface
from config import ChargeBankConfig

load_dotenv()


async def demo_poml_features():
    """Demonstrate POML-specific features"""
    
    print("🎯 POML Features Demonstration")
    print("=" * 40)
    
    agent = ChargeBankAgentInterface()
    
    # Demo 1: Structured role-based analysis
    print("\n1. POML Structured Analysis:")
    print("Query: 'Analyze charging options for fleet vehicles in Chicago'")
    
    response = await agent.process_query(
        "Analyze charging options for fleet vehicles in Chicago",
        location="Chicago, IL",
        vehicle_type="Commercial Fleet",
        business_type="delivery service",
        fleet_size="25 vehicles"
    )
    print(f"Response: {response[:200]}...")
    
    # Demo 2: Context-aware planning
    print("\n2. POML Context-Aware Planning:")
    print("Query: 'Plan charging for weekend road trip'")
    
    response = await agent.process_query(
        "Plan charging for weekend road trip",
        start_location="San Francisco, CA",
        destination="Lake Tahoe, CA",
        departure_time="Friday evening",
        return_time="Sunday evening",
        vehicle_type="Tesla Model Y"
    )
    print(f"Response: {response[:200]}...")
    
    # Demo 3: Structured troubleshooting
    print("\n3. POML Structured Troubleshooting:")
    print("Query: 'Charging session keeps timing out'")
    
    response = await agent.process_query(
        "Charging session keeps timing out after 10 minutes",
        issue_type="session_timeout",
        station_name="Electrify America",
        vehicle_model="Ford Mustang Mach-E",
        error_details="Session ends with 'Communication Error'"
    )
    print(f"Response: {response[:200]}...")


async def demo_langgraph_workflow():
    """Demonstrate LangGraph workflow features"""
    
    print("\n\n🔄 LangGraph Workflow Demonstration")
    print("=" * 40)
    
    agent = ChargeBankAgentInterface()
    
    # Demo different query types and routing
    queries = [
        ("Analysis Query", "What are the fastest charging stations in Austin?"),
        ("Planning Query", "Plan charging stops from Dallas to Houston"),
        ("Troubleshooting Query", "My ChargePoint card isn't working")
    ]
    
    for query_type, query in queries:
        print(f"\n{query_type}: '{query}'")
        response = await agent.process_query(query)
        print(f"Workflow routing and response: {response[:150]}...")


async def demo_openrouter_models():
    """Demonstrate different OpenRouter model capabilities"""
    
    print("\n\n🌐 OpenRouter Model Comparison")
    print("=" * 40)
    
    query = "Quick charging station recommendation for Tesla in downtown LA"
    
    models_to_test = [
        ("Fast Model", "anthropic/claude-3-haiku"),
        ("Balanced Model", "openai/gpt-3.5-turbo"),
        ("Premium Model", "anthropic/claude-3.5-sonnet")
    ]
    
    for model_desc, model_name in models_to_test:
        print(f"\n{model_desc} ({model_name}):")
        try:
            agent = ChargeBankAgentInterface(model_name=model_name, temperature=0.5)
            response = await agent.process_query(query, location="Los Angeles, CA")
            print(f"Response: {response[:150]}...")
        except Exception as e:
            print(f"Error with {model_name}: {e}")


async def demo_session_persistence():
    """Demonstrate session management and context persistence"""
    
    print("\n\n💾 Session Management Demonstration")
    print("=" * 40)
    
    agent = ChargeBankAgentInterface()
    session_id = "demo_session"
    
    # Create session with preferences
    session = agent.create_session(
        session_id,
        user_preferences={
            "preferred_networks": ["Tesla Supercharger", "Electrify America"],
            "vehicle": "Tesla Model 3",
            "charging_speed_preference": "fast",
            "budget": "moderate"
        }
    )
    
    print(f"Created session: {session_id}")
    
    # First query
    print("\nFirst query: 'Find charging near my hotel'")
    response1 = await agent.process_query(
        "Find charging near my hotel",
        session_id=session_id,
        location="Miami Beach, FL"
    )
    print(f"Response 1: {response1[:150]}...")
    
    # Follow-up query (should use context)
    print("\nFollow-up query: 'What about pricing at those stations?'")
    response2 = await agent.process_query(
        "What about pricing at those stations?",
        session_id=session_id
    )
    print(f"Response 2: {response2[:150]}...")
    
    # Session info
    session_info = agent.get_session_info(session_id)
    print(f"\nSession info: {session_info}")


def demo_configuration():
    """Demonstrate configuration options"""
    
    print("\n\n⚙️  Configuration Demonstration")
    print("=" * 40)
    
    # Show available models
    print("Available Models:")
    for key, config in ChargeBankConfig.MODELS.items():
        print(f"• {key}: {config.description}")
        print(f"  Cost: ${config.cost_per_1k_tokens}/1K tokens, Tier: {config.tier.value}")
    
    # Show charging networks
    print(f"\nSupported Charging Networks:")
    for key, network in ChargeBankConfig.CHARGING_NETWORKS.items():
        print(f"• {network['name']}: {', '.join(network['connectors'])}")
    
    # Environment validation
    print(f"\nEnvironment Validation:")
    validation = ChargeBankConfig.validate_environment()
    for var, is_set in validation.items():
        status = "✅" if is_set else "❌"
        print(f"  {status} {var}")


async def main():
    """Run all demonstrations"""
    
    print("🔋 ChargeBankAgent Demo")
    print("Microsoft POML + LangGraph + OpenRouter Integration")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  Warning: OPENROUTER_API_KEY not set.")
        print("Some demonstrations will be limited.")
        print("Set your API key in .env file for full functionality.\n")
    
    try:
        # Run configuration demo first (doesn't require API)
        demo_configuration()
        
        # Run API-dependent demos if key is available
        if os.getenv("OPENROUTER_API_KEY"):
            await demo_poml_features()
            await demo_langgraph_workflow()
            await demo_openrouter_models()
            await demo_session_persistence()
        else:
            print("\n🔑 Set OPENROUTER_API_KEY to see live demonstrations")
        
        print("\n\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("• Run 'python run_agent.py' for interactive CLI")
        print("• Run 'python examples.py' for more examples")
        print("• Run 'python test_agent.py' for testing")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("Make sure dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    asyncio.run(main())