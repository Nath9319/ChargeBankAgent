"""
Example Usage of ChargeBankAgent with POML and LangGraph
Demonstrates various use cases and integration patterns
"""

import asyncio
import os
from typing import Dict, Any

from charge_bank_agent import (
    ChargeBankAgentInterface,
    ChargeBankAgentFactory,
    quick_analysis,
    plan_charging_route,
    troubleshoot_charging
)


async def example_basic_usage():
    """Basic usage example with POML-structured prompts"""
    
    print("=== Basic ChargeBankAgent Usage ===\n")
    
    # Create agent instance
    agent = ChargeBankAgentInterface(
        model_name="anthropic/claude-3.5-sonnet",
        temperature=0.7
    )
    
    # Example 1: Charging station analysis
    print("1. Charging Station Analysis:")
    query1 = "What are the best charging stations near downtown Portland, Oregon for a Tesla Model Y?"
    response1 = await agent.process_query(
        query1,
        location="Portland, Oregon",
        vehicle_type="Tesla Model Y",
        budget="moderate"
    )
    print(f"Query: {query1}")
    print(f"Response: {response1}\n")
    
    # Example 2: Route planning
    print("2. Route Planning:")
    query2 = "Plan charging stops for a trip from Seattle to Los Angeles"
    response2 = await agent.process_query(
        query2,
        start_location="Seattle, WA",
        destination="Los Angeles, CA",
        vehicle_type="Tesla Model 3",
        departure_time="morning"
    )
    print(f"Query: {query2}")
    print(f"Response: {response2}\n")
    
    # Example 3: Troubleshooting
    print("3. Troubleshooting:")
    query3 = "My charging session keeps failing at the EVgo station"
    response3 = await agent.process_query(
        query3,
        issue_type="charging_failure",
        station_name="EVgo",
        error_details="Session stops after 5 minutes"
    )
    print(f"Query: {query3}")
    print(f"Response: {response3}\n")


async def example_session_management():
    """Example of session-based interactions with context preservation"""
    
    print("=== Session Management Example ===\n")
    
    agent = ChargeBankAgentInterface()
    session_id = "user_123"
    
    # Create session with user preferences
    session = agent.create_session(
        session_id=session_id,
        user_preferences={
            "preferred_networks": ["Tesla Supercharger", "Electrify America"],
            "max_charging_time": "30 minutes",
            "budget_preference": "cost-effective",
            "vehicle": "Tesla Model S"
        }
    )
    
    # First interaction
    response1 = await agent.process_query(
        "I need charging stations in San Francisco",
        session_id=session_id,
        location="San Francisco, CA"
    )
    print(f"First Query Response: {response1}\n")
    
    # Follow-up interaction (context preserved)
    response2 = await agent.process_query(
        "What about pricing at those stations?",
        session_id=session_id
    )
    print(f"Follow-up Query Response: {response2}\n")
    
    # Check session info
    session_info = agent.get_session_info(session_id)
    print(f"Session Info: {session_info}")


async def example_specialized_agents():
    """Example of creating specialized agents for different use cases"""
    
    print("=== Specialized Agents Example ===\n")
    
    # Cost optimization specialist
    cost_agent = ChargeBankAgentFactory.create_specialized_agent("cost_optimizer")
    cost_response = await cost_agent.process_query(
        "Find the cheapest charging options in Austin, Texas",
        location="Austin, TX",
        priority="lowest_cost"
    )
    print(f"Cost Optimizer Response: {cost_response}\n")
    
    # Route planning specialist
    route_agent = ChargeBankAgentFactory.create_specialized_agent("route_planner")
    route_response = await route_agent.process_query(
        "Optimize charging stops for a cross-country trip",
        start_location="New York, NY",
        destination="Los Angeles, CA",
        vehicle_type="Lucid Air"
    )
    print(f"Route Planner Response: {route_response}\n")
    
    # Technical support specialist
    tech_agent = ChargeBankAgentFactory.create_specialized_agent("technical_support")
    tech_response = await tech_agent.process_query(
        "Charging cable won't connect properly to CCS port",
        issue_type="hardware_problem",
        connector_type="CCS",
        vehicle_model="BMW i4"
    )
    print(f"Technical Support Response: {tech_response}\n")


async def example_convenience_functions():
    """Example of using convenience functions for common tasks"""
    
    print("=== Convenience Functions Example ===\n")
    
    # Quick analysis
    analysis = await quick_analysis(
        "Best fast charging stations in Denver",
        location="Denver, CO"
    )
    print(f"Quick Analysis: {analysis}\n")
    
    # Route planning
    route_plan = await plan_charging_route(
        start="Phoenix, AZ",
        destination="Las Vegas, NV",
        vehicle_type="Ford Mustang Mach-E"
    )
    print(f"Route Plan: {route_plan}\n")
    
    # Troubleshooting
    troubleshoot_help = await troubleshoot_charging(
        "Payment keeps getting declined at charging station",
        station_name="ChargePoint",
        issue_type="payment_problem"
    )
    print(f"Troubleshooting Help: {troubleshoot_help}\n")


def example_poml_template_customization():
    """Example of how to customize POML templates"""
    
    print("=== POML Template Customization ===\n")
    
    from poml_templates import POMLTemplate, PromptRole, POMLContext
    
    class CustomChargeBankPrompt(POMLTemplate):
        """Custom POML prompt for specific business requirements"""
        
        def __init__(self):
            super().__init__("custom_charge_bank", PromptRole.ADVISOR)
            self.metadata = {
                "version": "1.0",
                "description": "Custom prompt for enterprise charging solutions",
                "capabilities": ["fleet_management", "enterprise_planning"]
            }
        
        def format(self, context: POMLContext, **kwargs) -> str:
            fleet_size = kwargs.get('fleet_size', 'small fleet')
            business_type = kwargs.get('business_type', 'general business')
            
            return f"""<poml>
<role>
You are an Enterprise Electric Vehicle Fleet Charging Consultant specializing in:
- Large-scale charging infrastructure planning
- Fleet electrification strategies
- Cost optimization for commercial operations
- Regulatory compliance and incentive programs
</role>

<task>
Provide enterprise-level charging recommendations for fleet operations.
User Request: {context.user_query}
</task>

<enterprise-context>
<fleet-size>{fleet_size}</fleet-size>
<business-type>{business_type}</business-type>
<operational-requirements>{kwargs.get('operations', 'standard business hours')}</operational-requirements>
</enterprise-context>

<analysis-framework>
1. **Infrastructure Assessment**: Evaluate current and required charging capacity
2. **Financial Analysis**: ROI calculations and incentive opportunities
3. **Operational Integration**: Workflow and scheduling optimization
4. **Scalability Planning**: Future expansion considerations
5. **Compliance Review**: Regulatory requirements and standards
</analysis-framework>

<deliverables>
- Executive summary with key recommendations
- Detailed infrastructure plan
- Financial projections and ROI analysis
- Implementation timeline
- Risk assessment and mitigation strategies
</deliverables>
</poml>"""
    
    # Example of using the custom template
    custom_template = CustomChargeBankPrompt()
    context = POMLContext(user_query="Plan charging infrastructure for 50-vehicle delivery fleet")
    
    formatted_prompt = custom_template.format(
        context,
        fleet_size="50 vehicles",
        business_type="delivery service",
        operations="24/7 operations"
    )
    
    print("Custom POML Template:")
    print(formatted_prompt)


async def main():
    """Run all examples"""
    
    # Check if API key is available
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY not set. Some examples may not work.")
        print("Set your API key in .env file or environment variables.\n")
    
    try:
        await example_basic_usage()
        await example_session_management()
        await example_specialized_agents()
        await example_convenience_functions()
        example_poml_template_customization()
        
    except Exception as e:
        print(f"Example execution error: {e}")
        print("Make sure to set up your OpenRouter API key and install dependencies.")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())