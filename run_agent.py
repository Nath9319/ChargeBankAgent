#!/usr/bin/env python3
"""
Simple CLI interface for running the ChargeBankAgent
Demonstrates POML + LangGraph + OpenRouter integration
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

from charge_bank_agent import ChargeBankAgentInterface

# Load environment variables
load_dotenv()


async def interactive_mode():
    """Run the agent in interactive CLI mode"""
    
    print("🔋 ChargeBankAgent - POML + LangGraph + OpenRouter")
    print("=" * 50)
    print("Ask me about:")
    print("• Charging station locations and recommendations")
    print("• Route planning with charging stops")
    print("• Troubleshooting charging issues")
    print("• Cost optimization strategies")
    print("\nType 'quit' to exit, 'help' for more options\n")
    
    # Initialize agent
    try:
        agent = ChargeBankAgentInterface(
            model_name=os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet"),
            temperature=float(os.getenv("TEMPERATURE", "0.7"))
        )
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("Make sure your OPENROUTER_API_KEY is set correctly.")
        return
    
    session_id = "cli_session"
    
    while True:
        try:
            # Get user input
            user_input = input("\n🤖 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if user_input.lower() == 'clear':
                # Clear session
                if session_id in agent.sessions:
                    del agent.sessions[session_id]
                print("🧹 Session cleared!")
                continue
            
            if not user_input:
                continue
            
            print("\n🔄 Processing your request...")
            
            # Process the query
            response = await agent.process_query(
                user_input,
                session_id=session_id
            )
            
            print(f"\n💡 Response:\n{response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'help' for assistance.")


def print_help():
    """Print help information"""
    print("""
🔋 ChargeBankAgent Help
=====================

Example Queries:
• "Find Tesla Supercharger stations in Miami"
• "Plan route from Boston to Washington DC with charging stops"
• "My Electrify America session won't start, what should I do?"
• "Compare charging costs between different networks in California"
• "Best charging strategy for a Rivian R1T road trip"

Commands:
• help - Show this help message
• clear - Clear conversation history
• quit/exit/q - Exit the program

Tips:
• Be specific about your location and vehicle type
• Include any error messages for troubleshooting
• Mention budget constraints for cost optimization
""")


async def single_query_mode(query: str):
    """Process a single query and exit"""
    
    try:
        agent = ChargeBankAgentInterface()
        response = await agent.process_query(query)
        print(response)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        asyncio.run(single_query_mode(query))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()