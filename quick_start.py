#!/usr/bin/env python3
"""
Quick Start Script for POML-Enhanced LangGraph Agent

This script provides a simple way to test the POML agent setup
and run a basic example.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()


def check_setup():
    """Check if the environment is properly set up"""
    print("🔍 Checking setup...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        print("   Please set it in .env file or export it:")
        print("   export OPENROUTER_API_KEY='your-key-here'")
        return False
    print("✅ OpenRouter API key found")
    
    # Check for required files
    required_files = [
        "poml_prompts.py",
        "langgraph_poml_agent.py",
        "openrouter_config.py"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file missing: {file}")
            return False
    print("✅ All required files present")
    
    # Try importing modules
    try:
        import langchain
        import langgraph
        print("✅ LangChain and LangGraph installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True


async def run_simple_example():
    """Run a simple example with the POML agent"""
    from langgraph_poml_agent import SimplePOMLAgent
    from openrouter_config import get_recommended_model
    
    print("\n🚀 Running Simple POML Agent Example")
    print("=" * 50)
    
    # Create agent with budget-friendly model
    model = get_recommended_model("budget", budget_conscious=True)
    print(f"Using model: {model}")
    
    agent = SimplePOMLAgent(
        model_name=model,
        temperature=0.7
    )
    
    # Simple task
    task = """
    Explain the concept of recursion in programming using a simple analogy 
    that a non-programmer could understand.
    """
    
    print(f"\n📝 Task: {task}")
    print("\n⏳ Processing with POML reasoning...")
    
    try:
        result = await agent.process(task)
        print("\n✨ Response:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def run_full_example():
    """Run a full POML workflow example"""
    from langgraph_poml_agent import POMLAgent
    
    print("\n🚀 Running Full POML Agent Example")
    print("=" * 50)
    
    # Create agent
    agent = POMLAgent(
        model_name="anthropic/claude-3.5-sonnet",
        temperature=0.5,
        max_iterations=3
    )
    
    # More complex task
    task = """
    A company needs to process 10 million records daily. Each record takes 
    100ms to process sequentially. How can they complete processing within 
    8 hours? Provide specific architectural recommendations.
    """
    
    print(f"\n📝 Task: {task}")
    print("\n⏳ Processing through POML workflow...")
    print("   Phases: Understand → Decompose → Analyze → Synthesize → Verify")
    
    try:
        result = await agent.process(task)
        
        print("\n🧠 Reasoning Chain:")
        for step in result["reasoning_chain"]:
            print(f"\n[{step['step'].upper()}]")
            # Show first 200 chars of each step
            preview = step["output"][:200] + "..." if len(step["output"]) > 200 else step["output"]
            print(preview)
        
        print("\n✨ Final Solution:")
        print("-" * 50)
        print(result["final_output"])
        print("-" * 50)
        
        if result["verification_results"]:
            print(f"\n✅ Verification: {'Passed' if result['verification_results']['passed'] else 'Failed'}")
        
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def interactive_mode():
    """Run in interactive mode"""
    from langgraph_poml_agent import SimplePOMLAgent
    from openrouter_config import get_recommended_model, estimate_cost
    
    print("\n🎮 Interactive POML Agent")
    print("=" * 50)
    print("Type 'quit' to exit, 'help' for commands")
    
    # Initialize agent
    model = get_recommended_model("reasoning")
    agent = SimplePOMLAgent(model_name=model, temperature=0.7)
    
    while True:
        print("\n> ", end="")
        user_input = input().strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'help':
            print("""
Commands:
  quit     - Exit the program
  help     - Show this help
  model    - Show current model
  cost     - Estimate cost of last query
  [text]   - Process any text with POML reasoning
            """)
            continue
        elif user_input.lower() == 'model':
            print(f"Current model: {model}")
            continue
        elif user_input.lower() == 'cost':
            # Rough estimation
            tokens = len(user_input.split()) * 10
            cost = estimate_cost(model, tokens, tokens)
            print(f"Estimated cost: ${cost['total_cost']:.6f}")
            continue
        elif not user_input:
            continue
        
        print("\n⏳ Processing...")
        try:
            result = await agent.process(user_input)
            print("\n💡 Response:")
            print(result)
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════╗
║     POML-Enhanced LangGraph Agent - Quick Start     ║
╚══════════════════════════════════════════════════════╝
    """)
    
    # Check setup
    if not check_setup():
        print("\n⚠️  Please fix the setup issues above and try again.")
        sys.exit(1)
    
    print("\n✅ Setup complete!")
    
    # Menu
    print("""
Choose an option:
1. Run simple example (fast, uses budget model)
2. Run full POML workflow example (comprehensive)
3. Interactive mode (chat with the agent)
4. Exit

Enter choice (1-4): """, end="")
    
    choice = input().strip()
    
    if choice == "1":
        success = asyncio.run(run_simple_example())
    elif choice == "2":
        success = asyncio.run(run_full_example())
    elif choice == "3":
        asyncio.run(interactive_mode())
        success = True
    elif choice == "4":
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice, running simple example...")
        success = asyncio.run(run_simple_example())
    
    if success:
        print("\n✅ Example completed successfully!")
        print("\nNext steps:")
        print("  - Try more examples: python examples.py")
        print("  - Run tests: python test_poml_agent.py")
        print("  - Read the documentation in README.md")
    else:
        print("\n⚠️  Example failed. Please check your setup and API key.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)