"""
Example usage of the POML-enhanced LangGraph agent with OpenRouter

This file demonstrates various ways to use the POML agent for different
types of problems and reasoning tasks.
"""

import asyncio
import os
from dotenv import load_dotenv
from langgraph_poml_agent import POMLAgent, SimplePOMLAgent
from openrouter_config import get_recommended_model, estimate_cost
from poml_prompts import POMLPromptBuilder, create_poml_enhanced_prompt

# Load environment variables
load_dotenv()


async def example_math_problem():
    """
    Example: Solving a complex math problem with POML reasoning
    """
    print("=" * 60)
    print("Example 1: Math Problem Solving with POML")
    print("=" * 60)
    
    # Initialize the agent
    agent = POMLAgent(
        model_name=get_recommended_model("reasoning"),
        temperature=0.3  # Lower temperature for math
    )
    
    # Define the problem
    problem = """
    A farmer has a rectangular field. If he increases the length by 20% and 
    decreases the width by 20%, the area remains 96 square meters. 
    If he instead increases both dimensions by 10%, the area becomes 121 square meters.
    What are the original dimensions of the field?
    """
    
    # Process with POML reasoning
    result = await agent.process(problem)
    
    print("\n📝 Problem:")
    print(problem)
    print("\n🧠 POML Reasoning Chain:")
    for step in result["reasoning_chain"]:
        print(f"\n[{step['step'].upper()}]")
        print(step["output"][:500] + "..." if len(step["output"]) > 500 else step["output"])
    
    print("\n✅ Final Answer:")
    print(result["final_output"])
    
    print("\n🔍 Verification Results:")
    print(f"Passed: {result['verification_results']['passed']}")
    
    return result


async def example_code_generation():
    """
    Example: Generating code with POML-enhanced reasoning
    """
    print("\n" + "=" * 60)
    print("Example 2: Code Generation with POML")
    print("=" * 60)
    
    # Use a coding-optimized model
    agent = POMLAgent(
        model_name=get_recommended_model("coding"),
        temperature=0.5
    )
    
    task = """
    Create a Python class for a priority queue that supports:
    1. Insert with priority (lower number = higher priority)
    2. Extract minimum (highest priority)
    3. Peek at minimum without removing
    4. Update priority of an existing element
    5. Check if empty
    
    The implementation should be efficient and handle edge cases properly.
    Include proper error handling and documentation.
    """
    
    result = await agent.process(task)
    
    print("\n📝 Task:")
    print(task)
    print("\n💻 Generated Code:")
    print(result["final_output"])
    
    return result


async def example_analysis():
    """
    Example: Complex analysis task with POML
    """
    print("\n" + "=" * 60)
    print("Example 3: System Design Analysis with POML")
    print("=" * 60)
    
    agent = POMLAgent(
        model_name=get_recommended_model("analysis"),
        temperature=0.6
    )
    
    task = """
    Design a distributed caching system for a social media platform that:
    - Handles 1 million requests per second
    - Supports user profiles, posts, and comments
    - Has 99.99% availability
    - Provides sub-100ms response times
    - Handles cache invalidation properly
    
    Analyze the trade-offs between different approaches and recommend
    the best architecture with justification.
    """
    
    result = await agent.process(task)
    
    print("\n📝 Analysis Task:")
    print(task)
    print("\n🏗️ System Design:")
    print(result["final_output"])
    
    return result


async def example_simple_poml():
    """
    Example: Using the simplified POML agent for quick tasks
    """
    print("\n" + "=" * 60)
    print("Example 4: Simple POML Agent (Single-shot)")
    print("=" * 60)
    
    # Use the simplified agent for faster responses
    agent = SimplePOMLAgent(
        model_name=get_recommended_model("budget", budget_conscious=True),
        temperature=0.7
    )
    
    task = "Explain the concept of recursion with a simple example"
    
    result = await agent.process(task)
    
    print("\n📝 Task:")
    print(task)
    print("\n💡 POML-Enhanced Response:")
    print(result)
    
    return result


async def example_creative_writing():
    """
    Example: Creative writing with POML structure
    """
    print("\n" + "=" * 60)
    print("Example 5: Creative Writing with POML Structure")
    print("=" * 60)
    
    agent = POMLAgent(
        model_name=get_recommended_model("creative"),
        temperature=0.8  # Higher temperature for creativity
    )
    
    task = """
    Write a short science fiction story (300 words) about a world where 
    memories can be traded like currency. The story should explore the 
    philosophical implications and include a twist ending.
    """
    
    result = await agent.process(task)
    
    print("\n📝 Creative Task:")
    print(task)
    print("\n📖 Story:")
    print(result["final_output"])
    
    return result


async def example_debugging():
    """
    Example: Debugging code with POML reasoning
    """
    print("\n" + "=" * 60)
    print("Example 6: Code Debugging with POML")
    print("=" * 60)
    
    agent = POMLAgent(
        model_name=get_recommended_model("coding"),
        temperature=0.3
    )
    
    task = """
    Debug this Python code that's supposed to find all prime numbers up to n:
    
    ```python
    def find_primes(n):
        primes = []
        for num in range(2, n):
            is_prime = True
            for i in range(2, num):
                if num % i == 0:
                    is_prime = False
            if is_prime:
                primes.append(num)
        return primes
    ```
    
    The code works but is very slow for large n. Identify the issues and 
    provide an optimized version with explanation.
    """
    
    result = await agent.process(task)
    
    print("\n🐛 Debugging Task:")
    print(task)
    print("\n🔧 Solution:")
    print(result["final_output"])
    
    return result


async def example_custom_prompt():
    """
    Example: Using custom POML prompts
    """
    print("\n" + "=" * 60)
    print("Example 7: Custom POML Prompt")
    print("=" * 60)
    
    # Create a custom POML-enhanced prompt
    base_task = "Compare REST APIs vs GraphQL for a mobile app backend"
    enhanced_prompt = create_poml_enhanced_prompt(
        base_prompt=base_task,
        include_examples=True
    )
    
    # Use with simple agent
    agent = SimplePOMLAgent(
        model_name=get_recommended_model("analysis"),
        temperature=0.6
    )
    
    result = await agent.process(enhanced_prompt)
    
    print("\n📝 Enhanced Task:")
    print(enhanced_prompt[:500] + "...")
    print("\n📊 Analysis:")
    print(result)
    
    return result


async def example_with_cost_estimation():
    """
    Example: Running a task with cost estimation
    """
    print("\n" + "=" * 60)
    print("Example 8: Task with Cost Estimation")
    print("=" * 60)
    
    model_name = "meta-llama/llama-3.1-70b-instruct"  # Budget model
    
    agent = SimplePOMLAgent(
        model_name=model_name,
        temperature=0.7
    )
    
    task = "Write a haiku about artificial intelligence"
    
    # Process the task
    result = await agent.process(task)
    
    # Estimate cost (rough approximation)
    input_tokens = len(task.split()) * 2  # Rough estimate
    output_tokens = len(result.split()) * 2  # Rough estimate
    
    cost = estimate_cost(model_name, input_tokens, output_tokens)
    
    print(f"\n📝 Task: {task}")
    print(f"\n🎋 Haiku:\n{result}")
    print(f"\n💰 Estimated Cost:")
    print(f"  - Input tokens: ~{input_tokens}")
    print(f"  - Output tokens: ~{output_tokens}")
    print(f"  - Total cost: ~${cost['total_cost']:.6f}")
    
    return result


async def run_all_examples():
    """
    Run all examples sequentially
    """
    print("\n🚀 Running POML Agent Examples\n")
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  Warning: OPENROUTER_API_KEY not found in environment")
        print("Please set your OpenRouter API key in .env file")
        return
    
    examples = [
        ("Math Problem", example_math_problem),
        ("Code Generation", example_code_generation),
        ("System Analysis", example_analysis),
        ("Simple POML", example_simple_poml),
        ("Creative Writing", example_creative_writing),
        ("Code Debugging", example_debugging),
        ("Custom Prompt", example_custom_prompt),
        ("Cost Estimation", example_with_cost_estimation)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n🔄 Running: {name}")
            await example_func()
            print(f"\n✅ Completed: {name}")
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")
        
        # Small delay between examples
        await asyncio.sleep(2)
    
    print("\n🎉 All examples completed!")


def main():
    """
    Main entry point for running examples
    """
    # You can run specific examples or all of them
    print("""
POML Agent Examples
===================

Choose an example to run:
1. Math Problem Solving
2. Code Generation
3. System Design Analysis
4. Simple POML (Fast)
5. Creative Writing
6. Code Debugging
7. Custom POML Prompt
8. With Cost Estimation
9. Run All Examples

Enter your choice (1-9): """, end="")
    
    choice = input().strip()
    
    example_map = {
        "1": example_math_problem,
        "2": example_code_generation,
        "3": example_analysis,
        "4": example_simple_poml,
        "5": example_creative_writing,
        "6": example_debugging,
        "7": example_custom_prompt,
        "8": example_with_cost_estimation,
        "9": run_all_examples
    }
    
    if choice in example_map:
        asyncio.run(example_map[choice]())
    else:
        print("Invalid choice. Running simple example...")
        asyncio.run(example_simple_poml())


if __name__ == "__main__":
    main()