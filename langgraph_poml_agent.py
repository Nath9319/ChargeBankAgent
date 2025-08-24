"""
LangGraph Agent with POML (Parrot-Olympiad-Math-Logic) Integration

This module implements a LangGraph-based agent that uses Microsoft's POML
prompting technique for enhanced reasoning and problem-solving.
"""

import os
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint import MemorySaver
from pydantic import BaseModel, Field

from poml_prompts import (
    POMLPromptBuilder, 
    ReasoningStep,
    create_poml_enhanced_prompt,
    validate_reasoning_chain
)


# State definition for the graph
class AgentState(TypedDict):
    """State structure for the POML-enhanced agent"""
    messages: Sequence[BaseMessage]
    current_step: str
    reasoning_chain: List[Dict[str, Any]]
    verification_results: Optional[Dict[str, Any]]
    final_output: Optional[str]
    metadata: Dict[str, Any]


class POMLNode(Enum):
    """Nodes in the POML reasoning graph"""
    UNDERSTAND = "understand"
    DECOMPOSE = "decompose"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"
    REFLECT = "reflect"
    OUTPUT = "output"


class POMLAgent:
    """
    LangGraph agent enhanced with POML reasoning framework
    """
    
    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        model_name: str = "anthropic/claude-3.5-sonnet",
        temperature: float = 0.7,
        max_iterations: int = 10
    ):
        """
        Initialize the POML-enhanced agent
        
        Args:
            openrouter_api_key: API key for OpenRouter
            model_name: Model to use via OpenRouter
            temperature: Temperature for generation
            max_iterations: Maximum reasoning iterations
        """
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        
        # Initialize the LLM with OpenRouter
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            model=model_name,
            temperature=temperature,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000"),
                "X-Title": os.getenv("OPENROUTER_SITE_NAME", "POML-Agent"),
            }
        )
        
        # Initialize prompt builder
        self.prompt_builder = POMLPromptBuilder()
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Initialize memory
        self.memory = MemorySaver()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow with POML nodes
        """
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each POML step
        workflow.add_node(POMLNode.UNDERSTAND.value, self._understand_node)
        workflow.add_node(POMLNode.DECOMPOSE.value, self._decompose_node)
        workflow.add_node(POMLNode.ANALYZE.value, self._analyze_node)
        workflow.add_node(POMLNode.SYNTHESIZE.value, self._synthesize_node)
        workflow.add_node(POMLNode.VERIFY.value, self._verify_node)
        workflow.add_node(POMLNode.REFLECT.value, self._reflect_node)
        workflow.add_node(POMLNode.OUTPUT.value, self._output_node)
        
        # Set entry point
        workflow.set_entry_point(POMLNode.UNDERSTAND.value)
        
        # Add edges
        workflow.add_edge(POMLNode.UNDERSTAND.value, POMLNode.DECOMPOSE.value)
        workflow.add_edge(POMLNode.DECOMPOSE.value, POMLNode.ANALYZE.value)
        workflow.add_edge(POMLNode.ANALYZE.value, POMLNode.SYNTHESIZE.value)
        workflow.add_edge(POMLNode.SYNTHESIZE.value, POMLNode.VERIFY.value)
        
        # Conditional edge from VERIFY
        workflow.add_conditional_edges(
            POMLNode.VERIFY.value,
            self._should_reflect_or_output,
            {
                POMLNode.REFLECT.value: POMLNode.REFLECT.value,
                POMLNode.OUTPUT.value: POMLNode.OUTPUT.value
            }
        )
        
        # Edge from REFLECT back to ANALYZE for iteration
        workflow.add_conditional_edges(
            POMLNode.REFLECT.value,
            self._should_continue_or_output,
            {
                POMLNode.ANALYZE.value: POMLNode.ANALYZE.value,
                POMLNode.OUTPUT.value: POMLNode.OUTPUT.value
            }
        )
        
        # End after OUTPUT
        workflow.add_edge(POMLNode.OUTPUT.value, END)
        
        return workflow.compile(checkpointer=self.memory)
    
    async def _understand_node(self, state: AgentState) -> AgentState:
        """
        UNDERSTAND phase: Comprehend and restate the problem
        """
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Create POML understanding prompt
        system_prompt = self.prompt_builder.build_system_prompt(
            role="reasoning agent",
            domain="problem analysis"
        )
        
        understand_prompt = f"""
## POML Phase: UNDERSTAND (Parrot)

### Input:
{last_message}

### Your Task:
1. Carefully read and comprehend the input
2. Identify key information, constraints, and requirements
3. Restate the problem in clear, structured terms
4. Note any ambiguities or missing information

### Output Format:
**Problem Restatement**: [Clear restatement of the problem]
**Key Requirements**: [List of identified requirements]
**Constraints**: [Any limitations or constraints]
**Assumptions**: [Any assumptions being made]
**Clarifications Needed**: [Questions or ambiguities, if any]
"""
        
        # Get LLM response
        response = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=understand_prompt)
        ])
        
        # Update state
        state["messages"].append(response)
        state["current_step"] = POMLNode.UNDERSTAND.value
        state["reasoning_chain"].append({
            "step": POMLNode.UNDERSTAND.value,
            "output": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        return state
    
    async def _decompose_node(self, state: AgentState) -> AgentState:
        """
        DECOMPOSE phase: Break down the problem into components
        """
        understanding = state["reasoning_chain"][-1]["output"]
        
        decompose_prompt = f"""
## POML Phase: DECOMPOSE (Olympiad)

### Previous Understanding:
{understanding}

### Your Task:
1. Break down the problem into smaller, manageable components
2. Identify patterns, relationships, and dependencies
3. Create a structured approach for each component
4. Consider multiple solution strategies

### Output Format:
**Component Breakdown**:
- Component 1: [Description and approach]
- Component 2: [Description and approach]
- ...

**Dependencies**: [How components relate to each other]

**Solution Strategies**:
- Strategy A: [Description]
- Strategy B: [Alternative approach]

**Selected Approach**: [Which strategy and why]
"""
        
        response = await self.llm.ainvoke([
            HumanMessage(content=decompose_prompt)
        ])
        
        state["messages"].append(response)
        state["current_step"] = POMLNode.DECOMPOSE.value
        state["reasoning_chain"].append({
            "step": POMLNode.DECOMPOSE.value,
            "output": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        return state
    
    async def _analyze_node(self, state: AgentState) -> AgentState:
        """
        ANALYZE phase: Apply logical reasoning to each component
        """
        decomposition = state["reasoning_chain"][-1]["output"]
        
        analyze_prompt = f"""
## POML Phase: ANALYZE (Math)

### Previous Decomposition:
{decomposition}

### Your Task:
1. Apply logical reasoning to each component
2. Use appropriate analytical tools and methods
3. Show your work step-by-step
4. Maintain mathematical/logical rigor

### Output Format:
**Component Analysis**:

For each component:
- **Analysis**: [Detailed logical analysis]
- **Method**: [Specific approach used]
- **Steps**: [Step-by-step reasoning]
- **Result**: [Component solution]

**Intermediate Validation**: [Check each component result]
"""
        
        response = await self.llm.ainvoke([
            HumanMessage(content=analyze_prompt)
        ])
        
        state["messages"].append(response)
        state["current_step"] = POMLNode.ANALYZE.value
        state["reasoning_chain"].append({
            "step": POMLNode.ANALYZE.value,
            "output": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        return state
    
    async def _synthesize_node(self, state: AgentState) -> AgentState:
        """
        SYNTHESIZE phase: Combine component solutions
        """
        analysis = state["reasoning_chain"][-1]["output"]
        
        synthesize_prompt = f"""
## POML Phase: SYNTHESIZE (Math-Logic)

### Previous Analysis:
{analysis}

### Your Task:
1. Combine component solutions into a coherent whole
2. Ensure consistency across all parts
3. Build the complete solution systematically
4. Document your reasoning chain

### Output Format:
**Integration Process**:
- Step 1: [How components are combined]
- Step 2: [Ensuring consistency]
- ...

**Complete Solution**:
[Full, integrated solution]

**Reasoning Chain Summary**:
[Clear explanation of how we arrived at this solution]

**Confidence Assessment**: [How confident in the solution, 0-100%]
"""
        
        response = await self.llm.ainvoke([
            HumanMessage(content=synthesize_prompt)
        ])
        
        state["messages"].append(response)
        state["current_step"] = POMLNode.SYNTHESIZE.value
        state["reasoning_chain"].append({
            "step": POMLNode.SYNTHESIZE.value,
            "output": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        return state
    
    async def _verify_node(self, state: AgentState) -> AgentState:
        """
        VERIFY phase: Validate the solution
        """
        synthesis = state["reasoning_chain"][-1]["output"]
        original_problem = state["messages"][0].content if state["messages"] else ""
        
        verify_prompt = f"""
## POML Phase: VERIFY (Logic)

### Original Problem:
{original_problem}

### Proposed Solution:
{synthesis}

### Your Task:
1. Check solution against original requirements
2. Validate each step of reasoning
3. Test edge cases and boundary conditions
4. Ensure logical consistency

### Verification Checklist:
- [ ] Addresses all requirements
- [ ] Logically sound
- [ ] Handles edge cases
- [ ] Efficient approach
- [ ] Clear and complete

### Output Format:
**Verification Results**:
- Requirement Coverage: [Pass/Fail with details]
- Logical Consistency: [Pass/Fail with details]
- Edge Case Handling: [Pass/Fail with details]
- Overall Assessment: [Pass/Fail]

**Issues Found**: [List any problems]

**Confidence Score**: [0-100%]
"""
        
        response = await self.llm.ainvoke([
            HumanMessage(content=verify_prompt)
        ])
        
        # Parse verification results
        verification_passed = "Pass" in response.content and "Overall Assessment: Pass" in response.content
        
        state["messages"].append(response)
        state["current_step"] = POMLNode.VERIFY.value
        state["reasoning_chain"].append({
            "step": POMLNode.VERIFY.value,
            "output": response.content,
            "timestamp": datetime.now().isoformat()
        })
        state["verification_results"] = {
            "passed": verification_passed,
            "details": response.content
        }
        
        return state
    
    async def _reflect_node(self, state: AgentState) -> AgentState:
        """
        REFLECT phase: Consider improvements and alternatives
        """
        full_chain = state["reasoning_chain"]
        verification = state["verification_results"]
        
        reflect_prompt = f"""
## POML Phase: REFLECT

### Reasoning Chain Summary:
{json.dumps([{
    "step": item["step"],
    "summary": item["output"][:200] + "..."
} for item in full_chain], indent=2)}

### Verification Results:
{verification["details"]}

### Your Task:
1. Consider alternative approaches
2. Identify potential improvements
3. Learn from the problem-solving process
4. Decide if another iteration is needed

### Output Format:
**Alternative Approaches**: [Other possible solutions]

**Potential Improvements**: [How to enhance the solution]

**Lessons Learned**: [Key insights]

**Iteration Decision**: [CONTINUE for another round, or FINALIZE if satisfied]

**If CONTINUE, specify**:
- What to change: [Specific improvements]
- Which phase to revisit: [ANALYZE/DECOMPOSE]
"""
        
        response = await self.llm.ainvoke([
            HumanMessage(content=reflect_prompt)
        ])
        
        state["messages"].append(response)
        state["current_step"] = POMLNode.REFLECT.value
        state["reasoning_chain"].append({
            "step": POMLNode.REFLECT.value,
            "output": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check iteration count
        iteration_count = state["metadata"].get("iteration_count", 0)
        state["metadata"]["iteration_count"] = iteration_count + 1
        
        return state
    
    async def _output_node(self, state: AgentState) -> AgentState:
        """
        OUTPUT phase: Generate final response
        """
        # Find the synthesis step
        synthesis = None
        for item in reversed(state["reasoning_chain"]):
            if item["step"] == POMLNode.SYNTHESIZE.value:
                synthesis = item["output"]
                break
        
        output_prompt = f"""
## Final Output Generation

### Complete Solution:
{synthesis}

### Task:
Provide a clear, well-formatted final answer to the original problem.
Focus on the solution itself, not the reasoning process (which has already been documented).

Make the output:
- Clear and concise
- Directly actionable
- Professional and polished
- Free of internal reasoning markers
"""
        
        response = await self.llm.ainvoke([
            HumanMessage(content=output_prompt)
        ])
        
        state["final_output"] = response.content
        state["messages"].append(response)
        state["current_step"] = POMLNode.OUTPUT.value
        
        return state
    
    def _should_reflect_or_output(self, state: AgentState) -> str:
        """
        Decide whether to reflect or output based on verification
        """
        verification = state.get("verification_results", {})
        
        if verification.get("passed", False):
            # Verification passed, go to output
            return POMLNode.OUTPUT.value
        else:
            # Verification failed, reflect on improvements
            return POMLNode.REFLECT.value
    
    def _should_continue_or_output(self, state: AgentState) -> str:
        """
        Decide whether to continue iterating or output final result
        """
        # Check iteration count
        iteration_count = state["metadata"].get("iteration_count", 0)
        if iteration_count >= self.max_iterations:
            return POMLNode.OUTPUT.value
        
        # Check if reflection suggests continuing
        last_reflection = state["reasoning_chain"][-1]["output"]
        if "CONTINUE" in last_reflection:
            return POMLNode.ANALYZE.value
        else:
            return POMLNode.OUTPUT.value
    
    async def process(
        self,
        input_text: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an input through the POML reasoning pipeline
        
        Args:
            input_text: The problem or task to solve
            config: Optional configuration for the graph execution
        
        Returns:
            Dictionary containing the final output and reasoning chain
        """
        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=input_text)],
            "current_step": "",
            "reasoning_chain": [],
            "verification_results": None,
            "final_output": None,
            "metadata": {"iteration_count": 0}
        }
        
        # Run the graph
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        final_state = await self.graph.ainvoke(initial_state, config)
        
        return {
            "final_output": final_state.get("final_output", ""),
            "reasoning_chain": final_state.get("reasoning_chain", []),
            "verification_results": final_state.get("verification_results", {}),
            "messages": [msg.content for msg in final_state.get("messages", [])]
        }
    
    def process_sync(
        self,
        input_text: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for process method
        """
        return asyncio.run(self.process(input_text, config))


class SimplePOMLAgent:
    """
    A simplified version of the POML agent for quick tasks
    """
    
    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        model_name: str = "anthropic/claude-3.5-sonnet",
        temperature: float = 0.7
    ):
        """
        Initialize the simplified POML agent
        """
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            model=model_name,
            temperature=temperature,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000"),
                "X-Title": os.getenv("OPENROUTER_SITE_NAME", "POML-Agent"),
            }
        )
        
        self.prompt_builder = POMLPromptBuilder()
    
    async def process(self, input_text: str) -> str:
        """
        Process input with POML-enhanced single-shot reasoning
        """
        system_prompt = self.prompt_builder.build_system_prompt()
        task_prompt = self.prompt_builder.build_task_prompt(input_text)
        
        response = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt)
        ])
        
        return response.content
    
    def process_sync(self, input_text: str) -> str:
        """
        Synchronous wrapper for process method
        """
        return asyncio.run(self.process(input_text))