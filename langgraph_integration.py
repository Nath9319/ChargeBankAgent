"""
LangGraph Integration for POML Prompts with OpenRouter
This module provides seamless integration between POML prompts, LangGraph, and OpenRouter
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv

from poml_prompts import POMLPromptEngine, POMLPrompt, CommonPOMLPrompts

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """State for the LangGraph agent"""
    messages: List[Any]
    current_prompt: Optional[str]
    prompt_variables: Dict[str, Any]
    prompt_output: Optional[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]


class POMLLangGraphAgent:
    """LangGraph agent that uses POML prompts with OpenRouter"""
    
    def __init__(self, openrouter_api_key: Optional[str] = None, model: str = "openai/gpt-4"):
        self.prompt_engine = POMLPromptEngine()
        self.model = model
        
        # Initialize OpenRouter client
        api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            headers={
                "HTTP-Referer": "https://github.com/your-repo/chargebank-agent",
                "X-Title": "ChargeBankAgent"
            }
        )
        
        # Register common prompts
        self._register_common_prompts()
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _register_common_prompts(self):
        """Register common POML prompts"""
        common_prompts = [
            CommonPOMLPrompts.create_analysis_prompt(),
            CommonPOMLPrompts.create_code_review_prompt(),
            CommonPOMLPrompts.create_planning_prompt()
        ]
        
        for prompt in common_prompts:
            self.prompt_engine.register_prompt(prompt)
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_prompt", self._process_prompt_node)
        workflow.add_node("execute_llm", self._execute_llm_node)
        workflow.add_node("validate_output", self._validate_output_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Add edges
        workflow.add_edge("process_prompt", "execute_llm")
        workflow.add_edge("execute_llm", "validate_output")
        workflow.add_edge("validate_output", END)
        workflow.add_edge("handle_error", END)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "process_prompt",
            self._should_continue,
            {
                "continue": "execute_llm",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_llm",
            self._should_continue,
            {
                "continue": "validate_output",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_output",
            self._should_continue,
            {
                "continue": END,
                "error": "handle_error"
            }
        )
        
        return workflow.compile()
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if the workflow should continue or handle errors"""
        if state.get("error"):
            return "error"
        return "continue"
    
    def _process_prompt_node(self, state: AgentState) -> AgentState:
        """Process the POML prompt and prepare for LLM execution"""
        try:
            current_prompt = state.get("current_prompt")
            if not current_prompt:
                raise ValueError("No prompt specified")
            
            # Render the prompt
            rendered_prompt = self.prompt_engine.render_prompt(
                current_prompt, 
                state.get("prompt_variables", {})
            )
            
            # Add system and user messages to state
            messages = [
                SystemMessage(content=rendered_prompt["system_message"]),
                HumanMessage(content=rendered_prompt["user_message"])
            ]
            
            state["messages"] = messages
            state["metadata"]["rendered_prompt"] = rendered_prompt
            
            return state
            
        except Exception as e:
            state["error"] = f"Failed to process prompt: {str(e)}"
            return state
    
    def _execute_llm_node(self, state: AgentState) -> AgentState:
        """Execute the LLM with the rendered prompt"""
        try:
            messages = state["messages"]
            
            # Execute with function calling if available
            prompt = self.prompt_engine.prompts.get(state["current_prompt"])
            if prompt and prompt.functions:
                # Use function calling
                response = self.llm.invoke(
                    messages,
                    tools=self._convert_functions_to_tools(prompt.functions)
                )
            else:
                # Regular completion
                response = self.llm.invoke(messages)
            
            # Add AI response to messages
            state["messages"].append(response)
            state["prompt_output"] = self._extract_output(response)
            
            return state
            
        except Exception as e:
            state["error"] = f"Failed to execute LLM: {str(e)}"
            return state
    
    def _validate_output_node(self, state: AgentState) -> AgentState:
        """Validate the LLM output against expected format"""
        try:
            prompt = self.prompt_engine.prompts.get(state["current_prompt"])
            if not prompt:
                return state
            
            output = state.get("prompt_output", {})
            expected_format = prompt.output_format
            
            # Basic validation - check if output has expected structure
            if expected_format and isinstance(expected_format, dict):
                if "properties" in expected_format:
                    for key in expected_format["properties"]:
                        if key not in output:
                            state["error"] = f"Missing required output field: {key}"
                            return state
            
            return state
            
        except Exception as e:
            state["error"] = f"Failed to validate output: {str(e)}"
            return state
    
    def _handle_error_node(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        error = state.get("error", "Unknown error")
        print(f"Error in workflow: {error}")
        return state
    
    def _convert_functions_to_tools(self, functions) -> List:
        """Convert POML functions to LangChain tools"""
        tools = []
        
        for func in functions:
            @tool
            def dynamic_tool(**kwargs):
                """Dynamic tool based on POML function"""
                # This is a simplified implementation
                # In practice, you'd want to implement actual function logic
                return f"Executed {func.name} with parameters: {kwargs}"
            
            # Set tool attributes
            dynamic_tool.name = func.name
            dynamic_tool.description = func.description
            
            tools.append(dynamic_tool)
        
        return tools
    
    def _extract_output(self, response) -> Dict[str, Any]:
        """Extract output from LLM response"""
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                try:
                    # Try to parse as JSON
                    return json.loads(content)
                except:
                    return {"text": content}
            elif isinstance(content, dict):
                return content
        return {"text": str(response)}
    
    def execute_prompt(self, prompt_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a POML prompt with given variables"""
        initial_state = AgentState(
            messages=[],
            current_prompt=prompt_name,
            prompt_variables=variables,
            prompt_output=None,
            error=None,
            metadata={}
        )
        
        # Execute the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "success": result.get("error") is None,
            "output": result.get("prompt_output"),
            "messages": result.get("messages"),
            "error": result.get("error"),
            "metadata": result.get("metadata", {})
        }
    
    def add_custom_prompt(self, prompt: POMLPrompt) -> None:
        """Add a custom POML prompt to the agent"""
        self.prompt_engine.register_prompt(prompt)
    
    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt names"""
        return list(self.prompt_engine.prompts.keys())


# Example usage and demonstration
def create_chargebank_agent() -> POMLLangGraphAgent:
    """Create a ChargeBank agent with custom prompts"""
    agent = POMLLangGraphAgent()
    
    # Add custom ChargeBank-specific prompts
    from poml_prompts import PromptVariable, PromptFunction
    
    chargebank_prompt = POMLPrompt(
        name="chargebank_analysis",
        description="Analyze ChargeBank transaction data and provide insights",
        system_message="You are a financial analyst specializing in ChargeBank transactions. Analyze the data and provide actionable financial insights.",
        user_message_template="""
        Analyze the following ChargeBank transaction data:
        
        Transactions: {{transactions}}
        Time Period: {{time_period}}
        Account Type: {{account_type}}
        
        Provide analysis covering:
        - Spending patterns
        - Category analysis
        - Budget recommendations
        - Fraud detection insights
        - Financial health score
        """,
        variables=[
            PromptVariable("transactions", "Transaction data in JSON format", "string", True),
            PromptVariable("time_period", "Analysis time period", "string", True),
            PromptVariable("account_type", "Type of account", "string", False, "personal")
        ],
        output_format={
            "type": "object",
            "properties": {
                "spending_patterns": {"type": "array", "items": {"type": "string"}},
                "category_analysis": {"type": "object"},
                "budget_recommendations": {"type": "array", "items": {"type": "string"}},
                "fraud_insights": {"type": "array", "items": {"type": "string"}},
                "financial_health_score": {"type": "number"},
                "risk_level": {"type": "string"}
            }
        }
    )
    
    agent.add_custom_prompt(chargebank_prompt)
    return agent


if __name__ == "__main__":
    # Example usage
    try:
        agent = create_chargebank_agent()
        print(f"Available prompts: {agent.get_available_prompts()}")
        
        # Example execution
        result = agent.execute_prompt("data_analysis", {
            "data": "Sample data for analysis",
            "context": "Financial context"
        })
        
        print(f"Execution result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set OPENROUTER_API_KEY environment variable")