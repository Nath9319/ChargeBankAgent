"""
LangGraph Agent Implementation with POML Integration
Combines Microsoft POML techniques with LangGraph for structured agent workflows
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
import operator

from poml_templates import (
    POMLTemplateManager, 
    POMLContext, 
    ChargeBankAnalystPrompt,
    ChargeBankPlannerPrompt,
    ChargeBankTroubleshooterPrompt
)
from openrouter_client import OpenRouterClient


class AgentState(TypedDict):
    """State for the ChargeBankAgent LangGraph workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    context: Dict[str, Any]
    current_step: str
    analysis_result: Optional[str]
    plan_result: Optional[str]
    final_response: Optional[str]
    error_state: Optional[str]


class ChargeBankAgent:
    """
    LangGraph-powered agent using Microsoft POML techniques for charge bank assistance
    """
    
    def __init__(
        self,
        openrouter_client: Optional[OpenRouterClient] = None,
        model_name: str = "anthropic/claude-3.5-sonnet"
    ):
        self.client = openrouter_client or OpenRouterClient(model_name=model_name)
        self.template_manager = POMLTemplateManager()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for the charge bank agent"""
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for different processing steps
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("analyze_charging_needs", self._analyze_charging_needs)
        workflow.add_node("plan_charging_route", self._plan_charging_route)
        workflow.add_node("troubleshoot_issues", self._troubleshoot_issues)
        workflow.add_node("synthesize_response", self._synthesize_response)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("classify_query")
        
        # Define conditional edges based on query classification
        workflow.add_conditional_edges(
            "classify_query",
            self._route_query,
            {
                "analysis": "analyze_charging_needs",
                "planning": "plan_charging_route", 
                "troubleshooting": "troubleshoot_issues",
                "error": "handle_error"
            }
        )
        
        # Connect processing nodes to synthesis
        workflow.add_edge("analyze_charging_needs", "synthesize_response")
        workflow.add_edge("plan_charging_route", "synthesize_response")
        workflow.add_edge("troubleshoot_issues", "synthesize_response")
        
        # Connect synthesis and error handling to end
        workflow.add_edge("synthesize_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def _classify_query(self, state: AgentState) -> AgentState:
        """Classify the user query to determine the appropriate workflow path"""
        
        classification_prompt = """<poml>
<role>
You are a Query Classification Specialist for an Electric Vehicle Charging Assistant.
Your job is to categorize user queries into the appropriate workflow.
</role>

<task>
Classify the following user query into one of these categories:
- "analysis": Questions about charging stations, costs, availability, comparisons
- "planning": Route planning, trip charging needs, scheduling
- "troubleshooting": Problems with charging, technical issues, error resolution
</task>

<classification-criteria>
**Analysis**: "Where are the best charging stations?", "How much does charging cost?", "Compare charging networks"
**Planning**: "Plan my route to...", "When should I charge?", "Optimize my charging schedule"
**Troubleshooting**: "Charging station not working", "Payment failed", "Connector doesn't fit"
</classification-criteria>

<output-format>
Respond with ONLY one word: "analysis", "planning", or "troubleshooting"
</output-format>
</poml>"""
        
        try:
            messages = [
                SystemMessage(content=classification_prompt),
                HumanMessage(content=state["user_query"])
            ]
            
            classification = self.client.generate_response_sync(messages).strip().lower()
            
            # Validate classification
            valid_classifications = ["analysis", "planning", "troubleshooting"]
            if classification not in valid_classifications:
                classification = "analysis"  # Default fallback
            
            state["current_step"] = classification
            return state
            
        except Exception as e:
            state["error_state"] = f"Classification error: {str(e)}"
            state["current_step"] = "error"
            return state
    
    def _route_query(self, state: AgentState) -> str:
        """Route the query based on classification"""
        if state.get("error_state"):
            return "error"
        return state.get("current_step", "analysis")
    
    async def _analyze_charging_needs(self, state: AgentState) -> AgentState:
        """Analyze charging infrastructure and provide recommendations"""
        
        context = POMLContext(
            user_query=state["user_query"],
            session_id=state["context"].get("session_id"),
            user_preferences=state["context"].get("preferences")
        )
        
        try:
            analyst_prompt = self.template_manager.format_prompt(
                "analyst", 
                context,
                **state["context"]
            )
            
            messages = self.client.create_messages_from_poml(analyst_prompt, state["user_query"])
            response = self.client.generate_response_sync(messages)
            
            state["analysis_result"] = response
            return state
            
        except Exception as e:
            state["error_state"] = f"Analysis error: {str(e)}"
            return state
    
    async def _plan_charging_route(self, state: AgentState) -> AgentState:
        """Plan optimal charging routes and schedules"""
        
        context = POMLContext(
            user_query=state["user_query"],
            session_id=state["context"].get("session_id"),
            user_preferences=state["context"].get("preferences")
        )
        
        try:
            planner_prompt = self.template_manager.format_prompt(
                "planner",
                context,
                **state["context"]
            )
            
            messages = self.client.create_messages_from_poml(planner_prompt, state["user_query"])
            response = self.client.generate_response_sync(messages)
            
            state["plan_result"] = response
            return state
            
        except Exception as e:
            state["error_state"] = f"Planning error: {str(e)}"
            return state
    
    async def _troubleshoot_issues(self, state: AgentState) -> AgentState:
        """Diagnose and resolve charging-related issues"""
        
        context = POMLContext(
            user_query=state["user_query"],
            session_id=state["context"].get("session_id"),
            user_preferences=state["context"].get("preferences")
        )
        
        try:
            troubleshooter_prompt = self.template_manager.format_prompt(
                "troubleshooter",
                context,
                **state["context"]
            )
            
            messages = self.client.create_messages_from_poml(troubleshooter_prompt, state["user_query"])
            response = self.client.generate_response_sync(messages)
            
            state["analysis_result"] = response  # Store in analysis_result for consistency
            return state
            
        except Exception as e:
            state["error_state"] = f"Troubleshooting error: {str(e)}"
            return state
    
    async def _synthesize_response(self, state: AgentState) -> AgentState:
        """Synthesize the final response from processed results"""
        
        # Determine which result to use based on the workflow path
        result = (
            state.get("analysis_result") or 
            state.get("plan_result") or 
            "I apologize, but I couldn't process your request properly."
        )
        
        state["final_response"] = result
        
        # Add the final response to messages
        state["messages"].append(AIMessage(content=result))
        
        return state
    
    async def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        
        error_message = f"""I encountered an issue while processing your request: {state.get('error_state', 'Unknown error')}

Please try rephrasing your question or provide more specific details about your charging needs. 

For immediate assistance, you can:
1. Check if your query includes specific location information
2. Specify your vehicle type and charging requirements
3. Describe any error messages you're seeing

I'm here to help with:
- Finding charging stations and comparing options
- Planning routes with charging stops
- Troubleshooting charging issues"""
        
        state["final_response"] = error_message
        state["messages"].append(AIMessage(content=error_message))
        
        return state
    
    async def process_query(
        self, 
        user_query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a user query through the LangGraph workflow"""
        
        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=user_query)],
            user_query=user_query,
            context=context or {},
            current_step="",
            analysis_result=None,
            plan_result=None,
            final_response=None,
            error_state=None
        )
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        return final_state.get("final_response", "I couldn't process your request.")
    
    def process_query_sync(
        self, 
        user_query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Synchronous version of query processing"""
        
        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=user_query)],
            user_query=user_query,
            context=context or {},
            current_step="",
            analysis_result=None,
            plan_result=None,
            final_response=None,
            error_state=None
        )
        
        # Run the graph synchronously
        final_state = self.graph.invoke(initial_state)
        
        return final_state.get("final_response", "I couldn't process your request.")


class ChargeBankAgentFactory:
    """Factory class for creating configured ChargeBankAgent instances"""
    
    @staticmethod
    def create_agent(
        model_name: str = "anthropic/claude-3.5-sonnet",
        temperature: float = 0.7,
        **kwargs
    ) -> ChargeBankAgent:
        """Create a pre-configured ChargeBankAgent"""
        
        client = OpenRouterClient(
            model_name=model_name,
            temperature=temperature,
            **kwargs
        )
        
        return ChargeBankAgent(openrouter_client=client)
    
    @staticmethod
    def create_specialized_agent(
        specialization: str,
        **kwargs
    ) -> ChargeBankAgent:
        """Create an agent optimized for specific use cases"""
        
        specialization_configs = {
            "cost_optimizer": {
                "model_name": "anthropic/claude-3-haiku",
                "temperature": 0.3
            },
            "route_planner": {
                "model_name": "anthropic/claude-3.5-sonnet", 
                "temperature": 0.5
            },
            "technical_support": {
                "model_name": "openai/gpt-4-turbo",
                "temperature": 0.2
            }
        }
        
        config = specialization_configs.get(specialization, {})
        config.update(kwargs)
        
        return ChargeBankAgentFactory.create_agent(**config)