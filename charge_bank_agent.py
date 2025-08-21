"""
Main ChargeBankAgent Interface
Provides a unified interface for charge bank assistance using POML and LangGraph
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

from langgraph_agent import ChargeBankAgent as LangGraphAgent, ChargeBankAgentFactory
from poml_templates import POMLContext
from openrouter_client import OpenRouterClient


@dataclass
class ChargeBankSession:
    """Represents a user session with the ChargeBankAgent"""
    session_id: str
    user_preferences: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    context: Dict[str, Any]


class ChargeBankAgentInterface:
    """
    Main interface for the ChargeBankAgent system
    Combines POML prompting with LangGraph workflows and OpenRouter LLM access
    """
    
    def __init__(
        self,
        model_name: str = "anthropic/claude-3.5-sonnet",
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize the LangGraph agent
        self.agent = ChargeBankAgentFactory.create_agent(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key
        )
        
        # Session management
        self.sessions: Dict[str, ChargeBankSession] = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def process_query(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        location: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        budget: Optional[str] = None,
        **additional_context
    ) -> str:
        """
        Process a user query with POML-structured prompting
        
        Args:
            user_query: The user's question or request
            session_id: Optional session identifier for context
            location: User's location or area of interest
            vehicle_type: Type of electric vehicle
            budget: Budget constraints
            **additional_context: Additional context parameters
        
        Returns:
            Formatted response from the agent
        """
        
        try:
            # Prepare context for the agent
            context = {
                "location": location,
                "vehicle_type": vehicle_type,
                "budget": budget,
                "session_id": session_id,
                **additional_context
            }
            
            # Update session if provided
            if session_id:
                self._update_session(session_id, user_query, context)
                context["conversation_history"] = self.sessions[session_id].conversation_history
                context["preferences"] = self.sessions[session_id].user_preferences
            
            # Process through LangGraph workflow
            response = await self.agent.process_query(user_query, context)
            
            # Update session with response
            if session_id and session_id in self.sessions:
                self.sessions[session_id].conversation_history.append({
                    "user": user_query,
                    "assistant": response,
                    "timestamp": str(asyncio.get_event_loop().time())
                })
            
            self.logger.info(f"Successfully processed query for session {session_id}")
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.logger.error(error_msg)
            return self._generate_error_response(error_msg)
    
    def process_query_sync(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        **context
    ) -> str:
        """Synchronous version of query processing"""
        
        try:
            # Prepare context
            full_context = context.copy()
            full_context["session_id"] = session_id
            
            if session_id:
                self._update_session(session_id, user_query, full_context)
                full_context["conversation_history"] = self.sessions[session_id].conversation_history
                full_context["preferences"] = self.sessions[session_id].user_preferences
            
            # Process through LangGraph workflow
            response = self.agent.process_query_sync(user_query, full_context)
            
            # Update session with response
            if session_id and session_id in self.sessions:
                self.sessions[session_id].conversation_history.append({
                    "user": user_query,
                    "assistant": response
                })
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.logger.error(error_msg)
            return self._generate_error_response(error_msg)
    
    def create_session(
        self,
        session_id: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> ChargeBankSession:
        """Create a new user session"""
        
        session = ChargeBankSession(
            session_id=session_id,
            user_preferences=user_preferences or {},
            conversation_history=[],
            context={}
        )
        
        self.sessions[session_id] = session
        self.logger.info(f"Created new session: {session_id}")
        
        return session
    
    def _update_session(
        self,
        session_id: str,
        user_query: str,
        context: Dict[str, Any]
    ):
        """Update or create session with new interaction"""
        
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        # Update context
        self.sessions[session_id].context.update(context)
    
    def _generate_error_response(self, error_msg: str) -> str:
        """Generate a user-friendly error response"""
        
        return f"""I apologize, but I encountered an issue while processing your request. 

**Error Details**: {error_msg}

**What you can try**:
1. Rephrase your question with more specific details
2. Include location information if asking about charging stations
3. Specify your vehicle type for compatibility checks
4. Check your internet connection and try again

**Example queries that work well**:
- "Find Tesla Supercharger stations near downtown Seattle"
- "Plan charging stops for a road trip from Los Angeles to San Francisco"
- "My charging session failed at EVgo station, what should I do?"

Please try again with a more specific question, and I'll be happy to help!"""
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session"""
        
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "preferences": session.user_preferences,
            "conversation_count": len(session.conversation_history),
            "last_interaction": session.conversation_history[-1] if session.conversation_history else None
        }
    
    def update_user_preferences(
        self,
        session_id: str,
        preferences: Dict[str, Any]
    ):
        """Update user preferences for a session"""
        
        if session_id not in self.sessions:
            self.create_session(session_id, preferences)
        else:
            self.sessions[session_id].user_preferences.update(preferences)
        
        self.logger.info(f"Updated preferences for session {session_id}")


# Convenience functions for common use cases
async def quick_analysis(query: str, location: str = None, **kwargs) -> str:
    """Quick charging station analysis without session management"""
    agent = ChargeBankAgentInterface()
    return await agent.process_query(query, location=location, **kwargs)


async def plan_charging_route(
    start: str,
    destination: str,
    vehicle_type: str = "Tesla Model 3",
    **kwargs
) -> str:
    """Plan a charging route between two locations"""
    agent = ChargeBankAgentInterface()
    query = f"Plan charging stops from {start} to {destination} for {vehicle_type}"
    return await agent.process_query(
        query,
        start_location=start,
        destination=destination,
        vehicle_type=vehicle_type,
        **kwargs
    )


async def troubleshoot_charging(
    issue_description: str,
    station_name: str = None,
    vehicle_type: str = None,
    **kwargs
) -> str:
    """Get help with charging issues"""
    agent = ChargeBankAgentInterface()
    context = {
        "issue_type": "charging_problem",
        "station_name": station_name,
        "vehicle_type": vehicle_type,
        **kwargs
    }
    return await agent.process_query(issue_description, **context)