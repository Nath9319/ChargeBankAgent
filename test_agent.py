"""
Test script for ChargeBankAgent
Validates POML templates, LangGraph workflows, and OpenRouter integration
"""

import asyncio
import pytest
from unittest.mock import Mock, patch
import os

from charge_bank_agent import ChargeBankAgentInterface, ChargeBankAgentFactory
from poml_templates import POMLTemplateManager, POMLContext, PromptRole
from openrouter_client import OpenRouterClient
from config import ChargeBankConfig


class TestPOMLTemplates:
    """Test POML template functionality"""
    
    def test_template_manager_initialization(self):
        """Test that template manager initializes correctly"""
        manager = POMLTemplateManager()
        templates = manager.list_templates()
        
        assert "analyst" in templates
        assert "planner" in templates
        assert "troubleshooter" in templates
    
    def test_analyst_template_formatting(self):
        """Test analyst template formatting with POML structure"""
        manager = POMLTemplateManager()
        context = POMLContext(
            user_query="Find charging stations in Seattle",
            session_id="test_session"
        )
        
        formatted_prompt = manager.format_prompt(
            "analyst",
            context,
            location="Seattle, WA",
            vehicle_type="Tesla Model 3"
        )
        
        # Check POML structure
        assert "<poml>" in formatted_prompt
        assert "<role>" in formatted_prompt
        assert "<task>" in formatted_prompt
        assert "<context>" in formatted_prompt
        assert "<output-format>" in formatted_prompt
        assert "Seattle, WA" in formatted_prompt
        assert "Tesla Model 3" in formatted_prompt
    
    def test_planner_template_formatting(self):
        """Test planner template formatting"""
        manager = POMLTemplateManager()
        context = POMLContext(user_query="Plan route from A to B")
        
        formatted_prompt = manager.format_prompt(
            "planner",
            context,
            start_location="Portland, OR",
            destination="Seattle, WA"
        )
        
        assert "<journey-parameters>" in formatted_prompt
        assert "<planning-methodology>" in formatted_prompt
        assert "Portland, OR" in formatted_prompt
        assert "Seattle, WA" in formatted_prompt
    
    def test_troubleshooter_template_formatting(self):
        """Test troubleshooter template formatting"""
        manager = POMLTemplateManager()
        context = POMLContext(user_query="Charging station not working")
        
        formatted_prompt = manager.format_prompt(
            "troubleshooter",
            context,
            issue_type="hardware_failure",
            vehicle_model="BMW i4"
        )
        
        assert "<problem-context>" in formatted_prompt
        assert "<diagnostic-framework>" in formatted_prompt
        assert "hardware_failure" in formatted_prompt
        assert "BMW i4" in formatted_prompt


class TestOpenRouterClient:
    """Test OpenRouter client functionality"""
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"})
    def test_client_initialization(self):
        """Test OpenRouter client initialization"""
        client = OpenRouterClient()
        
        assert client.api_key == "test_key"
        assert client.llm is not None
        assert "anthropic/claude-3.5-sonnet" in client.llm.model_name
    
    def test_available_models(self):
        """Test that available models list is populated"""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}):
            client = OpenRouterClient()
            models = client.get_available_models()
            
            assert len(models) > 0
            assert "anthropic/claude-3.5-sonnet" in models
            assert "openai/gpt-4-turbo" in models
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"})
    def test_message_creation(self):
        """Test POML prompt to LangChain message conversion"""
        client = OpenRouterClient()
        
        poml_prompt = "<poml><role>Test role</role><task>Test task</task></poml>"
        user_message = "Test user message"
        
        messages = client.create_messages_from_poml(poml_prompt, user_message)
        
        assert len(messages) == 2
        assert messages[0].content == poml_prompt
        assert messages[1].content == user_message


class TestChargeBankAgent:
    """Test main agent functionality"""
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"})
    @patch('openrouter_client.ChatOpenAI')
    def test_agent_initialization(self, mock_chat):
        """Test agent initialization"""
        mock_chat.return_value.generate.return_value.generations = [[Mock(text="analysis")]]
        
        agent = ChargeBankAgentInterface()
        
        assert agent.agent is not None
        assert len(agent.sessions) == 0
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"})
    def test_session_creation(self):
        """Test session management"""
        agent = ChargeBankAgentInterface()
        
        session = agent.create_session(
            "test_session",
            {"preferred_network": "Tesla Supercharger"}
        )
        
        assert session.session_id == "test_session"
        assert session.user_preferences["preferred_network"] == "Tesla Supercharger"
        assert "test_session" in agent.sessions
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"})
    def test_session_info_retrieval(self):
        """Test session information retrieval"""
        agent = ChargeBankAgentInterface()
        
        # Test non-existent session
        info = agent.get_session_info("non_existent")
        assert info is None
        
        # Test existing session
        agent.create_session("test_session")
        info = agent.get_session_info("test_session")
        assert info is not None
        assert info["session_id"] == "test_session"


class TestAgentFactory:
    """Test agent factory functionality"""
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"})
    @patch('openrouter_client.ChatOpenAI')
    def test_create_agent(self, mock_chat):
        """Test basic agent creation"""
        mock_chat.return_value.generate.return_value.generations = [[Mock(text="test")]]
        
        agent = ChargeBankAgentFactory.create_agent()
        assert agent is not None
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"})
    @patch('openrouter_client.ChatOpenAI')
    def test_create_specialized_agent(self, mock_chat):
        """Test specialized agent creation"""
        mock_chat.return_value.generate.return_value.generations = [[Mock(text="test")]]
        
        cost_agent = ChargeBankAgentFactory.create_specialized_agent("cost_optimizer")
        route_agent = ChargeBankAgentFactory.create_specialized_agent("route_planner")
        tech_agent = ChargeBankAgentFactory.create_specialized_agent("technical_support")
        
        assert cost_agent is not None
        assert route_agent is not None
        assert tech_agent is not None


class TestConfiguration:
    """Test configuration and environment validation"""
    
    def test_model_configuration(self):
        """Test model configuration retrieval"""
        config = ChargeBankConfig.get_model_config("claude-3.5-sonnet")
        
        assert config.name == "anthropic/claude-3.5-sonnet"
        assert config.tier.value == "premium"
        assert config.temperature == 0.7
    
    def test_model_by_tier(self):
        """Test model selection by tier"""
        from config import ModelTier
        
        fast_model = ChargeBankConfig.get_model_by_tier(ModelTier.FAST)
        premium_model = ChargeBankConfig.get_model_by_tier(ModelTier.PREMIUM)
        
        assert fast_model.tier == ModelTier.FAST
        assert premium_model.tier == ModelTier.PREMIUM
    
    def test_environment_validation(self):
        """Test environment variable validation"""
        validation = ChargeBankConfig.validate_environment()
        
        assert "OPENROUTER_API_KEY" in validation
        assert isinstance(validation["OPENROUTER_API_KEY"], bool)


# Integration tests (require actual API key)
class TestIntegration:
    """Integration tests - require valid API key"""
    
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY environment variable"
    )
    async def test_real_query_processing(self):
        """Test actual query processing with real API"""
        agent = ChargeBankAgentInterface()
        
        response = await agent.process_query(
            "Find charging stations in a test location",
            location="San Francisco, CA",
            vehicle_type="Tesla Model 3"
        )
        
        assert len(response) > 0
        assert "charging" in response.lower()
    
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY environment variable"
    )
    async def test_real_route_planning(self):
        """Test actual route planning with real API"""
        from charge_bank_agent import plan_charging_route
        
        response = await plan_charging_route(
            start="Test City A",
            destination="Test City B",
            vehicle_type="Tesla Model Y"
        )
        
        assert len(response) > 0
        assert any(word in response.lower() for word in ["route", "charging", "stop"])


async def run_manual_tests():
    """Run manual tests for development"""
    
    print("🧪 Running ChargeBankAgent Tests")
    print("=" * 40)
    
    # Test environment
    print("1. Environment Validation:")
    validation = ChargeBankConfig.validate_environment()
    for var, is_set in validation.items():
        status = "✅" if is_set else "❌"
        print(f"   {status} {var}")
    
    if not validation.get("OPENROUTER_API_KEY"):
        print("\n⚠️  OpenRouter API key not set. Skipping live tests.")
        return
    
    # Test basic functionality
    print("\n2. Basic Functionality Test:")
    try:
        agent = ChargeBankAgentInterface()
        response = await agent.process_query(
            "Test query: Find charging stations",
            location="Test Location"
        )
        print(f"   ✅ Basic query processing works")
        print(f"   Response length: {len(response)} characters")
    except Exception as e:
        print(f"   ❌ Basic query failed: {e}")
    
    # Test POML template formatting
    print("\n3. POML Template Test:")
    try:
        manager = POMLTemplateManager()
        context = POMLContext(user_query="Test query")
        formatted = manager.format_prompt("analyst", context, location="Test")
        
        assert "<poml>" in formatted
        assert "<role>" in formatted
        print("   ✅ POML template formatting works")
    except Exception as e:
        print(f"   ❌ POML template test failed: {e}")
    
    print("\n✅ Manual tests completed!")


if __name__ == "__main__":
    # Run manual tests
    asyncio.run(run_manual_tests())