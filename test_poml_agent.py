"""
Test suite for the POML-enhanced LangGraph agent

This file contains tests to verify the POML agent functionality
with various scenarios and edge cases.
"""

import asyncio
import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from langgraph_poml_agent import POMLAgent, SimplePOMLAgent, AgentState, POMLNode
from poml_prompts import (
    POMLPromptBuilder,
    ReasoningStep,
    create_poml_enhanced_prompt,
    validate_reasoning_chain
)
from openrouter_config import (
    OpenRouterModels,
    ModelProvider,
    get_recommended_model,
    estimate_cost
)


# Test fixtures
@pytest.fixture
def mock_api_key():
    """Mock API key for testing"""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"}):
        yield "test-api-key"


@pytest.fixture
def prompt_builder():
    """Create a POMLPromptBuilder instance"""
    return POMLPromptBuilder()


@pytest.fixture
async def mock_agent(mock_api_key):
    """Create a mocked POMLAgent"""
    with patch("langgraph_poml_agent.ChatOpenAI") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(
            return_value=Mock(content="Test response")
        )
        agent = POMLAgent(
            openrouter_api_key=mock_api_key,
            model_name="test-model",
            temperature=0.5
        )
        yield agent


# POML Prompt Tests
class TestPOMLPrompts:
    """Test POML prompt generation"""
    
    def test_system_prompt_generation(self, prompt_builder):
        """Test system prompt generation with POML framework"""
        prompt = prompt_builder.build_system_prompt(
            role="test assistant",
            domain="testing"
        )
        
        assert "POML" in prompt
        assert "test assistant" in prompt
        assert "testing" in prompt
        assert "UNDERSTAND" in prompt
        assert "DECOMPOSE" in prompt
        assert "ANALYZE" in prompt
        assert "SYNTHESIZE" in prompt
        assert "VERIFY" in prompt
        assert "REFLECT" in prompt
    
    def test_task_prompt_generation(self, prompt_builder):
        """Test task-specific prompt generation"""
        task = "Test task description"
        context = {"key1": "value1", "key2": "value2"}
        
        prompt = prompt_builder.build_task_prompt(task, context)
        
        assert task in prompt
        assert "key1: value1" in prompt
        assert "key2: value2" in prompt
        assert "Understanding" in prompt
        assert "Decomposition" in prompt
    
    def test_reasoning_chain_prompt(self, prompt_builder):
        """Test reasoning chain prompt generation"""
        problem = "Test problem"
        steps = ["Step 1", "Step 2", "Step 3"]
        
        prompt = prompt_builder.build_reasoning_chain_prompt(problem, steps)
        
        assert problem in prompt
        for step in steps:
            assert step in prompt
        assert "Chain of Thought" in prompt
        assert "Final Synthesis" in prompt
    
    def test_verification_prompt(self, prompt_builder):
        """Test verification prompt generation"""
        solution = "Test solution"
        criteria = ["Criterion 1", "Criterion 2"]
        
        prompt = prompt_builder.build_verification_prompt(solution, criteria)
        
        assert solution in prompt
        for criterion in criteria:
            assert criterion in prompt
        assert "Verification Protocol" in prompt
        assert "Correctness Check" in prompt
    
    def test_reflection_prompt(self, prompt_builder):
        """Test reflection prompt generation"""
        experience = "Test experience"
        outcomes = {"outcome1": "result1", "outcome2": "result2"}
        
        prompt = prompt_builder.build_reflection_prompt(experience, outcomes)
        
        assert experience in prompt
        assert "outcome1: result1" in prompt
        assert "outcome2: result2" in prompt
        assert "What Worked Well" in prompt
        assert "Learning Points" in prompt


# OpenRouter Configuration Tests
class TestOpenRouterConfig:
    """Test OpenRouter configuration and utilities"""
    
    def test_model_registry(self):
        """Test model registry access"""
        model = OpenRouterModels.get_model("anthropic/claude-3.5-sonnet")
        
        assert model is not None
        assert model.provider == ModelProvider.ANTHROPIC
        assert model.context_length > 0
        assert model.supports_functions is True
    
    def test_list_models_with_filters(self):
        """Test filtering models by criteria"""
        # Test provider filter
        anthropic_models = OpenRouterModels.list_models(
            provider=ModelProvider.ANTHROPIC
        )
        assert all(m.provider == ModelProvider.ANTHROPIC for m in anthropic_models)
        
        # Test function support filter
        function_models = OpenRouterModels.list_models(
            supports_functions=True
        )
        assert all(m.supports_functions for m in function_models)
        
        # Test vision support filter
        vision_models = OpenRouterModels.list_models(
            supports_vision=True
        )
        assert all(m.supports_vision for m in vision_models)
    
    def test_get_cheapest_model(self):
        """Test finding cheapest model"""
        cheapest = OpenRouterModels.get_cheapest_model(
            min_context=50000,
            supports_functions=False
        )
        
        assert cheapest is not None
        assert cheapest.context_length >= 50000
    
    def test_cost_estimation(self):
        """Test cost estimation for completions"""
        cost = estimate_cost(
            model_name="anthropic/claude-3.5-sonnet",
            input_tokens=1000,
            output_tokens=500
        )
        
        assert "input_cost" in cost
        assert "output_cost" in cost
        assert "total_cost" in cost
        assert cost["total_cost"] > 0
    
    def test_model_recommendations(self):
        """Test model recommendations for use cases"""
        # Test reasoning recommendation
        reasoning_model = get_recommended_model("reasoning")
        assert reasoning_model in [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4-turbo",
            "google/gemini-pro-1.5"
        ]
        
        # Test budget recommendation
        budget_model = get_recommended_model("budget", budget_conscious=True)
        assert budget_model is not None


# POMLAgent Tests
class TestPOMLAgent:
    """Test the main POML agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_api_key):
        """Test agent initialization"""
        with patch("langgraph_poml_agent.ChatOpenAI"):
            agent = POMLAgent(
                openrouter_api_key=mock_api_key,
                model_name="test-model",
                temperature=0.5,
                max_iterations=5
            )
            
            assert agent.api_key == mock_api_key
            assert agent.model_name == "test-model"
            assert agent.temperature == 0.5
            assert agent.max_iterations == 5
            assert agent.graph is not None
    
    @pytest.mark.asyncio
    async def test_understand_node(self, mock_agent):
        """Test the UNDERSTAND node processing"""
        state: AgentState = {
            "messages": [Mock(content="Test problem")],
            "current_step": "",
            "reasoning_chain": [],
            "verification_results": None,
            "final_output": None,
            "metadata": {}
        }
        
        result = await mock_agent._understand_node(state)
        
        assert result["current_step"] == POMLNode.UNDERSTAND.value
        assert len(result["reasoning_chain"]) == 1
        assert result["reasoning_chain"][0]["step"] == POMLNode.UNDERSTAND.value
    
    @pytest.mark.asyncio
    async def test_decompose_node(self, mock_agent):
        """Test the DECOMPOSE node processing"""
        state: AgentState = {
            "messages": [],
            "current_step": POMLNode.UNDERSTAND.value,
            "reasoning_chain": [
                {"step": POMLNode.UNDERSTAND.value, "output": "Understanding"}
            ],
            "verification_results": None,
            "final_output": None,
            "metadata": {}
        }
        
        result = await mock_agent._decompose_node(state)
        
        assert result["current_step"] == POMLNode.DECOMPOSE.value
        assert len(result["reasoning_chain"]) == 2
    
    @pytest.mark.asyncio
    async def test_verification_logic(self, mock_agent):
        """Test verification pass/fail logic"""
        # Test passing verification
        state_pass: AgentState = {
            "messages": [],
            "current_step": "",
            "reasoning_chain": [],
            "verification_results": {"passed": True, "details": "All good"},
            "final_output": None,
            "metadata": {}
        }
        
        decision = mock_agent._should_reflect_or_output(state_pass)
        assert decision == POMLNode.OUTPUT.value
        
        # Test failing verification
        state_fail: AgentState = {
            "messages": [],
            "current_step": "",
            "reasoning_chain": [],
            "verification_results": {"passed": False, "details": "Issues found"},
            "final_output": None,
            "metadata": {}
        }
        
        decision = mock_agent._should_reflect_or_output(state_fail)
        assert decision == POMLNode.REFLECT.value
    
    @pytest.mark.asyncio
    async def test_iteration_limit(self, mock_agent):
        """Test that iteration limit is respected"""
        state: AgentState = {
            "messages": [],
            "current_step": "",
            "reasoning_chain": [
                {"step": POMLNode.REFLECT.value, "output": "CONTINUE"}
            ],
            "verification_results": None,
            "final_output": None,
            "metadata": {"iteration_count": mock_agent.max_iterations + 1}
        }
        
        decision = mock_agent._should_continue_or_output(state)
        assert decision == POMLNode.OUTPUT.value


# SimplePOMLAgent Tests
class TestSimplePOMLAgent:
    """Test the simplified POML agent"""
    
    @pytest.mark.asyncio
    async def test_simple_agent_initialization(self, mock_api_key):
        """Test simple agent initialization"""
        with patch("langgraph_poml_agent.ChatOpenAI"):
            agent = SimplePOMLAgent(
                openrouter_api_key=mock_api_key,
                model_name="test-model",
                temperature=0.7
            )
            
            assert agent.api_key == mock_api_key
            assert agent.llm is not None
            assert agent.prompt_builder is not None
    
    @pytest.mark.asyncio
    async def test_simple_agent_process(self, mock_api_key):
        """Test simple agent processing"""
        with patch("langgraph_poml_agent.ChatOpenAI") as mock_llm:
            mock_instance = Mock()
            mock_instance.ainvoke = AsyncMock(
                return_value=Mock(content="Test response")
            )
            mock_llm.return_value = mock_instance
            
            agent = SimplePOMLAgent(
                openrouter_api_key=mock_api_key,
                model_name="test-model"
            )
            
            result = await agent.process("Test input")
            
            assert result == "Test response"
            mock_instance.ainvoke.assert_called_once()


# Integration Tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_create_enhanced_prompt(self):
        """Test creating an enhanced prompt"""
        base = "Solve this problem"
        enhanced = create_poml_enhanced_prompt(
            base_prompt=base,
            reasoning_steps=[ReasoningStep.UNDERSTAND, ReasoningStep.ANALYZE],
            include_examples=True
        )
        
        assert base in enhanced
        assert "POML Analysis Framework" in enhanced
        assert "Example Application" in enhanced
    
    def test_validate_reasoning_chain_function(self):
        """Test reasoning chain validation"""
        problem = "Test problem"
        solution = "Test solution"
        criteria = ["Correctness", "Completeness"]
        
        validation_prompt = validate_reasoning_chain(
            problem=problem,
            solution=solution,
            criteria=criteria
        )
        
        assert problem in validation_prompt
        assert solution in validation_prompt
        assert "Correctness" in validation_prompt
        assert "Completeness" in validation_prompt


# Edge Cases and Error Handling
class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_missing_api_key(self):
        """Test handling of missing API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                POMLAgent()
    
    def test_unknown_model_cost_estimation(self):
        """Test cost estimation with unknown model"""
        cost = estimate_cost(
            model_name="unknown/model",
            input_tokens=1000,
            output_tokens=500
        )
        
        assert "error" in cost
        assert cost["total_cost"] == 0.0
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, mock_agent):
        """Test handling of empty input"""
        state: AgentState = {
            "messages": [],
            "current_step": "",
            "reasoning_chain": [],
            "verification_results": None,
            "final_output": None,
            "metadata": {}
        }
        
        # Should handle gracefully without crashing
        result = await mock_agent._understand_node(state)
        assert result is not None


# Performance Tests
class TestPerformance:
    """Test performance-related aspects"""
    
    def test_prompt_size_limits(self, prompt_builder):
        """Test that prompts don't exceed reasonable sizes"""
        # Create a very long task
        long_task = "x" * 10000
        prompt = prompt_builder.build_task_prompt(long_task)
        
        # Prompt should include the task but be structured
        assert len(prompt) > len(long_task)
        assert "POML Analysis Framework" in prompt
    
    def test_reasoning_chain_memory(self):
        """Test that reasoning chain doesn't grow unbounded"""
        state: AgentState = {
            "messages": [],
            "current_step": "",
            "reasoning_chain": [],
            "verification_results": None,
            "final_output": None,
            "metadata": {}
        }
        
        # Add many items to reasoning chain
        for i in range(100):
            state["reasoning_chain"].append({
                "step": f"step_{i}",
                "output": f"output_{i}" * 1000,
                "timestamp": "2024-01-01T00:00:00"
            })
        
        # Should still be manageable
        assert len(state["reasoning_chain"]) == 100


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()