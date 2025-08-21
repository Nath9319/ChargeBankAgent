"""
OpenRouter Configuration for POML Prompts
This module provides configuration options for different OpenRouter models and API settings
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OpenRouterModel:
    """Configuration for an OpenRouter model"""
    name: str
    provider: str
    context_length: int
    max_tokens: int
    pricing: Dict[str, float]  # input/output tokens per 1M tokens
    capabilities: List[str]
    description: str


class OpenRouterConfig:
    """Configuration manager for OpenRouter"""
    
    # Available models
    MODELS = {
        "gpt-4": OpenRouterModel(
            name="openai/gpt-4",
            provider="OpenAI",
            context_length=8192,
            max_tokens=4096,
            pricing={"input": 30.0, "output": 60.0},
            capabilities=["function_calling", "json_mode", "vision"],
            description="GPT-4 with function calling and JSON mode"
        ),
        "gpt-4-turbo": OpenRouterModel(
            name="openai/gpt-4-turbo-preview",
            provider="OpenAI",
            context_length=128000,
            max_tokens=4096,
            pricing={"input": 10.0, "output": 30.0},
            capabilities=["function_calling", "json_mode", "vision"],
            description="GPT-4 Turbo with extended context"
        ),
        "claude-3": OpenRouterModel(
            name="anthropic/claude-3-opus",
            provider="Anthropic",
            context_length=200000,
            max_tokens=4096,
            pricing={"input": 15.0, "output": 75.0},
            capabilities=["function_calling", "json_mode"],
            description="Claude 3 Opus with function calling"
        ),
        "claude-3-sonnet": OpenRouterModel(
            name="anthropic/claude-3-sonnet",
            provider="Anthropic",
            context_length=200000,
            max_tokens=4096,
            pricing={"input": 3.0, "output": 15.0},
            capabilities=["function_calling", "json_mode"],
            description="Claude 3 Sonnet with function calling"
        ),
        "gemini-pro": OpenRouterModel(
            name="google/gemini-pro",
            provider="Google",
            context_length=32768,
            max_tokens=2048,
            pricing={"input": 0.5, "output": 1.5},
            capabilities=["function_calling"],
            description="Gemini Pro with function calling"
        ),
        "llama-3": OpenRouterModel(
            name="meta-llama/llama-3-70b-instruct",
            provider="Meta",
            context_length=8192,
            max_tokens=4096,
            pricing={"input": 0.6, "output": 0.8},
            capabilities=["function_calling"],
            description="Llama 3 70B with function calling"
        )
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_headers = {
            "HTTP-Referer": "https://github.com/your-repo/chargebank-agent",
            "X-Title": "ChargeBankAgent"
        }
    
    def get_model_config(self, model_name: str) -> OpenRouterModel:
        """Get configuration for a specific model"""
        if model_name in self.MODELS:
            return self.MODELS[model_name]
        
        # Try to find by full name
        for key, model in self.MODELS.items():
            if model.name == model_name:
                return model
        
        raise ValueError(f"Model '{model_name}' not found")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.MODELS.keys())
    
    def get_models_by_capability(self, capability: str) -> List[OpenRouterModel]:
        """Get models that support a specific capability"""
        return [model for model in self.MODELS.values() if capability in model.capabilities]
    
    def get_cost_estimate(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a specific model and token usage"""
        model = self.get_model_config(model_name)
        
        input_cost = (input_tokens / 1_000_000) * model.pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model.pricing["output"]
        
        return input_cost + output_cost
    
    def get_headers(self, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = self.default_headers.copy()
        if custom_headers:
            headers.update(custom_headers)
        return headers
    
    def validate_model_for_prompt(self, model_name: str, prompt_requirements: List[str]) -> bool:
        """Validate if a model supports the requirements of a prompt"""
        model = self.get_model_config(model_name)
        
        for requirement in prompt_requirements:
            if requirement not in model.capabilities:
                return False
        
        return True


# Model selection helpers
def select_model_for_prompt(
    prompt_requirements: List[str],
    budget_constraint: Optional[float] = None,
    context_length_needed: Optional[int] = None
) -> str:
    """Select the best model for a prompt based on requirements and constraints"""
    config = OpenRouterConfig()
    
    # Filter models by capabilities
    suitable_models = []
    for model_name, model in config.MODELS.items():
        if config.validate_model_for_prompt(model_name, prompt_requirements):
            suitable_models.append((model_name, model))
    
    if not suitable_models:
        raise ValueError(f"No models found that support requirements: {prompt_requirements}")
    
    # Filter by context length if specified
    if context_length_needed:
        suitable_models = [
            (name, model) for name, model in suitable_models
            if model.context_length >= context_length_needed
        ]
    
    if not suitable_models:
        raise ValueError(f"No models found with sufficient context length: {context_length_needed}")
    
    # Sort by cost (cheapest first)
    suitable_models.sort(key=lambda x: x[1].pricing["input"] + x[1].pricing["output"])
    
    # Apply budget constraint if specified
    if budget_constraint:
        for model_name, model in suitable_models:
            # Estimate cost for typical usage (1K input, 500 output tokens)
            estimated_cost = config.get_cost_estimate(model_name, 1000, 500)
            if estimated_cost <= budget_constraint:
                return model_name
        
        raise ValueError(f"No models found within budget constraint: ${budget_constraint}")
    
    # Return cheapest suitable model
    return suitable_models[0][0]


# Example usage
if __name__ == "__main__":
    try:
        config = OpenRouterConfig()
        
        print("Available models:")
        for name, model in config.MODELS.items():
            print(f"- {name}: {model.description}")
        
        print(f"\nModels with function calling:")
        function_calling_models = config.get_models_by_capability("function_calling")
        for model in function_calling_models:
            print(f"- {model.name}")
        
        # Example model selection
        requirements = ["function_calling", "json_mode"]
        selected_model = select_model_for_prompt(requirements, budget_constraint=0.01)
        print(f"\nSelected model for requirements {requirements}: {selected_model}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set OPENROUTER_API_KEY environment variable")