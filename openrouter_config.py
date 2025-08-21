"""
OpenRouter Configuration and Utilities

This module provides configuration and helper functions for integrating
with OpenRouter API for accessing various LLM models.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import httpx
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ModelProvider(Enum):
    """Supported model providers via OpenRouter"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    META = "meta-llama"
    GOOGLE = "google"
    MISTRAL = "mistralai"
    COHERE = "cohere"
    PERPLEXITY = "perplexity"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    provider: ModelProvider
    context_length: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    supports_functions: bool
    supports_vision: bool
    description: str


class OpenRouterModels:
    """Registry of available models via OpenRouter"""
    
    MODELS = {
        "anthropic/claude-3.5-sonnet": ModelConfig(
            name="anthropic/claude-3.5-sonnet",
            provider=ModelProvider.ANTHROPIC,
            context_length=200000,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            supports_functions=True,
            supports_vision=True,
            description="Most capable Claude model with excellent reasoning"
        ),
        "anthropic/claude-3-opus": ModelConfig(
            name="anthropic/claude-3-opus",
            provider=ModelProvider.ANTHROPIC,
            context_length=200000,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            supports_functions=True,
            supports_vision=True,
            description="Previous flagship Claude model"
        ),
        "openai/gpt-4-turbo": ModelConfig(
            name="openai/gpt-4-turbo",
            provider=ModelProvider.OPENAI,
            context_length=128000,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            supports_functions=True,
            supports_vision=True,
            description="OpenAI's GPT-4 Turbo with vision"
        ),
        "openai/gpt-4o": ModelConfig(
            name="openai/gpt-4o",
            provider=ModelProvider.OPENAI,
            context_length=128000,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            supports_functions=True,
            supports_vision=True,
            description="OpenAI's multimodal GPT-4 Omni model"
        ),
        "meta-llama/llama-3.1-405b-instruct": ModelConfig(
            name="meta-llama/llama-3.1-405b-instruct",
            provider=ModelProvider.META,
            context_length=131072,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.003,
            supports_functions=False,
            supports_vision=False,
            description="Meta's largest Llama 3.1 model"
        ),
        "meta-llama/llama-3.1-70b-instruct": ModelConfig(
            name="meta-llama/llama-3.1-70b-instruct",
            provider=ModelProvider.META,
            context_length=131072,
            cost_per_1k_input=0.00052,
            cost_per_1k_output=0.00075,
            supports_functions=False,
            supports_vision=False,
            description="Meta's efficient Llama 3.1 70B model"
        ),
        "google/gemini-pro-1.5": ModelConfig(
            name="google/gemini-pro-1.5",
            provider=ModelProvider.GOOGLE,
            context_length=2097152,
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.005,
            supports_functions=True,
            supports_vision=True,
            description="Google's Gemini Pro 1.5 with massive context"
        ),
        "mistralai/mixtral-8x22b-instruct": ModelConfig(
            name="mistralai/mixtral-8x22b-instruct",
            provider=ModelProvider.MISTRAL,
            context_length=65536,
            cost_per_1k_input=0.0009,
            cost_per_1k_output=0.0009,
            supports_functions=True,
            supports_vision=False,
            description="Mistral's MoE model with 8x22B experts"
        ),
        "cohere/command-r-plus": ModelConfig(
            name="cohere/command-r-plus",
            provider=ModelProvider.COHERE,
            context_length=128000,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            supports_functions=True,
            supports_vision=False,
            description="Cohere's Command R+ for RAG and tool use"
        ),
        "perplexity/llama-3.1-sonar-large-128k-online": ModelConfig(
            name="perplexity/llama-3.1-sonar-large-128k-online",
            provider=ModelProvider.PERPLEXITY,
            context_length=127072,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.001,
            supports_functions=False,
            supports_vision=False,
            description="Perplexity's online search-enabled model"
        )
    }
    
    @classmethod
    def get_model(cls, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return cls.MODELS.get(model_name)
    
    @classmethod
    def list_models(cls, 
                   provider: Optional[ModelProvider] = None,
                   supports_functions: Optional[bool] = None,
                   supports_vision: Optional[bool] = None) -> List[ModelConfig]:
        """
        List models with optional filters
        
        Args:
            provider: Filter by provider
            supports_functions: Filter by function calling support
            supports_vision: Filter by vision support
        
        Returns:
            List of matching model configurations
        """
        models = list(cls.MODELS.values())
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        if supports_functions is not None:
            models = [m for m in models if m.supports_functions == supports_functions]
        
        if supports_vision is not None:
            models = [m for m in models if m.supports_vision == supports_vision]
        
        return models
    
    @classmethod
    def get_cheapest_model(cls, 
                          min_context: int = 0,
                          supports_functions: Optional[bool] = None) -> Optional[ModelConfig]:
        """
        Get the cheapest model meeting requirements
        
        Args:
            min_context: Minimum required context length
            supports_functions: Whether function calling is required
        
        Returns:
            Cheapest matching model or None
        """
        models = cls.list_models(supports_functions=supports_functions)
        models = [m for m in models if m.context_length >= min_context]
        
        if not models:
            return None
        
        # Sort by average cost
        models.sort(key=lambda m: (m.cost_per_1k_input + m.cost_per_1k_output) / 2)
        return models[0]


class OpenRouterClient:
    """
    Client for interacting with OpenRouter API directly
    (For cases where you need more control than LangChain provides)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        """
        Initialize OpenRouter client
        
        Args:
            api_key: OpenRouter API key
            base_url: Base URL for OpenRouter API
            site_url: Your site URL for tracking
            site_name: Your site name for tracking
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.base_url = base_url
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000")
        self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "POML-Agent")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json"
        }
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion via OpenRouter
        
        Args:
            model: Model name (e.g., "anthropic/claude-3.5-sonnet")
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters
        
        Returns:
            API response dictionary
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from OpenRouter
        
        Returns:
            List of model information dictionaries
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json().get("data", [])
    
    async def get_usage(self) -> Dict[str, Any]:
        """
        Get usage statistics for your API key
        
        Returns:
            Usage information dictionary
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()


def create_openrouter_llm(
    model_name: str = "anthropic/claude-3.5-sonnet",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    **kwargs
) -> ChatOpenAI:
    """
    Create a LangChain ChatOpenAI instance configured for OpenRouter
    
    Args:
        model_name: Name of the model to use
        temperature: Sampling temperature
        api_key: OpenRouter API key
        **kwargs: Additional parameters for ChatOpenAI
    
    Returns:
        Configured ChatOpenAI instance
    """
    from langchain_community.chat_models import ChatOpenAI
    
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key is required")
    
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000"),
            "X-Title": os.getenv("OPENROUTER_SITE_NAME", "POML-Agent"),
        },
        **kwargs
    )


def estimate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int
) -> Dict[str, float]:
    """
    Estimate the cost of a completion
    
    Args:
        model_name: Name of the model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    
    Returns:
        Dictionary with cost breakdown
    """
    model = OpenRouterModels.get_model(model_name)
    if not model:
        return {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "error": f"Unknown model: {model_name}"
        }
    
    input_cost = (input_tokens / 1000) * model.cost_per_1k_input
    output_cost = (output_tokens / 1000) * model.cost_per_1k_output
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }


# Model recommendations for different use cases
MODEL_RECOMMENDATIONS = {
    "reasoning": [
        "anthropic/claude-3.5-sonnet",  # Best reasoning
        "openai/gpt-4-turbo",  # Strong alternative
        "google/gemini-pro-1.5"  # Good with huge context
    ],
    "coding": [
        "anthropic/claude-3.5-sonnet",  # Excellent for code
        "openai/gpt-4-turbo",  # Strong coding abilities
        "meta-llama/llama-3.1-405b-instruct"  # Open source alternative
    ],
    "creative": [
        "anthropic/claude-3.5-sonnet",  # Creative and coherent
        "openai/gpt-4o",  # Multimodal creativity
        "mistralai/mixtral-8x22b-instruct"  # Good creative writing
    ],
    "analysis": [
        "anthropic/claude-3.5-sonnet",  # Deep analysis
        "google/gemini-pro-1.5",  # Huge context for documents
        "cohere/command-r-plus"  # Optimized for RAG
    ],
    "budget": [
        "meta-llama/llama-3.1-70b-instruct",  # Very cost-effective
        "mistralai/mixtral-8x22b-instruct",  # Good balance
        "perplexity/llama-3.1-sonar-large-128k-online"  # With web search
    ],
    "search": [
        "perplexity/llama-3.1-sonar-large-128k-online",  # Built-in web search
        "cohere/command-r-plus",  # Good for RAG
        "google/gemini-pro-1.5"  # Large context for documents
    ]
}


def get_recommended_model(use_case: str, budget_conscious: bool = False) -> str:
    """
    Get recommended model for a specific use case
    
    Args:
        use_case: Type of task (reasoning, coding, creative, analysis, budget, search)
        budget_conscious: If True, prefer cheaper options
    
    Returns:
        Recommended model name
    """
    recommendations = MODEL_RECOMMENDATIONS.get(use_case, MODEL_RECOMMENDATIONS["reasoning"])
    
    if budget_conscious:
        # Return the cheapest option from recommendations
        models = [OpenRouterModels.get_model(m) for m in recommendations]
        models = [m for m in models if m]  # Filter None
        if models:
            models.sort(key=lambda m: (m.cost_per_1k_input + m.cost_per_1k_output) / 2)
            return models[0].name
    
    return recommendations[0]