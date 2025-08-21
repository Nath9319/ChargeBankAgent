"""
OpenRouter Client Configuration for ChargeBankAgent
Handles LLM communication through OpenRouter API
"""

import os
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


class OpenRouterClient:
    """Client for interacting with OpenRouter API using LangChain"""
    
    def __init__(
        self,
        model_name: str = "anthropic/claude-3.5-sonnet",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000")
        self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "ChargeBankAgent")
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        # Configure ChatOpenAI for OpenRouter
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=self.api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model_kwargs={
                "headers": {
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                }
            }
        )
    
    async def generate_response(self, messages: List[BaseMessage]) -> str:
        """Generate response using OpenRouter LLM"""
        try:
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}")
    
    def generate_response_sync(self, messages: List[BaseMessage]) -> str:
        """Synchronous version of response generation"""
        try:
            response = self.llm.generate([messages])
            return response.generations[0][0].text
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}")
    
    def create_messages_from_poml(self, poml_prompt: str, user_message: str) -> List[BaseMessage]:
        """Convert POML prompt and user message to LangChain message format"""
        messages = [
            SystemMessage(content=poml_prompt),
            HumanMessage(content=user_message)
        ]
        return messages
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from OpenRouter"""
        # Common OpenRouter models for reference
        return [
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-haiku",
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemini-pro",
            "mistralai/mixtral-8x7b-instruct"
        ]
    
    def update_model(self, model_name: str, temperature: Optional[float] = None):
        """Update the model configuration"""
        if temperature is not None:
            self.llm.temperature = temperature
        self.llm.model_name = model_name