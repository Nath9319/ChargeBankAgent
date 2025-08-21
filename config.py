"""
Configuration settings for ChargeBankAgent
POML template configurations and OpenRouter model settings
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class ModelTier(Enum):
    """Different model tiers for various use cases"""
    FAST = "fast"          # Quick responses, lower cost
    BALANCED = "balanced"   # Good balance of speed and quality
    PREMIUM = "premium"     # Highest quality responses


@dataclass
class ModelConfig:
    """Configuration for OpenRouter models"""
    name: str
    temperature: float
    max_tokens: int
    tier: ModelTier
    cost_per_1k_tokens: float
    description: str


class ChargeBankConfig:
    """Central configuration for the ChargeBankAgent system"""
    
    # OpenRouter Model Configurations
    MODELS = {
        "claude-3.5-sonnet": ModelConfig(
            name="anthropic/claude-3.5-sonnet",
            temperature=0.7,
            max_tokens=4000,
            tier=ModelTier.PREMIUM,
            cost_per_1k_tokens=0.003,
            description="Best for complex analysis and detailed responses"
        ),
        "claude-3-haiku": ModelConfig(
            name="anthropic/claude-3-haiku",
            temperature=0.5,
            max_tokens=2000,
            tier=ModelTier.FAST,
            cost_per_1k_tokens=0.0005,
            description="Fast responses for simple queries"
        ),
        "gpt-4-turbo": ModelConfig(
            name="openai/gpt-4-turbo",
            temperature=0.6,
            max_tokens=3000,
            tier=ModelTier.PREMIUM,
            cost_per_1k_tokens=0.01,
            description="Excellent for technical troubleshooting"
        ),
        "gpt-3.5-turbo": ModelConfig(
            name="openai/gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2000,
            tier=ModelTier.BALANCED,
            cost_per_1k_tokens=0.002,
            description="Good balance of cost and performance"
        )
    }
    
    # POML Template Settings
    POML_CONFIG = {
        "enable_validation": True,
        "strict_formatting": True,
        "include_metadata": True,
        "context_window_size": 8000,
        "max_examples_per_prompt": 3
    }
    
    # LangGraph Workflow Settings
    LANGGRAPH_CONFIG = {
        "max_iterations": 10,
        "timeout_seconds": 60,
        "enable_streaming": True,
        "checkpoint_enabled": True
    }
    
    # Default context parameters
    DEFAULT_CONTEXT = {
        "location": "United States",
        "vehicle_type": "Electric Vehicle",
        "budget": "moderate",
        "urgency": "normal",
        "preferences": "balanced cost and convenience"
    }
    
    # Charging network information
    CHARGING_NETWORKS = {
        "tesla": {
            "name": "Tesla Supercharger",
            "connectors": ["Tesla", "NACS"],
            "typical_speed": "150-250kW",
            "payment_methods": ["Tesla app", "Credit card"]
        },
        "electrify_america": {
            "name": "Electrify America",
            "connectors": ["CCS", "CHAdeMO"],
            "typical_speed": "150-350kW",
            "payment_methods": ["EA app", "Credit card", "RFID"]
        },
        "evgo": {
            "name": "EVgo",
            "connectors": ["CCS", "CHAdeMO"],
            "typical_speed": "50-100kW",
            "payment_methods": ["EVgo app", "Credit card"]
        },
        "chargepoint": {
            "name": "ChargePoint",
            "connectors": ["J1772", "CCS"],
            "typical_speed": "6-50kW",
            "payment_methods": ["ChargePoint app", "RFID", "Credit card"]
        }
    }
    
    @classmethod
    def get_model_config(cls, model_key: str) -> ModelConfig:
        """Get configuration for a specific model"""
        return cls.MODELS.get(model_key, cls.MODELS["claude-3.5-sonnet"])
    
    @classmethod
    def get_model_by_tier(cls, tier: ModelTier) -> ModelConfig:
        """Get the best model for a specific tier"""
        tier_models = {
            ModelTier.FAST: "claude-3-haiku",
            ModelTier.BALANCED: "gpt-3.5-turbo", 
            ModelTier.PREMIUM: "claude-3.5-sonnet"
        }
        return cls.get_model_config(tier_models[tier])
    
    @classmethod
    def validate_environment(cls) -> Dict[str, bool]:
        """Validate that required environment variables are set"""
        required_vars = [
            "OPENROUTER_API_KEY"
        ]
        
        optional_vars = [
            "OPENROUTER_SITE_URL",
            "OPENROUTER_SITE_NAME",
            "MODEL_NAME",
            "TEMPERATURE"
        ]
        
        validation = {}
        
        for var in required_vars:
            validation[var] = bool(os.getenv(var))
        
        for var in optional_vars:
            validation[f"{var} (optional)"] = bool(os.getenv(var))
        
        return validation


# Environment validation on import
if __name__ == "__main__":
    print("ChargeBankAgent Configuration")
    print("=" * 30)
    
    validation = ChargeBankConfig.validate_environment()
    for var, is_set in validation.items():
        status = "✅" if is_set else "❌"
        print(f"{status} {var}")
    
    print(f"\nAvailable Models:")
    for key, config in ChargeBankConfig.MODELS.items():
        print(f"• {key}: {config.description}")
    
    print(f"\nCharging Networks Supported:")
    for key, network in ChargeBankConfig.CHARGING_NETWORKS.items():
        print(f"• {network['name']}: {', '.join(network['connectors'])}")