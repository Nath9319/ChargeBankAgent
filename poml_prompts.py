"""
Microsoft POML-like Prompting System for LangGraph and OpenRouter
This module provides structured prompting techniques similar to Microsoft's POML
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from jinja2 import Template, Environment, BaseLoader
from pydantic import BaseModel, Field
import re


@dataclass
class PromptVariable:
    """Represents a variable in a POML prompt"""
    name: str
    description: str
    type: str = "string"
    required: bool = True
    default: Optional[Any] = None
    constraints: Optional[List[str]] = None


@dataclass
class PromptFunction:
    """Represents a function call in a POML prompt"""
    name: str
    description: str
    parameters: List[PromptVariable]
    returns: str


@dataclass
class PromptContext:
    """Represents context information for a POML prompt"""
    source: str
    timestamp: str
    metadata: Dict[str, Any]


class POMLPrompt(BaseModel):
    """Main POML prompt structure"""
    name: str = Field(description="Name of the prompt")
    version: str = Field(default="1.0.0", description="POML version")
    description: str = Field(description="Description of what this prompt does")
    
    # Prompt structure
    system_message: str = Field(description="System message/role definition")
    user_message_template: str = Field(description="User message template with variables")
    
    # Variables and functions
    variables: List[PromptVariable] = Field(default_factory=list, description="Input variables")
    functions: List[PromptFunction] = Field(default_factory=list, description="Available functions")
    
    # Context and examples
    context: Optional[PromptContext] = Field(default=None, description="Context information")
    examples: List[Dict[str, str]] = Field(default_factory=list, description="Example conversations")
    
    # Output format
    output_format: Dict[str, Any] = Field(default_factory=dict, description="Expected output format")
    
    # Constraints and validation
    constraints: List[str] = Field(default_factory=list, description="Prompt constraints")
    validation_rules: List[str] = Field(default_factory=list, description="Validation rules")


class POMLPromptEngine:
    """Engine for processing POML prompts with LangGraph and OpenRouter"""
    
    def __init__(self, jinja_env: Optional[Environment] = None):
        self.jinja_env = jinja_env or Environment(loader=BaseLoader())
        self.prompts: Dict[str, POMLPrompt] = {}
    
    def load_prompt_from_yaml(self, yaml_content: str) -> POMLPrompt:
        """Load a POML prompt from YAML content"""
        try:
            data = yaml.safe_load(yaml_content)
            return POMLPrompt(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse POML YAML: {e}")
    
    def load_prompt_from_json(self, json_content: str) -> POMLPrompt:
        """Load a POML prompt from JSON content"""
        try:
            data = json.loads(json_content)
            return POMLPrompt(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse POML JSON: {e}")
    
    def register_prompt(self, prompt: POMLPrompt) -> None:
        """Register a prompt in the engine"""
        self.prompts[prompt.name] = prompt
    
    def render_prompt(self, prompt_name: str, variables: Dict[str, Any]) -> Dict[str, str]:
        """Render a POML prompt with given variables"""
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        
        prompt = self.prompts[prompt_name]
        
        # Validate required variables
        self._validate_variables(prompt, variables)
        
        # Render templates
        system_message = self._render_template(prompt.system_message, variables)
        user_message = self._render_template(prompt.user_message_template, variables)
        
        return {
            "system_message": system_message,
            "user_message": user_message,
            "functions": [asdict(f) for f in prompt.functions],
            "output_format": prompt.output_format
        }
    
    def _validate_variables(self, prompt: POMLPrompt, variables: Dict[str, Any]) -> None:
        """Validate that all required variables are provided"""
        for var in prompt.variables:
            if var.required and var.name not in variables:
                raise ValueError(f"Required variable '{var.name}' not provided")
    
    def _render_template(self, template_str: str, variables: Dict[str, Any]) -> str:
        """Render a Jinja2 template with variables"""
        try:
            template = self.jinja_env.from_string(template_str)
            return template.render(**variables)
        except Exception as e:
            raise ValueError(f"Failed to render template: {e}")
    
    def get_prompt_schema(self, prompt_name: str) -> Dict[str, Any]:
        """Get the schema for a prompt (useful for OpenRouter function calling)"""
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        
        prompt = self.prompts[prompt_name]
        
        # Convert to OpenRouter function calling format
        properties = {}
        required = []
        
        for var in prompt.variables:
            properties[var.name] = {
                "type": var.type,
                "description": var.description
            }
            if var.required:
                required.append(var.name)
        
        return {
            "name": prompt_name,
            "description": prompt.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


# Pre-built POML prompts for common use cases
class CommonPOMLPrompts:
    """Collection of common POML prompts"""
    
    @staticmethod
    def create_analysis_prompt() -> POMLPrompt:
        """Create a data analysis prompt"""
        return POMLPrompt(
            name="data_analysis",
            description="Analyze data and provide insights",
            system_message="You are a data analyst expert. Analyze the provided data and give clear, actionable insights.",
            user_message_template="Please analyze the following data:\n\n{{data}}\n\nContext: {{context}}\n\nProvide analysis in the following format:\n- Key findings\n- Trends\n- Recommendations",
            variables=[
                PromptVariable("data", "The data to analyze", "string", True),
                PromptVariable("context", "Additional context for analysis", "string", False, "No additional context")
            ],
            output_format={
                "type": "object",
                "properties": {
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "trends": {"type": "array", "items": {"type": "string"}},
                    "recommendations": {"type": "array", "items": {"type": "string"}}
                }
            }
        )
    
    @staticmethod
    def create_code_review_prompt() -> POMLPrompt:
        """Create a code review prompt"""
        return POMLPrompt(
            name="code_review",
            description="Review code and provide feedback",
            system_message="You are a senior software engineer conducting a code review. Provide constructive feedback focusing on code quality, security, and best practices.",
            user_message_template="Please review the following code:\n\n```{{language}}\n{{code}}\n```\n\nFocus on: {{focus_areas}}\n\nProvide feedback in the following format:\n- Code quality issues\n- Security concerns\n- Performance improvements\n- Best practices",
            variables=[
                PromptVariable("code", "The code to review", "string", True),
                PromptVariable("language", "Programming language", "string", True),
                PromptVariable("focus_areas", "Areas to focus on", "string", False, "Code quality, security, performance")
            ],
            output_format={
                "type": "object",
                "properties": {
                    "code_quality_issues": {"type": "array", "items": {"type": "string"}},
                    "security_concerns": {"type": "array", "items": {"type": "string"}},
                    "performance_improvements": {"type": "array", "items": {"type": "string"}},
                    "best_practices": {"type": "array", "items": {"type": "string"}}
                }
            }
        )
    
    @staticmethod
    def create_planning_prompt() -> POMLPrompt:
        """Create a planning and strategy prompt"""
        return POMLPrompt(
            name="planning_strategy",
            description="Create a strategic plan based on requirements",
            system_message="You are a strategic planning expert. Help create comprehensive plans and strategies based on requirements and constraints.",
            user_message_template="Create a strategic plan for:\n\nGoal: {{goal}}\n\nRequirements: {{requirements}}\n\nConstraints: {{constraints}}\n\nProvide a plan with:\n- Objectives\n- Timeline\n- Resources needed\n- Risk assessment\n- Success metrics",
            variables=[
                PromptVariable("goal", "The main goal to achieve", "string", True),
                PromptVariable("requirements", "Key requirements", "string", True),
                PromptVariable("constraints", "Limitations and constraints", "string", False, "No specific constraints")
            ],
            output_format={
                "type": "object",
                "properties": {
                    "objectives": {"type": "array", "items": {"type": "string"}},
                    "timeline": {"type": "object", "properties": {"phases": {"type": "array", "items": {"type": "string"}}}},
                    "resources": {"type": "array", "items": {"type": "string"}},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "success_metrics": {"type": "array", "items": {"type": "string"}}
                }
            }
        )


# Utility functions for working with POML prompts
def create_poml_from_template(template_name: str, **kwargs) -> POMLPrompt:
    """Create a POML prompt from a template"""
    templates = {
        "analysis": CommonPOMLPrompts.create_analysis_prompt,
        "code_review": CommonPOMLPrompts.create_code_review_prompt,
        "planning": CommonPOMLPrompts.create_planning_prompt
    }
    
    if template_name not in templates:
        raise ValueError(f"Template '{template_name}' not found. Available: {list(templates.keys())}")
    
    return templates[template_name]()


def export_poml_to_yaml(prompt: POMLPrompt) -> str:
    """Export a POML prompt to YAML format"""
    return yaml.dump(asdict(prompt), default_flow_style=False, sort_keys=False)


def export_poml_to_json(prompt: POMLPrompt) -> str:
    """Export a POML prompt to JSON format"""
    return json.dumps(asdict(prompt), indent=2)