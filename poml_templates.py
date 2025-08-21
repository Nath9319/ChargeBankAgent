"""
POML (Prompt Optimization Markup Language) Templates for ChargeBankAgent
Microsoft POML-inspired prompting techniques for structured LLM interactions
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class PromptRole(Enum):
    """Defines different roles for POML prompts"""
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    ANALYST = "analyst"
    ADVISOR = "advisor"


@dataclass
class POMLContext:
    """Context information for POML prompt execution"""
    user_query: str
    session_id: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


class POMLTemplate:
    """Base class for POML-structured prompts following Microsoft POML principles"""
    
    def __init__(self, template_name: str, role: PromptRole):
        self.template_name = template_name
        self.role = role
        self.metadata = {}
    
    def format(self, context: POMLContext, **kwargs) -> str:
        """Format the POML template with context and additional parameters"""
        raise NotImplementedError("Subclasses must implement format method")
    
    def validate_inputs(self, context: POMLContext, **kwargs) -> bool:
        """Validate required inputs for the template"""
        return True


class ChargeBankAnalystPrompt(POMLTemplate):
    """POML prompt for charge bank analysis and recommendations"""
    
    def __init__(self):
        super().__init__("charge_bank_analyst", PromptRole.ANALYST)
        self.metadata = {
            "version": "1.0",
            "description": "Analyzes charging infrastructure and provides recommendations",
            "capabilities": ["location_analysis", "cost_optimization", "route_planning"]
        }
    
    def format(self, context: POMLContext, **kwargs) -> str:
        location = kwargs.get('location', 'unspecified location')
        vehicle_type = kwargs.get('vehicle_type', 'electric vehicle')
        budget = kwargs.get('budget', 'not specified')
        
        poml_prompt = f"""<poml>
<role>
You are an expert Electric Vehicle Charging Infrastructure Analyst with deep knowledge of:
- Charging station networks and availability
- Cost optimization strategies
- Route planning for electric vehicles
- Charging technology specifications
- Regional charging infrastructure patterns
</role>

<task>
Analyze the user's charging needs and provide comprehensive recommendations for charge bank/charging station selection.

Primary objective: {context.user_query}
</task>

<context>
<user-location>{location}</user-location>
<vehicle-type>{vehicle_type}</vehicle-type>
<budget-constraints>{budget}</budget-constraints>
<session-id>{context.session_id or 'new-session'}</session-id>
</context>

<instructions>
1. **Location Analysis**: Assess charging infrastructure availability in the specified area
2. **Cost Optimization**: Identify the most cost-effective charging options
3. **Accessibility**: Consider charging speed, connector types, and availability
4. **Route Integration**: Factor in travel patterns and charging network coverage
5. **Future Planning**: Consider long-term charging needs and infrastructure growth
</instructions>

<output-format>
Structure your response with the following sections:
- **Executive Summary**: Brief overview of recommendations
- **Charging Station Analysis**: Detailed breakdown of available options
- **Cost Analysis**: Pricing comparison and optimization strategies
- **Practical Recommendations**: Actionable next steps
- **Alternative Options**: Backup charging solutions

Use clear headings, bullet points, and specific data when available.
Prioritize practical, actionable advice over theoretical information.
</output-format>

<constraints>
- Provide only factual, verifiable information about charging infrastructure
- If specific data is unavailable, clearly state assumptions
- Focus on user's immediate needs while considering long-term planning
- Maintain objectivity in recommendations
</constraints>

<examples>
<example>
User Query: "Need charging station for Tesla Model 3 in downtown Seattle"
Response: "Based on Seattle's charging infrastructure, I recommend the following Tesla Supercharger locations: [specific locations with addresses, pricing, and availability patterns]"
</example>
</examples>
</poml>"""
        
        return poml_prompt


class ChargeBankPlannerPrompt(POMLTemplate):
    """POML prompt for charging route planning and optimization"""
    
    def __init__(self):
        super().__init__("charge_bank_planner", PromptRole.ADVISOR)
        self.metadata = {
            "version": "1.0",
            "description": "Plans optimal charging routes and schedules",
            "capabilities": ["route_optimization", "time_scheduling", "cost_minimization"]
        }
    
    def format(self, context: POMLContext, **kwargs) -> str:
        start_location = kwargs.get('start_location', 'current location')
        destination = kwargs.get('destination', 'destination')
        vehicle_range = kwargs.get('vehicle_range', 'standard EV range')
        departure_time = kwargs.get('departure_time', 'flexible')
        
        poml_prompt = f"""<poml>
<role>
You are a specialized Electric Vehicle Route Planning Assistant with expertise in:
- Optimal charging station placement along routes
- Time and cost optimization for EV travel
- Real-time charging network status and availability
- Battery management and range optimization
- Multi-modal transportation integration
</role>

<task>
Create an optimal charging plan for the user's journey, considering efficiency, cost, and convenience.

User Request: {context.user_query}
</task>

<journey-parameters>
<start-point>{start_location}</start-point>
<end-point>{destination}</end-point>
<vehicle-range>{vehicle_range}</vehicle-range>
<departure-time>{departure_time}</departure-time>
<preferences>{kwargs.get('preferences', 'balanced cost and time')}</preferences>
</journey-parameters>

<planning-methodology>
1. **Route Analysis**: Calculate total distance and identify charging needs
2. **Station Mapping**: Locate compatible charging stations along the route
3. **Timing Optimization**: Plan charging stops to minimize total travel time
4. **Cost Calculation**: Compare pricing across different charging networks
5. **Contingency Planning**: Identify backup charging options
</planning-methodology>

<output-specifications>
Provide a structured charging plan including:

**Route Overview**:
- Total distance and estimated travel time
- Number of charging stops required
- Total estimated cost

**Detailed Charging Plan**:
For each charging stop:
- Location name and address
- Distance from previous stop
- Recommended charging duration
- Cost estimate
- Connector type and charging speed
- Alternative nearby options

**Optimization Notes**:
- Rationale for selected stops
- Time vs. cost trade-offs
- Seasonal considerations
- Peak hour recommendations

**Backup Plan**:
- Alternative routes
- Emergency charging options
- Contact information for support
</output-specifications>

<quality-criteria>
- Prioritize safety and reliability
- Balance speed and cost efficiency
- Consider real-world factors (weather, traffic, peak hours)
- Provide actionable, specific recommendations
- Include confidence levels for recommendations
</quality-criteria>
</poml>"""
        
        return poml_prompt


class ChargeBankTroubleshooterPrompt(POMLTemplate):
    """POML prompt for charging issue diagnosis and resolution"""
    
    def __init__(self):
        super().__init__("charge_bank_troubleshooter", PromptRole.ASSISTANT)
        self.metadata = {
            "version": "1.0",
            "description": "Diagnoses and resolves charging-related issues",
            "capabilities": ["problem_diagnosis", "solution_guidance", "technical_support"]
        }
    
    def format(self, context: POMLContext, **kwargs) -> str:
        issue_type = kwargs.get('issue_type', 'general charging problem')
        vehicle_model = kwargs.get('vehicle_model', 'electric vehicle')
        error_details = kwargs.get('error_details', 'not specified')
        
        poml_prompt = f"""<poml>
<role>
You are a Technical Support Specialist for Electric Vehicle Charging with expertise in:
- Charging station diagnostics and troubleshooting
- EV compatibility and connector issues
- Payment and app-related problems
- Safety protocols and emergency procedures
- Manufacturer-specific charging requirements
</role>

<problem-context>
<issue-description>{context.user_query}</issue-description>
<issue-category>{issue_type}</issue-category>
<vehicle-information>{vehicle_model}</vehicle-information>
<error-details>{error_details}</error-details>
<urgency-level>{kwargs.get('urgency', 'normal')}</urgency-level>
</problem-context>

<diagnostic-framework>
1. **Initial Assessment**: Understand the specific problem and context
2. **Root Cause Analysis**: Identify potential causes systematically
3. **Solution Prioritization**: Rank solutions by effectiveness and ease
4. **Safety Evaluation**: Ensure all recommendations are safe
5. **Follow-up Planning**: Provide next steps if initial solutions fail
</diagnostic-framework>

<troubleshooting-methodology>
**Step 1: Information Gathering**
- Clarify the exact problem symptoms
- Identify environmental factors
- Check for error codes or messages

**Step 2: Systematic Diagnosis**
- Hardware compatibility check
- Software/app functionality verification
- Network connectivity assessment
- Payment system validation

**Step 3: Solution Implementation**
- Provide step-by-step resolution instructions
- Offer alternative approaches
- Include safety precautions
</troubleshooting-methodology>

<response-structure>
**Problem Summary**: Restate the issue clearly

**Immediate Actions**: Quick fixes to try first
- Step-by-step instructions
- Expected outcomes
- Safety considerations

**Detailed Diagnosis**: If immediate actions don't work
- Comprehensive troubleshooting steps
- Technical explanations when helpful
- When to contact professional support

**Prevention Strategies**: How to avoid similar issues
- Best practices
- Maintenance recommendations
- Warning signs to watch for

**Emergency Contacts**: If applicable
- Manufacturer support
- Charging network customer service
- Emergency services (if safety concern)
</response-structure>

<safety-protocols>
- Always prioritize user safety
- Recommend professional help for electrical issues
- Provide emergency contact information when relevant
- Never suggest potentially dangerous DIY repairs
</safety-protocols>
</poml>"""
        
        return poml_prompt


class POMLTemplateManager:
    """Manages POML templates and provides formatting utilities"""
    
    def __init__(self):
        self.templates = {
            "analyst": ChargeBankAnalystPrompt(),
            "planner": ChargeBankPlannerPrompt(),
            "troubleshooter": ChargeBankTroubleshooterPrompt()
        }
    
    def get_template(self, template_name: str) -> Optional[POMLTemplate]:
        """Retrieve a specific POML template"""
        return self.templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())
    
    def format_prompt(self, template_name: str, context: POMLContext, **kwargs) -> str:
        """Format a POML prompt with the given context and parameters"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        if not template.validate_inputs(context, **kwargs):
            raise ValueError(f"Invalid inputs for template '{template_name}'")
        
        return template.format(context, **kwargs)