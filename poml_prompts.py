"""
Microsoft POML (Parrot-Olympiad-Math-Logic) Prompting System

This module implements the POML prompting technique which enhances reasoning
through structured problem decomposition, logical analysis, and step-by-step verification.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ReasoningStep(Enum):
    """Types of reasoning steps in POML methodology"""
    UNDERSTAND = "understand"
    DECOMPOSE = "decompose"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"
    REFLECT = "reflect"


@dataclass
class POMLPrompt:
    """Structure for POML-enhanced prompts"""
    base_instruction: str
    reasoning_steps: List[ReasoningStep]
    examples: Optional[List[Dict[str, str]]] = None
    constraints: Optional[List[str]] = None
    verification_criteria: Optional[List[str]] = None


class POMLPromptBuilder:
    """Builder for constructing POML-enhanced prompts"""
    
    @staticmethod
    def build_system_prompt(role: str = "assistant", domain: str = "general") -> str:
        """
        Build a POML-enhanced system prompt with structured reasoning framework
        """
        return f"""You are an advanced AI {role} that uses the POML (Parrot-Olympiad-Math-Logic) reasoning framework.

## POML Reasoning Framework

### Core Principles:
1. **Parrot Phase**: First, accurately understand and restate the problem
2. **Olympiad Phase**: Apply systematic problem-solving strategies
3. **Math Phase**: Use logical, step-by-step reasoning
4. **Logic Phase**: Verify conclusions through formal validation

### Your Reasoning Process:

#### 1. UNDERSTAND (Parrot)
- Carefully read and comprehend the input
- Identify key information, constraints, and requirements
- Restate the problem in your own words to ensure understanding
- Ask clarifying questions if needed

#### 2. DECOMPOSE (Olympiad)
- Break down complex problems into smaller, manageable components
- Identify patterns, relationships, and dependencies
- Create a structured approach to solving each component
- Consider multiple solution strategies

#### 3. ANALYZE (Math)
- Apply logical reasoning to each component
- Use appropriate analytical tools and methods
- Show your work step-by-step
- Maintain mathematical/logical rigor

#### 4. SYNTHESIZE (Math-Logic)
- Combine component solutions into a coherent whole
- Ensure consistency across all parts
- Build the complete solution systematically
- Document your reasoning chain

#### 5. VERIFY (Logic)
- Check your solution against the original requirements
- Validate each step of your reasoning
- Test edge cases and boundary conditions
- Ensure logical consistency

#### 6. REFLECT
- Consider alternative approaches
- Identify potential improvements
- Learn from the problem-solving process
- Document insights for future reference

### Domain-Specific Context: {domain}

You are specialized in {domain}. Apply domain-specific knowledge while maintaining the POML framework's systematic approach.

### Output Format:
When solving problems, structure your response as follows:

**Understanding**: [Restate the problem and key requirements]

**Approach**: [Outline your solution strategy]

**Solution**: [Step-by-step solution with clear reasoning]

**Verification**: [Validate your solution]

**Reflection**: [Key insights and potential improvements]

Remember: Quality of reasoning is more important than speed. Take time to think through problems systematically."""

    @staticmethod
    def build_task_prompt(task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a POML-enhanced prompt for a specific task
        """
        prompt = f"""## Task
{task}

## POML Analysis Framework

### Step 1: Understanding
Let me first understand what you're asking for:
- Primary objective: [Identify main goal]
- Key requirements: [List specific requirements]
- Constraints: [Note any limitations or constraints]
- Success criteria: [Define what constitutes a successful solution]

### Step 2: Decomposition
Breaking this down into manageable components:
- Component A: [First major component]
- Component B: [Second major component]
- Dependencies: [How components relate]

### Step 3: Analytical Approach
For each component, I'll apply:
- Relevant methods: [Specific techniques or algorithms]
- Tools needed: [Required resources or tools]
- Step-by-step process: [Detailed approach]

### Step 4: Synthesis Strategy
Combining the components:
- Integration plan: [How to combine solutions]
- Consistency checks: [Ensure alignment]
- Final assembly: [Complete solution structure]

### Step 5: Verification Plan
To ensure correctness:
- Test cases: [Specific scenarios to validate]
- Edge cases: [Boundary conditions to check]
- Performance criteria: [Metrics to evaluate]"""

        if context:
            prompt += "\n\n## Additional Context\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"

        return prompt

    @staticmethod
    def build_reasoning_chain_prompt(problem: str, steps: List[str]) -> str:
        """
        Build a prompt that guides through a specific reasoning chain
        """
        prompt = f"""## Problem
{problem}

## POML Reasoning Chain

Let's solve this step-by-step using the POML framework:

### Chain of Thought:
"""
        for i, step in enumerate(steps, 1):
            prompt += f"""
**Step {i}**: {step}
- Analysis: [Detailed analysis for this step]
- Reasoning: [Logical justification]
- Result: [Outcome of this step]
- Validation: [Check correctness]
"""
        
        prompt += """
### Final Synthesis:
Combining all steps to reach the conclusion:
- Integration: [How steps connect]
- Final answer: [Complete solution]
- Confidence level: [Assessment of solution quality]
- Alternative approaches: [Other possible solutions]"""
        
        return prompt

    @staticmethod
    def build_verification_prompt(solution: str, criteria: List[str]) -> str:
        """
        Build a prompt for verifying a solution using POML logic phase
        """
        prompt = f"""## Solution to Verify
{solution}

## POML Verification Protocol

### Verification Criteria:
"""
        for i, criterion in enumerate(criteria, 1):
            prompt += f"{i}. {criterion}\n"
        
        prompt += """
### Verification Process:

#### 1. Correctness Check
- Does the solution address the original problem?
- Are all requirements satisfied?
- Is the logic sound and consistent?

#### 2. Completeness Check
- Are all edge cases handled?
- Is the solution comprehensive?
- Are there any missing components?

#### 3. Optimization Check
- Is this the most efficient approach?
- Can the solution be simplified?
- Are there performance improvements possible?

#### 4. Robustness Check
- How does the solution handle errors?
- Is it resilient to unexpected inputs?
- Are there potential failure modes?

### Verification Results:
- Overall assessment: [Pass/Fail with reasoning]
- Strengths: [What works well]
- Weaknesses: [Areas for improvement]
- Recommendations: [Suggested enhancements]"""
        
        return prompt

    @staticmethod
    def build_reflection_prompt(experience: str, outcomes: Dict[str, Any]) -> str:
        """
        Build a prompt for reflection and learning using POML
        """
        prompt = f"""## Experience
{experience}

## Outcomes
"""
        for key, value in outcomes.items():
            prompt += f"- {key}: {value}\n"
        
        prompt += """
## POML Reflection Framework

### 1. What Worked Well
- Successful strategies: [Identify effective approaches]
- Key insights: [Important discoveries]
- Reusable patterns: [Patterns to apply in future]

### 2. Challenges Encountered
- Difficulties: [Problems faced]
- Root causes: [Why these occurred]
- Mitigation strategies: [How to avoid in future]

### 3. Learning Points
- New knowledge: [What was learned]
- Skill development: [Capabilities improved]
- Framework refinements: [How to enhance POML application]

### 4. Future Applications
- Similar problems: [Where this applies]
- Transferable skills: [Generalizable lessons]
- Process improvements: [How to do better next time]

### 5. Knowledge Integration
- Connection to existing knowledge: [How this relates to prior experience]
- Updated mental models: [How understanding has evolved]
- Documentation for future reference: [Key takeaways to remember]"""
        
        return prompt


class POMLExamples:
    """Collection of POML prompt examples for different scenarios"""
    
    @staticmethod
    def get_code_generation_example() -> str:
        """Example of POML-enhanced code generation prompt"""
        return """## Task: Generate a Python function to calculate fibonacci numbers

### POML Analysis:

**Understanding**: 
- Need to create a function that calculates the nth Fibonacci number
- Fibonacci sequence: each number is sum of two preceding ones
- Starts with 0, 1, 1, 2, 3, 5, 8, 13...

**Decomposition**:
1. Handle base cases (n=0, n=1)
2. Implement recursive or iterative logic
3. Consider optimization (memoization/dynamic programming)
4. Add input validation

**Solution Approach**:
```python
def fibonacci(n: int, memo: dict = None) -> int:
    \"\"\"
    Calculate the nth Fibonacci number using memoization.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        memo: Dictionary for memoization (internal use)
    
    Returns:
        The nth Fibonacci number
    
    Raises:
        ValueError: If n is negative
    \"\"\"
    # Input validation
    if n < 0:
        raise ValueError("n must be non-negative")
    
    # Initialize memoization dictionary
    if memo is None:
        memo = {}
    
    # Base cases
    if n in (0, 1):
        return n
    
    # Check if already computed
    if n in memo:
        return memo[n]
    
    # Recursive calculation with memoization
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]
```

**Verification**:
- Base cases: fib(0)=0, fib(1)=1 ✓
- Sequence: fib(5)=5, fib(10)=55 ✓
- Performance: O(n) time with memoization ✓
- Edge cases: Handles negative input ✓

**Reflection**:
- Memoization prevents redundant calculations
- Could also implement iterative version for space efficiency
- Consider adding support for large numbers using arbitrary precision"""

    @staticmethod
    def get_problem_solving_example() -> str:
        """Example of POML-enhanced problem solving prompt"""
        return """## Problem: Design a rate limiter for an API

### POML Analysis:

**Understanding**:
- Need to limit API requests per user/IP
- Prevent abuse while allowing legitimate traffic
- Must be performant and scalable
- Common algorithms: Token bucket, Sliding window, Fixed window

**Decomposition**:
1. Choose appropriate algorithm
2. Design data structure for tracking
3. Implement request validation logic
4. Handle edge cases and errors
5. Consider distributed systems aspects

**Analytical Solution**:

Component 1: Algorithm Selection
- Token Bucket: Best for allowing bursts
- Sliding Window: Most accurate but memory intensive
- Fixed Window: Simple but has boundary issues
- Decision: Token Bucket for flexibility

Component 2: Implementation Design
```python
class TokenBucketRateLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
    
    def _refill(self):
        now = time.time()
        tokens_to_add = (now - self.last_refill) * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def allow_request(self, tokens_required: int = 1) -> bool:
        self._refill()
        if self.tokens >= tokens_required:
            self.tokens -= tokens_required
            return True
        return False
```

**Verification**:
- Burst handling: Allows up to capacity requests ✓
- Rate limiting: Enforces average rate ✓
- Thread safety: Need to add locks ⚠️
- Distributed: Need Redis/shared storage ⚠️

**Reflection**:
- Token bucket provides good balance
- Need synchronization for production
- Consider using Redis for distributed systems
- Could add request queuing for better UX"""


# Helper functions for using POML prompts with LangGraph

def create_poml_enhanced_prompt(
    base_prompt: str,
    reasoning_steps: Optional[List[ReasoningStep]] = None,
    include_examples: bool = False
) -> str:
    """
    Enhance a base prompt with POML structure
    
    Args:
        base_prompt: The original prompt
        reasoning_steps: Specific reasoning steps to emphasize
        include_examples: Whether to include example applications
    
    Returns:
        POML-enhanced prompt string
    """
    if reasoning_steps is None:
        reasoning_steps = list(ReasoningStep)
    
    builder = POMLPromptBuilder()
    enhanced = builder.build_task_prompt(base_prompt)
    
    if include_examples:
        enhanced += "\n\n## Example Application:\n"
        enhanced += POMLExamples.get_code_generation_example()
    
    return enhanced


def validate_reasoning_chain(
    problem: str,
    solution: str,
    criteria: Optional[List[str]] = None
) -> str:
    """
    Create a validation prompt for a reasoning chain
    
    Args:
        problem: Original problem statement
        solution: Proposed solution
        criteria: Specific validation criteria
    
    Returns:
        Validation prompt string
    """
    if criteria is None:
        criteria = [
            "Correctness: Does the solution solve the stated problem?",
            "Completeness: Are all requirements addressed?",
            "Efficiency: Is the solution optimal?",
            "Clarity: Is the reasoning clear and logical?",
            "Robustness: Does it handle edge cases?"
        ]
    
    builder = POMLPromptBuilder()
    return builder.build_verification_prompt(
        f"Problem: {problem}\n\nProposed Solution: {solution}",
        criteria
    )