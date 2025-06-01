"""
analyze_payoffs.py
Generate strategic analysis of payoff patterns and actor behavior.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from api.openai.infer_payoffs import ActorEntry


class StrategyAnalysis(BaseModel):
    strategy_id: str = Field(description="Strategy ID (e.g., 'CG-1')")
    actor_sector: str = Field(description="Actor sector")
    commitment_level: str = Field(description="High/Medium/Low commitment")
    payoff_category: str = Field(description="Best Performer/Floor Strategy/Middle Ground/etc.")
    economic_attractiveness: str = Field(description="Assessment of economic viability")
    key_insights: List[str] = Field(description="List of 2-3 key analytical points")


class PayoffAnalysisResponse(BaseModel):
    overall_patterns: List[str] = Field(description="3-4 high-level patterns across all strategies")
    strategy_analyses: List[StrategyAnalysis] = Field(description="Analysis of 3-4 representative strategies")
    strategic_implications: List[str] = Field(description="2-3 implications for policy/intervention design")


class PayoffAnalysisContainer(BaseModel):
    """Container for payoff analysis results"""
    analysis: PayoffAnalysisResponse = Field(description="The analysis results")
    
    @property 
    def overall_patterns(self):
        return self.analysis.overall_patterns
    
    @property
    def strategy_analyses(self):
        return self.analysis.strategy_analyses
    
    @property
    def strategic_implications(self):
        return self.analysis.strategic_implications


# Prompt template
_ANALYSIS_PROMPT = """
You are a UK social-policy analyst specializing in evolutionary game theory and strategic behavior analysis.

**Context**
You have calculated payoffs for various actors and strategies to address: {problem_description}

The system objective being optimized is: {system_objective}

**Your Task**
Analyze the payoff patterns to provide strategic insights. Focus on:

1. **Overall Patterns**: Identify 3-4 high-level patterns across all strategies (e.g., "High-commitment strategies dominate", "Private sector shows limited engagement", etc.)

2. **Representative Strategy Analysis**: Select 3-4 strategies that illustrate different payoff patterns:
   - Choose strategies with different payoff ranges (high, medium, floor values)
   - Include different actor types and commitment levels
   - Focus on what makes each strategy economically attractive/unattractive

3. **Strategic Implications**: Identify 2-3 key implications for policy makers and intervention designers

**Payoff Data Analysis**
For each strategy, consider:
- **Economic attractiveness**: Is the payoff high enough to motivate action?
- **Cost-benefit ratio**: Does the social gain justify the private cost?
- **Actor commitment**: How does the actor's weight affect their motivation?
- **Commitment level**: How does High/Medium/Low commitment affect outcomes?

**Categories for Strategy Analysis**:
- "Best Performer" - highest payoffs, most attractive
- "Floor Strategy" - hitting minimum payoff floor (EPSILON)
- "Middle Ground" - moderate, viable payoffs
- "Marginal Strategy" - barely viable but positive
- "Cost-Prohibitive" - high costs relative to benefits

**Input Data**
{payoffs_data}

**Output Requirements**
- Provide concrete, actionable insights
- Reference specific payoff values and calculations
- Explain the economic logic behind actor decisions
- Use clear, policy-relevant language
- Focus on practical implications for intervention design

{format_instructions}
"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def _get_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=PayoffAnalysisResponse)


def _get_analysis_chain():
    """Return an LLM chain for payoff analysis."""
    llm = _get_llm()
    parser = _get_parser()
    prompt = ChatPromptTemplate.from_template(
        _ANALYSIS_PROMPT,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt | llm | parser


def format_payoffs_for_analysis(actors: List[ActorEntry]) -> str:
    """Format the payoff data for analysis."""
    formatted_data = []
    
    for actor in actors:
        actor_data = f"\n**{actor.sector} ({actor.actor_id})**"
        for strategy in actor.strategies:
            payoff = getattr(strategy, 'payoff_epoch_0', 0.0)
            actor_data += f"\n  - {strategy.id} ({strategy.commitment_level}): "
            actor_data += f"Δ={strategy.delta:.3f}, Cost={strategy.private_cost:.3f}, "
            actor_data += f"Weight={strategy.weight:.3f}, Payoff={payoff:.6f}"
            actor_data += f"\n    Description: {strategy.description}"
        formatted_data.append(actor_data)
    
    return "\n".join(formatted_data)


def analyze_payoffs(problem_description: str, actors: List[ActorEntry], system_objective: str = "the social problem") -> PayoffAnalysisContainer:
    """
    Generate strategic analysis of payoff patterns.
    """
    chain = _get_analysis_chain()
    
    try:
        payoffs_data = format_payoffs_for_analysis(actors)
        
        print(f"Analyzing payoffs for {len(actors)} actors...")
        print(f"Sending data: {payoffs_data[:200]}...")  # Debug print
        
        result = chain.invoke({
            "problem_description": problem_description,
            "system_objective": system_objective,
            "payoffs_data": payoffs_data
        })
        
        print(f"Generated analysis with {len(result.strategy_analyses)} strategy analyses")
        
        # Wrap in container
        return PayoffAnalysisContainer(analysis=result)
        
    except Exception as e:
        print(f"Error in payoff analysis: {e}")
        import traceback
        traceback.print_exc()
        # Return container with empty analysis on error
        empty_analysis = PayoffAnalysisResponse(
            overall_patterns=["Analysis temporarily unavailable"],
            strategy_analyses=[],
            strategic_implications=["Analysis temporarily unavailable"]
        )
        return PayoffAnalysisContainer(analysis=empty_analysis)