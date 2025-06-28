"""
infer_payoffs.py
Generate Δ-effect, private cost, and overall payoff for every
(actor, strategy) tuple produced by the earlier `identify_actors` step.

The file exposes one public function:
    infer_payoffs(problem_description: str, actors_json: str) -> List[ActorEntry]
"""

from __future__ import annotations

import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Input models (for parsing existing data)
class StrategyInput(BaseModel):
    id: str = Field(description="Strategy ID in the format '[actorID-index]' (e.g., 'CG-1')")
    description: str = Field(description="Brief description of the strategy")
    commitment_level: str = Field(description="Level of commitment: 'High', 'Medium', or 'Low'")
    # Optional fields that might not exist in input
    delta: float = Field(default=None, description="Estimated change in target metric")
    private_cost: float = Field(default=None, description="Estimated cost to actor")
    payoff_epoch_0: float = Field(default=None, description="Calculated payoff")
    behavior_share_epoch_0: float = Field(default=None, description="Behavior share")

class ActorInputEntry(BaseModel):
    sector: str
    role_in_alleviating_child_poverty: str
    actor_index: str
    actor_id: str
    weight: float = Field(default=None, description="Actor weight - will be inferred if not present")
    strategies: List[StrategyInput]

class PayoffsInputResponse(BaseModel):
    actors: List[ActorInputEntry] = Field(description="List of input actors")

# Output models (existing Strategy and ActorEntry remain the same)
class Strategy(BaseModel):
    id: str = Field(description="Strategy ID in the format '[actorID-index]' (e.g., 'CG-1')")
    description: str = Field(description="Brief description of the strategy")
    commitment_level: str = Field(description="Level of commitment: 'High', 'Medium', or 'Low'")
    delta: float = Field(description="Estimated change in target metric")
    private_cost: float = Field(description="Estimated cost to actor")
    payoff_epoch_0: float = Field(default=None, description="Calculated payoff")
    behavior_share_epoch_0: float = Field(default=None, description="Behavior share")

class ActorEntry(BaseModel):
    sector: str
    role_in_alleviating_child_poverty: str
    actor_index: str
    actor_id: str
    weight: float = Field(description="How strongly the actor values improvement in the outcome target")
    strategies: List[Strategy]


# Container class for the parser
class PayoffsResponse(BaseModel):
    actors: List[ActorEntry] = Field(description="List of actors with payoff data")


# Prompt template
_PAYOFF_PROMPT = """
You are a UK social-policy analyst using evolutionary game theory to model social interventions.

**Context**
The target is to reduce this specific social problem: {problem_description}. 

The specific system objective being optimized is: {system_objective}

You will estimate two numbers for each strategy and one weight per actor:

**Task**  
For each strategy in the JSON below, add these two numeric fields:

1. **delta**: How much the strategy changes the target system objective: {system_objective}
   - Negative values = improvement (moves toward the objective)
   - Positive values = worsening (moves away from the objective)  
   - Scale: typically between -0.15 to +0.05 (stronger effects, more negative for improvements)
   - High commitment strategies should have larger negative deltas (-0.08 to -0.15)
   - Medium commitment strategies: (-0.04 to -0.08)
   - Low commitment strategies: (-0.01 to -0.04)
   - Use exactly 3 decimal places

2. **private_cost**: Estimate the financial, reputational or political cost of the strategy to the actor
   - Higher values = more costly for the actor
   - Scale: value between 0.005 to 0.080 (reduced maximum to create better cost-benefit ratio)
   - High commitment strategies: 0.020-0.080
   - Medium commitment strategies: 0.010-0.040
   - Low commitment strategies: 0.005-0.020
   - Use exactly 3 decimal places

**Actor Weight (one per actor)**
For each actor, add this field at the actor level:

3. **weight**: How strongly this actor genuinely values improvement in the outcome target
   - Research the actor's historical actions, policies, and investments in the UK related to this issue
   - Consider their track record of sustained commitment vs. rhetoric
   - Look at their actual resource allocation and policy priorities
   - Value between 0.0 and 1.0 where:
     * 0.8-1.0 = Strong genuine commitment (consistent actions over time)
     * 0.5-0.7 = Moderate genuine interest (some actions but not priority)
     * 0.2-0.4 = Limited genuine interest (mostly rhetoric, minimal action)
     * 0.0-0.1 = Minimal genuine interest (indifferent or conflicted priorities)
   - Base estimates on observable UK policy actions, not stated intentions
   - Use exactly 3 decimal places
   - **This weight applies to ALL strategies for this actor**

**Commitment Level Guidelines:**
- **High commitment strategies**: Should have strong negative deltas (-0.08 to -0.15) but also high costs (0.040-0.080)
- **Medium commitment strategies**: Moderate negative deltas (-0.04 to -0.08) with moderate costs (0.020-0.040)  
- **Low commitment strategies**: Small negative deltas (-0.01 to -0.04) with low costs (0.005-0.020)

This ensures a realistic trade-off where more ambitious strategies cost more but also achieve more.

**Balance Guidelines:**
- Ensure that for each actor, at least one strategy has positive raw payoff (where weight × (-delta) > private_cost)
- Create a realistic cost-benefit spectrum across commitment levels
- Aim for calculated payoffs distributed across a meaningful range, not clustered at minimum values
- Consider that actors need at least some viable strategies to participate meaningfully

**Research Guidelines for Weight Estimation**
For each actor, consider their UK track record:
- Government departments: Look at budget allocations, policy consistency, and outcomes
- Local authorities: Examine their actual spending priorities and policy implementation
- Charities/NGOs: Review their resource allocation and campaign focus areas
- Private sector: Assess their genuine CSR investments vs. marketing claims
- Consider whether their actions align with stated commitments over multiple years

**Input Data**
{actors_block}

**Output Requirements**
- Return the EXACT same JSON structure with:
  - Two new numeric fields (delta, private_cost) added to each strategy
  - One weight field added at the actor level (NOT at strategy level)
- Keep all existing fields (id, description, commitment_level) unchanged
- Ensure all weight values are between 0.0 and 1.0
- Use exactly 3 decimal places for all numbers
- Follow the commitment level guidelines to create realistic cost-benefit relationships
- **Important**: The weight field should be at the actor level and apply to all strategies for that actor

Return the result as a JSON object with an "actors" field containing the array of ActorEntry objects with the enhanced strategies.

{format_instructions}
"""

# LangChain helpers
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

def _get_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=PayoffsResponse)

def _get_payoff_chain():
    """Return an LLM chain that maps actors → payoffs."""
    llm = _get_llm()
    parser = _get_parser()
    prompt = ChatPromptTemplate.from_template(
        _PAYOFF_PROMPT,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt | llm | parser

# Public API
def infer_payoffs(problem_description: str, actors_json: str, system_objective: str = "the social problem") -> List[ActorEntry]:
    """
    Parameters
    ----------
    problem_description : str
        The same free-text context fed into the actor-identification step.
    actors_json : str
        The raw JSON (list of ActorEntry) returned by that step.
    system_objective : str
        The selected system objective/target metric.

    Returns
    -------
    List[ActorEntry]
        Actors with `delta`, `private_cost`, `weight`, and calculated `payoff_epoch_0`
        for each of their three strategies.
    """
    chain = _get_payoff_chain()
    
    try:
        print(f"Sending to OpenAI:\n{actors_json[:500]}...")  # Debug print
        result = chain.invoke({
            "problem_description": problem_description,
            "system_objective": system_objective,
            "actors_block": actors_json
        })
        print(f"Received from OpenAI: {len(result.actors) if result and result.actors else 0} actors")  # Debug print
        
        # Calculate payoffs using the algorithm from pay-off-formula.md
        if result and result.actors:
            from maths.calculate_payoffs import process_payoffs_data
            _, _, updated_actors = process_payoffs_data(result.actors)
            return updated_actors
        else:
            return []
            
    except Exception as e:
        print(f"Error in payoffs chain: {e}")
        import traceback
        traceback.print_exc()
        return []

# Example CLI usage (can be zapped later)
if __name__ == "__main__":
    import json, sys
    _problem = sys.argv[1]
    _actors = json.loads(sys.stdin.read())
    enriched = infer_payoffs(_problem, json.dumps(_actors, indent=2))
    print(json.dumps([e.model_dump() for e in enriched], indent=2))
