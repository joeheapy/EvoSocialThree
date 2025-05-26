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

# Data models – Strategy now has three extra numeric fields
class Strategy(BaseModel):
    id: str = Field(description="Strategy ID in the format '[actorID-index]' (e.g., 'CG-1')")
    description: str = Field(description="Brief description of the strategy")
    commitment_level: str = Field(description="Level of commitment: 'High', 'Medium', or 'Low'")
    # Newly inferred numbers
    delta: float = Field(
        description="Estimated change the strategy causes in the target metric "
                    "(negative number improves the situation, positive makes it worse)."
    )
    private_cost: float = Field(
        description="Estimated financial or political cost to the actor, "
                    "expressed in arbitrary 'cost units'."
    )
    payoff: float = Field(
        description="Net payoff combining the actor's private cost and its weight "
                    "on the social outcome. Must be strictly positive."
    )


class ActorEntry(BaseModel):
    sector: str
    role_in_alleviating_child_poverty: str
    actor_index: str
    actor_id: str
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

You will estimate three numbers for each strategy:

**Task**  
For each strategy in the JSON below, add these three numeric fields:

1. **delta**: How much the strategy changes the target system objective: {system_objective}
   - Negative values = improvement (moves toward the objective)
   - Positive values = worsening (moves away from the objective)  
   - Scale: typically between -0.1 to +0.1
   - Use exactly 3 decimal places

2. **private_cost**: Estimate a cost vector for the strategy (financial, reputational or political)
   - Higher values = more expensive for the actor
   - Scale: typically between 0.001 to 0.1
   - Return one value per strategy§
   - Use exactly 3 decimal places

3. **payoff**: Net benefit the actor receives (must be positive)
   - Combines social benefit and private cost considerations
   - Higher when delta is very negative (helps achieve the system objective) 
   - Lower when private_cost is high
   - Always > 0.01 minimum
   - Use exactly 3 decimal places

**Input Data**x
{actors_block}

**Output Requirements**
- Return the EXACT same JSON structure with the three new numeric fields added to each strategy
- Keep all existing fields (id, description, commitment_level) unchanged
- Ensure all payoff values are strictly positive
- Use exactly 2 decimal places for all numbers

Return the result as a JSON object with an "actors" field containing the array of ActorEntry objects with the enhanced strategies.

{format_instructions}
"""

# LangChain helpers
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.3,
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
        Actors with `delta`, `private_cost`, and `payoff`
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
        return result.actors if result else []
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
