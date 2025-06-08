"""
infer_behavior_shares.py
Estimate how much of each actor's behavior is allocated to each strategy at epoch t.
Behavior shares must sum to 1.0 for each actor and no strategy can have 0% share.
"""

from __future__ import annotations

import os
from typing import List
import json

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Import the existing models from infer_payoffs - no need to extend them now
from api.openai.infer_payoffs import Strategy, ActorEntry

class BehaviorSharesResponse(BaseModel):
    actors: List[ActorEntry] = Field(description="List of actors with behavior share data")

# Prompt template
_BEHAVIOR_SHARES_PROMPT = """
You are a UK policy analyst using evolutionary game theory to model how actors allocate their behavior across different strategies.

**Context**
The target problem is: {problem_description}

You need to estimate how each actor currently allocates their behavior across their three strategies at epoch 0 (present day). Use current, up-to-date information about each actor's actual policies, spending, and actions in the UK.

**Task**
For each strategy in the JSON below, add the field:

**behavior_share_epoch_0**: The proportion of the actor's total relevant behavior currently allocated to this strategy
- Must be a decimal between 0.001 and 1.000 (no strategy can have 0% share)
- All three strategies for each actor MUST sum to exactly 1.000
- Use exactly 3 decimal places
- Base estimates on current, observable behavior in the UK (2024-2025)

**Research Guidelines**
Search for and use current information about each actor's:
- Recent policy announcements and implementations
- Current budget allocations and spending priorities
- Active programs and initiatives
- Public statements backed by concrete actions
- Resource deployment patterns over the last 1-2 years

**Allocation Logic**
Consider that actors typically:
- Allocate MORE behavior to strategies with higher payoffs (unless constrained)
- May be constrained by political feasibility, existing commitments, or institutional capacity
- Often maintain some level of activity across all strategies rather than going "all in" on one
- Balance short-term political considerations with longer-term effectiveness

**High-commitment strategies** tend to get smaller behavior shares initially due to political/resource constraints, even if they have good long-term payoffs.

**Medium and low-commitment strategies** often get larger current behavior shares as they're easier to implement and maintain.

**Realism Requirements**
- Government departments: Look at actual current spending patterns and active policy initiatives
- Local authorities: Examine their current service delivery and investment priorities  
- Charities/NGOs: Review their current campaign focus and resource allocation
- Private sector: Assess their actual CSR spending and program emphasis (not just PR)

**Quality Control**
- Verify that each actor's three behavior_share_epoch_0 values sum to exactly 1.000
- Ensure no strategy has a share below 0.001 or above 1.000
- Cross-reference with recent UK policy developments and current events
- Use the most recent available data (prioritize 2024-2025 information)

**Input Data**
{actors_json}

**Output Requirements**
- Return a JSON object with an "actors" field containing the array of actors
- Each actor should have behavior_share_epoch_0 added to each strategy
- Keep all existing fields unchanged
- Ensure all behavior shares sum to 1.000 for each actor
- Use exactly 3 decimal places
- Base estimates on current, verifiable UK policy and spending data

Return the result in this exact format:
{{
  "actors": [
    // ... array of actor objects with behavior_share_epoch_0 added to each strategy
  ]
}}

{format_instructions}
"""

# LangChain helpers
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name="gpt-4o",  # Using gpt-4o which has internet access
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

def _get_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=BehaviorSharesResponse)

def _get_behavior_shares_chain():
    """Return an LLM chain that estimates behavior shares."""
    llm = _get_llm()
    parser = _get_parser()
    prompt = ChatPromptTemplate.from_template(
        _BEHAVIOR_SHARES_PROMPT,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt | llm | parser

# Public API
def infer_behavior_shares(problem_description: str, actors_with_payoffs: List[ActorEntry], epoch: int = 0) -> List[ActorEntry]:
    """
    Parameters
    ----------
    problem_description : str
        The same free-text context fed into earlier steps.
    actors_with_payoffs : List[ActorEntry]
        Actors with payoff data already calculated.
    epoch : int
        The time epoch (currently only supports epoch 0).

    Returns
    -------
    List[ActorEntry]
        Actors with behavior_share_epoch_0 added to each strategy.
    """
    
    # Debug: Check input data
    print(f"DEBUG: Received {len(actors_with_payoffs)} actors for behavior shares")
    for i, actor in enumerate(actors_with_payoffs):
        print(f"DEBUG: Actor {i}: {actor.actor_id}, strategies: {len(actor.strategies)}")
        for j, strategy in enumerate(actor.strategies):
            current_share = getattr(strategy, 'behavior_share_epoch_0', None)
            print(f"DEBUG: Strategy {j}: {strategy.id}, current behavior_share: {current_share}")
    
    # Convert actors to JSON for the prompt
    actors_json = json.dumps([actor.model_dump() for actor in actors_with_payoffs], indent=2)
    
    chain = _get_behavior_shares_chain()
    
    try:
        print(f"Sending behavior shares request to OpenAI for {len(actors_with_payoffs)} actors...")
        result = chain.invoke({
            "problem_description": problem_description,
            "actors_json": actors_json
        })
        print(f"Received behavior shares from OpenAI: {len(result.actors) if result and result.actors else 0} actors")
        
        if result and result.actors:
            # Debug: Check output data
            print("DEBUG: Behavior shares results:")
            for i, actor in enumerate(result.actors):
                print(f"DEBUG: Actor {i}: {actor.actor_id}")
                total_share = 0
                for j, strategy in enumerate(actor.strategies):
                    share = getattr(strategy, 'behavior_share_epoch_0', None)
                    if share is not None:
                        total_share += share
                    print(f"DEBUG: Strategy {j}: {strategy.id}, behavior_share: {share}")
                print(f"DEBUG: Total share for {actor.actor_id}: {total_share}")
            
            # Validate that behavior shares sum to 1.0 for each actor
            for actor in result.actors:
                total_share = 0
                for strategy in actor.strategies:
                    share = getattr(strategy, 'behavior_share_epoch_0', None)
                    if share is not None:
                        total_share += share
                
                if abs(total_share - 1.0) > 0.001:  # Allow small floating point errors
                    print(f"Warning: Actor {actor.actor_id} behavior shares sum to {total_share:.3f}, not 1.000")
            
            return result.actors
        else:
            print("DEBUG: No result or no actors in result")
            return actors_with_payoffs
            
    except Exception as e:
        print(f"Error in behavior shares chain: {e}")
        import traceback
        traceback.print_exc()
        return actors_with_payoffs

# Example CLI usage (can be zapped later)
if __name__ == "__main__":
    import json, sys
    _problem = sys.argv[1]
    _actors = json.loads(sys.stdin.read())
    # Convert dict to ActorEntry objects
    actor_objects = [ActorEntry(**actor) for actor in _actors]
    enriched = infer_behavior_shares(_problem, actor_objects)
    print(json.dumps([e.model_dump() for e in enriched], indent=2))