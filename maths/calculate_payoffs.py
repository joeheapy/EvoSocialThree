"""
calculate_payoffs.py
Convert OpenAI payoff results into Pandas DataFrames and calculate overall payoffs
using the algorithm from pay-off-formula.md.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from api.openai.infer_payoffs import ActorEntry

# Constants
EPSILON = 1e-4  # tiny positive constant to avoid zero payoffs and divide by zero errors later.


def convert_to_dataframes(actors: List[ActorEntry]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert ActorEntry list to two DataFrames: actors_df and strategies_df.
    """
    # Prepare data for actors DataFrame
    actors_data = []
    strategies_data = []
    
    for g, actor in enumerate(actors):
        # Use actor.weight instead of actor.strategies[0].weight
        actors_data.append({
            "g": g,
            "actor_id": actor.actor_id,
            "sector": actor.sector,
            "weight": actor.weight  # Changed from actor.strategies[0].weight
        })
        
        # Add strategies to strategies DataFrame WITH ACTOR'S WEIGHT (not individual strategy weights)
        commitment_to_k = {"High": 0, "Medium": 1, "Low": 2}
        
        for strategy in actor.strategies:
            k = commitment_to_k.get(strategy.commitment_level, 0)
            strategies_data.append({
                "g": g,
                "k": k,
                "strategy_id": strategy.id,
                "commitment": strategy.commitment_level,
                "delta": strategy.delta,
                "cost": strategy.private_cost,
                "weight": actor.weight,  # Use ACTOR's weight, not strategy.weight
                "description": strategy.description
            })
    
    # Create DataFrames
    actors_df = pd.DataFrame(actors_data).set_index("g")
    strategies_df = pd.DataFrame(strategies_data).set_index(["g", "k"])
    
    return actors_df, strategies_df


def calculate_payoffs_epoch_0(actors_df: pd.DataFrame, strategies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate payoffs for epoch 0 using actor-level weights.
    
    for each actor g in 1 … G:
        for each strategy k in 1 … K:
            social_gain  =  weight[g]  *  (- delta[g][k])  # Note: weight per actor now
            raw_payoff   =  social_gain  -  cost[g][k]
            payoff[g][k][0] =  max(raw_payoff + EPSILON, EPSILON)
    
    Parameters
    ----------
    actors_df : pd.DataFrame
        DataFrame with actor information including weights
    strategies_df : pd.DataFrame
        DataFrame with strategy information including delta and cost
        
    Returns
    -------
    pd.DataFrame
        strategies_df with added 'payoff_epoch_0' column
    """
    # Create a copy to avoid modifying the original
    result_df = strategies_df.copy()
    
    # DEBUG: Print the dataframe with actor-level weights
    print("DEBUG: Strategies DataFrame with actor-level weights:")
    print(result_df.head(10))
    print()
    
    # Apply the algorithm using actor-level weights
    # social_gain = weight[g] * (- delta[g][k])  # Note: weight per actor now
    result_df['social_gain'] = result_df['weight'] * (-result_df['delta'])
    
    # DEBUG: Print social gain calculations
    print("DEBUG: Social gain calculations:")
    for idx, row in result_df.iterrows():
        print(f"  {row['strategy_id']}: weight={row['weight']:.3f} × (-delta={-row['delta']:.3f}) = {row['social_gain']:.6f}")
    print()
    
    # raw_payoff = social_gain - cost[g][k]
    result_df['raw_payoff'] = result_df['social_gain'] - result_df['cost']
    
    # DEBUG: Print raw payoff calculations
    print("DEBUG: Raw payoff calculations:")
    for idx, row in result_df.iterrows():
        print(f"  {row['strategy_id']}: social_gain={row['social_gain']:.6f} - cost={row['cost']:.3f} = {row['raw_payoff']:.6f}")
    print()
    
    # payoff[g][k][0] = max(raw_payoff + EPSILON, EPSILON)
    result_df['payoff_epoch_0'] = np.maximum(result_df['raw_payoff'] + EPSILON, EPSILON)
    
    # DEBUG: Print final payoff calculations
    print("DEBUG: Final payoff calculations:")
    for idx, row in result_df.iterrows():
        print(f"  {row['strategy_id']}: max({row['raw_payoff']:.6f} + {EPSILON:.3f}, {EPSILON:.3f}) = {row['payoff_epoch_0']:.6f}")
    print()
    
    # Drop intermediate calculation columns
    result_df = result_df.drop(['social_gain', 'raw_payoff'], axis=1)
    
    return result_df


def add_payoffs_to_actors(actors: List[ActorEntry], strategies_df: pd.DataFrame) -> List[ActorEntry]:
    """
    Add calculated payoffs back to the ActorEntry objects for use in templates.
    """
    # DEBUG: Print what OpenAI originally returned
    print("DEBUG: Original data from OpenAI:")
    for g, actor in enumerate(actors):
        print(f"  Actor {g} ({actor.actor_id}): weight={actor.weight:.3f}")
        for strategy in actor.strategies:
            print(f"    {strategy.id}: delta={strategy.delta:.3f}, cost={strategy.private_cost:.3f}")
    print()
    
    # Create a copy of actors to avoid modifying the original
    updated_actors = []
    
    for g, actor in enumerate(actors):
        # Create new actor with updated strategies
        updated_strategies = []
        commitment_to_k = {"High": 0, "Medium": 1, "Low": 2}
        
        for strategy in actor.strategies:
            k = commitment_to_k.get(strategy.commitment_level, 0)
            
            # Get payoff from DataFrame
            try:
                payoff = strategies_df.loc[(g, k), 'payoff_epoch_0']
                print(f"DEBUG: Found payoff for {strategy.id} at ({g},{k}): {payoff:.6f}")
            except KeyError:
                payoff = EPSILON  # Fallback
                print(f"DEBUG: KeyError for {strategy.id} at ({g},{k}), using EPSILON: {payoff:.6f}")
            
            # Create new strategy with payoff using model_copy with update
            updated_strategy = strategy.model_copy(update={'payoff_epoch_0': payoff})
            updated_strategies.append(updated_strategy)
        
        # Create updated actor
        updated_actor = actor.model_copy(update={'strategies': updated_strategies})
        updated_actors.append(updated_actor)
    
    return updated_actors


def process_payoffs_data(actors: List[ActorEntry]) -> Tuple[pd.DataFrame, pd.DataFrame, List[ActorEntry]]:
    """
    Main function to convert ActorEntry list to DataFrames with calculated payoffs
    and return updated actors with payoff information.
    
    Parameters
    ----------
    actors : List[ActorEntry]
        List of actors with strategies from infer_payoffs
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[ActorEntry]]
        (actors_df, strategies_df_with_payoffs, updated_actors)
    """
    # Convert to DataFrames
    actors_df, strategies_df = convert_to_dataframes(actors)
    
    # Calculate payoffs using the algorithm from pay-off-formula.md
    strategies_df_with_payoffs = calculate_payoffs_epoch_0(actors_df, strategies_df)
    
    # Add payoffs back to ActorEntry objects for template use
    updated_actors = add_payoffs_to_actors(actors, strategies_df_with_payoffs)
    
    return actors_df, strategies_df_with_payoffs, updated_actors

