import numpy as np
from typing import List, Optional, Tuple
from simulation import run_simulation, parse_rows_to_arrays, SimulationResult

def find_optimum_random(
    rows: List[List],
    P_baseline: float,
    P_target: float,
    max_epochs: int,
    *,
    scale: Optional[float] = None,
    subsidy_cap: float = 0.15,  # Search for subsidies
    penalty_cap: float = 0.10,  # Search for penalties
    trials: int = 10000,        # Number of random trials
    early_exit: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Optional[SimulationResult], Optional[np.ndarray]]:
    
    # Parse input data
    delta_raw, private_cost, weight, payoff_base, initial_shares, sector_names, strategy_ids = parse_rows_to_arrays(rows)
    
    G, K = delta_raw.shape
    rng = np.random.default_rng(seed)
    target_direction = P_target < P_baseline
    
    # Calculate strategy importance based on delta magnitude and current shares
    strategy_importance = np.abs(delta_raw) * initial_shares
    
    best_result = None
    best_pi = None
    best_budget = float('inf')
    
    for trial in range(trials):
        pi_sub = np.zeros((G, K))
        pi_pen = np.zeros((G, K))
        
        for g in range(G):
            for k in range(K):
                # Weight incentives by strategy importance
                importance_weight = strategy_importance[g, k] + 0.1
                
                # More aggressive incentives for helpful strategies
                if target_direction and delta_raw[g, k] < 0:  # Want to reduce P
                    # Give larger subsidies to more impactful strategies
                    max_subsidy = subsidy_cap * min(2.0, importance_weight * 5.0)  # Increased multiplier
                    pi_sub[g, k] = rng.uniform(subsidy_cap * 0.3, min(max_subsidy, subsidy_cap))  # Higher minimum
                elif target_direction and delta_raw[g, k] > 0:  # Penalize harmful strategies
                    max_penalty = penalty_cap * min(2.0, importance_weight * 3.0)
                    pi_pen[g, k] = rng.uniform(penalty_cap * 0.2, min(max_penalty, penalty_cap))
                elif not target_direction and delta_raw[g, k] > 0:  # Want to increase P
                    max_subsidy = subsidy_cap * importance_weight
                    pi_sub[g, k] = rng.uniform(0, min(max_subsidy, subsidy_cap))
                elif not target_direction and delta_raw[g, k] < 0:
                    max_penalty = penalty_cap * importance_weight
                    pi_pen[g, k] = rng.uniform(0, min(max_penalty, penalty_cap))
        
        # Calculate modified costs: cost' = private_cost - subsidies + penalties
        cost_override = private_cost - pi_sub + pi_pen
        cost_override = np.clip(cost_override, 0, None)  # Ensure non-negative
        
        # Calculate total budget
        total_budget = pi_sub.sum() + pi_pen.sum()
        
        try:
            # Run simulation with modified costs
            result = run_simulation(
                rows=rows,
                P_baseline=P_baseline,
                P_target=P_target,
                max_epochs=max_epochs,
                scale=scale,
                cost_override=cost_override
            )
            
            # Check if target was hit and budget is better
            if result.t_hit is not None and total_budget < best_budget:
                best_result = result
                best_pi = pi_sub - pi_pen  # Store net incentives (positive = subsidy, negative = penalty)
                best_budget = total_budget
                
                print(f"Trial {trial}: Success at epoch {result.t_hit}, budget {total_budget:.6f}")
                
                if early_exit:
                    break
            
        except Exception as e:
            # Skip failed simulations
            if trial % 1000 == 0:
                print(f"Trial {trial}: Simulation failed - {e}")
            continue
        
        if trial % 1000 == 0:
            print(f"Completed {trial} trials, best budget so far: {best_budget:.6f}")
    
    if best_result is not None:
        print(f"Search completed. Best solution: epoch {best_result.t_hit}, budget {best_budget:.6f}")
    else:
        print("Search completed. No solution found.")
    
    return best_result, best_pi