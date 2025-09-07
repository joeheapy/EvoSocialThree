import numpy as np
from typing import List, Optional, Tuple
from simulation import run_simulation, parse_rows_to_arrays, SimulationResult
from config import SIMULATION_TRIALS, SIMULATION_SUBSIDY_CAP, SIMULATION_PENALTY_CAP, SIMULATION_MAX_EPOCHS, SIMULATION_ADAPTIVE_CAP_S_MULTIPLIER, SIMULATION_ADAPTIVE_CAP_P_MULTIPLIER

def find_optimum_random(
    rows: List[List],
    P_baseline: float,
    P_target: float,
    max_epochs: int = SIMULATION_MAX_EPOCHS,
    *,
    scale: Optional[float] = None,
    subsidy_cap: float = SIMULATION_SUBSIDY_CAP,
    penalty_cap: float = SIMULATION_PENALTY_CAP,
    trials: int = SIMULATION_TRIALS,
    early_exit: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Optional[SimulationResult], Optional[np.ndarray], Optional[List[str]]]:
    
    # Parse input data
    delta_raw, private_cost, weight, payoff_base, initial_shares, sector_names, strategy_ids = parse_rows_to_arrays(rows)
    
    G, K = delta_raw.shape
    rng = np.random.default_rng(seed)
    target_direction = P_target < P_baseline
    
    # Calculate strategy importance
    strategy_importance = np.abs(delta_raw) * initial_shares
    
    # OPTION D: Scale incentive caps based on cost magnitudes and target distance
    avg_cost = np.mean(private_cost[private_cost > 0]) if np.any(private_cost > 0) else 1.0
    target_distance = abs(P_target - P_baseline)
    
    # Scale caps based on average costs and target ambition
    # If avg_cost is 0.1, factor is 1. If avg_cost is 10, factor is 100.
    cost_scale_factor = max(1.0, avg_cost / 0.1)
    # If target_distance is 10k, factor is 1. If 100k, factor is 10.
    distance_scale_factor = max(1.0, target_distance / 10000)
    
    # Adaptive caps - can become much larger for large-scale problems
    adaptive_subsidy_cap = subsidy_cap * cost_scale_factor * distance_scale_factor * SIMULATION_ADAPTIVE_CAP_S_MULTIPLIER
    adaptive_penalty_cap = penalty_cap * cost_scale_factor * distance_scale_factor * SIMULATION_ADAPTIVE_CAP_P_MULTIPLIER
    
    print(f"DEBUG INCENTIVES: Original caps - Subsidy: {subsidy_cap:.3f}, Penalty: {penalty_cap:.3f}")
    print(f"DEBUG INCENTIVES: Cost scale factor: {cost_scale_factor:.3f}, Distance scale factor: {distance_scale_factor:.3f}")
    print(f"DEBUG INCENTIVES: Adaptive caps - Subsidy: {adaptive_subsidy_cap:.3f}, Penalty: {adaptive_penalty_cap:.3f}")
    print(f"DEBUG INCENTIVES: Average private cost: {avg_cost:.6f}")
    
    best_result = None
    best_incentives = None
    best_budget = float('inf')
    best_distance = float('inf')  # Track best distance to target
    
    for trial in range(trials):
        pi_sub = np.zeros((G, K))
        pi_pen = np.zeros((G, K))
        
        for g in range(G):
            for k in range(K):
                importance_weight = strategy_importance[g, k] + 0.1
                
                # Determine if a strategy is helpful or harmful based on target direction
                is_helpful = (target_direction and delta_raw[g, k] < 0) or \
                             (not target_direction and delta_raw[g, k] > 0)
                is_harmful = (target_direction and delta_raw[g, k] > 0) or \
                             (not target_direction and delta_raw[g, k] < 0)

                if is_helpful:
                    # Generate a subsidy for helpful strategies
                    # The random subsidy is scaled by the adaptive cap and importance
                    max_s = adaptive_subsidy_cap * importance_weight * 5.0
                    pi_sub[g, k] = rng.uniform(0, min(max_s, adaptive_subsidy_cap * 2.0))
                elif is_harmful:
                    # Generate a penalty for harmful strategies
                    # The random penalty is scaled by the adaptive cap and importance
                    max_p = adaptive_penalty_cap * importance_weight * 4.0
                    pi_pen[g, k] = rng.uniform(0, min(max_p, adaptive_penalty_cap * 2.0))
        
        # Net incentives for simulation
        incentive_net = pi_sub - pi_pen
        total_budget = pi_sub.sum() + pi_pen.sum()
        
        # Debug incentive magnitudes on first trial
        if trial == 0:
            print(f"DEBUG INCENTIVES: Sample incentive_net values: {incentive_net.flatten()[:5].round(6)}")
            print(f"DEBUG INCENTIVES: Sample subsidy values: {pi_sub.flatten()[:5].round(6)}")
            print(f"DEBUG INCENTIVES: Sample penalty values: {pi_pen.flatten()[:5].round(6)}")
            print(f"DEBUG INCENTIVES: Total budget for trial 0: {total_budget:.6f}")
        
        try:
            result = run_simulation(
                rows=rows,
                P_baseline=P_baseline,
                P_target=P_target,
                max_epochs=max_epochs,
                scale=scale,
                incentive_adjustments=incentive_net
            )
            
            # Calculate distance to target
            final_value = result.P_series[-1]
            distance_to_target = abs(final_value - P_target)
            
            # Update best result based on success first, then distance, then budget
            is_better = False
            
            if result.t_hit is not None:  # This trial succeeded
                if best_result is None or best_result.t_hit is None:
                    # First successful trial, or previous best didn't succeed
                    is_better = True
                elif total_budget < best_budget:
                    # Both succeeded, prefer lower budget
                    is_better = True
            else:  # This trial didn't succeed
                if best_result is None or best_result.t_hit is None:
                    # No successful trial yet, prefer closer to target
                    if distance_to_target < best_distance:
                        is_better = True
                # If we already have a successful trial, don't replace it with a failed one
            
            if is_better:
                best_result = result
                best_incentives = incentive_net
                best_budget = total_budget
                best_distance = distance_to_target
                
                if result.t_hit is not None:
                    print(f"Trial {trial}: Success at epoch {result.t_hit}, budget {total_budget:.6f}")
                    if early_exit:
                        break
                else:
                    print(f"Trial {trial}: Best attempt so far, distance {distance_to_target:.6f}, budget {total_budget:.6f}")
            
        except Exception as e:
            if trial % 1000 == 0:
                print(f"Trial {trial}: Simulation failed - {e}")
            continue
        
        if trial % 1000 == 0:
            status = "successful" if best_result and best_result.t_hit is not None else "closest"
            print(f"Completed {trial} trials, best {status} distance: {best_distance:.6f}, budget: {best_budget:.6f}")
    
    final_status = "solution found" if best_result and best_result.t_hit is not None else "closest attempt found"
    print(f"Search completed. {final_status.capitalize()}")
    
    return best_result, best_incentives, sector_names