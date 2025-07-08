import numpy as np
import json
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import uuid
import os

# Constants
EPSILON = 0.01

class SimulationResult(BaseModel):
    P_series: List[float] = Field(description="Headline metric over time")
    share: List[List[List[float]]] = Field(description="Strategy shares [actor][strategy][epoch]")
    payoff: List[List[List[float]]] = Field(description="Payoffs [actor][strategy][epoch]")
    t_hit: Optional[int] = Field(description="Epoch when target was hit, None if not reached")
    
    class Config:
        arbitrary_types_allowed = True

def parse_rows_to_arrays(rows: List[List]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Convert list of rows to structured arrays."""
    # Group by actor
    actors = {}
    sector_names = []
    strategy_ids = []
    
    for row in rows:
        if len(row) < 9:
            print(f"Warning: Row has insufficient data: {row}")
            continue
            
        sector, strategy_id, commitment_level, delta, private_cost, weight, payoff_epoch_0, behavior_share_epoch_0, description = row
        
        if sector not in actors:
            actors[sector] = []
            if sector not in sector_names:
                sector_names.append(sector)
        
        # Handle null values and type conversion
        try:
            delta_val = float(delta) if delta is not None else 0.0
            private_cost_val = float(private_cost) if private_cost is not None else 0.0
            weight_val = float(weight) if weight is not None else 1.0
            payoff_base_val = float(payoff_epoch_0) if payoff_epoch_0 not in [None, 'N/A', 'null'] else 0.1
            behavior_share_val = float(behavior_share_epoch_0) if behavior_share_epoch_0 not in [None, 'N/A', 'null'] else 1/3
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert values in row {row}: {e}")
            continue
        
        actors[sector].append({
            'strategy_id': strategy_id,
            'delta': delta_val,
            'private_cost': private_cost_val,
            'weight': weight_val,
            'payoff_base': payoff_base_val,
            'behavior_share': behavior_share_val
        })
    
    if not actors:
        raise ValueError("No valid actor data found")
    
    G = len(actors)  # Number of actors
    K = max(len(strategies) for strategies in actors.values())  # Max strategies per actor
    if K > 3:
        K = 3  # Limit to 3 strategies
    
    # Initialize arrays - weight should be (G,) not (G, K)
    delta_raw = np.zeros((G, K))
    private_cost = np.zeros((G, K))
    weight = np.zeros(G)  # Changed: weight per actor, not per strategy
    payoff_base = np.zeros((G, K))
    initial_shares = np.zeros((G, K))
    
    # Fill arrays
    for g, sector in enumerate(sector_names):
        strategies = actors[sector][:K]
        # Set actor weight once per actor
        if strategies:
            weight[g] = strategies[0]['weight']  # All strategies have same actor weight
        
        for k, strategy in enumerate(strategies):
            delta_raw[g, k] = strategy['delta']
            private_cost[g, k] = strategy['private_cost']
            # Don't set weight[g, k] here - it's now weight[g]
            payoff_base[g, k] = strategy['payoff_base']
            initial_shares[g, k] = strategy['behavior_share']
    
    # Fill strategy_ids if not enough
    while len(strategy_ids) < K:
        strategy_ids.append(f"Strategy_{len(strategy_ids)+1}")
    
    # Normalize shares to sum to 1 per actor
    for g in range(G):
        row_sum = np.sum(initial_shares[g, :])
        if row_sum > 0:
            initial_shares[g, :] /= row_sum
        else:
            initial_shares[g, :] = 1/K
    
    return delta_raw, private_cost, weight, payoff_base, initial_shares, sector_names, strategy_ids


def compute_payoff_simple(payoff_base: np.ndarray, incentive_adjustments: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate current payoffs by adding incentives to the base payoff.
    This is much simpler and more correct.
    """
    # Start with the normalized base payoffs
    payoff = payoff_base.copy()
    
    # Add the effect of incentives if they exist
    if incentive_adjustments is not None:
        # Incentives are structured as (subsidy - penalty).
        # A subsidy increases payoff, a penalty decreases it.
        # The incentive_adjustments are already scaled, so we just add them.
        payoff += incentive_adjustments
        
    return payoff

# The value 0.01 is used as a tolerance for checking convergence
# This can be adjusted based on the precision required for the target metric.
def check_success_condition(P_t: float, P_target: float, P_baseline: float, tolerance_percent: float = 0.10) -> bool:
    """
    Check if current value is close enough to target using adaptive tolerance.
    
    Args:
        P_t: Current value
        P_target: Target value  
        P_baseline: Starting baseline value
        tolerance_percent: Tolerance as percentage of total change needed (default 10%)
    
    Returns:
        True if within tolerance of target
    """
    # Calculate total change needed from baseline to target
    total_change_needed = abs(P_target - P_baseline)
    
    # Handle edge case where baseline equals target
    if total_change_needed == 0:
        return abs(P_t - P_target) <= 0.001  # Very small absolute tolerance
    
    # Calculate tolerance as percentage of total change
    tolerance_value = tolerance_percent * total_change_needed
    
    # Check if current value is within tolerance of target
    distance_from_target = abs(P_t - P_target)
    
    return distance_from_target <= tolerance_value

def evaluate_solution(result: SimulationResult, P_target: float, P_baseline: float) -> Tuple[bool, float, float]:
    """Evaluate solution quality with adaptive tolerance."""
    final_value = result.P_series[-1]
    distance = abs(final_value - P_target)
    
    # Success using adaptive tolerance (10% of total change)
    success = check_success_condition(final_value, P_target, P_baseline, tolerance_percent=0.10)
    
    # Score combines distance and convergence speed
    speed_bonus = 1.0 / (result.t_hit + 1) if result.t_hit is not None else 0
    score = 1.0 / (distance + 0.001) + speed_bonus
    
    return success, score, distance

def normalize_simulation_data(delta_raw: np.ndarray, private_cost: np.ndarray, payoff_base: np.ndarray, P_baseline: float, P_target: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Normalize ALL simulation data consistently."""
    
    # Calculate the change needed
    target_change = P_target - P_baseline
    
    # Find scale based on delta magnitudes - we want deltas to be able to drive meaningful change
    max_delta_impact = np.sum(np.abs(delta_raw))
    
    if max_delta_impact > 0:
        # Scale so that maximum delta impact could drive the full baseline->target change
        scale_factor = abs(target_change) / max_delta_impact if max_delta_impact > 0 else 1.0
        # Add a multiplier to ensure deltas are large enough to drive change
        scale_factor *= 2.0  # 2x multiplier for stronger effects
    else:
        scale_factor = 1.0
    
    # Normalize deltas
    normalized_delta = delta_raw * scale_factor
    
    # Normalize costs and base payoffs proportionally 
    normalized_cost = private_cost * scale_factor
    normalized_payoff_base = payoff_base * scale_factor
    
    # FIXED: Normalize baseline and target to a standard range (0-100)
    if abs(target_change) > 0:
        normalized_baseline = 50.0  # Standard baseline
        normalized_target = 50.0 + (target_change / abs(target_change)) * 25.0  # Target 25 units away
    else:
        normalized_baseline = 50.0
        normalized_target = 50.0
    
    print(f"NORMALIZATION: Original range {P_baseline:.0f} -> {P_target:.0f}")
    print(f"NORMALIZATION: Normalized range {normalized_baseline:.1f} -> {normalized_target:.1f}")
    print(f"NORMALIZATION: Scale factor = {scale_factor:.8f}")
    print(f"NORMALIZATION: Delta range scaled by {scale_factor:.6f}")
    
    return normalized_delta, normalized_cost, normalized_payoff_base, normalized_baseline, normalized_target, scale_factor

def run_simulation(rows: List[List], P_baseline: float, P_target: float, max_epochs: int, 
                  scale: Optional[float] = None, incentive_adjustments: Optional[np.ndarray] = None) -> SimulationResult:
    """Run evolutionary game theory simulation with proper normalization."""
    
    if not rows:
        raise ValueError("No data provided for simulation")
    
    # Parse input data
    delta_raw, private_cost, weight, payoff_base, initial_shares, sector_names, strategy_ids = parse_rows_to_arrays(rows)
    
    # NORMALIZE ALL data consistently
    normalized_delta, normalized_cost, normalized_payoff_base, norm_baseline, norm_target, scale_factor = normalize_simulation_data(
        delta_raw, private_cost, payoff_base, P_baseline, P_target
    )
    
    # Scale incentives to match normalized costs
    normalized_incentives = None
    if incentive_adjustments is not None:
        normalized_incentives = incentive_adjustments * scale_factor
    
    G, K = normalized_delta.shape
    
    # DEBUG: Print simulation setup
    print(f"DEBUG SIMULATION: Baseline={P_baseline:.0f}, Target={P_target:.0f}")
    print(f"DEBUG SIMULATION: Normalized baseline={norm_baseline:.1f}, target={norm_target:.1f}")
    print(f"DEBUG SIMULATION: {G} actors, {K} strategies per actor")
    print(f"DEBUG SIMULATION: Target direction={'DOWN' if P_target < P_baseline else 'UP'}")
    print(f"DEBUG SIMULATION: Scale factor={scale_factor:.6f}")
    
    # Initialize storage
    P_series = []
    share_history = np.zeros((G, K, max_epochs))
    payoff_history = np.zeros((G, K, max_epochs))
    
    # Set initial shares
    share = initial_shares.copy()
    t_hit = None
    
    # Adaptive learning rate parameters
    base_learning_rate = 0.05  # Lower base rate
    max_learning_rate = 0.3    # Higher max rate for far distances
    min_learning_rate = 0.01   # Minimum rate when very close
    
    for t in range(max_epochs):
        # Calculate current headline metric in normalized space
        P_t_normalized = norm_baseline + np.sum(normalized_delta * share)
        
        # Convert back to original scale for reporting and target checking
        normalized_change = P_t_normalized - norm_baseline
        original_change = normalized_change / scale_factor
        P_t = P_baseline + original_change
        P_series.append(float(P_t))
        
        # Calculate adaptive learning rate based on distance to target (in original scale)
        distance_to_target = abs(P_t - P_target)
        total_distance_needed = abs(P_target - P_baseline)
        
        if total_distance_needed > 0:
            # Normalize distance (0 = at target, 1 = at baseline)
            normalized_distance = distance_to_target / total_distance_needed
            # Scale learning rate: higher when far, lower when close
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * normalized_distance
            learning_rate = max(learning_rate, min_learning_rate)
        else:
            learning_rate = base_learning_rate
        
        # DEBUG: Print every 10 epochs  
        if t % 10 == 0:
            print(f"DEBUG SIMULATION: Epoch {t}, P_t={P_t:.0f} (normalized: {P_t_normalized:.3f})")
            print(f"  Distance to target: {distance_to_target:.1f}, Learning rate: {learning_rate:.4f}")
            print(f"  Normalized change: {normalized_change:.3f}, Original change: {original_change:.1f}")
        
        # Calculate payoffs using normalized values
        payoff = compute_payoff_simple(normalized_payoff_base, normalized_incentives)
        
        # Additional debug info
        if t % 10 == 0:
            print(f"  Sample payoffs: {payoff[0, :].round(4)}")
            print(f"  Sample shares: {share[0, :].round(4)}")
            avg_payoff = np.mean(payoff[0, :])
            fitness_diffs = payoff[0, :] - avg_payoff
            print(f"  Fitness diffs: {fitness_diffs.round(6)}")
        
        # Store current state
        share_history[:, :, t] = share
        payoff_history[:, :, t] = payoff
        
        # Check stopping condition using original scale
        if check_success_condition(P_t, P_target, P_baseline, tolerance_percent=0.10):
            t_hit = t
            print(f"DEBUG SIMULATION: Target reached at epoch {t}!")
            break
        
        # Replicator dynamics update with adaptive learning rate
        if t < max_epochs - 1:
            new_share = share.copy()
            
            for g in range(G):
                avg_payoff_g = np.sum(share[g, :] * payoff[g, :])
                
                # Modified: Allow dynamics even with negative average payoffs
                if abs(avg_payoff_g) > EPSILON:
                    for k in range(K):
                        fitness_diff = payoff[g, k] - avg_payoff_g
                        # Use adaptive learning rate and absolute value of avg_payoff_g for scaling
                        new_share[g, k] = share[g, k] * (1 + learning_rate * fitness_diff / abs(avg_payoff_g))
                        new_share[g, k] = max(new_share[g, k], EPSILON)
                
                # Renormalize each actor's shares
                row_sum = np.sum(new_share[g, :])
                if row_sum > EPSILON:
                    new_share[g, :] /= row_sum
                else:
                    new_share[g, :] = 1/K
            
            share = new_share
    
    # Return results in ORIGINAL scale
    actual_length = len(P_series)
    share_trimmed = share_history[:, :, :actual_length]
    payoff_trimmed = payoff_history[:, :, :actual_length]
    
    return SimulationResult(
        P_series=P_series,  # Already in original scale
        share=share_trimmed.tolist(),
        payoff=payoff_trimmed.tolist(),
        t_hit=t_hit
    )

def generate_plots(result: SimulationResult, P_baseline: float, P_target: float, sector_names: List[str]) -> Tuple[str, str, str]:
    """Generate matplotlib plots and return filenames."""
    
    plot_dir = "static/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Convert back to numpy for plotting
    share_array = np.array(result.share)
    payoff_array = np.array(result.payoff)
    epochs = list(range(len(result.P_series)))
    
    # Plot 1: Line plot of P_series
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, result.P_series, linewidth=2, label='Headline Metric')
    ax1.axhline(y=P_target, color='red', linestyle='--', label=f'Target: {P_target:.3f}')
    ax1.axhline(y=P_baseline, color='gray', linestyle=':', alpha=0.7, label=f'Baseline: {P_baseline:.3f}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Metric Value')
    # ax1.set_title('Headline Metric Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    filename1 = f"{uuid.uuid4().hex}_metric.png"
    fig1.savefig(os.path.join(plot_dir, filename1), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot 2: Stacked area charts of shares
    G = len(sector_names)
    K = share_array.shape[1]  # Number of strategies
    
    fig2, axes2 = plt.subplots(G, 1, figsize=(12, 2*G), sharex=True)
    if G == 1:
        axes2 = [axes2]
    
    for g in range(G):
        ax = axes2[g]
        shares_g = share_array[g, :, :]  # K x T
        
        # Create stacked area plot
        ax.stackplot(epochs, *shares_g, alpha=0.7, labels=[f'Strategy {k+1}' for k in range(K)])
        
        ax.set_ylabel('Share')
        ax.set_title(f'{sector_names[g]} - Strategy Shares')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    if G > 0:
        axes2[-1].set_xlabel('Epoch')
    
    filename2 = f"{uuid.uuid4().hex}_shares.png"
    fig2.savefig(os.path.join(plot_dir, filename2), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # Plot 3: Heatmap of payoffs
    fig3, axes3 = plt.subplots(1, G, figsize=(4*G, 6))
    if G == 1:
        axes3 = [axes3]
    
    for g in range(G):
        ax = axes3[g]
        payoffs_g = payoff_array[g, :, :]  # K x T
        
        im = ax.imshow(payoffs_g, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'{sector_names[g][:15]}...' if len(sector_names[g]) > 15 else sector_names[g])
        ax.set_xlabel('Epoch')
        if g == 0:
            ax.set_ylabel('Strategy')
        ax.set_yticks(range(K))
        ax.set_yticklabels([f'Strategy {k+1}' for k in range(K)])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Payoff')
    
    filename3 = f"{uuid.uuid4().hex}_payoffs.png"
    fig3.savefig(os.path.join(plot_dir, filename3), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    return filename1, filename2, filename3

def generate_sample_json(rows_path: str, out_path: str):
    """Testing helper function."""
    with open(rows_path, 'r') as f:
        rows = json.load(f)
    
    # Default test parameters
    result = run_simulation(
        rows=rows,
        P_baseline=100.0,
        P_target=85.0,
        max_epochs=50
    )
    
    with open(out_path, 'w') as f:
        json.dump(result.model_dump(), f, indent=2)