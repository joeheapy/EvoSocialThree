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
EPSILON = 1e-3

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
    
    # Initialize arrays
    delta_raw = np.zeros((G, K))
    private_cost = np.zeros((G, K))
    weight = np.zeros((G, K))
    payoff_base = np.zeros((G, K))  # Base payoffs from infer_payoffs
    initial_shares = np.zeros((G, K))
    
    # Fill arrays
    for g, sector in enumerate(sector_names):
        strategies = actors[sector][:K]  # Take first K strategies
        for k, strategy in enumerate(strategies):
            delta_raw[g, k] = strategy['delta']
            private_cost[g, k] = strategy['private_cost']
            weight[g, k] = strategy['weight']
            payoff_base[g, k] = strategy['payoff_base']
            initial_shares[g, k] = strategy['behavior_share']
            
            if g == 0 and k < len(strategy_ids):  # Collect strategy IDs from first actor
                strategy_ids.append(strategy['strategy_id'])
    
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

def run_simulation(rows: List[List], P_baseline: float, P_target: float, max_epochs: int, scale: Optional[float] = None) -> SimulationResult:
    """Run evolutionary game theory simulation."""
    
    if not rows:
        raise ValueError("No data provided for simulation")
    
    # Parse input data - includes base payoffs
    delta_raw, private_cost, weight, payoff_base, initial_shares, sector_names, strategy_ids = parse_rows_to_arrays(rows)
    
    G, K = delta_raw.shape
    
    # DEBUG: Print simulation setup
    print(f"DEBUG SIMULATION: Baseline={P_baseline:.3f}, Target={P_target:.3f}")
    print(f"DEBUG SIMULATION: {G} actors, {K} strategies per actor")
    print(f"DEBUG SIMULATION: Target direction={'DOWN' if P_target < P_baseline else 'UP'}")
    
    # Determine scale for normalization
    if scale is None:
        if 0 <= abs(P_baseline) <= 1:
            scale = 1.0
        else:
            scale = abs(P_baseline) if P_baseline != 0 else 1.0
    
    # Initialize storage arrays
    P_series = []
    share_history = np.zeros((G, K, max_epochs))
    payoff_history = np.zeros((G, K, max_epochs))
    
    # Set initial shares
    share = initial_shares.copy()
    t_hit = None
    
    # Determine if we're moving towards target (up or down)
    target_direction = P_target < P_baseline
    progress_needed = abs(P_target - P_baseline)
    
    # Higher learning rate for more dynamic behavior
    learning_rate = 0.3
    
    # Initialize progress_made to avoid UnboundLocalError
    progress_made = 0.0
    
    for t in range(max_epochs):
        # Calculate current headline metric
        P_t = P_baseline + np.sum(delta_raw * share)
        P_series.append(float(P_t))
        
        # Better progress calculation
        if progress_needed > 0:
            if target_direction:  # Moving down (P_target < P_baseline)
                progress_made = max(0, (P_baseline - P_t) / progress_needed)
            else:  # Moving up (P_target > P_baseline)
                progress_made = max(0, (P_t - P_baseline) / progress_needed)
        else:
            progress_made = 1.0
        
        # Cap progress at 1.0
        progress_made = min(progress_made, 1.0)
        
        # DEBUG: Print every 10 epochs
        if t % 10 == 0:
            print(f"DEBUG SIMULATION: Epoch {t}, P_t={P_t:.6f}, Progress={(progress_made*100):.1f}%")
        
        # Calculate dynamic payoffs with enhanced bonuses
        payoff = np.zeros((G, K))
        
        for g in range(G):
            for k in range(K):
                # Start with the pre-computed base payoff from infer_payoffs
                base_payoff = payoff_base[g, k]
                
                # Dynamic payoff components
                # 1. System progress bonus (rewards collective progress)
                progress_bonus = progress_made * 0.5
                
                # 2. Strategy effectiveness bonus (rewards strategies that help reach target)
                strategy_effectiveness = 0.0
                if target_direction and delta_raw[g, k] < 0:  # Strategy helps move toward lower target
                    strategy_effectiveness = abs(delta_raw[g, k]) * share[g, k] * 2.0
                elif not target_direction and delta_raw[g, k] > 0:  # Strategy helps move toward higher target
                    strategy_effectiveness = delta_raw[g, k] * share[g, k] * 2.0
                
                # 3. Coordination bonus (slight bonus for strategies being used by others)
                coordination_bonus = np.mean(share[:, k]) * 0.1
                
                # Final payoff calculation
                payoff[g, k] = base_payoff + progress_bonus + strategy_effectiveness + coordination_bonus
                
                # Ensure minimum positive payoff for stability
                payoff[g, k] = max(payoff[g, k], EPSILON)
        
        # Additional debug info after payoff calculation
        if t % 10 == 0:
            print(f"  Sample payoffs: {payoff[0, :].round(6)}")
            print(f"  Sample shares: {share[0, :].round(3)}")
        
        # Store current state
        share_history[:, :, t] = share
        payoff_history[:, :, t] = payoff
        
        # Check stopping condition
        if target_direction and P_t <= P_target:
            t_hit = t
            break
        elif not target_direction and P_t >= P_target:
            t_hit = t
            break
        
        # Replicator dynamics update with stronger response
        if t < max_epochs - 1:
            new_share = share.copy()
            
            for g in range(G):
                # Calculate average payoff for this actor
                avg_payoff_g = np.sum(share[g, :] * payoff[g, :])
                
                if avg_payoff_g > EPSILON:
                    # Enhanced replicator dynamics with stronger fitness differences
                    for k in range(K):
                        fitness_diff = payoff[g, k] - avg_payoff_g
                        # Amplify fitness differences for more dynamic behavior
                        amplified_diff = fitness_diff * 1.5
                        new_share[g, k] = share[g, k] + learning_rate * share[g, k] * amplified_diff
                        
                        # Ensure non-negative with higher minimum
                        new_share[g, k] = max(new_share[g, k], EPSILON * 10)
                
                # Renormalize to ensure shares sum to 1
                row_sum = np.sum(new_share[g, :])
                if row_sum > EPSILON:
                    new_share[g, :] /= row_sum
                else:
                    new_share[g, :] = 1/K
            
            share = new_share
    
    # Trim arrays to actual simulation length
    actual_length = len(P_series)
    share_trimmed = share_history[:, :, :actual_length]
    payoff_trimmed = payoff_history[:, :, :actual_length]
    
    return SimulationResult(
        P_series=P_series,
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