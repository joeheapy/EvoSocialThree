from flask import Blueprint, request, jsonify, current_app, render_template, g
from simulation import generate_plots
from simulation_random import find_optimum_random

sim_bp = Blueprint('simulation', __name__)

@sim_bp.route('/simulate/random', methods=['POST'])
def simulate_random():
    """Run random incentive search simulation."""
    try:
        data = request.get_json()
        if not data:
            g.random_result = None
            g.simulation_params = {"trials": 0, "P_baseline": 0, "P_target": 0}
            return render_template('simulation_random_results.html')
        
        # Parse parameters
        rows = data.get('rows', [])
        P_baseline = float(data.get('P_baseline', 100.0))
        P_target = float(data.get('P_target', 85.0))
        max_epochs = int(data.get('max_epochs', 50))
        subsidy_cap = float(data.get('subsidy_cap', 0.15))
        penalty_cap = float(data.get('penalty_cap', 0.10))
        trials = int(data.get('trials', 10000))
        seed = data.get('seed')
        if seed is not None:
            seed = int(seed)
        
        # Store params for template
        g.simulation_params = {
            "P_baseline": P_baseline,
            "P_target": P_target,
            "trials": trials,
            "max_epochs": max_epochs,
            "subsidy_cap": subsidy_cap,
            "penalty_cap": penalty_cap
        }
        
        if not rows:
            g.random_result = None
            return render_template('simulation_random_results.html')
        
        # This code is calling the find_optimum_random function to run a random search optimization for finding optimal incentive structures.
        result, incentive_matrix, sector_names = find_optimum_random(
            rows=rows,
            P_baseline=P_baseline,
            P_target=P_target,
            max_epochs=max_epochs,
            subsidy_cap=subsidy_cap,
            penalty_cap=penalty_cap,
            trials=trials,
            seed=seed
        )
        
        # Store results for template
        g.random_result = result
        g.incentive_matrix = incentive_matrix
        g.sector_names = sector_names if sector_names else []
        g.total_budget = float(abs(incentive_matrix).sum()) if incentive_matrix is not None else 0.0
        
        # Always generate plots if we have a result (successful or best attempt)
        if result is not None:
            # Generate plots
            plot1, plot2, plot3 = generate_plots(result, P_baseline, P_target, g.sector_names)
            g.plot_files = {
                "metric_plot": plot1,
                "shares_plot": plot2,
                "payoffs_plot": plot3
            }
        else:
            g.plot_files = {}
        
        return render_template('simulation_random_results.html')
        
    except Exception as e:
        if current_app.config.get("DEBUG"):
            import traceback
            traceback.print_exc()
        
        g.random_result = None
        g.simulation_params = {
            "trials": trials if 'trials' in locals() else 0,
            "P_baseline": P_baseline if 'P_baseline' in locals() else 0,
            "P_target": P_target if 'P_target' in locals() else 0
        }
        return render_template('simulation_random_results.html')

# Test route to verify blueprint is working
@sim_bp.route('/test_simulation')
def test_simulation():
    """Test route to verify blueprint is working"""
    return jsonify({"status": "success", "message": "Simulation blueprint is working"})