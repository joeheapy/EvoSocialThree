from flask import Blueprint, request, jsonify, current_app, render_template, g
from simulation import generate_plots
from simulation_random import find_optimum_random

sim_bp = Blueprint('simulation', __name__)

@sim_bp.route('/simulate/random', methods=['POST'])
def simulate_random():
    """Run random incentive search simulation."""
    try:
        # Parse JSON payload
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        rows = data.get('rows', [])
        P_baseline = float(data.get('P_baseline', 100.0))
        P_target = float(data.get('P_target', 85.0))
        max_epochs = int(data.get('max_epochs', 50))
        subsidy_cap = float(data.get('subsidy_cap', 0.08))
        penalty_cap = float(data.get('penalty_cap', 0.04))
        trials = int(data.get('trials', 5000))
        seed = data.get('seed')
        if seed is not None:
            seed = int(seed)
        
        if current_app.config.get("DEBUG"):
            print(f"Running random search: baseline={P_baseline}, target={P_target}")
            print(f"Trials: {trials}, subsidy_cap: {subsidy_cap}, penalty_cap: {penalty_cap}")
        
        # Validate input
        if not rows:
            return jsonify({"error": "No strategy data provided"}), 400
        
        if P_baseline == P_target:
            return jsonify({"error": "Baseline and target cannot be equal"}), 400
        
        # Run random search
        result, incentive_matrix = find_optimum_random(
            rows=rows,
            P_baseline=P_baseline,
            P_target=P_target,
            max_epochs=max_epochs,
            subsidy_cap=subsidy_cap,
            penalty_cap=penalty_cap,
            trials=trials,
            seed=seed
        )
        
        if result is None:
            # Store error state in g for template
            g.random_result = None
            g.simulation_params = {
                'P_baseline': P_baseline,
                'P_target': P_target,
                'trials': trials
            }
            return render_template('simulation_random_results.html')
        
        # Extract sector names for plotting
        sector_names = []
        seen_sectors = set()
        for row in rows:
            if len(row) > 0:
                sector = row[0]
                if sector not in seen_sectors:
                    sector_names.append(sector)
                    seen_sectors.add(sector)
        
        # Generate plots
        plot1, plot2, plot3 = generate_plots(result, P_baseline, P_target, sector_names)
        
        # Calculate total budget
        total_budget = float(incentive_matrix.sum()) if incentive_matrix is not None else 0.0
        
        if current_app.config.get("DEBUG"):
            print(f"Random search completed: t_hit={result.t_hit}, budget={total_budget:.6f}")
        
        # Store results in g for template access
        g.random_result = result
        g.incentive_matrix = incentive_matrix
        g.sector_names = sector_names
        g.total_budget = total_budget
        g.simulation_params = {
            'P_baseline': P_baseline,
            'P_target': P_target,
            'max_epochs': max_epochs,
            'subsidy_cap': subsidy_cap,
            'penalty_cap': penalty_cap,
            'trials': trials
        }
        g.plot_files = {
            'metric_plot': plot1,
            'shares_plot': plot2,
            'payoffs_plot': plot3
        }
        
        return render_template('simulation_random_results.html')
        
    except ValueError as e:
        if current_app.config.get("DEBUG"):
            print(f"ValueError in random simulation: {e}")
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400
    except Exception as e:
        if current_app.config.get("DEBUG"):
            print(f"Error in random simulation: {e}")
            import traceback
            traceback.print_exc()
        return jsonify({"error": f"Random search failed: {str(e)}"}), 500

# Test route to verify blueprint is working
@sim_bp.route('/test_simulation')
def test_simulation():
    """Test route to verify blueprint is working"""
    return jsonify({"status": "success", "message": "Simulation blueprint is working"})