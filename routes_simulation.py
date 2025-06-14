from flask import Blueprint, request, jsonify, current_app
import json
import os
from simulation import run_simulation, generate_plots, SimulationResult

sim_bp = Blueprint('simulation', __name__)

@sim_bp.route('/simulate', methods=['POST'])
def simulate():
    """Run evolutionary game theory simulation."""
    try:
        # Parse JSON payload
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        rows = data.get('rows', [])
        P_baseline = float(data.get('P_baseline', 100.0))
        P_target = float(data.get('P_target', 85.0))
        max_epochs = int(data.get('max_epochs', 50))
        scale = data.get('scale')
        if scale is not None:
            scale = float(scale)
        
        if current_app.config.get("DEBUG"):
            print(f"Running simulation: baseline={P_baseline}, target={P_target}, epochs={max_epochs}")
            print(f"Number of rows: {len(rows)}")
            print(f"Sample row: {rows[0] if rows else 'None'}")
        
        # Validate input
        if not rows:
            return jsonify({"error": "No strategy data provided"}), 400
        
        if P_baseline == P_target:
            return jsonify({"error": "Baseline and target cannot be equal"}), 400
        
        # Run simulation
        result = run_simulation(rows, P_baseline, P_target, max_epochs, scale)
        
        # Extract sector names for plotting
        sector_names = []
        seen_sectors = set()
        for row in rows:
            if len(row) > 0:
                sector = row[0]  # First element is sector name
                if sector not in seen_sectors:
                    sector_names.append(sector)
                    seen_sectors.add(sector)
        
        # Generate plots
        plot1, plot2, plot3 = generate_plots(result, P_baseline, P_target, sector_names)
        
        if current_app.config.get("DEBUG"):
            print(f"Simulation completed: t_hit={result.t_hit}, final_P={result.P_series[-1]:.3f}")
        
        # Return JSON response
        return jsonify({
            "success": True,
            "t_hit": result.t_hit,
            "final_value": result.P_series[-1],
            "total_epochs": len(result.P_series),
            "plot_files": {
                'metric_plot': plot1,
                'shares_plot': plot2,
                'payoffs_plot': plot3
            },
            "simulation_params": {
                'P_baseline': P_baseline,
                'P_target': P_target,
                'max_epochs': max_epochs,
                'actual_epochs': len(result.P_series)
            }
        })
        
    except ValueError as e:
        if current_app.config.get("DEBUG"):
            print(f"ValueError in simulation: {e}")
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400
    except Exception as e:
        if current_app.config.get("DEBUG"):
            print(f"Error in simulation: {e}")
            import traceback
            traceback.print_exc()
        return jsonify({"error": f"Simulation failed: {str(e)}"}), 500

# Test route to verify blueprint is working
@sim_bp.route('/test_simulation')
def test_simulation():
    """Test route to verify blueprint is working"""
    return jsonify({"status": "success", "message": "Simulation blueprint is working"})