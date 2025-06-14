from flask import Blueprint, request, render_template, jsonify, current_app, g, Response
import json
import os
import csv
from io import StringIO
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
        
        # Store result in Flask g for template access
        g.simulation_result = result
        g.simulation_plots = {
            'metric_plot': plot1,
            'shares_plot': plot2,
            'payoffs_plot': plot3
        }
        g.simulation_params = {
            'P_baseline': P_baseline,
            'P_target': P_target,
            'max_epochs': max_epochs,
            'actual_epochs': len(result.P_series)
        }
        g.sector_names = sector_names
        
        if current_app.config.get("DEBUG"):
            print(f"Simulation completed: t_hit={result.t_hit}, final_P={result.P_series[-1]:.3f}")
        
        return render_template('simulation_results.html')
        
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

@sim_bp.route('/download_csv/<data_type>')
def download_csv(data_type):
    """Download simulation data as CSV."""
    try:
        if not hasattr(g, 'simulation_result') or not g.simulation_result:
            return "No simulation data available", 404
        
        result = g.simulation_result
        
        if data_type == 'shares':
            # Generate shares CSV
            output = StringIO()
            writer = csv.writer(output)
            
            # Header
            header = ['Actor', 'Strategy', 'Epoch', 'Share']
            writer.writerow(header)
            
            # Data
            if hasattr(g, 'sector_names') and g.sector_names:
                for g_idx, actor in enumerate(g.sector_names):
                    if g_idx < len(result.share):
                        for k in range(len(result.share[g_idx])):
                            for t in range(len(result.share[g_idx][k])):
                                writer.writerow([
                                    actor,
                                    f'Strategy_{k+1}',
                                    t,
                                    result.share[g_idx][k][t]
                                ])
            
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=simulation_shares.csv'}
            )
            
        elif data_type == 'payoffs':
            # Generate payoffs CSV
            output = StringIO()
            writer = csv.writer(output)
            
            # Header
            header = ['Actor', 'Strategy', 'Epoch', 'Payoff']
            writer.writerow(header)
            
            # Data
            if hasattr(g, 'sector_names') and g.sector_names:
                for g_idx, actor in enumerate(g.sector_names):
                    if g_idx < len(result.payoff):
                        for k in range(len(result.payoff[g_idx])):
                            for t in range(len(result.payoff[g_idx][k])):
                                writer.writerow([
                                    actor,
                                    f'Strategy_{k+1}',
                                    t,
                                    result.payoff[g_idx][k][t]
                                ])
            
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=simulation_payoffs.csv'}
            )
        
        else:
            return "Invalid data type", 400
            
    except Exception as e:
        if current_app.config.get("DEBUG"):
            print(f"Error generating CSV: {e}")
        return f"Error generating CSV: {str(e)}", 500

# Add this debug route to test if the blueprint is working
@sim_bp.route('/test_simulation')
def test_simulation():
    """Test route to verify blueprint is working"""
    return jsonify({"status": "success", "message": "Simulation blueprint is working"})