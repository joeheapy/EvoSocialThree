import webbrowser
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import os
from dotenv import load_dotenv
import html
from config import DEFAULT_PROBLEM_TEXT
from api.openai.infer_actors import infer_actors_from_problem
from api.openai.infer_outcome_target import infer_outcome_targets_from_problem
import json
from routes_simulation import sim_bp

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.register_blueprint(sim_bp)

# Add custom Jinja2 filter for number formatting
@app.template_filter('format_number')
def format_number(value):
    """Format numbers with commas and handle percentages"""
    try:
        # Convert to float if it's a string
        num = float(value)
        
        # Check if it's a percentage (between 0-100 and likely represents a percentage)
        # You might want to adjust this logic based on your data
        if 0 <= num <= 100 and '.' in str(value):
            return f"{num:.1f}"
        else:
            # Format with commas for thousands
            return f"{num:,.0f}"
    except (ValueError, TypeError):
        return str(value)

# Store form results (in a real app, you'd use a database)
results = {
    'problem': DEFAULT_PROBLEM_TEXT,
    'problem_submitted': False,
    'actors_table': None,
    'actors_table_error': False,
    'outcome_targets': None,
    'outcome_targets_error': False,
    'system_objective_selected': False,
    'selected_objective_index': None,
    'payoffs_table': None,
    'payoffs_table_error': False,
    'payoffs_analysis': None,
    'payoffs_analysis_error': False,
    # 'simulation_results': None,
    # 'simulation_error': False
}

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html', results=results, DEFAULT_PROBLEM_TEXT=DEFAULT_PROBLEM_TEXT)

@app.route('/submit', methods=['POST'])
def submit_problem():
    if request.method == 'POST':
        # Get the problem description from the form
        problem = request.form.get('problem', '')

        # Print the problem to the terminal
        print("\n--- SUBMITTED PROBLEM ---")
        print(problem)
        print("------------------------\n")
        
        # Store the problem in our results dictionary and reset all analysis results
        results['problem'] = problem
        results['problem_submitted'] = True
        results['actors_table'] = None
        results['actors_table_error'] = False
        results['outcome_targets'] = None
        results['outcome_targets_error'] = False
        results['system_objective_selected'] = False
        results['selected_objective_index'] = None
        results['payoffs_table'] = None
        results['payoffs_table_error'] = False
        results['payoffs_analysis'] = None
        results['payoffs_analysis_error'] = False
        
        # Redirect to the home page to display the form results
        return redirect(url_for('hello_world'))

@app.route('/reset', methods=['POST'])
def reset_app():
    """Reset the entire application to initial state"""
    print("\n--- RESETTING APPLICATION ---")
    
    # Reset all results to initial state
    results['problem'] = DEFAULT_PROBLEM_TEXT
    results['problem_submitted'] = False
    results['actors_table'] = None
    results['actors_table_error'] = False
    results['outcome_targets'] = None
    results['outcome_targets_error'] = False
    results['system_objective_selected'] = False
    results['selected_objective_index'] = None
    results['payoffs_table'] = None
    results['payoffs_table_error'] = False
    results['payoffs_analysis'] = None
    results['payoffs_analysis_error'] = False
    results['simulation_results'] = None  # NEW
    results['simulation_error'] = False   # NEW
    
    print("Application reset to initial state")
    print("--------------------------------\n")
    
    # Redirect to home page
    return redirect(url_for('hello_world'))

@app.route('/analyze_actors', methods=['POST'])
def analyze_actors():
    """Endpoint specifically for analyzing actors"""
    problem = results.get('problem', '')
    if problem:
        print("\n--- ANALYZING ACTORS ---")
        actors_data = infer_actors_from_problem(problem)
        if actors_data:
            results['actors_table'] = actors_data
            results['actors_table_error'] = False
        else:
            results['actors_table_error'] = True
    
    return redirect(url_for('hello_world') + '#step-2-actors-analysis')

@app.route('/analyze_outcome_targets', methods=['POST'])
def analyze_outcome_targets():
    """Endpoint specifically for analyzing outcome targets"""
    problem = results.get('problem', '')
    if problem:
        print("\n--- ANALYZING OUTCOME TARGETS ---")
        outcome_targets_data = infer_outcome_targets_from_problem(problem)
        if outcome_targets_data:
            results['outcome_targets'] = outcome_targets_data
            results['outcome_targets_error'] = False
        else:
            results['outcome_targets_error'] = True
    
    return redirect(url_for('hello_world') + '#step-3-outcome-targets')


@app.route('/select_objective', methods=['POST'])
def select_objective():
    """Endpoint for selecting system objective"""
    objective_index = request.form.get('objective_index')
    if objective_index is not None:
        print(f"\n--- SELECTING SYSTEM OBJECTIVE {objective_index} ---")
        results['system_objective_selected'] = True
        results['selected_objective_index'] = int(objective_index)
        # Reset payoffs and analysis when objective changes
        results['payoffs_table'] = None
        results['payoffs_table_error'] = False
        results['payoffs_analysis'] = None
        results['payoffs_analysis_error'] = False
        print(f"System objective {objective_index} selected")
    
        return redirect(url_for('hello_world') + '#step-4-payoffs')


# Change the route to accept GET requests for EventSource
@app.route('/infer_payoffs', methods=['GET', 'POST'])
def infer_payoffs():
    """Endpoint for inferring payoffs from actors data with streaming support"""
    # Check if this is a GET request (EventSource) - streaming
    if request.method == 'GET':
        return infer_payoffs_stream()
    else:
        # POST request - keep existing non-streaming behavior for backwards compatibility
        return infer_payoffs_non_stream()

def infer_payoffs_stream():
    """Streaming version of payoffs inference"""
    def generate():
        try:
            if not results.get('actors_table'):
                yield f"data: {json.dumps({'status': 'error', 'message': 'No actors data available'})}\n\n"
                return
                
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Initializing payoff calculations...'})}\n\n"
            
            print("\n--- INFERRING PAYOFFS (STREAMING) ---")
            
            # Convert actors data to JSON string for the API
            actors_json = json.dumps([actor.model_dump() for actor in results['actors_table'].actors], indent=2)
            problem = results.get('problem', '')
            
            # Get the selected system objective
            selected_index = results.get('selected_objective_index')
            system_objective = "the social problem"  # default
            
            if selected_index is not None and results.get('outcome_targets'):
                try:
                    system_objective = results['outcome_targets'].targets[selected_index].metric_name
                    print(f"Using system objective: {system_objective}")
                except (IndexError, AttributeError):
                    print("Could not retrieve system objective, using default")
            
            yield f"data: {json.dumps({'status': 'progress', 'message': 'Estimating values...', 'progress': 25})}\n\n"
            
            # Call the payoffs inference API
            from api.openai.infer_payoffs import infer_payoffs
            payoffs_data = infer_payoffs(problem, actors_json, system_objective)
            
            yield f"data: {json.dumps({'status': 'progress', 'message': 'Processing payoffs data...', 'progress': 50})}\n\n"
            
            if payoffs_data:
                # Create a simple container object to match the template expectations
                class PayoffsContainer:
                    def __init__(self, actors):
                        self.actors = actors
                
                results['payoffs_table'] = PayoffsContainer(payoffs_data)
                results['payoffs_table_error'] = False
                print("Payoffs inference successful")
                
                yield f"data: {json.dumps({'status': 'progress', 'message': 'Calculating behaviour shares...', 'progress': 70})}\n\n"
                
                # AUTOMATICALLY CALCULATE BEHAVIOR SHARES
                print("\n--- AUTO-INFERRING BEHAVIOR SHARES ---")
                
                from api.openai.infer_behavior_shares import infer_behavior_shares as infer_behavior_shares_fn
                try:
                    actors_with_behavior_shares = infer_behavior_shares_fn(problem, payoffs_data, epoch=0)
                    
                    if actors_with_behavior_shares:
                        results['payoffs_table'] = PayoffsContainer(actors_with_behavior_shares)
                        print("Behavior shares inference successful")
                    else:
                        print("Behavior shares inference returned no data, keeping payoffs data")
                        
                except Exception as e:
                    print(f"Error during automatic behavior shares inference: {e}")
                    print("Continuing with payoffs data only")
                
                yield f"data: {json.dumps({'status': 'complete', 'message': 'Payoffs calculation complete!', 'progress': 100})}\n\n"
                
            else:
                results['payoffs_table_error'] = True
                yield f"data: {json.dumps({'status': 'error', 'message': 'Payoffs inference returned no data'})}\n\n"
                
        except Exception as e:
            print(f"Error during payoffs inference: {e}")
            results['payoffs_table_error'] = True
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

def infer_payoffs_non_stream():
    """Non-streaming version (your existing code)"""
    if results.get('actors_table'):
        print("\n--- INFERRING PAYOFFS ---")
        
        # Convert actors data to JSON string for the API
        actors_json = json.dumps([actor.model_dump() for actor in results['actors_table'].actors], indent=2)
        problem = results.get('problem', '')
        
        # Get the selected system objective
        selected_index = results.get('selected_objective_index')
        system_objective = "the social problem"  # default
        
        if selected_index is not None and results.get('outcome_targets'):
            try:
                system_objective = results['outcome_targets'].targets[selected_index].metric_name
                print(f"Using system objective: {system_objective}")
            except (IndexError, AttributeError):
                print("Could not retrieve system objective, using default")
        
        # Call the payoffs inference API
        from api.openai.infer_payoffs import infer_payoffs
        try:
            payoffs_data = infer_payoffs(problem, actors_json, system_objective)
            if payoffs_data:
                # Create a simple container object to match the template expectations
                class PayoffsContainer:
                    def __init__(self, actors):
                        self.actors = actors
                
                results['payoffs_table'] = PayoffsContainer(payoffs_data)
                results['payoffs_table_error'] = False
                print("Payoffs inference successful")
                
                # AUTOMATICALLY CALCULATE BEHAVIOR SHARES
                print("\n--- AUTO-INFERRING BEHAVIOR SHARES ---")
                
                # Call the behavior shares inference API
                from api.openai.infer_behavior_shares import infer_behavior_shares as infer_behavior_shares_fn
                try:
                    actors_with_behavior_shares = infer_behavior_shares_fn(problem, payoffs_data, epoch=0)
                    
                    if actors_with_behavior_shares:
                        results['payoffs_table'] = PayoffsContainer(actors_with_behavior_shares)
                        print("Behavior shares inference successful")
                    else:
                        print("Behavior shares inference returned no data, keeping payoffs data")
                        
                except Exception as e:
                    print(f"Error during automatic behavior shares inference: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Continuing with payoffs data only")
                
            else:
                results['payoffs_table_error'] = True
                print("Payoffs inference returned no data")
        except Exception as e:
            print(f"Error during payoffs inference: {e}")
            results['payoffs_table_error'] = True
    else:
        print("No actors data available for payoffs inference")
        results['payoffs_table_error'] = True
    
    return redirect(url_for('hello_world') + '#step-4-payoffs')



@app.route('/analyze_payoffs', methods=['POST'])
def analyze_payoffs():
    """Endpoint for generating payoff analysis"""
    print(f"DEBUG: payoffs_table exists: {bool(results.get('payoffs_table'))}")
    print(f"DEBUG: payoffs_table has actors: {bool(results.get('payoffs_table') and results.get('payoffs_table').actors)}")
    
    if results.get('payoffs_table') and results.get('payoffs_table').actors:
        try:
            print(f"\n--- ANALYZING PAYOFF PATTERNS ---")
            print(f"DEBUG: Number of actors: {len(results['payoffs_table'].actors)}")
            
            from api.openai.analyze_payoffs import analyze_payoffs as analyze_payoffs_fn
            
            # Get the selected system objective
            selected_objective = "reducing child poverty"  # Default
            if results.get('outcome_targets') and results.get('selected_objective_index') is not None:
                idx = results['selected_objective_index']
                if 0 <= idx < len(results['outcome_targets'].targets):
                    selected_objective = results['outcome_targets'].targets[idx].metric_name
            
            print(f"DEBUG: Selected objective: {selected_objective}")
            
            # Generate analysis
            analysis = analyze_payoffs_fn(
                problem_description=results['problem'],
                actors=results['payoffs_table'].actors,
                system_objective=selected_objective
            )
            
            print(f"DEBUG: Analysis result type: {type(analysis)}")
            
            if analysis:
                results['payoffs_analysis'] = analysis
                results['payoffs_analysis_error'] = False
                print(f"Payoff analysis generated successfully")
            else:
                results['payoffs_analysis_error'] = True
                print("Payoff analysis returned no data")
            
        except Exception as e:
            print(f"Error during payoff analysis: {e}")
            import traceback
            traceback.print_exc()
            results['payoffs_analysis_error'] = True
    else:
        print("No payoff data available for analysis")
        results['payoffs_analysis_error'] = True
    
    return redirect(url_for('hello_world') + '#step-5-analyse-payoffs')

@app.route('/get_simulation_data')
def get_simulation_data():
    """API endpoint to get simulation data as JSON"""
    try:
        if not results.get('payoffs_table'):
            return jsonify({"error": "No payoffs data available"}), 404
        
        payoffs_table = results['payoffs_table']
        rows = []
        
        if hasattr(payoffs_table, 'actors') and payoffs_table.actors:
            for actor in payoffs_table.actors:
                if hasattr(actor, 'strategies') and actor.strategies:
                    for strategy in actor.strategies:
                        row = [
                            getattr(actor, 'sector', 'Unknown'),
                            getattr(strategy, 'strategy_id', 'Strategy'),
                            getattr(strategy, 'commitment_level', 'Medium'),
                            float(getattr(strategy, 'delta', 0)),
                            float(getattr(strategy, 'private_cost', 0)),
                            float(getattr(strategy, 'weight', 1)),
                            float(getattr(strategy, 'payoff_epoch_0', 0)) if hasattr(strategy, 'payoff_epoch_0') and strategy.payoff_epoch_0 not in [None, 'N/A'] else 0.0,
                            float(getattr(strategy, 'behavior_share_epoch_0', 0.333)) if hasattr(strategy, 'behavior_share_epoch_0') and strategy.behavior_share_epoch_0 not in [None, 'N/A'] else 0.333,
                            getattr(strategy, 'description', 'No description')
                        ]
                        rows.append(row)
        
        # Get baseline and target values - FIX: Use correct attribute names
        baseline = 100.0
        target = 85.0
        
        if (results.get('outcome_targets') and 
            results.get('selected_objective_index') is not None):
            selected_target = results['outcome_targets'].targets[results['selected_objective_index']]
            baseline = float(getattr(selected_target, 'from_value', 100))
            target = float(getattr(selected_target, 'to_value', 85))
        
        return jsonify({
            "rows": rows,
            "P_baseline": baseline,
            "P_target": target,
            "success": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/store_simulation_results', methods=['POST'])
def store_simulation_results():
    """Store simulation results in the results dictionary"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Store the simulation results
        results['simulation_results'] = data
        results['simulation_error'] = False
        
        return jsonify({"success": True})
        
    except Exception as e:
        print(f"Error storing simulation results: {e}")
        results['simulation_error'] = True
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Open browser in a separate thread
    import threading
    def open_browser():
        # Give the server a moment to start
        import time
        time.sleep(1.5)
        # Open the browser
        webbrowser.open('http://localhost:5001')

    # Start the browser thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Flask in the main thread (without debug for threading stability)
    print("Starting server at http://localhost:5001")
    print("Press Ctrl+C to exit...")
    
    try:
        app.run(host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("Shutting down...")