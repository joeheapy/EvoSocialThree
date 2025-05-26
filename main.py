import webbrowser
from flask import Flask, render_template, request, redirect, url_for
import os
from dotenv import load_dotenv
import html
from config import DEFAULT_PROBLEM_TEXT
from api.openai.infer_actors import infer_actors_from_problem
from api.openai.infer_outcome_target import infer_outcome_targets_from_problem

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

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
# Initialize with default text and placeholder for actors table
results = {
    'problem': DEFAULT_PROBLEM_TEXT,
    'problem_submitted': False,  # New flag to track if form has been submitted
    'actors_table': None,
    'actors_table_error': False,
    'outcome_targets': None,
    'outcome_targets_error': False,
    'system_objective_selected': False,  # Add this
    'selected_objective_index': None,    # Add this
    'payoffs_table': None,
    'payoffs_table_error': False
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
        results['problem_submitted'] = True  # Mark that the form has been submitted
        results['actors_table'] = None
        results['actors_table_error'] = False
        results['outcome_targets'] = None
        results['outcome_targets_error'] = False
        results['system_objective_selected'] = False  # Reset system objective
        results['selected_objective_index'] = None    # Reset selected index
        results['payoffs_table'] = None  # Reset payoffs table
        results['payoffs_table_error'] = False  # Reset payoffs table error flag
        
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
    results['system_objective_selected'] = False  # Add this
    results['selected_objective_index'] = None    # Add this
    results['payoffs_table'] = None
    results['payoffs_table_error'] = False
    
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
    
    return redirect(url_for('hello_world'))

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
    
    return redirect(url_for('hello_world'))

@app.route('/infer_payoffs', methods=['POST'])
def infer_payoffs():
    """Endpoint for inferring payoffs from actors data"""
    if results.get('actors_table'):
        print("\n--- INFERRING PAYOFFS ---")
        
        # Convert actors data to JSON string for the API
        import json
        actors_json = json.dumps([actor.model_dump() for actor in results['actors_table'].actors], indent=2)
        problem = results.get('problem', '')
        
        # Call the payoffs inference API
        from api.openai.infer_payoffs import infer_payoffs
        try:
            payoffs_data = infer_payoffs(problem, actors_json)
            if payoffs_data:
                # Create a simple container object to match the template expectations
                class PayoffsContainer:
                    def __init__(self, actors):
                        self.actors = actors
                
                results['payoffs_table'] = PayoffsContainer(payoffs_data)
                results['payoffs_table_error'] = False
                print("Payoffs inference successful")
            else:
                results['payoffs_table_error'] = True
                print("Payoffs inference returned no data")
        except Exception as e:
            print(f"Error during payoffs inference: {e}")
            results['payoffs_table_error'] = True
    else:
        print("No actors data available for payoffs inference")
        results['payoffs_table_error'] = True
    
    return redirect(url_for('hello_world'))

@app.route('/select_objective', methods=['POST'])
def select_objective():
    """Endpoint for selecting system objective"""
    objective_index = request.form.get('objective_index')
    if objective_index is not None:
        print(f"\n--- SELECTING SYSTEM OBJECTIVE {objective_index} ---")
        results['system_objective_selected'] = True
        results['selected_objective_index'] = int(objective_index)
        # Reset payoffs when objective changes
        results['payoffs_table'] = None
        results['payoffs_table_error'] = False
        print(f"System objective {objective_index} selected")
    
    return redirect(url_for('hello_world'))

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