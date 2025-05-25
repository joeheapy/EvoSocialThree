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

# Store form results (in a real app, you'd use a database)
# Initialize with default text and placeholder for actors table
results = {
    'problem': DEFAULT_PROBLEM_TEXT,
    'problem_submitted': False,  # New flag to track if form has been submitted
    'actors_table': None,
    'actors_table_error': False,
    'outcome_targets': None,
    'outcome_targets_error': False
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
        
        # Redirect to the home page to display the form results
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