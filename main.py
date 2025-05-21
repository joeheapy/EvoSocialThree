import webbrowser
from flask import Flask, render_template, request, redirect, url_for
import os
from dotenv import load_dotenv
import html
from config import DEFAULT_PROBLEM_TEXT
from api.openai.infer_actors import infer_actors_from_problem

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Store form results (in a real app, you'd use a database)
# Initialize with default text and placeholder for actors table
results = {
    'problem': DEFAULT_PROBLEM_TEXT,
    'actors_table': None,
    'actors_table_error': False
}

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html', results=results)

@app.route('/submit', methods=['POST'])
def submit_problem():
    if request.method == 'POST':
        # Get the problem description from the form
        problem = request.form.get('problem', '')

        # Print the problem to the terminal
        print("\n--- SUBMITTED PROBLEM ---")
        print(problem)
        print("------------------------\n")
        
        # Store the problem in our results dictionary
        results['problem'] = problem
        results['actors_table'] = None # Reset previous results
        results['actors_table_error'] = False
        
        # Infer actors using LangChain/OpenAI
        if problem: # Only infer if there's a problem description
            actors_data = infer_actors_from_problem(problem)
            if actors_data:
                results['actors_table'] = actors_data
            else:
                results['actors_table_error'] = True
        
        # Redirect to the home page to display the results
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