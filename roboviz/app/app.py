# app.py
from flask import Flask, render_template, jsonify, request, send_from_directory
import subprocess

app = Flask(__name__)

# Global variable to store user input
user_input = None

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Serve the plot.html (this is where the plot will be updated)
@app.route('/static/plot.html')
def get_plot():
    return send_from_directory('static', 'plot.html')

# Route to handle Entropy button click
@app.route('/run_entropy', methods=['GET'])
def run_entropy():
    if user_input:
        # Pass the user input to entropy.py via a subprocess call
        result = subprocess.run(['python', 'algos/entropy.py', user_input], capture_output=True, text=True)
        print(result.stdout)  # Print the output of entropy.py for debugging
        return jsonify({'status': 'success', 'output': result.stdout})
    else:
        return jsonify({'status': 'error', 'message': 'No input provided.'})

# Route to handle Kernel Density button click
@app.route('/run_kernel', methods=['GET'])
def run_kernel():
    if user_input:
        # Pass the user input (HDF5 file path) to kernel.py via a subprocess call
        result = subprocess.run(['python', 'algos/kernel.py', user_input], capture_output=True, text=True)
        print(result.stdout)  # Print the output of kernel.py for debugging
        return jsonify({'status': 'success', 'output': result.stdout})
    else:
        return jsonify({'status': 'error', 'message': 'No input provided.'})


# Route to handle sending input from the user
@app.route('/send_input', methods=['POST'])
def handle_input():
    global user_input  # Access the global variable
    user_input = request.json.get('input')  # Get input from JSON body
    if user_input:
        print(f"User Input: {user_input}")  # Print the input to the console
        return jsonify({'status': 'success', 'input_received': user_input})
    else:
        print("No input received.")
        return jsonify({'status': 'error', 'message': 'No input received.'})

# Route to handle Segmentation button click
@app.route('/run_segmentation', methods=['GET'])
def run_segmentation():
    if user_input:
        # Use the user input to pass as the file path to segmentation.py
        result = subprocess.run(['python', 'algos/segmentation.py', user_input], capture_output=True, text=True)
        print(result.stdout)  # Print the output of segmentation.py for debugging
        return jsonify({'status': 'success', 'output': result.stdout})
    else:
        return jsonify({'status': 'error', 'message': 'No input provided for segmentation.'})

if __name__ == '__main__':
    app.run(debug=True)
