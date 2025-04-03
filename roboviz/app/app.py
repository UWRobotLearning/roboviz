from flask import Flask, render_template, jsonify, send_from_directory
import subprocess

app = Flask(__name__)

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
    # Run the entropy.py script (assuming it's in the same folder)
    subprocess.run(['python', 'entropy.py'])
    return jsonify({'status': 'success'})
    print("ahahahah I got here first")

# Route to handle Segmentation button click
@app.route('/run_segmentation', methods=['GET'])
def run_segmentation():
    # Run the entropy.py script (assuming it's in the same folder)
    subprocess.run(['python', 'segmentation.py'])
    return jsonify({'status': 'success'})
    print("ahahahah I got here first")


# Route to handle Kernel Density button click
@app.route('/run_kernel', methods=['GET'])
def run_kernel():
    # Run the kernel.py script (assuming it's in the same folder)
    subprocess.run(['python', 'kernel.py'])
    return jsonify({'status': 'success'})
    print("ahahahah I got here")

if __name__ == '__main__':
    app.run(debug=True)
