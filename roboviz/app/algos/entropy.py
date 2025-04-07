# algos/entropy.py
import h5py
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from scipy.stats import entropy

def load_data_from_hdf5(file_path, demo_name, data_type='states', obs_type='obs'):
    with h5py.File(file_path, 'r') as f:
        data_path = f'data/{demo_name}/{obs_type}/{data_type}'
        if data_path not in f:
            raise KeyError(f"Key '{data_type}' not found at {data_path}.")
        states = f[data_path][:]
        return states

def extract_translation_data(states):
    return states[:, :3]  # X, Y, Z translations

def compute_entropy(data, bins=30):
    hist, edges = np.histogramdd(data, bins=bins, density=True)
    prob_dist = hist.flatten()
    prob_dist = prob_dist[prob_dist > 0]
    entropy_value = entropy(prob_dist)
    max_entropy = np.log(bins ** 3)
    normalized_entropy = entropy_value / max_entropy
    scaled_entropy = np.exp(normalized_entropy) - 1
    return np.clip(scaled_entropy, 0, 1)

def compute_diff_entropy(demo_translations, bins=30):
    differences = np.diff(demo_translations, axis=0)
    return compute_entropy(differences, bins)

def classify_dataset(std_entropy, threshold=0.10):
    return "Expert" if std_entropy <= threshold else "Play"

def create_3d_overlay_plot(all_translations, all_demo_names, entropy_values, overall_entropy, std_entropy, classification):
    traces = []
    for i, translations in enumerate(all_translations):
        trace = go.Scatter3d(
            x=translations[:, 0], y=translations[:, 1], z=translations[:, 2],
            mode='lines',
            name=f'Demo {all_demo_names[i]} (Entropy: {entropy_values[i]:.2f})',
            line=dict(width=4)
        )
        traces.append(trace)

    annotation = dict(
        text=f"Overall Entropy: {overall_entropy:.2f}<br>Std Dev: {std_entropy:.2f}<br>Class: {classification}",
        showarrow=False, x=0, y=0, xref='paper', yref='paper',
        font=dict(size=14, color='red')
    )
    layout = go.Layout(
        scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')),
        annotations=[annotation],
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig = go.Figure(data=traces, layout=layout)
    py.plot(fig, filename='static/plot.html', auto_open=False)

def main(user_input=None):
    if not user_input:
        print("No user input received!")
        return
    
    hdf5_file_path = user_input
    data_type = 'states'
    obs_type = 'obs'
    
    with h5py.File(hdf5_file_path, 'r') as f:
        demos = list(f['data'].keys())
        all_translations, all_demo_names, entropy_values = [], [], []
        
        for demo_name in demos:
            try:
                states = load_data_from_hdf5(hdf5_file_path, demo_name, data_type, obs_type)
                translations = extract_translation_data(states)
                all_translations.append(translations)
                all_demo_names.append(demo_name)
            except KeyError as e:
                print(f"Skipping {demo_name}: {e}")
        
        for translations in all_translations:
            entropy_values.append(compute_diff_entropy(translations))

        overall_entropy = np.mean(entropy_values)
        std_entropy = np.std(entropy_values)
        classification = classify_dataset(std_entropy)

        create_3d_overlay_plot(all_translations, all_demo_names, entropy_values, overall_entropy, std_entropy, classification)

if __name__ == "__main__":
    import sys
    user_input = sys.argv[1] if len(sys.argv) > 1 else None
    main(user_input)
