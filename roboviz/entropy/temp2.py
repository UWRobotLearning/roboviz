import h5py
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from scipy.stats import entropy

# Ensure plotly rendering in Jupyter notebooks
py.init_notebook_mode(connected=True)

def load_data_from_hdf5(file_path, demo_name, data_type='states', obs_type='obs'):
    with h5py.File(file_path, 'r') as f:
        data_path = f'data/{demo_name}/{obs_type}/{data_type}'
        if data_path not in f:
            raise KeyError(f"Key '{data_type}' not found at {data_path}.")
        states = f[data_path][:]
        return states

def extract_translation_data(states):
    return states[:, :3]  # since th first 3 columns are X, Y, Z translations

def compute_entropy(data, bins=30):
    hist, edges = np.histogramdd(data, bins=bins, density=True)
    prob_dist = hist.flatten()
    prob_dist = prob_dist[prob_dist > 0]
    entropy_value = entropy(prob_dist)
    max_entropy = np.log(bins ** 3)
    normalized_entropy = entropy_value / max_entropy
    scaled_entropy = np.exp(normalized_entropy) - 1
    scaled_entropy = np.clip(scaled_entropy, 0, 1)
    return scaled_entropy

def compute_diff_entropy(demo_translations, bins=30):
    differences = np.diff(demo_translations, axis=0)
    return compute_entropy(differences, bins)

def classify_dataset(std_entropy, threshold=0.10):
    if std_entropy <= threshold:
        return "Expert"
    else:
        return "Play"

def create_3d_overlay_plot(all_translations, all_demo_names, entropy_values, overall_entropy, std_entropy, classification):
    traces = []
    for i, translations in enumerate(all_translations):
        x, y, z = translations[:, 0], translations[:, 1], translations[:, 2]
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            name=f'Demo {all_demo_names[i]} (Entropy: {entropy_values[i]:.2f})',
            line=dict(width=3)
        )
        traces.append(trace)
    
    entropy_text = f'Overall Entropy: {overall_entropy:.2f}'
    std_diff_text = f'Standard Deviation: {std_entropy:.2f}'
    classification_text = f'Classification: {classification}'
    
    annotation = dict(
        text=f"{entropy_text}<br>{std_diff_text}<br>{classification_text}",
        showarrow=False,
        x=1, y=1,
        xref='paper', yref='paper',
        font=dict(size=14, color='red')
    )
    layout = go.Layout(
        title="Translation Data for Lampshade Demos",
        scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')),
        annotations=[annotation]
    )
    fig = go.Figure(data=traces, layout=layout)
    py.iplot(fig)  # Use py.iplot to display the plot inline in Jupyter Notebook

def main():
    hdf5_file_path = 'play_pushing.hdf5'
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
        
        # Compute entropy for each demo based on its difference from the average
        for translations in all_translations:
            entropy_values.append(compute_diff_entropy(translations))

        overall_entropy = np.mean(entropy_values)
        std_entropy = np.std(entropy_values)
        classification = classify_dataset(std_entropy)
        
        # Create the 3D plot with classification
        create_3d_overlay_plot(all_translations, all_demo_names, entropy_values, overall_entropy, std_entropy, classification)

if __name__ == "__main__":
    main()
