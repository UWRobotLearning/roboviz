# algos/entropy.py
import h5py
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from scipy.stats import entropy
import os
from roboviz.lerobot_reader.read_data import extract_states_grouped, extract_states_ungrouped
import argparse
from roboviz.s3_access.read_from_s3 import download_dataset

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
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static/entropy.html")
    py.plot(fig, filename=filename, auto_open=False)

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the dataset script; optionally fetch the file from S3 first."
    )
    p.add_argument(
        "dataset_path",
        help="Local path where the dataset file should live.",
    )
    p.add_argument(
        "--download",
        metavar="PATH",
        nargs="?",
        const="expert_lampshade2_demos.hdf5",
        help=(
            "Download the file from the given S3 file path if it is missing. "
            f"If PATH is omitted, defaults to 'expert_lampshade2_demos.hdf5'."
        ),
    )

    p.add_argument(
        "--creds",
        default="kopah_creds.json",
        help=f"Path to S3 credential JSON (default: 'kopah_creds.json').",
    )
    return p.parse_args()

def main():
    # decide whether or not to download from S3
    args = parse_cli()
    endpoint_url = "https://s3.kopah.uw.edu"
    bucket_name = 'roboviz-dataset'
    dataset_path = args.dataset_path
    if args.download is not None and not os.path.exists(args.dataset_path):
        if not os.path.exists(args.creds):
            print("credential file not found")
            return

        download_dataset(endpoint_url, bucket_name, args.download, args.dataset_path, args.creds)
    
    data_type = 'states'
    obs_type = 'obs'
    all_translations, all_demo_names, entropy_values = [], [], []
    
    if dataset_path.split('.')[-1] == 'hdf5':
        with h5py.File(dataset_path, 'r') as f:
            demos = list(f['data'].keys())
            for demo_name in demos:
                try:
                    states = load_data_from_hdf5(dataset_path, demo_name, data_type, obs_type)
                    translations = extract_translation_data(states)
                    all_translations.append(translations)
                    all_demo_names.append(demo_name)
                except KeyError as e:
                    print(f"Skipping {demo_name}: {e}")
            
        
    else:
        # lerobot dataset
        states = extract_states_grouped(dataset_path)
        all_translations = [extract_translation_data(state) for state in states]
        all_demo_names = [f'demo_{i}' for i in range(len(states))]

    for translations in all_translations:
        entropy_values.append(compute_diff_entropy(translations))
    overall_entropy = np.mean(entropy_values)
    std_entropy = np.std(entropy_values)
    classification = classify_dataset(std_entropy)

    create_3d_overlay_plot(all_translations, all_demo_names, entropy_values, overall_entropy, std_entropy, classification)

if __name__ == "__main__":
    main()
