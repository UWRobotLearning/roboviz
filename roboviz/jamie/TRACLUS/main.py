# Jamie Gashler
# Tranquil-Mayhem
# 02/18/25

import h5py
import numpy as np
import plotly.graph_objs as go
from geometry import Point
from coordination import run_traclus
import json

# Load data from HDF5 file
def load_data_from_hdf5(file_path, demo_name, data_type='states', obs_type='obs'):
    with h5py.File(file_path, 'r') as f:
        data_path = f'data/{demo_name}/{obs_type}/{data_type}'
        if data_path not in f:
            raise KeyError(f"Key '{data_type}' not found at {data_path}.")
        states = f[data_path][:]
        print(f"Loaded {data_type} from {obs_type} with shape: {states.shape} for {demo_name}")
        return states

# Extract translation data from the first 3 values from each state vector
def extract_translation_data(states):
    translations = states[:, :3]
    return translations

# Apply TRACLUS on the translations
def apply_traclus(translations, epsilon=1.0, min_neighbors=3, min_num_trajectories_in_cluster=3, min_vertical_lines=3, min_prev_dist=0.1):
    # Convert translations into Point objects
    trajs = [list(map(lambda pt: Point(x=pt[0], y=pt[1], z=pt[2]), traj)) for traj in translations]
    
    # Run TRACLUS
    result = run_traclus(point_iterable_list=trajs,
                         epsilon=epsilon,
                         min_neighbors=min_neighbors,
                         min_num_trajectories_in_cluster=min_num_trajectories_in_cluster,
                         min_vertical_lines=min_vertical_lines,
                         min_prev_dist=min_prev_dist)
    
    # Return clusters
    return result

# Create a 3D scatter plot with Plotly, overlaying all demos with TRACLUS clusters
def create_3d_overlay_plot_with_traclus(all_translations, all_demo_names, title="TRACLUS Clustering Summary Plot"):
    traces = []

    # Iterate through each demo's translations
    for i, translations in enumerate(all_translations):
        x = translations[:, 0]
        y = translations[:, 1]
        z = translations[:, 2]
        
        # Apply TRACLUS
        clusters = apply_traclus([translations])  # List of trajectories per demo
        
        # Create scatter plot for each demo with colored clusters
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=np.array([len(cluster) for cluster in clusters]),  # Assign cluster index
                colorscale='Viridis',
                colorbar=dict(title="Cluster")
            ),
            name=f"Demo {all_demo_names[i]}"
        )
        traces.append(trace)

    # Plot layout
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        legend=dict(title="Demos", x=0.8, y=0.9)
    )
    
    # Show the figure with all traces
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

# Main code to load the data and create a 3D visualization with TRACLUS
def main():
    hdf5_file_path = 'expert_lampshade2_demos.hdf5'  # Change this to the directory where expert/ play data is located
    data_type = 'states'  # The key that holds the state data (translation + quaternion)
    obs_type = 'obs'
    
    with h5py.File(hdf5_file_path, 'r') as f:
        demos = list(f['data'].keys())
        print("Demos found:", demos)
        
        all_translations = []
        all_demo_names = []
        
        for demo_name in demos:
            try:
                states = load_data_from_hdf5(hdf5_file_path, demo_name, data_type, obs_type)
                translations = extract_translation_data(states)
                
                # Project the 3D translations onto the XY plane
                xy_translations = translations[:, :2]
                
                # Apply the 2D TRACLUS clustering algorithm
                clusters = apply_traclus(xy_translations)
                
                # Store the results
                all_translations.append(xy_translations)
                all_demo_names.append(demo_name)
            except KeyError as e:
                print(f"Skipping {demo_name}: {e}")
        
        # Plot TRACLUS
        create_3d_overlay_plot_with_traclus(all_translations, all_demo_names)

if __name__ == "__main__":
    main()
