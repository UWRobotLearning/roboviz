# alt_KernelDensity.py
import sys
import h5py
import numpy as np
import plotly.graph_objs as go
from sklearn.neighbors import KernelDensity

# Load data from HDF5 file (states)
def load_data_from_hdf5(file_path, demo_name, data_type='states', obs_type='obs'):
    with h5py.File(file_path, 'r') as f:
        # Construct the path for the current observation (obs) or next observation (next_obs)
        data_path = f'data/{demo_name}/{obs_type}/{data_type}'
        
        # Check if the path exists first
        if data_path not in f:
            raise KeyError(f"Key '{data_type}' not found at {data_path}.")
        
        # Load the data
        states = f[data_path][:]
        print(f"Loaded {data_type} from {obs_type} with shape: {states.shape} for {demo_name}")
        
        return states

# Extract translation data (first 3 values from each state vector)
def extract_translation_data(states):
    translations = states[:, :3]  # Get the translation part of the state (first 3 values)
    return translations

# Compute KDE for translations
def compute_kde(translations, bandwidth=0.1):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(translations)
    return kde

# Create a solid shape with density coloring
def create_3d_overlay_plot_with_kde(all_translations, all_demo_names, title="3D KDE Solid Shape"):
    # Combine all translations into one big array
    combined_translations = np.vstack(all_translations)

    # Compute KDE on the combined data
    kde = compute_kde(combined_translations)

    # Define grid boundaries based on data
    x_min, y_min, z_min = combined_translations.min(axis=0)
    x_max, y_max, z_max = combined_translations.max(axis=0)

    # Create grid points for evaluation
    grid_x, grid_y, grid_z = np.mgrid[x_min:x_max:15j, y_min:y_max:15j, z_min:z_max:15j]

    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    # Evaluate KDE over the grid points
    kde_values = np.exp(kde.score_samples(grid_points))
    kde_values = kde_values.reshape(grid_x.shape)
  
    # Volume rendering of the density
    volume_trace = go.Volume(
        x=grid_x.ravel(),
        y=grid_y.ravel(),
        z=grid_z.ravel(),
        value=kde_values.ravel(),
        opacity=0.2,  # Transparency for better visibility
        surface_count=20,  # Number of surfaces for smoothness
        colorscale='Viridis',  # Colorscale for density
        colorbar=dict(title="Density"),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="KDE Volume"
    )

    # Scatter plot for the points themselves (Can be removed *check screenshots*)
    scatter = go.Scatter3d(
        x=combined_translations[:, 0],
        y=combined_translations[:, 1],
        z=combined_translations[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='darkblue',  # Color of the scatter points
            opacity=0.5  # Semi-transparent points
        ),
        name='Demo Points'
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[volume_trace, scatter], layout=layout)
    fig.write_html("static/plot.html")  # Save to HTML file
    print("Saved 3D plot to static/plot.html")


# Main code to load the data and create a 3D visualization with KDE
def main(file_path):
    data_type = 'states'  # The key that holds the state data (translation + quaternion)
    obs_type = 'obs'
    
    with h5py.File(file_path, 'r') as f:
        # List all the demos in the dataset
        demos = list(f['data'].keys())
        print("Demos found:", demos)
        
        # Store translation data and demo names
        all_translations = []
        all_demo_names = []
        
        # Iterate over all demos
        for demo_name in demos:
            try:
                states = load_data_from_hdf5(file_path, demo_name, data_type, obs_type)
                
                # Extract the translation part (x, y, z)
                translations = extract_translation_data(states)
                
                # Append the translations and demo name to the lists
                all_translations.append(translations)
                all_demo_names.append(demo_name)

            except KeyError as e:
                print(f"Skipping {demo_name}: {e}")
        
        # Plot
        create_3d_overlay_plot_with_kde(all_translations, all_demo_names)
        
        # Compute KDE
        kde = compute_kde(translations)
        x_min, y_min, z_min = translations.min(axis=0)
        x_max, y_max, z_max = translations.max(axis=0)

        grid_x, grid_y, grid_z = np.mgrid[x_min:x_max:15j, y_min:y_max:15j, z_min:z_max:15j]  # 15 points per axis

        # Stack grid points and evaluate the KDE
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        kde_values = np.exp(kde.score_samples(grid_points))  # KDE evaluation

        # Compute and print the min and max of the density values
        kde_min = kde_values.min()
        kde_max = kde_values.max()

        print(f"Min KDE Value: {kde_min}")
        print(f"Max KDE Value: {kde_max}")


if __name__ == "__main__":
    file_path = sys.argv[1]  # Get the HDF5 file path passed from the command line
    main(file_path)
