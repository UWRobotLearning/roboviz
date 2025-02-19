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

# Compute Kernel Density Estimation
def compute_kde(translations, bandwidth=0.1):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(translations)
    return kde

# Create a 3D scatter plot with Plotly, overlaying all demos with KDE contours
def create_3d_overlay_plot_with_kde(all_translations, all_demo_names, title="Kernel Density Summary Plot"):
    # Create a list of traces for each demo and KDE density
    traces = []
    
    # Iterate through each demo's translations
    for i, translations in enumerate(all_translations):
        
        # Extract x, y, z coordinates for each demo
        x = translations[:, 0]
        y = translations[:, 1]
        z = translations[:, 2]
        
        # Compute KDE for each demo's translations
        kde = compute_kde(translations)
        
        # Create the density data for each demo (density evaluated at the demo's points)
        kde_values = np.exp(kde.score_samples(translations))  # KDE evaluation for original points
        
        # Create scatter plot for each demo with density values as the color
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=kde_values,  # Use KDE values as color
                colorscale='Viridis',  # Set the colorscale
                colorbar=dict(title="Density")  # Optionally, add a colorbar
            ),
            name=f"Demo {all_demo_names[i]}"
        )
        traces.append(trace)

        # Grid for KDE evaluation 
        # use the min/max of x, y, and z for the grid
        x_min, y_min, z_min = translations.min(axis=0)
        x_max, y_max, z_max = translations.max(axis=0)
        
        grid_x, grid_y, grid_z = np.mgrid[x_min:x_max:30j, y_min:y_max:30j, z_min:z_max:30j]  # 30 points per axis
        
        # Evaluate the KDE on the grid and stack grid points
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        kde_values_grid = np.exp(kde.score_samples(grid_points))  # KDE evaluation on the grid

        # Match grid dimensions
        kde_values_reshaped = kde_values_grid.reshape(grid_x.shape)
        
        # Create a 3D surface plot for the KDE density
        contour_trace = go.Surface(
            x=grid_x,
            y=grid_y,
            z=kde_values_reshaped,  # KDE values (density)
            colorscale='Viridis',  # Color scale for the surface
            colorbar=dict(
                title="Density",  # Title for the color bar
                tickvals=[],
                ticktext=[],  # Remove tick labels
                ticks=""
            ),
            opacity=0.7,  # Set opacity for surface to make it visible
            name=f'KDE Density {all_demo_names[i]}'
        )
        traces.append(contour_trace)
    
    # Define layout for the plot
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        legend=dict(title="Demos", x=0.8, y=0.9)
    )
    
    # Create and show the figure with all traces
    fig = go.Figure(data=traces, layout=layout)
    fig.show()



# Main code to load the data and create a 3D visualization with KDE
def main():
    # play_pushing.hdf5
    hdf5_file_path = 'expert_lampshade2_demos.hdf5'  # Change this to the directory where expert/ play data is located
    data_type = 'states'  # The key that holds the state data (translation + quaternion)
    obs_type = 'obs'
    
    with h5py.File(hdf5_file_path, 'r') as f:
        # List all the demos in the dataset
        demos = list(f['data'].keys())
        print("Demos found:", demos)
        
        # Store translation data and demo names
        all_translations = []
        all_demo_names = []
        
        # Iterate over all demos
        for demo_name in demos:
            try:
                states = load_data_from_hdf5(hdf5_file_path, demo_name, data_type, obs_type)
                
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

        grid_x, grid_y, grid_z = np.mgrid[x_min:x_max:30j, y_min:y_max:30j, z_min:z_max:30j]  # 30 points per axis

        # Stack grid points and evaluate the KDE
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        kde_values = np.exp(kde.score_samples(grid_points))  # KDE evaluation

        # Compute and print the min and max of the density values
        kde_min = kde_values.min()
        kde_max = kde_values.max()

        print(f"Min KDE Value: {kde_min}")
        print(f"Max KDE Value: {kde_max}")


if __name__ == "__main__":
    main()
