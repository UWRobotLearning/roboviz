import h5py
import numpy as np
import plotly.graph_objs as go

# :oad data from HDF5 file
def load_data_from_hdf5(file_path, demo_name, data_type='states', obs_type='obs'):
    with h5py.File(file_path, 'r') as f:
        # Construct the path for the current observation
        data_path = f'data/{demo_name}/{obs_type}/{data_type}'
        
        # Check if the path exists
        if data_path not in f:
            raise KeyError(f"Key '{data_type}' not found at {data_path}.")
        
        # Load the data
        states = f[data_path][:]
        print(f"Loaded {data_type} from {obs_type} with shape: {states.shape} for {demo_name}")
        
        return states

# Extract translation data from the first 3 values from each state
def extract_translation_data(states):
    translations = states[:, :3]  # Get the translation part of the state
    return translations

# Create a 3D plot with all demos
def create_3d_overlay_plot(all_translations, all_demo_names, title="End Effector Pose -w- Translation for All Expert Lampshade Demos"):
    # Create a list of traces, one for each demo
    traces = []
    
    # Iterate through each demo's translations and create a trace for each one
    for i, translations in enumerate(all_translations):
        # Extract x, y, z coordinates for each demo
        x = translations[:, 0]
        y = translations[:, 1]
        z = translations[:, 2]
        
        # Create a trace for the 3D line plot (no markers)
        trace = go.Scatter3d(
            x=x, 
            y=y, 
            z=z,
            mode='lines',
            name=f'Demo {all_demo_names[i]}',
            marker=dict(
                size=5,
                opacity=0.8,
                colorscale='Viridis'
            ),
            line=dict(width=3)
        )
        
        traces.append(trace)
    
    # Layout for the plot
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

# Main function to load the data and create a 3D visualization of all demos
def main():
    hdf5_file_path = 'expert_lampshade2_demos.hdf5'
    data_type = 'states'  # The key that holds the state data (translation + quaternion)
    obs_type = 'obs'
    
    with h5py.File(hdf5_file_path, 'r') as f:

        # List all the demos in the dataset
        demos = list(f['data'].keys())
        print("Demos found:", demos)
        
        # Initialize lists to store translation data and demos
        all_translations = []
        all_demo_names = []
        
        for demo_name in demos:
            try:
                # Load the states data for each demo
                states = load_data_from_hdf5(hdf5_file_path, demo_name, data_type, obs_type)
                
                # Extract the translation data
                translations = extract_translation_data(states)
                all_translations.append(translations)
                all_demo_names.append(demo_name)

            except KeyError as e:
                print(f"Skipping {demo_name}: {e}")
        
        # Plot
        create_3d_overlay_plot(all_translations, all_demo_names)

if __name__ == "__main__":
    main()
