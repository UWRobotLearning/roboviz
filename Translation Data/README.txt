This program visualizes the translation data from multiple demos stored in an HDF5 file. 
It loads the data, extracts the translation components, and plots them in a 3D space, then overlays the trajectories from all demos in a single plot. 
The plot is interactive and color-coded according to the demo using Plotly.

Required Imports:
    
    h5py: For reading data from the HDF5 file
    numpy: For numerical calculations
    plotly: For 3D plotting and visualization
    
You can install these by using the command: pip install h5py numpy plotly


**Make sure to change the path to your HDF5 file in the main() function so that it accesses the play/expert data correctly**
