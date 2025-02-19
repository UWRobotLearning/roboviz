This program reads motion data stored in an HDF5 file and generates a 3D plot using Plotly. 
The plot visualizes the Kernel Density Estimation (KDE) of the translation data (x, y, z) from the given demos. 
The KDE helps visualize the density of state transitions across the given demos.

Features:

    Load Data from HDF5: The script loads state data from an HDF5 file (states key) containing multiple demos.
    
    Extract Translation Data: It extracts the translation part (x, y, z) from each state in the demo.
    
    KDE Calculation: A Kernel Density Estimation is computed for the translations of each demo.
    
    3D Plot with KDE Contours: It visualizes the KDE contours along with the data points in 3D space.
    
    Interactive Visualization: Uses Plotly for interactive 3D visualization, allowing users to rotate, zoom, and explore the KDE and demo data.
    
Required Imports:
    
    h5py: For reading data from the HDF5 file
    numpy: For numerical calculations
    plotly: For 3D plotting and visualization
    scikit-learn: For Kernel Density Estimation
    
You can install these by using the command: pip install h5py numpy plotly scikit-learn


**Make sure to change the path to your HDF5 file in the main() function so that it accesses the play/expert data correctly**
