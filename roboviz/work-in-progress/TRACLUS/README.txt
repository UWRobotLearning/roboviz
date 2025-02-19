This mess of code is an attempt to implement TRACLUS (Trajectory Clustering), a method for clustering trajectory data into groups based on their spatial relationships. 
The provided code should allow you to load trajectory data from an HDF5 file, extract translation and velocity data, apply the TRACLUS algorithm, and visualize the results in a 3D scatter plot. 
But it does not work at the moment.
:(

Run main.py to see the soon-to-be demo in action! 
(There's something funny with the file dependencies, so I'm still working on cleaning that up.)

Required Imports:
    
    h5py: For reading data from the HDF5 file
    numpy: For numerical calculations
    plotly: For 3D plotting and visualization
    
You can install these by using the command: pip install h5py numpy plotly


**Make sure to change the path to your HDF5 file in the main() function so that it accesses the play/expert data correctly**

Link to TRACLUS GitHub repo:
https://github.com/apolcyn/traclus_impl/tree/master
