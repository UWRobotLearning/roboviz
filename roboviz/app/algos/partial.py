import h5py
import numpy as np
import plotly.graph_objs as go
from sklearn.cluster import HDBSCAN
import pickle
import math

# Global variable to store the mapping of original trajectories.
original_trajectory_mapping = {}

# =============================================================================
# 1. Load trajectories from an HDF5 file and demo names from the file
# =============================================================================
def load_trajectories(file_path):
    """
    Loads trajectories and demo names from the provided HDF5 file.
    Assumes that each demonstration (demo) has a nested 'obs/states' dataset.
    Returns:
      - trajectories: list of NumPy arrays, each with shape (n_steps, state_dim)
      - demo_names: list of corresponding demo names from the file
    """
    trajectories = []
    demo_names = []
    with h5py.File(file_path, 'r') as hdf:
        data_group = hdf['data']
        for demo_key in data_group:
            demo_group = data_group[demo_key]
            if 'obs' in demo_group and 'states' in demo_group['obs']:
                traj = demo_group['obs']['states'][:]
                if traj.shape[0] > 0:
                    trajectories.append(traj)
                    demo_names.append(demo_key)
    return trajectories, demo_names

# =============================================================================
# 2. Choose target resample length (dynamically determined)
# =============================================================================
def choose_resample_length(trajectories):
    lengths = [traj.shape[0] for traj in trajectories]
    return int(np.median(lengths))

# =============================================================================
# 3. Resample a trajectory using linear interpolation to a fixed length
# =============================================================================
def resample_trajectory(traj, new_length):
    original_length = traj.shape[0]
    state_dim = traj.shape[1]
    old_indices = np.arange(original_length)
    new_indices = np.linspace(0, original_length - 1, new_length)
    new_traj = np.zeros((new_length, state_dim))
    for d in range(state_dim):
        new_traj[:, d] = np.interp(new_indices, old_indices, traj[:, d])
    return new_traj

# =============================================================================
# 4. Feature extraction: flatten the entire resampled trajectory
# =============================================================================
def extract_features(traj):
    return traj.flatten()

# =============================================================================
# 4.1 Compute the differential entropy for a set of features
# =============================================================================
def compute_cluster_entropy(cluster_features):
    cov = np.cov(cluster_features, rowvar=False)
    d = cluster_features.shape[1]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov += np.eye(d) * 1e-6
        sign, logdet = np.linalg.slogdet(cov)
    entropy_val = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
    return entropy_val

# =============================================================================
# 5. Distance calculation for trajectory comparison (XYZ only)
# =============================================================================
def calc_xyz_distance(traj):
    diff_xyz = np.diff(traj, axis=0)
    step_dists = np.linalg.norm(diff_xyz, axis=1)
    return np.sum(step_dists)

# =============================================================================
# 6. Calculate average trajectory for a cluster (used for scoring)
# =============================================================================
def average_trajectory(trajs):
    return np.mean(np.stack(trajs, axis=0), axis=0)

# =============================================================================
# 7. Return the original trajectories mapping
# =============================================================================
def get_original_trajectory_mapping():
    global original_trajectory_mapping
    return original_trajectory_mapping

# =============================================================================
# 8. Main processing function
# =============================================================================
def main():
    global original_trajectory_mapping

    # 1. Load the original trajectories and demo names.
    hdf_path_expert = '/gscratch/scrubbed/roboviz/app/data/expert_lampshade2_demos.hdf5'
    original_trajectories, demo_names = load_trajectories(hdf_path_expert)
    if not original_trajectories:
        
        return
    
    # 2. Determine target resample length using the median length.
    target_length = choose_resample_length(original_trajectories)
    
    
    # 3. Create resampled trajectories for clustering/feature extraction (but keep originals).
    resampled_trajectories = [resample_trajectory(traj, target_length) for traj in original_trajectories]
    
    # 4. Extract features using the resampled trajectories.
    features = np.array([extract_features(traj) for traj in resampled_trajectories])
    
    # 5. Cluster using HDBSCAN (all clusters are retained).
    clusterer = HDBSCAN(min_cluster_size=5, min_samples=2, metric='euclidean')
    cluster_labels = clusterer.fit_predict(features)
    unique_labels = np.unique(cluster_labels)
    
    
    # (Optional) Compute differential entropy per cluster. (used in my old algo)
    cluster_entropies = {}
    for lbl in unique_labels:
        cluster_features = features[cluster_labels == lbl]
        if len(cluster_features) == 0:
            continue
        entropy_val = compute_cluster_entropy(cluster_features)
        cluster_entropies[lbl] = entropy_val
        
    
    # 6. Compute average (resampled) trajectory and its total XYZ distance per cluster.
    cluster_avg_trajs = {}
    cluster_xyz_distances = {}
    for label in unique_labels:
        indices = [i for i, lab in enumerate(cluster_labels) if lab == label]
        if not indices:
            continue
        cluster_trajs = [resampled_trajectories[i] for i in indices]
        avg_traj = average_trajectory(cluster_trajs)
        cluster_avg_trajs[label] = avg_traj
        
        distance = calc_xyz_distance(avg_traj)
        cluster_xyz_distances[label] = distance
        
    
    # 7. Identify the main cluster as the one with the highest XYZ distance.
    main_cluster = max(cluster_xyz_distances, key=cluster_xyz_distances.get)
    
    
    # 8. Build a mapping from demo name to classification ("full" or "partial").
    mapping = {}
    for i, lab in enumerate(cluster_labels):
        classification = "full" if lab == main_cluster else "partial"
        mapping[demo_names[i]] = classification
    
    original_trajectory_mapping = mapping
    with open("/gscratch/scrubbed/roboviz/app/data/trajectory_mapping.pkl", "wb") as f:
        pickle.dump(mapping, f)
    
    # 9. Create a Plotly figure for average trajectories only.
    fig = go.Figure()
    for label, avg_traj in cluster_avg_trajs.items():
        if label == main_cluster:
            trace = go.Scatter3d(
                x=avg_traj[:, 0],
                y=avg_traj[:, 1],
                z=avg_traj[:, 2],
                mode='lines+markers',
                marker=dict(symbol='circle', size=5),
                line=dict(color='blue', width=3),
                name=f'Cluster {label} (Main Trajectory)'
            )
        else:
            trace = go.Scatter3d(
                x=avg_traj[:, 0],
                y=avg_traj[:, 1],
                z=avg_traj[:, 2],
                mode='lines+markers',
                marker=dict(symbol='x', size=5),
                line=dict(color='red', width=2, dash='dash'),
                name=f'Cluster {label} (Partial Trajectory)'
            )
        fig.add_trace(trace)
    
    fig.update_layout(
        title="Average Trajectories by Cluster",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        legend=dict(title="Trajectories"),
        height=800
    )
    
    # Save the plot as an HTML file (to be displayed on the webpage).
    fig.write_html("static/plot.html")

if __name__ == '__main__':
    main()