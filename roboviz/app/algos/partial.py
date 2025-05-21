import h5py
import numpy as np
import plotly.graph_objs as go
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
import sys
import argparse
from roboviz.s3_access.read_from_s3 import download_dataset
from roboviz.lerobot_reader.read_data import extract_states_grouped, extract_states_ungrouped

# =============================================================================
# Constants
# =============================================================================
MARGIN_RATIO     = 0.10   # ±10% around the median travelled distance
FULL_FRAC_THRESH = 0.90   # if ≥90% are “full,” override and mark all as full
MIN_CLUSTER_SIZE = 5      # min cluster size for HDBSCAN

# =============================================================================
# 1. Load trajectories
# =============================================================================
def load_trajectories(file_path):
    trajectories, demo_names = [], []
    with h5py.File(file_path, 'r') as hdf:
        for key in hdf['data']:
            grp = hdf['data'][key]
            if 'obs' in grp and 'states' in grp['obs']:
                traj = grp['obs']['states'][:]
                if traj.shape[0] > 0:
                    trajectories.append(traj)
                    demo_names.append(key)
    return trajectories, demo_names

# =============================================================================
# 2. Distance calculation
# =============================================================================
def calc_xyz_distance(traj):
    diffs = np.diff(traj, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))

# =============================================================================
# 3. Feature extraction (includes start/end positions)
# =============================================================================
def extract_features(trajs):
    feats = []
    for t in trajs:
        dist     = calc_xyz_distance(t)
        speeds   = np.linalg.norm(np.diff(t, axis=0), axis=1)
        mean_spd = speeds.mean() if speeds.size else 0
        std_spd  = speeds.std()  if speeds.size else 0
        start    = t[0]
        end      = t[-1]
        feats.append([
            dist,
            mean_spd,
            std_spd,
            start[0], start[1], start[2],
            end[0],   end[1],   end[2]
        ])
    return np.array(feats)

# =============================================================================
# 2-b. Resample helpers
# =============================================================================
def choose_resample_length(trajs):
    lengths = [t.shape[0] for t in trajs]
    return int(np.median(lengths))

def resample_trajectory(traj, new_length):
    orig_len, dim = traj.shape
    old_idx = np.arange(orig_len)
    new_idx = np.linspace(0, orig_len - 1, new_length)
    out = np.zeros((new_length, dim))
    for d in range(dim):
        out[:, d] = np.interp(new_idx, old_idx, traj[:, d])
    return out

# =============================================================================
# 3. Average helper
# =============================================================================
def average_trajectory(trajs):
    return np.mean(np.stack(trajs, axis=0), axis=0)

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

# =============================================================================
# 4. Full pipeline
# =============================================================================
def main():
    # download from s3 if necessary
    args = parse_cli()
    endpoint_url = "https://s3.kopah.uw.edu"
    bucket_name = 'roboviz-dataset'
    path = args.dataset_path
    if args.download is not None and not os.path.exists(args.dataset_path):
        if not os.path.exists(args.creds):
            print("credential file not found")
            return

        download_dataset(endpoint_url, bucket_name, args.download, args.dataset_path, args.creds)

    # 4.1 load ALL
    if path.split('.')[-1] == "hdf5":
        all_trajs, demo_names = load_trajectories(path)
    else:
        all_trajs = extract_states_grouped(path)
        
    if not all_trajs:
        print("No trajectories found; exiting.")
        return

    # 4.2 median‐distance filter
    dists       = [calc_xyz_distance(t) for t in all_trajs]
    med_dist    = np.median(dists)
    margin      = med_dist * MARGIN_RATIO
    print(f"Median travelled distance = {med_dist:.2f} ± {margin:.2f}")

    inlier_idxs  = [i for i, d in enumerate(dists) if abs(d - med_dist) <= margin]
    outlier_idxs = [i for i, d in enumerate(dists) if abs(d - med_dist) >  margin]

    mapping_all = {
        "full":      inlier_idxs.copy(),
        "partial":   [],
        "overshoot": []
    }

    proc_trajs = [all_trajs[i] for i in outlier_idxs]

    # 4.3 heuristic split
    endpoints       = np.array([t[-1] for t in proc_trajs])
    centroid        = endpoints.mean(axis=0)
    dists_end       = np.linalg.norm(endpoints - centroid, axis=1)
    radius          = np.percentile(dists_end, 75)
    xyz_dists_proc  = [calc_xyz_distance(t) for t in proc_trajs]
    med_xyz         = np.median(xyz_dists_proc)
    len_tol         = 1.5 * np.std(xyz_dists_proc)

    mapping_heur = {"full": [], "partial": [], "overshoot": []}
    for i, (de, L) in enumerate(zip(dists_end, xyz_dists_proc)):
        if de <= radius and abs(L - med_xyz) <= len_tol:
            mapping_heur["full"].append(i)
        elif L > med_xyz + len_tol:
            mapping_heur["overshoot"].append(i)
        else:
            mapping_heur["partial"].append(i)

    total = len(proc_trajs)
    if total > 0 and len(mapping_heur["full"]) / total >= FULL_FRAC_THRESH:
        print(f">= {int(FULL_FRAC_THRESH*100)}% full → marking ALL as full")
        mapping_heur = {"full": list(range(total)), "partial": [], "overshoot": []}

    # 4.4 clustering
    feats     = extract_features(proc_trajs)
    X         = StandardScaler().fit_transform(feats)
    clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE)
    labels    = clusterer.fit_predict(X)

    uniq, counts  = np.unique(labels, return_counts=True)
    main_cluster  = uniq[np.argmax(counts)]
    mapping_clust = {"full": [], "partial": [], "overshoot": []}
    for i, lab in enumerate(labels):
        if lab == main_cluster:
            mapping_clust["full"].append(i)
        else:
            L = xyz_dists_proc[i]
            if L > med_xyz + len_tol:
                mapping_clust["overshoot"].append(i)
            else:
                mapping_clust["partial"].append(i)

    # 4.5 weak‐label + decision tree
    y_weak = np.zeros(len(proc_trajs), dtype=int)
    for i in mapping_heur["partial"]:
        y_weak[i] = 1
    for i in mapping_heur["overshoot"]:
        y_weak[i] = 2

    clf    = DecisionTreeClassifier(max_depth=4).fit(X, y_weak)
    y_pred = clf.predict(X)

    mapping_temp = {"full": [], "partial": [], "overshoot": []}
    for i, lab in enumerate(y_pred):
        mapping_temp[["full","partial","overshoot"][lab]].append(i)

    # merge back into full index space
    for cls in mapping_temp:
        for j in mapping_temp[cls]:
            mapping_all[cls].append(outlier_idxs[j])

    # save mapping
    out_p = os.path.join(os.path.dirname(__file__), "../data/trajectory_mapping.pkl")
    with open(out_p, "wb") as f:
        pickle.dump(mapping_all, f)
    print("Final mapping saved to", out_p)

    # ──────────── 4.6 PLOT THE **AVERAGE** TRAJECTORIES ────────────
    # resample all originals to a fixed length for averaging
    target_len    = choose_resample_length(all_trajs)
    resampled_all = [resample_trajectory(t, target_len) for t in all_trajs]

    fig = go.Figure()
    styles = {
        "full":      dict(line=dict(color='blue',   dash='solid', width=3)),
        "partial":   dict(line=dict(color='red',    dash='dash',  width=2)),
        "overshoot": dict(line=dict(color='orange', dash='dot',   width=2))
    }

    for cls, idxs in mapping_all.items():
        if not idxs:
            continue
        # **average** the resampled curves in this class
        avg_curve = average_trajectory([resampled_all[i] for i in idxs])
        fig.add_trace(go.Scatter3d(
            x=avg_curve[:,0],
            y=avg_curve[:,1],
            z=avg_curve[:,2],
            mode='lines+markers',
            line=styles[cls]['line'],
            name=f"{cls.capitalize()} Trajectory"
        ))

    fig.update_layout(
        title="Average Trajectories by Class",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        legend=dict(title="Class"),
        height=800
    )

    html_out = os.path.join(os.path.dirname(__file__), "../static/partial.html")
    fig.write_html(html_out)
    print("Plot written to", html_out)


if __name__ == "__main__":
    main()