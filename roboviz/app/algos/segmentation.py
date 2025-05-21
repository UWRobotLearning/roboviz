import importlib
import os
import h5py
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.neighbors import KernelDensity
import numpy as np
from pathlib import Path
import sys
import pickle

import plotly.express as px
import plotly.graph_objects as go
import argparse
from botocore.exceptions import ClientError

from roboviz.lerobot_reader.read_data import extract_states_grouped, extract_states_ungrouped
from roboviz.s3_access.read_from_s3 import download_dataset


def extract_states(dataset_path, index):
  assert os.path.exists(dataset_path)

  f = h5py.File(dataset_path, "r")
  index = set(index)

  demos = list(f["data"].keys())
  num_demos = len(demos)

  inds = np.argsort([int(elem[5:]) for elem in demos])
  demos = [demos[i] for i in inds]
  print(f"Num demos loaded: {len(demos)}")

  demo_key = demos[0]
  demo_grp = f["data/{}".format(demo_key)]
  
  result = np.zeros((0, demo_grp["obs/states"].shape[1]))

  for i, demo_key in enumerate(demos):
    if i in index:
      demo_grp = f["data/{}".format(demo_key)]
      points = demo_grp["obs/states"]
      result = np.concatenate((result, points), axis=0)

  return result

def extract_one_demos(dataset_path):
  assert os.path.exists(dataset_path)

  f = h5py.File(dataset_path, "r")
  demos = list(f["data"].keys())
  num_demos = len(demos)

  inds = np.argsort([int(elem[5:]) for elem in demos])
  demos = [demos[i] for i in inds]

  demo_key = demos[0]
  demo_grp = f["data/{}".format(demo_key)]
  
  result = np.zeros((0, demo_grp["obs/states"].shape[1]))

  for i, demo_key in enumerate(demos):
    demo_grp = f["data/{}".format(demo_key)]
    points = demo_grp["obs/states"]
    result = np.concatenate((result, points), axis=0)
    break

  return result

def extract_states_trajectory_separated(dataset_path, index):
  assert os.path.exists(dataset_path)

  f = h5py.File(dataset_path, "r")
  index = set(index)

  demos = list(f["data"].keys())
  num_demos = len(demos)

  inds = np.argsort([int(elem[5:]) for elem in demos])
  demos = [demos[i] for i in inds]

  demo_key = demos[0]
  demo_grp = f["data/{}".format(demo_key)]
  
  result = {}
  for i, demo_key in enumerate(demos):
    if i in index:
      demo_grp = f["data/{}".format(demo_key)]
      points = demo_grp["obs/states"]
      result[i] = np.array(points)

  return result

def extract_states_dict(grouped_data):
  result = {}
  for i, item in enumerate(grouped_data):
    result[f'demo_{i}'] = item
  return result

def cluster(X):
  kmeans = KMeans(n_clusters=2, init='k-means++')
  kmeans.fit(X)

def density(X):
  kde = KernelDensity().fit(X)
  return kde

def hdbscan(X, min_cluster_size=200):
  clustering = HDBSCAN(min_cluster_size=min_cluster_size, store_centers="centroid").fit(X)
  return clustering

def hdbscan_predict(X, centroids, eps):
  labels = [-1] * X.shape[0]
  for i, center in enumerate(centroids):
    distances = np.linalg.norm(X - center, axis=1)
    if np.min(distances) <= eps[i]:
      labels[np.argmin(distances)] = i
  
  return labels

def plot_plotly(X, labels, centroids):
    X = X[:, :3]
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=labels,
            colorscale='Rainbow',
            cmin=np.min(labels),
            cmax=np.max(labels),
            colorbar=dict(title="labels")
        ),
        name='Data Points'
    ))

    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers',
        marker=dict(
            size=7,
            color='red',
            symbol='x'
        ),
        name='Centroids'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            zaxis_title="Z-axis"
        ),
        title="Clusters of captured points-expert play data"
    )

    fig.show()
    fig.write_html('plot.html')

"""
Sample input = {'Trajectory_1' : {
  edge_1 = [..],
  edge_2 = [..],
}, 'Trajectory_2' : ...}
"""
def plot_edges(multi_edges, centroids):
  colors = px.colors.qualitative.Plotly
  fig = go.Figure()
  for index, edges in multi_edges.items():
    for i, (edge, points) in enumerate(edges.items()):
      color = colors[i % len(colors)]
      
      fig.add_trace(go.Scatter3d(
          x=points[:, 0],
          y=points[:, 1],
          z=points[:, 2],
          mode='markers',
          marker=dict(color=color, size=8),
          name=f"Trajectory : {index}, Edge {edge}"
      ))

  for i, centroid in enumerate(centroids):
    fig.add_trace(go.Scatter3d(
          x=[centroid[0]],
          y=[centroid[1]],
          z=[centroid[2]],
          mode='markers',
          marker=dict(color=color, size=12, symbol="x"),
          name=f"Centroid {i}"
      ))
  fig.write_html(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static/segmentation.html"))

def calculate_centroids(X, labels):
  centroids = np.zeros((0, 3))
  label_set = set(labels)
  for label in label_set:
    if label == -1:
      continue
    
    mask = (labels == label)
    x = X[mask]
    centroids = np.concatenate((centroids, np.mean(x, axis=0, keepdims=True)), axis=0)

  return centroids

def calculate_eps(X, centroids, label_set, labels):
  epsilons = []
  for label in label_set:
    if label == -1:
      continue

    mask = (labels == label)
    x = X[mask]
    distances = np.linalg.norm(x - centroids[label], axis=1)
    epsilons.append(np.max(distances))

  return epsilons

def split_edges(X, labels):
  assert len(X) == len(labels)
  labels = np.array(labels)
  indices = list(np.argwhere(labels >= 0).flatten())
  
  indices.insert(0, 0)
  indices.append(len(labels))

  res = {}
  edges = 0
  for i in range(len(indices) - 1):
    if i < len(indices) - 1:
      res[edges] = X[indices[i]:indices[i+1], :]
      edges += 1
  

  # coalesce edges that are too short
  for edge in range(edges):
    # buggy implementation
    if edge == 0 and edge + 1 in res:
      if len(res[edge]) < len(X) / (len(indices) * 2):
        res[edge + 1] = np.concatenate((res[edge + 1], res.pop(edge)), axis = 0)
    else:
      if len(res[edge]) < len(X) / (len(indices) * 2):
        res[edge - 1] = np.concatenate((res[edge - 1], res.pop(edge)), axis=0)

  return res

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
  # decide whether or not to download from s3
  args = parse_cli()
  endpoint_url = "https://s3.kopah.uw.edu"
  bucket_name = 'roboviz-dataset'
  dataset_path = args.dataset_path
  cur_dir = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(cur_dir, "../data/trajectory_mapping.pkl"), "rb") as f:
    mapping = pickle.load(f)

  if args.download is not None and not os.path.exists(args.dataset_path):
      if not os.path.exists(args.creds):
          print("credential file not found")
          return

      download_dataset(endpoint_url, bucket_name, args.download, args.dataset_path, args.creds)
  if dataset_path.split('.')[-1] == 'hdf5':
      full_trajectory_indexes = mapping["full"]
      trajectories = extract_states_trajectory_separated(dataset_path, full_trajectory_indexes)
      states = extract_states(dataset_path, full_trajectory_indexes)
  else:
    # it is a lerobot dataset
    states = extract_states_ungrouped(dataset_path)
    trajectories = extract_states_dict(extract_states_grouped(dataset_path))

  min_cluster_size = int(0.1 * states.shape[0])
  X = states[:, :3]
  print(X.shape)
  
  # trainining the clusterer and obtain cluster centers
  clustering = hdbscan(X, min_cluster_size=min_cluster_size)
  
  labels = clustering.labels_
  centroids = clustering.centroids_
  epsilons = calculate_eps(X, centroids, set(labels), labels)
  mask = labels >= 0
  X = X[mask]
  labels = labels[mask]
  
  # inference step and split trajectories to multiple edges
  #predicted_labels = hdbscan_predict(X_demos, centroids, epsilons)
  trajectories_labels = {}
  multi_edges = {}
  for (demo_num, points) in trajectories.items():
    trajectories_labels[demo_num] = hdbscan_predict(points[:, :3], centroids, epsilons)

    multi_edges[demo_num] = split_edges(points[:, :3], trajectories_labels[demo_num])

  plot_edges(multi_edges, centroids)
  
if __name__ == "__main__":
  main()
  