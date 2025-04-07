# segmentation.py
import importlib
import os
import h5py
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.neighbors import KernelDensity
import numpy as np
from pathlib import Path
import sys

import plotly.express as px
import plotly.graph_objects as go

def extract_states(path):
  dataset_path = os.path.join("/home/marco/Roboviz", path)
  assert os.path.exists(dataset_path)

  f = h5py.File(dataset_path, "r")

  demos = list(f["data"].keys())
  num_demos = len(demos)

  inds = np.argsort([int(elem[5:]) for elem in demos])
  demos = [demos[i] for i in inds]
  print(f"Num demos loaded: {len(demos)}")

  demo_key = demos[0]
  demo_grp = f["data/{}".format(demo_key)]
  
  result = np.zeros((0, demo_grp["obs/states"].shape[1]))

  for i, demo_key in enumerate(demos):
    demo_grp = f["data/{}".format(demo_key)]
    points = demo_grp["obs/states"]
    result = np.concatenate((result, points), axis=0)

  return result

def extract_one_demos(path):
  dataset_path = os.path.join("/home/marco/Roboviz", path)
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

def extract_states_trajectory_separated(path):
  dataset_path = os.path.join("/home/marco/Roboviz", path)
  print(dataset_path)
  assert os.path.exists(dataset_path)

  f = h5py.File(dataset_path, "r")

  demos = list(f["data"].keys())
  num_demos = len(demos)

  inds = np.argsort([int(elem[5:]) for elem in demos])
  demos = [demos[i] for i in inds]

  demo_key = demos[0]
  demo_grp = f["data/{}".format(demo_key)]
  
  result = {}
  for i, demo_key in enumerate(demos):
    demo_grp = f["data/{}".format(demo_key)]
    points = demo_grp["obs/states"]
    result[i] = np.array(points)

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
    labels[np.argmin(np.linalg.norm(X - center, axis=1))] = i
  
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

def plot_edges(multi_edges):
  colors = px.colors.qualitative.Plotly
  fig = go.Figure()
  for _, edges in multi_edges.items():
    for i, (edge, points) in enumerate(edges.items()):
      color = colors[i % len(colors)]
      
      fig.add_trace(go.Scatter3d(
          x=points[:, 0],
          y=points[:, 1],
          z=points[:, 2],
          mode='markers',
          marker=dict(color=color, size=8),
          name=f"Edge {edge}"
      ))
  fig.write_html('static/plot.html')

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
    if edge == 0:
      continue
    else:
      if len(res[edge]) < len(X) / (len(indices) * 2):
        res[edge - 1] = np.concatenate((res[edge - 1], res.pop(edge)), axis=0)
  

  return res

def main(states, trajectories, min_cluster_size):
  X = states[:, :3]
  print(X.shape)
  
  clustering = hdbscan(X, min_cluster_size=min_cluster_size)
  
  labels = clustering.labels_
  centroids = clustering.centroids_
  epsilons = calculate_eps(X, centroids, set(labels), labels)
  mask = labels >= 0
  X = X[mask]
  labels = labels[mask]
  
  #predicted_labels = hdbscan_predict(X_demos, centroids, epsilons)
  trajectories_labels = {}
  multi_edges = {}
  for (demo_num, points) in trajectories.items():
    trajectories_labels[demo_num] = hdbscan_predict(points[:, :3], centroids, epsilons)

    multi_edges[demo_num] = split_edges(points[:, :3], trajectories_labels[demo_num])

  plot_edges(multi_edges)
  
if __name__ == "__main__":
  trajectory_separated = extract_states_trajectory_separated(sys.argv[1])
  states = extract_states(sys.argv[1])
  one_demo = extract_one_demos(sys.argv[1])
  min_cluster_size = int(0.1 * states.shape[0])
  main(states, trajectory_separated, min_cluster_size)
  


  





