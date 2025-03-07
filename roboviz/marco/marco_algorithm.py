from process_hdf5 import extract_states, extract_one_demos
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.neighbors import KernelDensity
import numpy as np
from pathlib import Path
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.express as px
import plotly.graph_objects as go

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

def plot(X, labels, centroids):
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection='3d')
  norm = mcolors.Normalize(vmin=np.min(labels), vmax=np.max(labels))
  color_map = plt.cm.rainbow(norm(labels))
  ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color_map, marker='o')

  ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='x', s=400)

  ax.set_xlabel("X-axis")
  ax.set_ylabel("Y-axis")
  ax.set_zlabel("Z-axis")
  ax.set_title("Clusters of captured points-expert play data")

  mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.rainbow)
  mappable.set_array(labels)
  cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=15, pad=0.1)
  cbar.set_label("labels")
  plt.show()

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

def plot_edges(edges):
  colors = px.colors.qualitative.Plotly
  fig = go.Figure()
  for i, (edge, points) in enumerate(edges.items()):
    # Extract x and y values from the list of points
    
    
    # Select a color from the palette; cycle if necessary
    color = colors[i % len(colors)]
    
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(color=color, size=8),
        name=f"Edge {edge}"
    ))
  return fig

def calculate_centroids(X, labels):
  centroids = np.zeros((0, 3))
  label_set = set(labels)
  for label in label_set:
    if label == -1:
      continue
    
    mask = (labels == label)
    x = X[mask]
    centroids = np.concat((centroids, np.mean(x, axis=0, keepdims=True)), axis=0)

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

"""
returns: (n, d) array where n is the number of edges. d is the dimension
"""
def split_edges(X, labels):
  assert len(X) == len(labels)
  labels = np.array(labels)
  indices = list(np.argwhere(labels >= 0).flatten())
  indices.insert(0, 0)
  indices.append(len(labels))
  res = {}
  edge = 0
  for i in range(len(indices) - 1):
    if i < len(indices) - 1:
      res[edge] = X[indices[i]:indices[i+1], :]
      edge += 1

  return res

def main(states, one_demo, min_cluster_size):
  X = states[:, :3]
  print(X.shape)
  X_demos = one_demo[:, :3]
  
  #TODO: tune the min_cluster_size
  clustering = hdbscan(X, min_cluster_size=min_cluster_size)
  
  labels = clustering.labels_
  centroids = clustering.centroids_
  epsilons = calculate_eps(X, centroids, set(labels), labels)
  mask = labels >= 0
  X = X[mask]
  labels = labels[mask]
  
  predicted_labels = hdbscan_predict(X_demos, centroids, epsilons)
  pred_mask = (predicted_labels != -1)
  edges = split_edges(X_demos, predicted_labels)

  #plot(X_demos, predicted_labels, centroids)
  #plot_plotly(X, labels, centroids)
  #plot_plotly(X_demos, predicted_labels, centroids)
  return plot_edges(edges)
  
if __name__ == "__main__":
  states = extract_states(sys.argv[1])
  print(states.shape)
  one_demo = extract_one_demos(sys.argv[1])
  min_cluster_size = int(0.1 * states.shape[0])
  main(states, one_demo, min_cluster_size)
  


  





