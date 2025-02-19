from process_hdf5 import extract_states, extract_one_demos
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.neighbors import KernelDensity
import numpy as np

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

def dbscan(X, min_cluster_size=200):
  clustering = HDBSCAN(min_cluster_size=min_cluster_size).fit(X)
  return clustering

def hdbscan_predict(X, centroids, eps):
  labels = [-1] * X.shape[0]
  for i, center in enumerate(centroids):
    labels[np.argmin(np.linalg.norm(X - center, axis=1))] = i
  #for i in range(len(X)):
  #  x = X[i]
  #  distances = np.linalg.norm(centroids - x, axis=1)
  #  min_distance = np.min(distances)
  #  label = np.argmin(distances)
  #  if min_distance > eps[label]:
  #    labels.append(-1)
  #  else:
  #    labels.append(label)
  
  return labels

def plot(X, labels, centroids):
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

def calculate_eps(X, centroids, label_set):
  epsilons = []
  for label in label_set:
    if label == -1:
      continue

    mask = (labels == label)
    x = X[mask]
    distances = np.linalg.norm(x - centroids[label], axis=1)
    epsilons.append(np.max(distances))

  return epsilons

def main(states, one_demo):
  X = states[:, :3]
  X_demos = one_demo[:, :3]
  
  #TODO: tune the min_cluster_size
  clustering = dbscan(X, min_cluster_size=30)

  # plot datapoints
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection='3d')
  
  labels = clustering.labels_
  centroids = calculate_centroids(X, labels)
  epsilons = calculate_eps(X, centroids, set(labels))
  print(epsilons)
  mask = labels != -1
  X = X[mask]
  labels = labels[mask]
  print(centroids)
  
  predicted_labels = hdbscan_predict(X_demos, centroids, epsilons)
  pred_mask = (predicted_labels != -1)
  
  print(predicted_labels)

  #plot(X_demos, predicted_labels, centroids)
  plot_plotly(X, labels, centroids)
  
if __name__ == "__main__":
  states = extract_states("data/expert_lampshade2_demos.hdf5")
  one_demo = extract_one_demos("data/expert_lampshade2_demos.hdf5")

  main(states, one_demo)
  


  





