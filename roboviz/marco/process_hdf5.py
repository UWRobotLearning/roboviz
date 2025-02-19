import os
import h5py
import numpy as np

def extract_states(path):
  dataset_path = os.path.join("../", path)
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
    result = np.concat((result, points), axis=0)

  return result

def extract_one_demos(path):
  dataset_path = os.path.join(".", path)
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
      result = np.concat((result, points), axis=0)
      break

  return result
    