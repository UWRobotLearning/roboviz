import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch import from_numpy

class TrajectoryDataset(Dataset):
  def __init__(self, path) -> None:
    super().__init__()
    self.states = self.extract_states(path)

  def __len__(self):
    return self.states.shape[0]
  
  def __getitem__(self, idx):
    return self.states[idx]
  

  def extract_states(self, path):
    dataset_path = path
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
      result = np.concat((result, points), axis=0)

    return from_numpy(result)
  

   

    