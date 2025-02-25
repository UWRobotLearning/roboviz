import os
import h5py
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from torch import from_numpy
from torch.utils.data import IterableDataset, get_worker_info
import h5py
import s3fs

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
  
class IterableTrajectoryDataset(IterableDataset):
    def __init__(self, s3_path):
        """
        s3_path: A string representing the S3 URI, e.g., "s3://your-bucket/path/to/file.hdf5"
        """
        self.s3_path = s3_path
        self.fs = s3fs.S3FileSystem(key="",
                                    secret="",
                                    client_kwargs={})  # Input S3 credentials
    
    def __iter__(self):
        with self.fs.open(self.s3_path, 'rb') as f:
            h5_file = h5py.File(f, 'r')
            demos = list(h5_file["data"].keys())
            
            inds = np.argsort([int(elem[5:]) for elem in demos])
            demos = [demos[i] for i in inds]
            
            # If using multiple workers, split the demos among them
            worker_info = get_worker_info()
            if worker_info is not None:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                demos = demos[worker_id::num_workers]
            
            # This is currently yielding point-wise.
            # We can modify this in the future to return whatever we want
            for demo_key in demos:
                demo_grp = h5_file["data/{}".format(demo_key)]
                points = demo_grp["obs/states"]
                for i in range(points.shape[0]):
                    yield from_numpy(points[i])
            h5_file.close()
  

   

    