import lerobot
import torch
from pprint import pprint
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# Extract observation states grouped into its own episode indexes
# Return: list of numpy array t, where t is of shape (n, j).
#         n = trajectory/episode length
#         j = number of observed points
def extract_states_grouped(dataset_path):
  ds = LeRobotDataset(repo_id="local", root=dataset_path)
  hf = ds.hf_dataset
  res = []
  states = torch.stack(hf['observation.state'])
  ep = torch.stack(hf['episode_index'])
  for ep_index in torch.unique(ep):
    mask = torch.argwhere(ep == ep_index).squeeze()
    a = states[mask]
    res.append(a.cpu().numpy())

  return res

# Extract observation states grouped into its own episode indexes
# Return: numpy array of size (n, j)
#         n = total frame count
#         j = number of observed points
def extract_states_ungrouped(dataset_path):
  ds = LeRobotDataset(repo_id="local", root=dataset_path)
  hf = ds.hf_dataset
  return torch.stack(hf['observation.state']).cpu().numpy()
