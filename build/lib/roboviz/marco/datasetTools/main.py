from roboviz.marco.datasetTools.TrajectoryDataset import TrajectoryDataset, IterableTrajectoryDataset
import sys
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, load_from_disk

# Use this to convert the Trajectory dataset to huggingface dataset.
# TODO: Need to make streaming available
def main(path):
  dataset = TrajectoryDataset(path)
  dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
  state = next(iter(dataloader))
  def data_generator():
    for sample in dataset:
      yield {"data": sample.numpy()}
  
  dset = Dataset.from_generator(data_generator)
  dset.save_to_disk("")  # change path to where you want to save the dataset

if __name__ == "__main__":
  s3_path = ""  # Path to S3 bucket
  dataset = IterableTrajectoryDataset(s3_path)
  dataloader = DataLoader(dataset, batch_size=32)
