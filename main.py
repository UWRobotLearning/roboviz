from roboviz.marco.datasetTools.TrajectoryDataset import TrajectoryDataset
import sys
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, load_from_disk



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
  main(sys.argv[1])
  