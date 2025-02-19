import streamlit as st
from roboviz.marco.process_hdf5 import extract_states, extract_one_demos
from roboviz.marco import marco_algorithm
import os
import pathlib

if __name__ == "__main__":
  container_root_path = os.getcwd()
  data_path = pathlib.Path(container_root_path) /  "expert_demos.hdf5"
  print(data_path)

  states = extract_states(data_path)
  one_demo = extract_one_demos(data_path)

  marco_algorithm.main(states, one_demo)
