import streamlit as st
from roboviz.marco.process_hdf5 import extract_states, extract_one_demos
from roboviz.marco import marco_algorithm

if __name__ == "__main__":
  states = extract_states("../expert_demos.hdf5")
  one_demo = extract_one_demos("../expert_demos.hdf5")

  marco_algorithm.main(states, one_demo)
