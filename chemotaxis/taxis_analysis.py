# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:59:38 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

import pickle
import gzip
import glob
import os

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import h5py

# %% from bacteria chemotaxis data, define states, then compute inverse Q-learning
### then simulate data constrained RL agent
### then replace states with experimental modes
### then extend to time varying states...
### does it say something about expectations?

# %% load mat file for the data structure
file_dir = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/chemotaxis/170222-RP437_gradient_4X_001.swimtracker.mat'
# Open the .mat file
with h5py.File(file_dir, 'r') as file:
    # Access the structure
    your_struct = file['tracks']
    col_k = list(your_struct.keys())
    print(col_k)
    
# %% load specific fields
extract_data = {}
keys_of_interest = ['RCD', 'x', 'y', 'dx', 'dy']
n_tracks = 2000

with h5py.File(file_dir, 'r') as f:
    struct = f['tracks']
    keys = list(struct.keys())  # list of subfield names

    # Initialize empty lists for each subfield
    for key in keys_of_interest:
        extract_data[key] = []

    n_entries = len(struct[keys[0]])  # how many elements (assume consistent size)

    for i in range(n_tracks): #range(n_entries):
        print(i)
        for key in keys_of_interest:
            # Get the reference to the i-th object in the subfield
            ref = struct[key][i][0]  # [i][0] because it's stored in MATLAB HDF5 weirdly
            # Dereference and read data
            if isinstance(ref, h5py.Reference):
                obj = f[ref]
                value = obj[()]
                extract_data[key].append(value)
            else:
                extract_data[key].append(ref)

# %% visualize bacteria traces
plt.figure()
for ii in range(n_tracks):
    xi,yi = extract_data['x'][ii].squeeze(), extract_data['y'][ii].squeeze()
    plt.plot(xi, yi)