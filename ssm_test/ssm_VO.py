# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:03:51 2026

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

import pickle
import gzip
import glob
import os

import ssm
import numpy.random as npr
import seaborn as sns

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% load V+O data
folder_path = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\Cooper_data'
pkl_files = glob.glob(os.path.join(folder_path, '*.pkl'))

# Print the list of .pkl files
for file in pkl_files:
    print(file)

with open(pkl_files[0], 'rb') as file:
    data = pickle.load(file)
    
# %% build features for HMM fit
n_tracks = len(data)
data4fit = []
xy_pos = []

for ii in range(n_tracks):
    track_i = data[ii]
    
    ### find features
    vision_i = track_i['view_angle']
    speed_i = track_i['speed']
    odor_i = track_i['ribbon_engagement']
    x_i = track_i['x_position']
    y_i = track_i['y_position']
    
    # vision_i = np.mod(vision_i, 90)  ### hacking for now
    
    ### conditions
    if x_i[0]>100:
        # temp = np.column_stack((vision_i, odor_i, speed_i))
        temp = np.column_stack((vision_i, speed_i))
        data4fit.append(temp)
        temp_xy = np.column_stack((x_i, y_i))
        xy_pos.append(temp_xy)

# %%
# %% quick ssm test
###############################################################################
# %% setup
num_states = 3
obs_dim = 2

# %% inference
# data = data4fit*1 # Treat observations generated above as synthetic data.
N_iters = 100

## testing the constrained transitions class
hmm = ssm.HMM(num_states, obs_dim, observations="gaussian")#,  transitions="sticky")

hmm_lls = hmm.fit(data4fit, method="em", num_iters=N_iters, init_method="kmeans")

plt.figure()
plt.plot(hmm_lls, label="EM")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")
plt.show()   

# %% analysis
##############################################################################
# %% filtering!
pick_id = 14  # 0,7
most_likely_states = hmm.most_likely_states(data4fit[pick_id])
track_i = xy_pos[pick_id]

most_likely_states = most_likely_states[:] #:6
track_i = track_i[:] #:6

# Create a colormap for the two states
colors = ['red', 'blue']  # You can choose different colors for the two states
unique_states = np.unique(most_likely_states)
cmap = plt.get_cmap('tab10')

plt.figure(figsize=(8, 6))

# Loop over the unique states and plot the corresponding segments
# for i, state in enumerate(unique_states):
for ii in range(num_states): #(len(unique_states)):
    state_mask = np.where(most_likely_states==ii)[0]
    # Find where the trajectory is in the current state
    # state_mask = (state==most_likely_states)
    
    # Plot the trajectory segment with a different color
    plt.plot(track_i[state_mask,0], track_i[state_mask,1], 'o', color=cmap(ii), alpha=0.5)
    
# Add labels and legends
plt.title("state-code trajectories")
plt.xlabel("X")
plt.ylabel("Y")

# %% all tracks
plt.figure()
for ii in range(0, len(data4fit), 2):
    pick_id = ii*1
    most_likely_states = hmm.most_likely_states(data4fit[pick_id])
    track_i = xy_pos[pick_id]

    most_likely_states = most_likely_states[:] #:6
    track_i = track_i[:] #:6

    # Create a colormap for the two states
    unique_states = np.unique(most_likely_states)
    cmap = plt.get_cmap('tab10')

    # Loop over the unique states and plot the corresponding segments
    # for i, state in enumerate(unique_states):
    for ii in range(num_states): #(len(unique_states)):
        state_mask = np.where(most_likely_states==ii)[0]
        # Find where the trajectory is in the current state
        # state_mask = (state==most_likely_states)
        
        # Plot the trajectory segment with a different color
        plt.plot(track_i[state_mask,0], track_i[state_mask,1], ',', color=cmap(ii), alpha=0.5)
        
plt.xlabel("X")
plt.ylabel("Y")

# %% stats

plt.figure()
for ii in range(0, len(data4fit), 2):
    pick_id = ii*1
    most_likely_states = hmm.most_likely_states(data4fit[pick_id])
    track_i = xy_pos[pick_id]
    
    temp = data4fit[ii]
    speed_i = temp[:,1]
    angle_i = temp[:,0]
    
    for ii in range(num_states): #(len(unique_states)):
        state_mask = np.where(most_likely_states==ii)[0]
        plt.plot(speed_i[state_mask], angle_i[state_mask], ',', color=cmap(ii), alpha=0.5)