# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:49:04 2024

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

# %% make a list of data
# list with tracks vx,vy
# future can include theta and the sensory input...

# %%
### cutoff for short tracks
threshold_track_l = 60 * 20  # look at long-enough tracks

# Define the folder path
folder_path = 'C:/Users/ksc75/Downloads/ribbon_data_kc/'

# Use glob to search for all .pkl files in the folder
pkl_files = glob.glob(os.path.join(folder_path, '*.pklz'))

# Print the list of .pkl files
for file in pkl_files:
    print(file)

# %% concatenate across files in a folder
data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
nf = len(pkl_files)
masks = []
track_id = []
rec_tracks = []
cond_id = 0

for ff in range(nf):
    ### load file
    with gzip.open(pkl_files[ff], 'rb') as f:
        data = pickle.load(f)
        
    ### extract tacks
    n_tracks = np.unique(data['trjn'])
    
    for ii in n_tracks:
        pos = np.where(data['trjn']==ii)[0] # find track elements
        if sum(data['behaving'][pos]):  # check if behaving
            if len(pos) > threshold_track_l:
                
                ### make per track data
                # temp = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos] , \
                                        # data['theta_smooth'][pos] , data['signal'][pos]))
                temp = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
                # temp = np.stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]),1)#######
                
                temp_xy = np.column_stack((data['x'][pos] , data['y'][pos]))
                
                
                mask_i = np.where(np.isnan(temp), 0, 1)
                if np.prod(mask_i)==1:  ######################################removing nan for now
                    data4fit.append(temp)  # get data for ssm fit
                    rec_tracks.append(temp_xy)  # get raw tracks
                    track_id.append(np.array([ff,ii]))  # get track id
                    cond_id += 1
                masks.append(mask_i)
# %%
# %% quick ssm test
###############################################################################

# %% setup
num_states = 5
obs_dim = 2

# %%
data = data4fit*1 # Treat observations generated above as synthetic data.
N_iters = 100

## testing the constrained transitions class
hmm = ssm.HMM(num_states, obs_dim, observations="gaussian",  transitions="sticky")

hmm_lls = hmm.fit(data, method="em", num_iters=N_iters, init_method="kmeans")

plt.plot(hmm_lls, label="EM")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")
plt.show()

# %% filtering!
pick_id = 25
most_likely_states = hmm.most_likely_states(data4fit[pick_id])
track_i = rec_tracks[pick_id]

most_likely_states = most_likely_states[::6]
track_i = track_i[::6]

# Create a colormap for the two states
colors = ['red', 'blue']  # You can choose different colors for the two states
unique_states = np.unique(most_likely_states)
cmap = plt.get_cmap('tab10')

plt.figure(figsize=(8, 6))

# Loop over the unique states and plot the corresponding segments
# for i, state in enumerate(unique_states):
for ii in range(len(unique_states)):
    state_mask = np.where(most_likely_states==ii)[0]
    # Find where the trajectory is in the current state
    # state_mask = (state==most_likely_states)
    
    # Plot the trajectory segment with a different color
    plt.plot(track_i[state_mask,0], track_i[state_mask,1], 'o', color=cmap(ii), alpha=0.5)
    
# Add labels and legends
plt.title("state-code trajectories")
plt.xlabel("X")
plt.ylabel("Y")


# %% state analysis upon stim
###############################################################################
# %%

