# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:44:54 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
import glob
import os

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% simple track loading demonstration
###############################################################################
# loads exp_matrix data
# loops through trials
# extracts variables
# makes initial plots
###############################################################################

# %% load data
# folder root-directory
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-5-1'  ### straight, jittered ribbon and OU data

# extract through exp names
exp_type = 'jitter0p0_'
target_file = "exp_matrix.pklz"

# List all subfolders in the root directory
subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
pkl_files = []

# Loop through each subfolder to search for the target file
for subfolder in subfolders:
    for dirpath, dirnames, filenames in os.walk(subfolder):
        # if target_file in filenames:
        if target_file in filenames and exp_type in dirpath:
            full_path = os.path.join(dirpath, target_file)
            pkl_files.append(full_path)
            print(full_path)

# print the pkl files
print(pkl_files) 
pkl_files = sorted(pkl_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

# %% setting criteria
threshold_track_l = 60 * 5  # min track length in time
min_speed = 0  # min average speed

# %% extract list of tracks
times = []
tracks = []
thetas = []
vxys = []
track_id = []
signals = []
speeds = []
msd_x = []

ff = np.arange(len(pkl_files))
for ii in range(len(ff)):  # loop for trials
    with gzip.open(pkl_files[ff[ii]], 'rb') as f:
        print(pkl_files[ff[ii]])
        data = pickle.load(f)
    ### extract tacks
    n_tracks = np.unique(data['trjn'])
    for jj in n_tracks-1:  # loop for tracks
        pos = np.where(data['trjn']==jj)[0] # find track elementss
        mean_speed = np.mean(np.sqrt(data['vx_smooth'][pos]**2+data['vy_smooth'][pos]**2))
        if mean_speed> min_speed:  # check if moving
            if len(pos) > threshold_track_l:  # check if long enough
                
                ### make per track data list
                theta = data['dtheta_smooth'][pos]
                theta = data['theta_smooth'][pos]
                temp_v = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
                temp_x = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                signal = data['signal'][pos]  ### if there is no signal, comment this out
                
                times.append(data['t'][pos])
                speeds.append(np.sqrt(data['vx_smooth'][pos]**2+data['vy_smooth'][pos]**2))
                tracks.append(temp_x)
                vxys.append(temp_v)
                track_id.append(np.zeros(len(pos))+jj) 
                thetas.append(theta)
                signals.append(signal)

# %% vectorize for all data
vec_time = np.concatenate(times)  # time in trial
vec_vxy = np.concatenate(vxys)  # velocity
vec_xy = np.concatenate(tracks)  # position
vec_ids = np.concatenate(track_id)  # track ID
vec_theta = np.concatenate(thetas)  # theta
vec_signal = np.concatenate(signals)  # odor signal
vec_speed = np.concatenate(speeds)  # for speed

# %% simple plots
### speed histogram
plt.figure()
plt.hist(vec_speed, bins=100)
plt.xlabel('speed (mm/s)'); plt.ylabel('count'); plt.xlim([0,30])

### see tracks
track_id = 98
plt.figure()
plt.plot(tracks[track_id][:,0], tracks[track_id][:,1],'.')
pos=np.where(signals[track_id]>0)[0]
plt.plot(tracks[track_id][pos,0], tracks[track_id][pos,1],'r.')
plt.plot(tracks[track_id][0,0], tracks[track_id][0,1],'*')

### plot ensemble
plt.figure()
for ii in range(500):
    xy_i = tracks[ii]
    plt.plot(xy_i[:,0], xy_i[:,1],'k',alpha=.5)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')