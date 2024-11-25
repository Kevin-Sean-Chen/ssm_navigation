# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 08:47:06 2024

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

import numpy.ma as ma
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

# %% check tracks
### inital analysis for tracks during bar presentation and with or without wind
### plot tracks and kinematics to visualize results...

# %% for perturbed data
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/odor_vision/2024-11-14/'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-11-22'
target_file = "exp_matrix.pklz"

# List all subfolders in the root directory
subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
pkl_files = []

# Loop through each subfolder to search for the target file
for subfolder in subfolders:
    for dirpath, dirnames, filenames in os.walk(subfolder):
        if target_file in filenames:
            full_path = os.path.join(dirpath, target_file)
            pkl_files.append(full_path)
            print(full_path)

# pkl_files = pkl_files[6:9]
# pkl_files = pkl_files[4:]
print(pkl_files) 

# %%
ff = np.array([1,2,3])+0 ### bar-only
# ff = np.array([4,5,6])+0 ### wind
# ff = np.array([7,0]) ### after

ff = np.array([7,8,9,10])  ### short edge
ff = np.array([1,2, 11])  ### inverted short
ff = np.array([3,4,5])  ### inverted long
ff = np.array([18,19,20])
ff = np.array([13,16,12])

threshold_track_l = 60 * 2 
times = []
tracks = []
thetas = []
vxys = []
track_id = []
for ii in range(len(ff)):  
    with gzip.open(pkl_files[ff[ii]], 'rb') as f:
        print(pkl_files[ff[ii]])
        data = pickle.load(f)
    ### extract tacks
    n_tracks = np.unique(data['trjn'])
    for jj in n_tracks:
        pos = np.where(data['trjn']==jj)[0] # find track elements
        if sum(data['behaving'][pos]):  # check if behaving
            if len(pos) > threshold_track_l:
                
                ### make per track data
                theta = data['theta_smooth'][pos]
                temp_v = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
                temp_x = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                times.append(data['t'][pos])
                tracks.append(temp_x)
                vxys.append(temp_v)
                track_id.append(np.zeros(len(pos))+jj) 
                thetas.append(theta)

# %% vectorize for all data
vec_time = np.concatenate(times)  # time in trial
vec_vxy = np.concatenate(vxys)  # velocity
vec_xy = np.concatenate(tracks)  # position
vec_ids = np.concatenate(track_id)  # track ID
vec_theta = np.concatenate(thetas)

# %%
plt.figure()
for ii in range(len(tracks)):
    xy_i = tracks[ii]
    time_i = times[ii]
    plt.plot(xy_i[:,0], xy_i[:,1],'k')
    # plt.scatter(xy_i[:,0],xy_i[:,1],c=time_i,cmap='coolwarm',s=.1,vmin=np.min(time_i),vmax=np.max(time_i))
    # plt.scatter(xy_i[:,0],xy_i[:,1],c=time_i,cmap='coolwarm',s=.1,vmin=np.min(vec_time),vmax=np.max(vec_time))
                
# %%
v_bins = np.arange(-15,15,.1)
plt.figure()
# aa,bb = np.histogram(vec_vxy[:,1], bins=v_bins);
# plt.plot(bb[:-1],aa)
plt.hist(vec_vxy[:,0],50)
plt.yscale('log')

# %% checking for speeding
plt.figure()
for ii in range(len(tracks)):
    xy_i = tracks[ii]
    vxy_i = vxys[ii]
    # plt.plot(xy_i[:,1], vxy_i[:,1],'k-.',alpha=0.2)
    plt.plot(xy_i[:,0], thetas[ii],'k-.',alpha=0.04)
# plt.ylim([-30,30])
# plt.xlabel('x (mm)'); plt.ylabel('vy (mm/s)')
plt.xlabel('x (mm)'); plt.ylabel('heading (degrees from wind)')
