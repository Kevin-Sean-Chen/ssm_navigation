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
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-11-22'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-11-25'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-11-29'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-12-2'
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
print(pkl_files) 

# %% filing
### 11/14
ff = np.array([1,2,3])+0 ### bar-only
# ff = np.array([4,5,6])+0 ### wind
# ff = np.array([7,0]) ### after

### 11/22
ff = np.array([7,8,9,10])  ### short edge
ff = np.array([1,2, 11])  ### inverted short
ff = np.array([3,4,5])  ### inverted long
ff = np.array([18,19,20])
ff = np.array([13,16,12])

### 11/25
ff = np.arange(22,30)  # iso-d1
# ff = np.arange(14,22)  # OCL flies

### 11/29
# ff = np.arange(8,22)  # LED on
# ff = np.arange(23,32)
# ff = np.arange(44,51)
# ff = np.arange(41,47)
# ff = np.arange(51,54)

### 12/2
ff = np.arange(21,25)
ff = np.arange(25,30)

threshold_track_l = 60 * 2 
times = []
tracks = []
thetas = []
vxys = []
track_id = []
signals = []

for ii in range(len(ff)):  
    with gzip.open(pkl_files[ff[ii]], 'rb') as f:
        print(pkl_files[ff[ii]])
        data = pickle.load(f)
    ### extract tacks
    n_tracks = np.unique(data['trjn'])
    for jj in n_tracks-1:
        pos = np.where(data['trjn']==jj)[0] # find track elements
        if sum(data['behaving'][pos]):  # check if behaving
            if len(pos) > threshold_track_l:
                
                ### make per track data
                theta = data['dtheta_smooth'][pos]
                temp_v = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
                temp_x = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                times.append(data['t'][pos])
                
                signal = data['signal'][pos]  ### if there is signal
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
# vec_signal = np.concatenate(signal)  # odor signal

# %% plot tracks
plt.figure()
for ii in range(len(tracks)):
    xy_i = tracks[ii]
    time_i = times[ii]
    # pos = np.where((times[ii]>60) & (times[ii]<90))[0]
    # pos = np.where((times[ii]<30))[0]
    pos = np.where((times[ii]>0))[0]
    plt.plot(xy_i[pos,0], xy_i[pos,1],'k',alpha=.5)
    
    # plt.plot(xy_i[:,0], xy_i[:,1],'k')
    # plt.scatter(xy_i[:,0],xy_i[:,1],c=time_i,cmap='coolwarm',s=.1,vmin=np.min(time_i),vmax=np.max(time_i))
    # plt.scatter(xy_i[:,0],xy_i[:,1],c=time_i,cmap='coolwarm',s=.1,vmin=np.min(vec_time),vmax=np.max(vec_time))

# %% turning analysis
###############################################################################
# %% align in time
time_align = []
dtheta_align = []
plt.figure()
for ii in range(len(tracks)):
    time_i = times[ii]
    dtheta_i = thetas[ii]
    pos = np.where(time_i<40)[0]
    plt.plot(time_i[pos], np.abs(dtheta_i[pos]),'k', alpha=0.2)
    time_align.append(time_i[pos])
    dtheta_align.append(dtheta_i[pos])
    
time_align = np.concatenate(time_align)
dtheta_align = np.concatenate(dtheta_align)

# %%
time_stim = np.arange(0,40,.4)
mean_dtheta = time_stim*0+np.nan
for tt in range(len(time_stim)-1):
    pos = np.where((time_align>time_stim[tt]) & (time_align<time_stim[tt+1]))[0]
    mean_dtheta[tt] = np.nanmean(np.abs(dtheta_align[pos]))

plt.figure()
plt.plot(time_stim, mean_dtheta)
plt.xlabel('time (s)'); plt.ylabel('|degree|/s'); plt.ylim([15, 50])

# %%
###############################################################################

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
