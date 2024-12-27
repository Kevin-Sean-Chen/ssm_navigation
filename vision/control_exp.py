# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:27:48 2024

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
### used to analyze a number of negative controls
### to plot tacks for showing bias to screen
### or to show time-aligned kinmetics to show response to stiumuli across trials

# %% for perturbed data
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-12-23\moving_bars'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-12-23\full_field_flash'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-12-23\blank'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-12-26'
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
pkl_files = sorted(pkl_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

# %% filing
### 12/23
specific_number = {0,1,2,3}#{4,5} #{0,3,4}#{1,2}#
### for blank
# ff = [
#     i for i, name in enumerate(pkl_files) 
#     if int(name.split("_")[-2][0]) in specific_number
# ]
### for flash
ff = [
    i for i, name in enumerate(pkl_files) 
    if int(name.split("_")[5][-1]) in specific_number
]  # 5,7

threshold_track_l = 60 * .51 #2 
times = []
tracks = []
thetas = []
vxys = []
track_id = []
signals = []
speeds = []
msd_x = []

for ii in range(len(ff)):  
    with gzip.open(pkl_files[ff[ii]], 'rb') as f:
        print(pkl_files[ff[ii]])
        data = pickle.load(f)
    ### extract tacks
    n_tracks = np.unique(data['trjn'])
    for jj in n_tracks-1:
        pos = np.where(data['trjn']==jj)[0] # find track elements
        # if sum(data['behaving'][pos]):  # check if behaving ##################### if applies
        mean_speed = np.mean(np.sqrt(data['vx_smooth'][pos]**2+data['vy_smooth'][pos]**2))
        # mean_speed = np.nanmean(np.abs(data['dtheta_smooth'][pos]))
        if mean_speed>0:  # check if behaving ##################### if applies
            if len(pos) > threshold_track_l:
                
                ### make per track data
                theta = data['dtheta_smooth'][pos]
                temp_v = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
                temp_x = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                times.append(data['t'][pos])
                
                # signal = data['signal'][pos]  ### if there is signal
                speeds.append(np.sqrt(data['vx_smooth'][pos]**2+data['vy_smooth'][pos]**2))
                tracks.append(temp_x)
                vxys.append(temp_v)
                track_id.append(np.zeros(len(pos))+jj) 
                thetas.append(theta)
                # signals.append(signal)
                
                n = len(pos)
                msd_x.append(np.array([np.mean((temp_x[t:,0] - temp_x[:n-t,0])**2) for t in range(n)]))

# %% vectorize for all data
vec_time = np.concatenate(times)  # time in trial
vec_vxy = np.concatenate(vxys)  # velocity
vec_xy = np.concatenate(tracks)  # position
vec_ids = np.concatenate(track_id)  # track ID
vec_theta = np.concatenate(thetas)  # theta
# vec_signal = np.concatenate(signal)  # odor signal
vec_msd_x = np.concatenate(msd_x)

# %% plot tracks
plt.figure()
for ii in range(len(tracks)):
    xy_i = tracks[ii]
    time_i = times[ii]
    # pos = np.where((times[ii]>30) & (times[ii]<90))[0]
    # pos = np.where((times[ii]<30))[0]
    pos = np.where((times[ii]>0))[0]
    plt.plot(xy_i[pos,0], xy_i[pos,1],'k',alpha=.5)
    
    # plt.plot(xy_i[:,0], xy_i[:,1],'k', alpha=.8)
    # plt.scatter(xy_i[:,0],xy_i[:,1],c=time_i,cmap='coolwarm',s=.1,vmin=np.min(time_i),vmax=np.max(time_i))
    # plt.scatter(xy_i[:,0],xy_i[:,1],c=time_i,cmap='coolwarm',s=.1,vmin=np.min(vec_time),vmax=np.max(vec_time))

# %% showing counts
plt.figure()
g = sns.jointplot(x=vec_xy[::30,0], y=vec_xy[::30,1], kind="kde")
g.set_axis_labels('x (mm)', 'y (mm)')

bias = len(np.where(vec_xy[:,1]>100)[0])/len(vec_xy)
print('bias: ', bias)

# %% turning analysis
###############################################################################
# %% align in time
time_align = []
dtheta_align = []
vel_align = []
speed_align = []
plt.figure()
for ii in range(len(tracks)):
    time_i = times[ii]
    dtheta_i = thetas[ii]
    speed_i = speeds[ii]
    vx_i = vxys[ii][:,0]
    pos = np.where(time_i<90)[0]
    # pos_v = np.where(vx_i>0)[0]
    # pos = np.intersect1d(pos, pos_v)
    # plt.plot(time_i[pos], np.abs(dtheta_i[pos]),'k', alpha=0.2)
    # plt.plot(time_i[pos], np.abs(speed_i[pos]),'k', alpha=0.2)
    time_align.append(time_i[pos])
    dtheta_align.append(dtheta_i[pos])
    vel_align.append(vx_i[pos])
    speed_align.append(speed_i[pos])
    
time_align = np.concatenate(time_align)
dtheta_align = np.concatenate(dtheta_align)
vel_align = np.concatenate(vel_align)
speed_align = np.concatenate(speed_align)

# %%
time_stim = np.arange(0,20,.1) ###
time_stim = np.arange(0,50,.5)
mean_dtheta = time_stim*0+np.nan
mean_speed = time_stim*0+np.nan
for tt in range(len(time_stim)-1):
    pos = np.where((time_align>time_stim[tt]) & (time_align<time_stim[tt+1]) )[0]
    mean_dtheta[tt] = np.nanmean(np.abs(dtheta_align[pos]))
    mean_speed[tt] = np.nanmean((speed_align[pos]))

plt.figure()
plt.plot(time_stim, mean_dtheta)
plt.xlabel('time (s)'); plt.ylabel('|degree|/s'); 
plt.ylim([10, 70])
plt.figure()
plt.plot(time_stim, mean_speed)
plt.xlabel('time (s)'); plt.ylabel('speed (mm/s)'); 
plt.ylim([2, 8])
