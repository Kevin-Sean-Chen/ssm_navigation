# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 11:48:30 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import joblib

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% meta analysis
### gap crossing analysis
### also the first test from new pipeline with stimuli

# %% find target files
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-09-06\kevin' ### gap crossing data
target_file = "exp_matrix.joblib"
exp_type = 'decreasing gap 60s'
# exp_type = 'increasing gap 60s'

subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
target_files = []
# Loop through each subfolder to search for the target file
for subfolder in subfolders:
    # Get files directly in this subfolder (no recursion)
    files = os.listdir(subfolder)
    if target_file in files and exp_type in subfolder:
        full_path = os.path.join(subfolder, target_file)
        target_files.append(full_path)
        print(full_path)
        
# %% load data
data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
nf = len(target_files)
masks = []   # where there is nan
track_id = []  # record track id (file and track)
rec_tracks = []  # record the full track x,y
rec_signal = []  # record opto signal
times = []   # record time in epoch
thetas = []
cond_id = 0
threshold_track_l = 60*10

for ff in range(nf):
    ### load file
    print(ff)
    data = joblib.load(target_files[ff])
        
    ### extract tacks
    n_tracks = np.unique(data['trjn'])
    
    for ii in n_tracks:
        pos = np.where(data['trjn']==ii)[0] # find track elements
        # if sum(data['behaving'][pos]):  # check if behaving
        if 1==1: 
            if len(pos) > threshold_track_l:
                
                ### make per track data
                # temp = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos] , \
                                        # data['theta_smooth'][pos] , data['signal'][pos]))
                theta = data['theta'][pos]
                temp = np.stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]),1)#######
                temp_xy = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                                
                ### criteria
                mask_i = np.where(np.isnan(temp), 0, 1)
                mask_j = np.where(np.isnan(theta), 0, 1)
                mean_v = np.nanmean(np.sum(temp**2,1)**0.5)
                max_v = np.max(np.sum(temp**2,1)**0.5)
                # print(mean_v)
                if np.prod(mask_i)==1 and np.prod(mask_j)==1 and mean_v>.1 and max_v<30: #max_v<20:  ###################################### removing nan for now
                    data4fit.append(temp)  # get data for ssm fit
                    rec_tracks.append(temp_xy)  # get raw tracks
                    track_id.append(np.zeros(len(pos))+ii) 
                    rec_signal.append(data['signal'][pos].squeeze())
                    # rec_signal.append(np.ones((len(pos),1)))   ########################## hacking if needed
                    cond_id += 1
                    times.append(data['t'][pos])
                    thetas.append(theta)

# %% vectorize for simpliciy
vec_signal = np.concatenate(rec_signal)  # odor signal
vec_time = np.concatenate(times)  # time in trial
vec_vxy = np.concatenate(data4fit)  # velocity
vec_xy = np.concatenate(rec_tracks)  # position
vec_ids = np.concatenate(track_id)  # track ID

# %% visualization
pos = np.where(vec_signal>0)[0]
plt.figure()
plt.plot(vec_xy[:,0], vec_xy[:,1],'k,')
plt.plot(vec_xy[pos,0], vec_xy[pos,1],'r,')

# %% upwind when in signal
ntracks = len(rec_tracks)
upwindx = []
thre_signalt = 60*3

for ii in range(ntracks):
    tracki = rec_tracks[ii]
    signali = rec_signal[ii]
    pos = np.where(signali>0)[0]
    if len(pos)>thre_signalt:
        dx = tracki[pos[0],0] - tracki[pos[-1],0]
        upwindx.append(dx)
        
# %% compare MSD
# plt.figure()
# plt.violinplot([upwindx, dec_dx], positions=[1.2, 2], showmeans=True)
# # Formatting
# plt.xticks([1.2, 2], ["increase", "decrease"])
# plt.ylabel("upwind via tracking (mm)")

# %% search during crossing
window = 60*2  # window size in frames
lossx = np.array([75, 131, 183, 233])  ### for increasing
lossx = np.array([45, 105, 168, 233])  ### for decreasing
crossing_indices = {i: [] for i in range(len(lossx))}  # Dictionary to store indices for each condition
crossing_segments = {i: [] for i in range(len(lossx))}  # Dictionary to store track segments

for ii in range(ntracks):
    ### load track
    tracki = rec_tracks[ii]
    signali = rec_signal[ii]
    pos = np.where(signali>0)[0]
    ### if crossing
    if len(pos)>thre_signalt:
        for ll in range(len(lossx)):
            xi = tracki[:,0]
            cross_idx = np.where((xi[:-1] > lossx[ll]) & (xi[1:] <= lossx[ll]))[0]
            ### if crossed this one
            if len(cross_idx) > 0:
                # Found crossing point(s)
                for idx in cross_idx:
                    # Store crossing indices
                    crossing_indices[ll].append((ii, idx))
                    # Store track segment after crossing
                    if idx + window <= len(tracki):
                        segment = tracki[idx:idx+window]
                        seg_signal = signali[idx:idx+window]
                        pos_signal = np.where(seg_signal!=0)[0]
                        segment[pos_signal,:] = np.nan
                        crossing_segments[ll].append(segment)

# %% plots
plt.figure()
mean_dx, std_dx = [],[]
mean_dy, std_dy = [],[]
kk = 0
for ii in range(len(lossx)):#-1, -1, -1):  # Changed to iterate in reverse
    plt.subplot(1,4,kk+1)
    displaceix = np.zeros(len(crossing_segments[ii]))
    displaceiy = np.zeros(len(crossing_segments[ii]))
    for jj in range(len(crossing_segments[ii])):
        trackj = crossing_segments[ii][jj]
        plt.plot(trackj[:,0]-trackj[0,0], trackj[:,1]-trackj[0,1],'k-', alpha=0.1)
        displaceix[jj] = np.nanmean((trackj[:,0]-trackj[0,0])**2)
        displaceiy[jj] = np.nanmean((trackj[:,1]-trackj[0,1])**2)
    kk += 1
    plt.title(f'crossing at {lossx[ii]}mm')
    # plt.xlim([lossx[ii]-20, lossx[ii]+100])
    plt.ylim([-50, 50])
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

    mean_dx.append(np.nanmean(displaceix))
    std_dx.append(np.nanstd(displaceix)/jj**0.5)
    mean_dy.append(np.nanmean(displaceiy))
    std_dy.append(np.nanstd(displaceiy)/jj**0.5)

plt.figure()
plt.subplot(1,2,1)
plt.errorbar([1,2,3,4], mean_dx, yerr=std_dx, fmt='o')
plt.xlabel('gap order'); plt.ylabel('up wind displacement (x)')
plt.subplot(1,2,2)
plt.errorbar([1,2,3,4], mean_dy, yerr=std_dy, fmt='o')
plt.xlabel('gap order'); plt.ylabel('cross wind displacement (y)')
