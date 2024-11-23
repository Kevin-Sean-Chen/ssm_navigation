# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:03:03 2024

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

# %% revisiting analysis of kinematics, conditioned on last encountrance
# start with vx,vy, dth
# analyze cross-wind and down-wind displacement
# systemetically condition on last encounter
### can later relate back to the modes...

# %% for Kiri's data
### cutoff for short tracks
threshold_track_l = 60 * 20  # 20 # look at long-enough tracks

# Define the folder path
folder_path = 'C:/Users/ksc75/Downloads/ribbon_data_kc/'

# Use glob to search for all .pkl files in the folder
pkl_files = glob.glob(os.path.join(folder_path, '*.pklz'))

# Print the list of .pkl files
for file in pkl_files:
    print(file)

# %% for perturbed data
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/100424_new/'  ### for OU-ribbons
root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kiri_choi/data/ribbon_sleap/2024-9-17/'  ### for lots of ribbon data
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/odor_vision/2024-11-5'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2024-11-7'  ### for full field
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2024-10-31'
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

# pkl_files = pkl_files[8:]
# pkl_files = pkl_files[:30]
print(pkl_files) 

    
# %% concatenate across files in a folder
data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
nf = len(pkl_files)
masks = []   # where there is nan
track_id = []  # record track id (file and track)
rec_tracks = []  # record the full track x,y
rec_signal = []  # record opto signal
times = []   # record time in epoch
cond_id = 0

for ff in range(nf):
    ### load file
    with gzip.open(pkl_files[ff], 'rb') as f:
        data = pickle.load(f)
        
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
                thetas = data['theta'][pos]
                # temp = np.column_stack((data['headx'][pos] , data['heady'][pos]))
                temp = np.stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]),1)#######
                
                temp_xy = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                # temp_xy = np.column_stack((data['x'][pos] , data['y'][pos]))
                                
                ### criteria
                mask_i = np.where(np.isnan(temp), 0, 1)
                mask_j = np.where(np.isnan(thetas), 0, 1)
                mean_v = np.nanmean(np.sum(temp**2,1)**0.5)
                max_v = np.max(np.sum(temp**2,1)**0.5)
                # print(mean_v)
                if np.prod(mask_i)==1 and np.prod(mask_j)==1:# and mean_v>1 and max_v<20:  ###################################### removing nan for now
                    data4fit.append(temp)  # get data for ssm fit
                    rec_tracks.append(temp_xy)  # get raw tracks
                    # track_id.append(np.array([ff,ii]))  # get track id
                    track_id.append(np.zeros(len(pos))+ii) 
                    rec_signal.append(data['signal'][pos])
                    # rec_signal.append(np.ones((len(pos),1)))   ########################## hacking for now...
                    cond_id += 1
                    masks.append(thetas)
                    times.append(data['t'][pos])
                # masks.append(mask_i)

# %% vectorize for simpliciy
vec_signal = np.concatenate(rec_signal)  # odor signal
vec_time = np.concatenate(times)  # time in trial
vec_vxy = np.concatenate(data4fit)  # velocity
vec_xy = np.concatenate(rec_tracks)  # position
vec_ids = np.concatenate(track_id)  # track ID

# %% measuring base on tracks
odor_feature = []
post_vxy = []
post_xy = []
for nn in range(len(data4fit)):
    time_i = times[nn]
    signal_i = rec_signal[nn]
    xy_i = rec_tracks[nn]
    vxy_i = data4fit[nn]
    pos_stim = np.where((time_i>45) & (time_i<45+30))[0]
    if np.nansum(signal_i)>0 and len(pos_stim)>0:  # some odor encounter
        pos = np.where(signal_i>0)[0][-1]  # last encounter
        # pos = pos_stim[-1]
        post_vxy.append(vxy_i[pos:,:])
        post_xy.append(xy_i[pos:,:])
        
        ### building features
        signal_vec = signal_i[pos_stim,0]*0
        signal_vec[signal_i[pos_stim,0]>0] = 1
        temp = np.diff(signal_vec)
        # odor_feature.append(np.nanmean(signal_i))
        
        ### pre-off behavior
        # xy_during = xy_i[pos_stim,:]
        # dxdy2 = np.linalg.norm(np.diff(xy_i), axis=0)
        # odor_feature.append(np.nanmean(dxdy2))
        
        ### number of encounter
        # odor_feature.append( len(np.where(temp>1)[0]) )
        # odor_feature.append(np.mean(vxy_i[:pos,:]**2))
        
        ### encounter time
        if len(np.where(temp>0)[0])>0:
            odor_feature.append( pos - np.where(temp>0)[0][-1])
        else:
            odor_feature.append( pos )
        
# %% sorted plots
dispy = 1
offset = 1
post_window = 15*60

sortt_id = np.argsort(odor_feature)[::-1]
import matplotlib.cm as cm
colors = cm.viridis(np.linspace(0, 1, len(sortt_id)))

plt.figure()
for kk in range(0,len(sortt_id),3):
    ### plot tracks
    traji = post_xy[sortt_id[kk]]
    if len(traji)<post_window:
        plt.plot(traji[:,0] - traji[0,0]*offset, kk*dispy + traji[0:,1]-traji[0,1]*offset, color=colors[kk])
    else:
        plt.plot(traji[:post_window,0] - traji[0,0]*offset, kk*dispy + traji[:post_window,1]-traji[0,1]*offset, color=colors[kk])
    plt.plot(traji[0,0]  - traji[0,0]*offset, kk*dispy +traji[0,1]-traji[0,1]*offset,'r.')
    
    ### plot dots
    # traji = post_xy[sortt_id[kk]][:post_window,:]
    # post_feature = np.sum(traji[:,0]**2)**.5
    # post_feature = np.sum((traji[0,0] - traji[-1,0])**2)**.5
    # post_feature = ((traji[0,1] - traji[-1,1]))
    # plt.plot(odor_feature[sortt_id[kk]], post_feature,'o')

# %%
###############################################################################    
# %% MSD analysis!

sortt_id = np.array(sortt_id, dtype=int)
# track_set = post_xy[sortt_id[:len(sortt_id)//2]]  ## compare sorted
track_set = [post_xy[i] for i in sortt_id[-len(sortt_id)//2:]]
max_lag = max(len(track) for track in track_set)

# Initialize arrays for MSD and counts
msd = np.zeros(max_lag)
counts = np.zeros(max_lag)

# Compute MSD
for track in track_set:
    n_points = len(track_set)
    for lag in range(1, n_points):
        displacements = track[lag:] - track[:-lag]  # Displacements for this lag
        squared_displacement = np.sum(displacements**2, axis=1)  # (dx^2 + dy^2)
        msd[lag] += np.sum(squared_displacement)  # Sum displacements
        counts[lag] += len(squared_displacement)  # Count valid pairs

# Normalize to get the average MSD for each lag
msd = msd / counts

# Plot MSD
lag_times = np.arange(max_lag)  # Lag times
plt.figure(figsize=(8, 6))
plt.plot(lag_times, msd, marker='o', linestyle='-', color='b')
plt.xlabel("Lag Time")
plt.ylabel("Mean Squared Displacement (MSD)")
plt.title("MSD vs Lag Time")
plt.grid(True)
plt.show()

