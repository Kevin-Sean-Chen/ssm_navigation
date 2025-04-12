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

import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% revisiting analysis of kinematics, conditioned on last encountrance
# start with vx,vy, dth
# analyze cross-wind and down-wind displacement
# systemetically condition on last encounter
### can later relate back to the modes...

# %% for Kiri's data
### cutoff for short tracks
threshold_track_l = 60 * 15  # 20 # look at long-enough tracks

# # Define the folder path
# folder_path = 'C:/Users/ksc75/Downloads/ribbon_data_kc/'

# # Use glob to search for all .pkl files in the folder
# pkl_files = glob.glob(os.path.join(folder_path, '*.pklz'))

# # Print the list of .pkl files
# for file in pkl_files:
#     print(file)

# %% for perturbed data
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/100424_new/'  ### for OU-ribbons
root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kiri_choi/data/ribbon_sleap/2024-9-17/'  ### for lots of ribbon data
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/odor_vision/2024-11-5'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2024-11-7'  ### for full field
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2024-10-31'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-3-20'  ### jittered ribbon
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-3-31'  ### jittered ribbon
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-4-4'  ### jittered ribbon
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
# pkl_files = pkl_files[:25]
# pkl_files = pkl_files[8:]
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
                if np.prod(mask_i)==1 and np.prod(mask_j)==1 and mean_v>.2 and max_v<20:  ###################################### removing nan for now
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

# %% visulization
ii = 0 #275 #tracks_stimed[50]
plt.figure()
plt.plot(rec_tracks[ii][:,0], rec_tracks[ii][:,1],'.'); 
pos=np.where(rec_signal[ii]>10)[0]; 
plt.plot(rec_tracks[ii][pos,0], rec_tracks[ii][pos,1],'r.'); plt.plot(rec_tracks[ii][0,0], rec_tracks[ii][0,1],'*')

# %% measuring base on tracks
pre_t = 30#45
stim_t = 30
odor_feature = []
post_vxy = []
post_xy = []
for nn in range(len(data4fit)):
    time_i = times[nn]
    signal_i = rec_signal[nn]
    xy_i = rec_tracks[nn]
    vxy_i = data4fit[nn]
    pos_stim = np.where((time_i>pre_t) & (time_i<pre_t+stim_t))[0]
    if np.nansum(signal_i)>0 and len(pos_stim)>0:  # some odor encounter
        pos = np.where(signal_i>0)[0][-1]  # last encounter
        # pos = pos_stim[-1]
        post_vxy.append(vxy_i[pos:,:])
        post_xy.append(xy_i[pos:,:])
        
        ### building features
        signal_vec = np.zeros_like(signal_i[pos_stim]) #,0]
        signal_vec[signal_i[pos_stim]>0] = 1  #,0]
        temp = (np.diff(signal_vec))
        # odor_feature.append(np.nanmean(signal_i))  # mean encounter
        
        ### pre-off behavior
        # xy_during = xy_i[pos_stim[0]:pos,:]
        # dxdy2 = np.linalg.norm(np.diff(xy_during), axis=0)
        # odor_feature.append(np.nanmean(dxdy2))
        
        ### number of encounter
        odor_feature.append( len(np.where(temp>0)[0]) )  # number of encounters
        # odor_feature.append(np.nanmean(vxy_i[pos_stim[0]:pos,0]**2))  # past behavior
        
        ### encounter time since last one
        # if len(np.where(temp>0)[0])>0:
        #     odor_feature.append( pos - np.where(temp>0)[0][-1] - 0*pos_stim[0])
        # else:
        #     odor_feature.append( pos - pos_stim[0])
        
# %% sorted plots
dispy = 3
offset = 1
post_window = 15*60

sortt_id = np.argsort(odor_feature)#[::-1]
import matplotlib.cm as cm
colors = cm.viridis(np.linspace(0, 1, len(sortt_id)))

plt.figure()
for kk in range(0,len(sortt_id),1):
    ### plot tracks
    traji = post_xy[sortt_id[kk]]
    if len(traji)<post_window:
        plt.plot(traji[:,0] - traji[0,0]*offset, kk*dispy + traji[0:,1]-traji[0,1]*offset, color=colors[kk])
    else:
        plt.plot(traji[:post_window,0] - traji[0,0]*offset, kk*dispy + traji[:post_window,1]-traji[0,1]*offset, color=colors[kk])
    plt.plot(traji[0,0]  - traji[0,0]*offset, kk*dispy +traji[0,1]-traji[0,1]*offset,'r.', markersize=2)
    
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
track_set = [post_xy[i] for i in sortt_id[:len(sortt_id)//2]]
# track_set = [post_xy[i] for i in sortt_id[len(sortt_id)//3:-len(sortt_id)//3]]
# track_set = [post_xy[i] for i in sortt_id[-len(sortt_id)//3:]]
max_lag = max(len(track) for track in track_set)
# max_lag = int(10*1/(1/60))
# Initialize arrays for MSD and counts
msd = np.zeros(max_lag)
counts = np.zeros(max_lag)
msd_std = msd*0

# Compute MSD
for track in track_set:
    n_points = len(track)
    for lag in range(1, n_points):
        displacements = track[lag:,:] - track[:-lag,:]  # Displacements for this lag
        squared_displacement = np.sum(displacements**2)#, axis=1)  # (dx^2 + dy^2)
        ##############################################################3 TRY <X^2> vs. <R^2>!!!
        msd[lag] += np.sum(squared_displacement)  # Sum displacements
        counts[lag] += len(displacements)  # Count valid pairs
        msd_std[lag] += np.sum(squared_displacement**2)
# Normalize to get the average MSD for each lag

# %%
msd_mean = msd / counts
variance_msd = (msd_std / counts) - (msd_mean**2)
sem_msd = np.sqrt(variance_msd) / counts**0.5 * 1
lag_times = np.arange(max_lag)*1/60  # Lag times

# %%
# Plot MSD
plt.figure(figsize=(8, 6))
plt.loglog(lag_times, msd_mean, marker='o', linestyle='-', color='k', label='long')
# plt.loglog(lag_times_mid, msd_mean_mid, marker='o', linestyle='-.', color='r', label='middle')
# plt.loglog(lag_times_last, msd_mean_last, marker='o', linestyle='--', color='b', label='short')
# plt.loglog(lag_times, lag_times**2 + msd_mean[1], marker='o', linestyle='-', color='g')
# plt.fill_between(
#     lag_times,
#     msd_mean - sem_msd,  # Lower bound of shaded region
#     msd_mean + sem_msd,  # Upper bound of shaded region
#     color='red',
#     alpha=0.5,
#     label='1 Std Dev'
# )
plt.xlabel("lag time (s)")
plt.ylabel(r"MSD (mm$^2$)")
plt.title("MSD scaling")
plt.grid(True)
plt.xlim([0,70]); plt.ylim([0.001, 13000])
# plt.show()
plt.legend()

