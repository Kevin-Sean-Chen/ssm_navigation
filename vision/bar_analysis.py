# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:21:58 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

import pickle
import gzip
import glob
import os
import re

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% analyze moving bar responses in the open arena
# stimulus onset at 3s and records for another 3s

# %% process single moving bar experiments
###############################################################################
# %% load all files
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\visual_behavior\2025-3-3'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\visual_behavior\2025-3-14'  ### processed with sleap tracking!
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\visual_behavior\2025-3-20'
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

print(pkl_files) 
pkl_files = sorted(pkl_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

# %% filing for relevant file names
### 3/3
# ff = np.array([1,2,3])+0 ### bar-only
# ff = np.arange(30,40)  ### 72 deg/s
# ff = np.arange(40,50)  ### 180 deg/s
# ff = np.arange(50,60)  ### 360 deg/s

### 3/14 data
def filter_files_by_number(file_list, target_number):
    target_number = str(target_number)
    pattern = rf"moving_bar_{target_number}_[^\\\/]+"  # Ensure number appears after "moving_bar_"
    # filtered_files = [f for f in file_list if re.search(pattern, f)]
    matching_indices = [i for i, f in enumerate(file_list) if re.search(pattern, f)]
    return matching_indices

### extract files
bar_speed = 72 ### 18, 72, 180, 360
stim_duration = 5  ### 1, 2, 5, 20 for 360, 180, 72, 18 deg/s
ff = filter_files_by_number(pkl_files, bar_speed)
threshold_track_l = 60 * 3 #2   #### set lenght criteria (in seconds)
min_mean_speed = 1
# ff = np.arange(30,40)

### sort the lists
times = []
tracks = []
thetas = []
dthetas = []
vxys = []
track_id = []
signals = []
speeds = []
msd_x = []
heading = []

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
        if mean_speed>min_mean_speed:  # check if behaving ##################### if applies
            if len(pos) > threshold_track_l:
                
                ### make per track data
                dtheta = data['dtheta_smooth'][pos]
                theta = data['theta'][pos]
                temp_v = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
                temp_x = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                temp_h = np.column_stack((data['headx_smooth'][pos] - data['x_smooth'][pos] , data['heady_smooth'][pos] - data['y_smooth'][pos])) ## heading vector
                times.append(data['t'][pos])
                
                signal = []#data['signal'][pos]  ### if there is signal
                speeds.append(np.sqrt(data['vx_smooth'][pos]**2+data['vy_smooth'][pos]**2))
                tracks.append(temp_x)
                vxys.append(temp_v)
                heading.append(temp_h)
                track_id.append(np.zeros(len(pos))+jj) 
                dthetas.append(dtheta)
                thetas.append(theta)
                signals.append(signal)
                
                n = len(pos)
                msd_x.append(np.array([np.mean((temp_x[t:,0] - temp_x[:n-t,0])**2) for t in range(n)]))

# %% vectorize for all data
vec_time = np.concatenate(times)  # time in trial
vec_vxy = np.concatenate(vxys)  # velocity
vec_xy = np.concatenate(tracks)  # position
vec_ids = np.concatenate(track_id)  # track ID
vec_dtheta = np.concatenate(dthetas)  # dtheta
vec_theta = np.concatenate(thetas)  # theta
# vec_signal = np.concatenate(signal)  # odor signal
vec_msd_x = np.concatenate(msd_x)
vec_heading = np.concatenate(heading)

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

# %% turning and running analysis
###############################################################################
# %% align in time
time_align = []
dtheta_align = []
vel_align = []
speed_align = []
plt.figure()
for ii in range(len(tracks)):
    time_i = times[ii]
    dtheta_i = dthetas[ii]
    speed_i = speeds[ii]
    vx_i = vxys[ii][:,0]
    # stim_i = signal[ii]
    pos = np.where(time_i>0)[0] #np.where((time_i<90) & (time_i>30))[0]  ### set extra criteria, not used here
    time_align.append(time_i[pos])
    dtheta_align.append(dtheta_i[pos])
    vel_align.append(vx_i[pos])
    speed_align.append(speed_i[pos])
    
time_align = np.concatenate(time_align)
dtheta_align = np.concatenate(dtheta_align)
vel_align = np.concatenate(vel_align)
speed_align = np.concatenate(speed_align)

# %% plot algined traces
dt = 0.05  # in seconds
# time_stim = np.arange(0,30,.1) ###
time_stim = np.arange(0,np.max(time_align),dt)
mean_dtheta = time_stim*0+np.nan
mean_speed = time_stim*0+np.nan
for tt in range(len(time_stim)-1):
    pos = np.where((time_align>time_stim[tt]) & (time_align<time_stim[tt+1]) )[0]
    mean_dtheta[tt] = np.nanmean(np.abs(dtheta_align[pos]))
    mean_speed[tt] = np.nanmean((speed_align[pos]))

plt.figure()
plt.plot(time_stim, mean_dtheta)
plt.xlabel('time (s)'); plt.ylabel('|degree|/s'); 
# plt.axvline(x=63, color='k', linestyle='--', linewidth=0.5); plt.axvline(x=63.5, color='k', linestyle='--', linewidth=0.5)
# plt.ylim([40, 96])
plt.figure()
plt.plot(time_stim, mean_speed)
plt.xlabel('time (s)'); plt.ylabel('speed (mm/s)'); 
# plt.axvline(x=63, color='k', linestyle='--', linewidth=0.5); plt.axvline(x=63.5, color='k', linestyle='--', linewidth=0.5)
# plt.ylim([5, 13])

# %% allocentric quadrant analysis
###############################################################################
# %% conditional analysis
stim_len = 1  # 1,2,5 for 360,180,72 deg/s
pre_stime = 3  # time before stimuli
pre_window = 30  # steps of 1/60 frame-rate
pre_angle = []
post_dtheta = []
post_time = []
for ii in range(len(tracks)):
    time_i = times[ii]
    dtheta_i = dthetas[ii]
    theta_i = thetas[ii]
    speed_i = speeds[ii]
    pos_pre = np.where(time_i<pre_stime)[0]
    pos_post = np.where(time_i>pre_stime+stim_len)[0]
    if len(pos_pre)>pre_window and len(pos_post)>0:  ### crossing stimuli zone
        pre_angle.append(np.nanmean(theta_i[pos_pre[-pre_window:]]))  # compute pre stim angle
        post_dtheta.append(dtheta_i[pos_pre[-1]:])  # dtheta_i
        post_time.append(time_i[pos_pre[-1]:])
        
pre_angle = np.array(pre_angle)
min_pos_wind = min([len(post_dtheta[ii]) for ii in range(len(post_dtheta))])

# %% conditional plot
conds = [[30, 330],
         [135, 45],
         [225, 135],
         [315, 225]] ### conditional angle bins
time_stim = np.arange(0,6,.1)+3
cols = ['r','g','b','k']
plt.figure()
for cc in range(0,len(conds)):
    angs_response = []
    time_response = []
    if cc==0:
        pos = np.where((pre_angle<conds[cc][0]) | (pre_angle>conds[cc][1]))[0]
    else:
        pos = np.where((pre_angle<conds[cc][0]) & (pre_angle>conds[cc][1]))[0]
    print(pos)
    for pp in range(len(pos)):
        # plt.plot(post_dtheta[pos[pp]], cols[cc],alpha=0.1)
        # angs_response.append(post_dtheta[pos[pp]][:min_pos_wind])
        angs_response.append(post_dtheta[pos[pp]][:])  ### for overall avrage
        time_response.append(post_time[pos[pp]])
    
    # angs_response = np.array(angs_response)
    # plt.plot(np.nanmean(angs_response,1), cols[cc])

    angs_response = np.concatenate(angs_response).reshape(-1)
    time_response = np.concatenate(time_response).reshape(-1)
    
    ## %% overall plot
    mean_dtheta = time_stim*0+np.nan
    for tt in range(len(time_stim)-1):
        pos = np.where((time_response>time_stim[tt]) & (time_response<time_stim[tt+1]) )[0]
        mean_dtheta[tt] = np.nanmean((angs_response[pos]))
    
    # plt.figure()
    plt.plot(time_stim, mean_dtheta, cols[cc])
    plt.xlabel('time (s)'); plt.ylabel('|degree|/s'); 
        
# %% egocentric analysis using heading
###############################################################################
# %% build stimulus vector
dt = 1/60  # in seconds
# stim_duration = 1  ### 1, 2, 5, 20 for 360, 180, 72, 18 deg/s
post_stim_duration = 3
pre_stim_duration = 3
time_full = np.arange(0,stim_duration + post_stim_duration,dt) + pre_stim_duration
time_stim = np.arange(0, stim_duration, dt)
stim_values = np.linspace(0, 360, num=len(time_stim))
post_stim_values = np.full(len(time_full)-len(time_stim), np.nan)
stim_vector = np.concatenate([post_stim_values, stim_values, post_stim_values])
time_vec = np.arange(0, pre_stim_duration+stim_duration+post_stim_duration, dt)

def find_stim_angle_at_time_t(time, time_vec=time_vec, stim=stim_vector):
    index = np.argmin(np.abs(time_vec - time))
    stim_angle = stim[index]
    return stim_angle
def wrap_heading(angle):
    return (angle + 180) % 360 - 180
def angle_between_vectors(u, v):
    u = np.asarray(u)
    v = np.asarray(v)
    angle = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrap_heading(angle*180/np.pi) ### return angles in degrees
def angle_between_vector_series(U, V):
    U = np.asarray(U)
    V = np.asarray(V)
    theta_u = np.arctan2(U[:, 1], U[:, 0])  # angle of each u_t
    theta_v = np.arctan2(V[:, 1], V[:, 0])  # angle of each v_t
    delta = theta_v - theta_u
    # delta = (delta + np.pi) % (2 * np.pi) - np.pi  # wrap to (-π, π]
    # return (np.degrees(delta))  #wrap_heading

    U = np.asarray(U)
    V = np.asarray(V)
    dot_products = np.einsum('ij,ij->i', U, V)                     # T-length array of dot products
    norms_U = np.linalg.norm(U, axis=1)
    norms_V = np.linalg.norm(V, axis=1)
    # Avoid division by zero and numerical issues
    cos_theta = np.clip(dot_products / (norms_U * norms_V), -1.0, 1.0)
    angles = np.arccos(cos_theta)
    cross = U[:, 0] * V[:, 1] - U[:, 1] * V[:, 0]  # Scalar value per pair
    signed_angles = angles * np.sign(cross)
    return np.degrees(signed_angles) 
    

# %% rebuild the stim angle
pre_window = 30  # steps of 1/60 frame-rate
delay = 20*1  ### test this here
stim_angle = []
fly_angle = []
rec_time = []
fly_response = []
stim_angle_vector = []
for ii in range(len(tracks)):
    time_i = times[ii]
    dtheta_i = dthetas[ii]
    theta_i = thetas[ii]
    speed_i = speeds[ii]
    head_i = heading[ii]
    pos_stim_ = np.where((time_i>pre_stim_duration) & (time_i<pre_stim_duration+stim_duration))[0]
    if len(pos_stim_)>delay:  # measurement during stim exists
    # if len(pos_stim)>0:
        pos_stim = pos_stim_[:-delay] ### make delays here
        pos_resp = pos_stim_[delay:]
        stim_angi = np.zeros(len(pos_stim))
        for tt in range(len(pos_stim)):
            stim_angi[tt] = find_stim_angle_at_time_t(time_i[pos_stim[tt]])
        
        fly_angle.append(theta_i[pos_resp])  # dtheta_i
        stim_angle.append( wrap_heading( stim_angi - theta_i[pos_resp]) )
        stim_angle_vector.append(angle_between_vector_series(np.array([np.cos(stim_angi), np.sin(stim_angi)]).T, head_i[pos_resp,:]))
        fly_response.append(dtheta_i[pos_stim])  # dtheta_i
        rec_time.append(time_i[pos_stim])
        
fly_angle = np.concatenate(fly_angle)
fly_response = np.concatenate(fly_response)
stim_angle = np.concatenate(stim_angle)
rec_time = np.concatenate(rec_time)
stim_angle_vector = np.concatenate(stim_angle_vector)

## %% test with simple binning
def bin_and_average_with_error(x, y, bins):
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid_indices]
    y = y[valid_indices]
    # Compute bin edges if bins is an integer
    if isinstance(bins, int):
        bin_edges = np.linspace(np.min(x), np.max(x), bins + 1)  # Ensure correct bin count
    else:
        bin_edges = np.asarray(bins)

    num_bins = len(bin_edges) - 1  # Ensure bins match bin centers

    # Digitize x values into bins (returns bin indices in range [1, num_bins]), subtract 1 for zero-based index
    bin_indices = np.digitize(x, bin_edges) - 0

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute mean y for each bin
    y_sums = np.bincount(bin_indices, weights=y, minlength=num_bins)[:-1]
    y_counts = np.bincount(bin_indices, minlength=num_bins)[:-1]

    # Avoid division by zero
    valid_bins = y_counts > 0  # Ensure the boolean mask matches bin_centers size
    y_means = np.full(num_bins+1, np.nan)  # Initialize with NaNs
    y_means[valid_bins] = y_sums[valid_bins] / y_counts[valid_bins]

    # Compute standard deviation and SEM for each bin
    y_squared_sums = np.bincount(bin_indices, weights=y**2, minlength=num_bins)[:-1]
    y_variance = np.full(num_bins+1, np.nan)  # Initialize with NaNs
    y_variance[valid_bins] = (y_squared_sums[valid_bins] / y_counts[valid_bins]) - y_means[valid_bins]**2
    y_std = np.sqrt(y_variance)  # Standard deviation
    y_sem = np.full(num_bins+1, np.nan)  # Initialize with NaNs
    y_sem[valid_bins] = y_std[valid_bins] / np.sqrt(y_counts[valid_bins])

    return bin_edges[valid_bins], y_means[valid_bins], y_sem[valid_bins]

# Bin and compute averages with error bars
delay = 1  ### need to think about how to choose this time lag for response
# bin_centers, y_means, y_sem = bin_and_average_with_error(stim_angle[:-delay], fly_response[delay:], bins=20) ### through angles
bin_centers, y_means, y_sem = bin_and_average_with_error(stim_angle_vector[:-delay], fly_response[delay:], bins=20)  ### through vectors

# Plot Results
plt.figure(figsize=(8, 5))
plt.errorbar(bin_centers, y_means, yerr=y_sem, fmt='o', capsize=5, label="Binned Averages")
plt.xlabel("stimulus angle to heading (deg)")
plt.ylabel("fly response angle (deg/s)")
# plt.ylim([-7,16])