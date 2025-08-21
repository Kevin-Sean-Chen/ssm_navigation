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
bar_speed = 360 ### 18, 72, 180, 360
stim_duration = 1  ### 1, 2, 5, 20 for 360, 180, 72, 18 deg/s
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
pre_window = int(1* 60)  # steps of 1/60 frame-rate
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
conds = [[45, 315],
         [135, 45],
         [225, 135],
         [315, 225]] ### conditional angle bins
time_stim = np.arange(0,6,.1)+0
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
time_full = np.arange(0, stim_duration + post_stim_duration,dt) + pre_stim_duration
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

# %% test code in real space
stim_pix = np.linspace(0, 640, num=len(time_stim))
post_stim_values = np.full(len(time_full)-len(time_stim), np.nan)
stim_pix_vector = np.concatenate([post_stim_values, stim_pix, post_stim_values])
time_vec_trial = np.arange(0, pre_stim_duration+stim_duration+post_stim_duration, dt)
pix2mm = 1.5 #1.5#1.5
def map_to_rectangle_path(n_points=641):
    """
    Map indices 0 to n_points-1 along a rectangle path.

    Rectangle vertices:
    (-96,64) -> (96,64) -> (96,-64) -> (-96,-64) -> (-96,64)
    """
    x = np.zeros(n_points)
    y = np.zeros(n_points)

    for idx in range(n_points):
        if idx <= 191:  # top edge
            x[idx] = -96 + (192 / 192) * idx  # linear from -96 to 96
            y[idx] = 64
        elif idx <= 319:  # right edge
            x[idx] = 96
            y[idx] = 64 - (128 / 128) * (idx - 192)  # linear from 64 to -64
        elif idx <= 511:  # bottom edge
            x[idx] = 96 - (192 / 192) * (idx - 320)  # linear from 96 to -96
            y[idx] = -64
        else:  # left edge
            x[idx] = -96
            y[idx] = -64 + (128 / 128) * (idx - 512)  # linear from -64 to 64

    return (x[:-1] + 96)*pix2mm, (y[:-1] + 64)*pix2mm  #### adding offset and rescale for now to apply to the same reference ###

x_bar,y_bar = map_to_rectangle_path()  ### get the real-space x,y coordinates

def find_stim_location_at_time_t(time,  time_vec=time_vec_trial, stim_pix_vector=stim_pix_vector, x=x_bar, y=y_bar):
    index = np.argmin(np.abs(time_vec - time)) # fine index with time
    # if np.isnan(stim_pix_vector[index]) is False:
        # print(stim_pix_vector[index])
    pixel_location = int(stim_pix_vector[index])-1  # find pixel index
    stim_location = np.array([x[pixel_location], y[pixel_location]]) # return x,y location
    # else:
    #     stim_location = np.ones(2)+np.nan
    return stim_location

# %% rebuild the stim angle
pre_window = 30  # steps of 1/60 frame-rate
delay = 1*1  ### test this here
stim_angle = []
fly_angle = []
rec_time = []
fly_response = []
stim_angle_vector = []
stim_ang_bar = []
valid_tracks = []
pseudo_lab_ang = []

for ii in range(len(tracks)):
    time_i = times[ii]
    dtheta_i = dthetas[ii]
    theta_i = thetas[ii]
    speed_i = speeds[ii]
    head_i = heading[ii]
    pos_i = tracks[ii]
    vxy_i = vxys[ii]
    pos_stim_ = np.where((time_i>pre_stim_duration) & (time_i<pre_stim_duration+stim_duration))[0]
    if len(pos_stim_)>delay+0:  # measurement during stim exists
    # if len(pos_stim)>0:
        valid_tracks.append(ii)
        pos_stim = pos_stim_[:-delay] ### make delays here
        pos_resp = pos_stim_[delay:]
        stim_angi = np.zeros(len(pos_stim))
        stim_ang_box = np.zeros(len(pos_stim))
        lab_fly_ang = np.zeros(len(pos_stim))
        test_bear = np.zeros(len(pos_stim))
        # dxy = np.diff(np.concatenate((pos_i,pos_i[0,:][None,:])).T).T
        dxy = vxy_i*1
        for tt in range(0,len(pos_stim)):
            stim_ang_box[tt] = find_stim_angle_at_time_t(time_i[pos_stim[tt]])  ### use angle directly

            ### test with bar location to calculate view angle
            stim_xyt = find_stim_location_at_time_t(time_i[pos_stim[tt]])
            # view_angle = angle_between_vectors(  head_i[pos_stim[tt],:] , stim_xyt - pos_i[pos_stim[tt]]) # angle between bar and heading
            # view_angle = angle_between_vectors(  stim_xyt - pos_i[pos_stim[tt]] , head_i[pos_stim[tt],:]) # angle between bar and heading, flipped
            view_angle = angle_between_vectors( np.array([1, 0]) , stim_xyt - pos_i[pos_stim[tt]]) # angle between bar vector and reference
            stim_angi[tt] = view_angle - theta_i[pos_stim[tt]] ### # difference between bar and heading
            lab_fly_ang[tt] = angle_between_vectors(  np.array([1, 0]) , stim_xyt - pos_i[pos_stim[tt]])
            test_bear[tt] = angle_between_vectors( dxy[pos_stim[tt],:] , dxy[pos_stim[tt-1],:] )
        
        test_bear[np.abs(test_bear)>50] = np.nan
        fly_angle.append(theta_i[pos_stim])  # dtheta_i
        stim_angle.append( wrap_heading( stim_angi - theta_i[pos_stim]) )
        # stim_angle_vector.append(angle_between_vector_series(np.array([np.cos(stim_angi), np.sin(stim_angi)]).T, head_i[pos_resp,:]))
        stim_angle_vector.append( wrap_heading(stim_angi) )
        fly_response.append(dtheta_i[pos_resp])  # dtheta_i
        # fly_response.append(test_bear) 
        rec_time.append(time_i[pos_stim])
        stim_ang_bar.append(stim_ang_box)
        pseudo_lab_ang.append(lab_fly_ang)
        
fly_angle_ = np.concatenate(fly_angle)
fly_response_ = np.concatenate(fly_response)
stim_angle_ = np.concatenate(stim_angle)
rec_time_ = np.concatenate(rec_time)
stim_angle_vector_ = np.concatenate(stim_angle_vector)

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
# delay = 5  ### need to think about how to choose this time lag for response
# bin_centers, y_means, y_sem = bin_and_average_with_error(stim_angle_[:-delay], fly_response_[delay:], bins=20) ### through angles
bin_centers, y_means, y_sem = bin_and_average_with_error(stim_angle_vector_[:-delay], fly_response_[delay:], bins=20)  ### through vectors

# Plot Results
plt.figure(figsize=(8, 5))
plt.errorbar(bin_centers, y_means, yerr=y_sem, fmt='o', capsize=5, label="Binned Averages")
plt.xlabel("stimulus angle to heading (deg)")
plt.ylabel("fly response angle (deg/s)")
# plt.ylim([-7,16])

# %% track based analysis
###############################################################################
# %% exp track
trk = 97 #41 #18,19 # 2 not moving 5 for good response... might have error in heading ??
plt.figure()
plt.subplot(311); plt.plot(fly_response[trk]); plt.ylabel(r'$d\theta$')
# plt.subplot(312); plt.plot(np.rad2deg(np.unwrap(np.deg2rad(stim_angle_vector[trk]))),'o')
plt.subplot(312); plt.plot(stim_angle_vector[trk],'o'); plt.plot([0, len(stim_angle_vector[trk])],[0,0],'k--'); plt.ylabel(r'$\phi$')
# plt.subplot(313); plt.plot(rec_time[trk] - rec_time[trk][0], stim_ang_bar[trk]); plt.ylabel('bar'); #plt.xlabel('time (s)')
plt.subplot(313); plt.plot(rec_time[trk] - rec_time[trk][0], pseudo_lab_ang[trk] , '.'); plt.ylabel('bar'); plt.xlabel('time (s)')

# %% show valid tacks
trid = valid_tracks[trk]
pos_stim_ = np.where((times[trid]>pre_stim_duration) & (times[trid]<pre_stim_duration+stim_duration))[0]
head_i = heading[trid][pos_stim_]; 
# head_i = vxys[trid][pos_stim_]; 
pos_i = tracks[trid][pos_stim_]
step = 1
x = pos_i[:, 0]
y = pos_i[:, 1]
dx = head_i[:, 0]
dy = head_i[:, 1]
plt.figure(figsize=(8, 6))
plt.plot(x, y, '-', label='Trajectory', color='gray')
plt.plot(x[0], y[0], 'k*')
plt.quiver(x[::step], y[::step], dx[::step], dy[::step], #scale=10,
           scale_units='xy', angles='xy',width=0.003, color='red', alpha=0.8, label='Heading')
plt.xlabel('X')
plt.ylabel('Y')

# %% testing the angle code
###############################################################################
# fact vectors
time_fake = np.arange(3, 4, 1/60)[:-1]
pos_fake = np.array([100, 100])
theta_fake = np.zeros(len(time_fake))-90
theta_fake[10:20] = 90
dtheta_fake = np.diff(theta_fake)
response_vec = np.zeros(len(time_fake))
view_vec = np.zeros(len(time_fake))
bar_vec = np.zeros(len(time_fake))
for tt in range(0,len(time_fake)):
    ### test with bar location to calculate view angle
    stim_xyt = find_stim_location_at_time_t(time_fake[tt])
    # view_angle = angle_between_vectors(  head_i[pos_stim[tt],:] , stim_xyt - pos_i[pos_stim[tt]]) # angle between bar and heading
    # view_angle = angle_between_vectors(  stim_xyt - pos_i[pos_stim[tt]] , head_i[pos_stim[tt],:]) # angle between bar and heading, flipped
    view_angle = angle_between_vectors( np.array([1, 0]) , stim_xyt - pos_fake) # angle between bar vector and reference
    view_vec[tt] = view_angle - theta_fake[tt] ### # difference between bar and heading
    bar_vec[tt] = angle_between_vectors(  np.array([1, 0]) , stim_xyt - pos_fake)

plt.figure()
plt.subplot(311); plt.plot(dtheta_fake); plt.ylabel(r'$d\theta$')
plt.subplot(312); plt.plot(view_vec,'o'); plt.plot([0, len(bar_vec)],[0,0],'k--'); plt.ylabel(r'$\phi$')
plt.subplot(313); plt.plot(bar_vec , '.'); plt.ylabel('bar'); plt.xlabel('time (s)')

# %% ##########################################################################
# %% find front-crossing
def find_first_pos_to_neg_crossing_circular(x, crossing_ang=0, wrap_limit=180):
    x = np.asarray(x)
    dx = np.diff(x)
    
    # Detect where the jump is a real sign change, not a wraparound
    valid_crossings = (x[:-1] >= crossing_ang) & (x[1:] < crossing_ang)
    
    # Exclude wraparound boundary jumps (e.g. from ~180 to -179)
    wrap_jumps = np.abs(dx) > 2 * wrap_limit - 10  # add tolerance if noisy

    # Only keep valid, non-wraparound crossings
    valid_crossings &= ~wrap_jumps

    indices = np.where(valid_crossings)[0]
    # indices = valid_crossings
    return indices[0] + 1 if len(indices) > 0 else None

# def find_first_pos_to_neg_crossing_circular(x, crossing_ang=0):
#     x = np.asarray(x)

#     # Unwrap angles so we handle circular jumps properly
#     x_unwrapped = np.unwrap(np.deg2rad(x))  # Convert to radians and unwrap
#     x_unwrapped_deg = np.rad2deg(x_unwrapped)  # Back to degrees for comparison

#     # Detect positive-to-negative crossing (relative to crossing_ang)
#     x0 = x_unwrapped_deg[:-1]
#     x1 = x_unwrapped_deg[1:]
    
#     crossings = (x0 >= crossing_ang) & (x1 < crossing_ang)

#     indices = np.where(crossings)[0]
#     return indices[0] + 1 if len(indices) > 0 else None

def circular_mean_std_deg(angles_deg):
    angles_deg = np.asarray(angles_deg)
    angles_deg = angles_deg[~np.isnan(angles_deg)]  # remove NaNs

    if len(angles_deg) == 0:
        return np.nan, np.nan  # or raise an error if preferred

    angles_rad = np.deg2rad(angles_deg)

    # Mean of unit vectors
    sin_sum = np.nanmean(np.sin(angles_rad))
    cos_sum = np.nanmean(np.cos(angles_rad))

    # Compute circular mean
    mean_angle_rad = np.arctan2(sin_sum, cos_sum)
    mean_angle_deg = np.rad2deg(mean_angle_rad)

    # Resultant vector length
    R = np.sqrt(sin_sum**2 + cos_sum**2)

    # Circular std: handle edge cases where R ~ 0
    circ_std_rad = np.sqrt(-2 * np.log(R)) if R > 0 else np.pi
    circ_std_deg = np.rad2deg(circ_std_rad)

    return mean_angle_deg, circ_std_deg

# %% tracks
kk=0
offset = 0## 1 or 0
time_vec = []
reps_vec = []
phi_vec = []
plt.figure()
for ii in range(len(fly_response)):
    pos = find_first_pos_to_neg_crossing_circular(stim_angle_vector[ii],crossing_ang=0)
    # print(stim_angle_vector[ii][1])
    if pos is not None and (stim_angle_vector[ii][0]>-45) and (stim_angle_vector[ii][0]<45):
    # if pos is not None and (stim_angle_vector[ii][0]>-0):
    # if pos is not None:# and (np.abs(stim_angle_vector[ii][-1])<100):
        plt.plot(rec_time[ii][pos:] - rec_time[ii][pos] , fly_response[ii][pos:] - offset*fly_response[ii][pos],'k-', alpha=0.2)
        # plt.plot(rec_time[ii][pos:] - rec_time[ii][pos] , stim_angle_vector[ii][pos:] - offset*stim_angle_vector[ii][pos],'k-', alpha=0.2)
        kk+=1
        time_vec.append( rec_time[ii][pos:] - rec_time[ii][pos] )
        reps_vec.append( fly_response[ii][pos:] - offset*fly_response[ii][pos] )
        phi_vec.append( stim_angle_vector[ii][pos-0:] - offset*stim_angle_vector[ii][pos] )
# plt.ylim([-500, 500])

time_vec = np.concatenate(time_vec)
reps_vec = np.concatenate(reps_vec)
phi_vec = np.concatenate(phi_vec)

plt.xlabel('Time (s)')
plt.ylabel(r'd$\theta$ post crossing threshold')

# %% binning
nbins = 30
min_samps = 50
t_bin = np.linspace(0, stim_duration, nbins)
response_mean = np.zeros(nbins) + np.nan
resposne_std = np.zeros(nbins) + np.nan
n_samps = np.zeros(nbins)

for bb in range(nbins-1):
    pos = np.where((time_vec>t_bin[bb]) & (time_vec<=t_bin[bb+1]))[0]
    if len(pos)>min_samps:
        
        response_mean[bb] = np.nanmean(reps_vec[pos])
        resposne_std[bb] = np.nanstd(reps_vec[pos])/np.sqrt(len(pos))
        
        # mm,ss = circular_mean_std_deg(phi_vec[pos])
        # response_mean[bb] = mm
        # resposne_std[bb] = ss/np.sqrt(len(pos))
        
        n_samps[bb] = len(pos)
    
plt.figure(figsize=(8, 5))
plt.plot(t_bin, response_mean, label='Mean', color='blue')
plt.plot([t_bin[0], 0.65], [0,0], 'k--')
plt.fill_between(t_bin,
                 response_mean - resposne_std,
                 response_mean + resposne_std,
                 color='blue',
                 alpha=0.3,
                 label='±1 SEM')

# Decorations
plt.xlabel('Time (s)')
plt.ylabel(r'd$\theta$ post crossing')
# plt.ylabel(r'$\phi$ post crossing')
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% NEW attempt for bar crossing mid-line
###############################################################################
def signed_angle(v1, theta):
    theta_ref_rad = theta/180*np.pi
    # Normalize v1
    v1 = np.asarray(v1)
    v1 = v1 / np.linalg.norm(v1)

    # Construct reference vector from angle
    v2 = np.array([np.cos(theta_ref_rad), np.sin(theta_ref_rad)])

    # Compute angles of each vector
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])

    # Difference
    angle_diff = np.rad2deg(angle1 - angle2)

    # Wrap to [-180, 180]
    angle_diff = (angle_diff + 180) % 360 - 180
    return angle_diff

# %% making the lists
valid_tracks = []  ### list of tracks that have stimuli and crossing the fly's eye
loc_crossing = []  ### location of the crossing along the time axis
temp_debug = []  ### a list of signal for debugging

### loop across tracks
for ii in range(len(tracks)):
    ### load measurements
    time_i = times[ii]
    dtheta_i = dthetas[ii]
    theta_i = thetas[ii]
    speed_i = speeds[ii]
    head_i = heading[ii]
    pos_i = tracks[ii]
    ### analyze the stimulated tracks
    was_stimed = np.where((time_i>pre_stim_duration) & (time_i<pre_stim_duration+stim_duration))[0]
    if len(was_stimed)>0:  # measurement during stim exists
        pos_stim = was_stimed*1
        stim_angi = np.zeros(len(pos_stim))  ### view angle (from fly heading)
        for tt in range(0,len(pos_stim)):
            ### test with bar location to calculate view angle
            stim_xyt = find_stim_location_at_time_t(time_i[pos_stim[tt]])  ### stimulus location

            ### using two angles with respect to 1,0 vector
            view_angle = angle_between_vectors( np.array([1, 0]) , stim_xyt - pos_i[pos_stim[tt]]) # angle between bar vector and reference
            stim_angi[tt] = wrap_heading(view_angle - theta_i[pos_stim[tt]]) ### # difference between bar and heading
            
            ### directly using two vectors (stim and heading)
            stim_vector = stim_xyt - pos_i[pos_stim[tt]]
            head_vector = head_i[pos_stim[tt],:]
            stim_angi[tt] = angle_between_vectors(head_vector, stim_vector)
            
            ### using a function to compute angle between angle and vector
            # stim_angi[tt] = signed_angle(stim_xyt, theta_i[pos_stim[tt]])  ### biased??
            
        ### record crossing
        pos = find_first_pos_to_neg_crossing_circular((stim_angi), crossing_ang=50) ## wrap_heading
        # print(stim_angle_vector[ii][1])
        if pos is not None:
            valid_tracks.append(ii)
            loc_crossing.append(pos + 1*was_stimed[0])
            temp_debug.append(stim_angi)
            
# %% plotting
aligned_dtheta = []
aligned_time = []
plt.figure()
for ii in range(len(valid_tracks)):
    vid = valid_tracks[ii]
    dthetai = dthetas[vid]
    time_aligned = times[vid] - times[vid][loc_crossing[ii]]
    # plt.plot(time_aligned, dthetai)
    plt.plot(temp_debug[ii],'ko', alpha=0.1)
    aligned_dtheta.append(dthetai)
    aligned_time.append(time_aligned)
plt.xlabel('time since crossing (s)')
plt.ylabel(r'd$\theta$')

# %%
x = np.concatenate(aligned_time)
y = np.concatenate(aligned_dtheta)
# Define bin edges (e.g. 3 bins between min and max of x)
num_bins = 60*4*2
bins = np.linspace(np.min(x), np.max(x), num_bins + 1)

# Digitize x values into bins (bin index ranges from 1 to len(bins)-1)
bin_indices = np.digitize(x, bins) - 1  # shift to 0-based index

# Initialize arrays to store statistics
bin_centers = (bins[:-1] + bins[1:]) / 2
mean_y = np.zeros(num_bins)
std_y = np.zeros(num_bins)

# Compute mean and std for y in each bin
for i in range(num_bins):
    mask = bin_indices == i
    values = y[mask]
    n = len(values)
    if np.any(mask):  # check if there are any values in the bin
        mean_y[i] = np.nanmean(y[mask])
        std_y[i] = np.nanstd(y[mask])/(n**0.5)
    else:
        mean_y[i] = np.nan
        std_y[i] = np.nan

# Plotting
plt.errorbar(bin_centers, mean_y, yerr=std_y, fmt='.-', capsize=5)
plt.xlabel('time since crossing (s)')
plt.ylabel(r'd$\theta$')
plt.grid(True)
plt.xlim([-3, 3]); plt.ylim([-70,70])
plt.show()
