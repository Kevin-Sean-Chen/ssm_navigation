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
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-12-2'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-12-5'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-12-10'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-12-12'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-12-19'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2024-12-23'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\visual_behavior\2025-1-24'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-2-3'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-2-5'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-2-6'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\visual_behavior\2025-2-6'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-2-13'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-2-14'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-2-17'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-2-20'   ### missing dv and dth
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\visual_behavior\2025-2-27'
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-3-6' 
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\visual_behavior\2025-3-20'
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-5-5' ### loom + ribbon
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\visual_behavior\2025-4-11'  ### visual loom + bar

exp_type = ['05s']
exclude_keywords = ['bars']
target_file = "exp_matrix.pklz"

# List all subfolders in the root directory
subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
pkl_files = []

# Loop through each subfolder to search for the target file
for subfolder in subfolders:
    for dirpath, dirnames, filenames in os.walk(subfolder):
        # if target_file in filenames:
        # if target_file in filenames and exp_type in dirpath:
        if (target_file in filenames and
            any(kw in dirpath for kw in exp_type) and
            not any(kw in dirpath for kw in exclude_keywords)):
            
            full_path = os.path.join(dirpath, target_file)
            pkl_files.append(full_path)
            print(full_path)

# pkl_files = pkl_files[6:9]
print(pkl_files) 
pkl_files = sorted(pkl_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

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
# ff = np.arange(0,1) #0,8
# # ff = np.arange(9,10) #9,16
# ff = np.arange(16,21) #21,25
# ff = np.arange(25,30)

### 12/5
# ff = np.array([1,3,4,5,6]) ### blue bar
# ff = np.array([10, 7,8,9,10]) ### invsere blue
# ff = np.arange(10,15) ### green with LED
# ff = np.arange(15,20) ### green without LED

### 12/10
# ff = np.arange(0,7)  ### only bar
# ff = np.arange(7,15)  ### bar+proj
# ff = np.arange(15,23) ### bar+proj
# ff = np.arange(7,23)
# ff = np.arange(23,31)  ### bar+proj w/ LED
# ff = np.arange(31,38)  ### proj w/ LED
# ff = np.arange(38,45)  ### proj w/o LED

### 12/12
# ff = np.arange(2,7)  ### 0.2
# ff = np.array([0,1,8,9]) ###1
# ff = np.arange(10,15) ### 0.4

### 12/20
# ff = np.arange(15,25)  # v+o w/ LED
# ff = np.arange(30,45) # v+o w/o LED
ff = np.arange(68,74)  # v negative w/ LED
# ff = np.arange(74,79)  # v negative  w/o LED
ff = np.arange(98,108)  # visual only 80,93
# ff = np.array([93,94,95,96,101,102,103,104,105,106,107])
# ff = np.arange(7,10)

### 12/23
ff = np.arange(0,10)  #f=1
# ff = np.arange(10,19)  #f=.5
# ff = np.arange(20,28)  #f=2
# ff = np.arange(29,38)  #f=0.2
ff = np.arange(41,47)
ff = np.arange(53,64)

### 01/24
ff = np.arange(0,7)
ff = np.arange(8,23)

### 02/03
ff = np.arange(0,7)  ### screen on right
# ff = np.arange(7,14) ## on the left
# ff = np.arange(29,40)## flashes

### 02/05
ff = np.arange(6,12)  ### 50, 100
ff = np.arange(12,20) ### 50, 1K
# ff = np.arange(20,28) ### 50, full blue
# ff = np.arange(28,36) ### 50, 0.5 blue 
# ff = np.arange(36,41) ### 255, 100
# ff = np.arange(42,49)
# ff = np.arange(50,56) ### 255, full blue
# ff = np.arange(57,64) ### 255, 0.5 blue

## 02/06
# ff = np.arange(0,16) ### bars
# ff = np.arange(16,26)  ### proj and bars
# ff = np.arange(26,35)  ### full blue
# ff = np.arange(34,42)  ### use LED
# ff = np.arange(42,56)  ### use LED at 255
### VISION
# ff = np.arange(0,10) ### 30, 2Hz
# ff = np.arange(11,18)  ### 1 Hz
# ff = np.arange(19,28)
# ff = np.arange(28,37)
# ff = np.arange(38,47)
# ff = np.arange(48,52) 
# ff = np.arange(53,60) ## 3 bars!

### 02/13
ff = np.arange(0,8)  ### blue bar
ff = np.arange(8,16)  ### black bar on blue
ff = np.arange(16,24)  ### black on green
ff = np.arange(24,32)  ### green bar
ff = np.arange(48,56)  ### cross at proj-50
ff = np.arange(56,64) ### cross at proj=150
# ff = np.arange(64,72)  ### combined
ff = np.arange(72,80)
# ff = np.arange(80,88)

### 02/17
ff = np.arange(24,32)
# ff = np.arange(40,48)
# ff = np.arange(56,64)

### 02/20
ff = np.arange(0,16)
ff = np.arange(26,35)
ff = np.arange(38,48)

### 03/06
# ff = np.arange(0,8)  ### odor +BP
# ff = np.arange(9,16) ### odor
# ff = np.arange(32,40)  ### w/o ATR +BP
ff = np.arange(40,56) ### w/o ATR

ff = np.arange(30,40)  ### loom with bars
ff = np.arange(20,30) ### fast looms

ff = np.arange(0, len(pkl_files))  ### if all
# ff= np.arange(0,17)

threshold_track_l = 60 * 1#5 #2 
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
                theta = data['theta_smooth'][pos]
                temp_v = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
                temp_x = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                times.append(data['t'][pos])
                
                signal = []#data['signal'][pos]  ### if there is signal
                speeds.append(np.sqrt(data['vx_smooth'][pos]**2+data['vy_smooth'][pos]**2))
                tracks.append(temp_x)
                vxys.append(temp_v)
                track_id.append(np.zeros(len(pos))+jj) 
                thetas.append(theta)
                signals.append(signal)
                
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
vec_speed = np.concatenate(speeds)

# %% plot tracks
plt.figure()
for ii in range(len(tracks)):
    xy_i = tracks[ii]
    time_i = times[ii]
    # pos = np.where((times[ii]>30) & (times[ii]<50))[0]
    # pos = np.where((times[ii]>63) & (times[ii]<63+5))[0]
    pos = np.where((times[ii]<126))[0]
    # pos = np.where((times[ii]>0))[0]
    plt.plot(xy_i[pos,0], xy_i[pos,1],'k',alpha=.5)
    
#     if len(pos)>0:
#         diff_pos = xy_i[pos[0],0] - xy_i[pos[-1],0]
#         if diff_pos>0:
#             plt.plot(xy_i[pos,0], xy_i[pos,1],'r',alpha=.5)
#         else:
#             plt.plot(xy_i[pos,0], xy_i[pos,1],'b',alpha=.5)

# plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title(r'example tracks ($\rightarrow$ in red, $\leftarrow$ in blue)')
    # # plt.plot(xy_i[:,0], xy_i[:,1],'k', alpha=.8)
    # plt.scatter(xy_i[:,0],xy_i[:,1],c=time_i,cmap='coolwarm',s=.1,vmin=np.min(time_i),vmax=np.max(time_i))
    # plt.scatter(xy_i[:,0],xy_i[:,1],c=time_i,cmap='coolwarm',s=.1,vmin=np.min(vec_time),vmax=np.max(vec_time))

# %% all dot-x for visual work
# x_vec = np.zeros((len(vec_vxy),2))
# x_vec[:,0] = 1
# # test = np.sum(np.abs(vec_vxy/np.linalg.norm(vec_vxy,axis=1)[:,None]) * x_vec, axis=1)
# test = np.sum((vec_vxy) * x_vec, axis=1)

# # %%
# bins = np.arange(-20,20,1)
# plt.figure()
# x_ = [np.zeros(len(test_02))+0.2, np.zeros(len(test_04))+0.4,np.zeros(len(test_10))+1]
# y_ = [test_02, test_04, test_10]
# label = [0.2,0.4,1.0]
# for ii in range(3):
#     plt.subplot(3,1,ii+1)
#     plt.hist(y_[ii],bins=bins, density='True', alpha=0.5)
#     plt.xlim([-20,20])
#     plt.yscale('log')
# plt.xlabel('vx (mm/s)')

# %% showing counts
# plt.figure()
# g = sns.jointplot(x=vec_xy[::30,0], y=vec_xy[::30,1], kind="kde")
# g.set_axis_labels('x (mm)', 'y (mm)')

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
    # stim_i = signal[ii]
    pos = np.where(time_i>0)[0]
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
time_stim = np.arange(0,123,.2) ###
# time_stim = np.arange(0,30,.2) ###
# time_stim = np.arange(0,120*2,.4)
mean_dtheta = time_stim*0+np.nan
mean_speed = time_stim*0+np.nan
for tt in range(len(time_stim)-1):
    pos = np.where((time_align>time_stim[tt]) & (time_align<time_stim[tt+1]) )[0]
    mean_dtheta[tt] = np.nanmean(np.abs(dtheta_align[pos]))
    mean_speed[tt] = np.nanmean((speed_align[pos]))

plt.figure()
plt.plot(time_stim, mean_dtheta)
plt.xlabel('time (s)'); plt.ylabel('|degree|/s'); 
# plt.ylim([25, 90])
plt.figure()
plt.plot(time_stim, mean_speed)
plt.xlabel('time (s)'); plt.ylabel('speed (mm/s)'); 
# plt.ylim([3, 11])

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

# %% time/space aligned tracks
###############################################################################
# %% iterate across tracks
pre_time = 10 ### 3,2
stop_threshold = 5
not_stop = 5
max_jump = 10
time_all, speed_all, stop_all = [], [], []
for ii in range(len(tracks)):
    xy_i = tracks[ii]
    time_i = times[ii]
    vxy_i = vxys[ii]
    speed_i = speeds[ii]
    pos_time = np.where((time_i>25) & (time_i<30))[0]  ### condition in time
    pos_time = np.where((time_i>63-pre_time) & (time_i<73))[0]
    pos_time = np.where((time_i>13-pre_time) & (time_i<28))[0]
    # pos_time = np.where((time_i>0))[0]   ### for just loom
    pos_space = np.where((xy_i[:,0]>50) & (xy_i[:,0]<250) & (xy_i[:,1]>50) & (xy_i[:,1]<150))[0]  ### condition in space
    
    pos = np.intersect1d(pos_time, pos_space)
    pos_end = np.where(np.diff(pos)>max_jump)[0]
    
    if len(pos_end)>0:
        pos = pos[:pos_end[0]]
    if len(pos)>0 and np.nanmean(speed_i[pos])>not_stop:
        # print(ii)
        plt.figure(1)
        plt.plot(np.abs(xy_i[pos,0] - xy_i[pos[0],0]), np.abs(xy_i[pos,1]-185) - xy_i[pos[0],1]*0, 'k', alpha=0.5); 
        plt.xlim([0,125]); #plt.ylim([-50, 50]); 
        plt.title(' 5s aligned tracks w/o loom'); plt.xlabel('aligned along bar direction, X (mm)'); plt.ylabel('distance to loom side, Y (mm)'); 
        
        plt.figure(2)
        plt.plot(np.arange(0, len(pos))/60 - pre_time, vxy_i[pos,0], 'k', alpha=0.45)
        # plt.plot(np.arange(0, len(pos))/60 - pre_time, speed_i[pos], 'k', alpha=0.45)
        plt.ylim([-50,50]); plt.ylabel(r'$V_x$ (mm/s)'); plt.xlabel('time since loom (s)')
        x = [0, 0.5,  0.5, 0]  # x-coordinates of corners
        y = [-35, -35, 35, 35]  # y-coordinates of corners
        
        time_all.append(time_i[pos])
        speed_all.append(speed_i[pos])
        temp_stop = speed_i[pos]*0 + 1
        temp_stop[speed_i[pos]>stop_threshold] = 0
        stop_all.append(temp_stop)

time_all = np.concatenate(time_all)
speed_all = np.concatenate(speed_all)
stop_all = np.concatenate(stop_all)

# Plot the gray area
plt.figure(1)
plt.gca().invert_yaxis()
plt.figure(2)
plt.fill(x, y, color='gray', alpha=0.5)
# plt.xlim([0,300]); plt.ylim([0, 180])

# %% average traces
###############################################################################
# %% stopping and speed
nbins = 90*2
at,bt = np.histogram(time_all, nbins)
mean_speed = np.zeros(nbins)
std_speed = np.zeros(nbins)
for ii in range(1,nbins):
    pos = np.where((time_all>bt[ii-1]) & (time_all<bt[ii]))[0]
    if len(pos)>0:
        mean_speed[ii-1] = np.mean(stop_all[pos])
        std_speed[ii-1] = np.std(stop_all[pos])/len(pos)**0.5
        mean_speed[ii-1] = np.mean(speed_all[pos])
        std_speed[ii-1] = np.std(speed_all[pos])/len(pos)**0.5

tt, mean, error = bt[:-2] - bt[0] - pre_time, mean_speed[:-1], std_speed[:-1]
plt.figure()
# plt.plot(bt[:-2], mean_speed[:-1], '-o')
plt.plot(tt, mean)
plt.fill_between(tt, mean - error, mean + error, color='blue', alpha=0.3, label='± Error')
plt.fill(x, y, color='gray', alpha=0.5)
plt.ylim([0,1]); plt.xlabel('time since loom (s)'); plt.ylabel('P(stop)')
plt.xlabel('time since loom (s)'); plt.ylabel('speed (mm/s)'); plt.ylim([2, 18])

# %% angular functions
def circular_mean_deg(angles_deg):
    # angles_rad = np.deg2rad(angles_deg)
    # mean_angle_rad = np.arctan2(np.nanmean(np.sin(angles_rad)), np.nanmean(np.cos(angles_rad)))
    # mean_angle_deg = np.rad2deg(mean_angle_rad - 0) % 360
    # return mean_angle_deg
    angles_deg = np.asarray(angles_deg)
    angles_rad = np.deg2rad(angles_deg)

    sin_mean = np.nanmean(np.sin(angles_rad))
    cos_mean = np.nanmean(np.cos(angles_rad))

    mean_angle_rad = np.arctan2(sin_mean, cos_mean)
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 360

    R = np.hypot(sin_mean, cos_mean)  # mean resultant length (vector strength)
    std_angle_rad = np.sqrt(-2 * np.log(R)) if R > 0 else np.nan
    std_angle_deg = np.rad2deg(std_angle_rad)

    return mean_angle_deg, std_angle_deg

def circular_mean_deg_complex(angles_deg):
    angles_rad = np.deg2rad(angles_deg)
    vectors = np.exp(1j * angles_rad)
    mean_vector = np.nanmean(vectors)
    mean_angle_rad = np.angle(mean_vector)
    mean_angle_deg = np.rad2deg(mean_angle_rad - 0) % 360
    return mean_angle_deg

def unwrap_angles_deg(angles_deg):
    angles_rad = np.deg2rad(angles_deg)
    unwrapped_rad = np.unwrap(angles_rad)
    unwrapped_deg = np.rad2deg(unwrapped_rad)
    return unwrapped_deg

# %% turning rate
pre_time = 5 ## 5,3
post_time = 10
not_stop = 0 #5
max_jump = 10
time_all, turn_all = [], []
hist_window = 5
time_time_point = 63

cnt = 0
plt.figure()
for ii in range(len(tracks)):
    xy_i = tracks[ii]
    time_i = times[ii]
    vxy_i = vxys[ii]
    speed_i = speeds[ii]
    theta_i = thetas[ii]
    # pos_time = np.where((time_i>25) & (time_i<30))[0]  ### condition in time
    pos_time = np.where((time_i>time_time_point - pre_time) & (time_i<time_time_point+post_time))[0]
    pos_time = np.where((time_i>13.-pre_time) & (time_i<26))[0]
    # pos_time = np.where((time_i>0))[0]   ### for just loom
    pos_space = np.where((xy_i[:,0]>50) & (xy_i[:,0]<250) & (xy_i[:,1]>50) & (xy_i[:,1]<150))[0]  ### condition in space
    # pos_space = np.where((xy_i[:,0]>70) & (xy_i[:,0]<230) & (xy_i[:,1]>70) & (xy_i[:,1]<130))[0]  ### condition in space
    
    pos = np.intersect1d(pos_time, pos_space)
    pos_end = np.where(np.diff(pos)>max_jump)[0]
    pos_pre = np.where((time_i>time_time_point-hist_window) & (time_i<time_time_point))[0]
    
    if len(pos_end)>0:
        pos = pos[:pos_end[0]]
    if len(pos)>0 and np.nanmean(speed_i[pos])>not_stop:# and len(pos_pre)>0: #xy_i[pos[0],0] < xy_i[pos[-1],0] 
        direction = xy_i[pos[0],0] - xy_i[pos[-1],0] #np.mean(vxy_i[pos,0])
        if direction > 0:
                time_vec = np.arange(0, len(pos))/60 - pre_time
                theta_vec = (theta_i[pos])
                plt.plot(time_vec, theta_vec, 'k', alpha=0.1)
                # plt.ylim([-35,35]); plt.ylabel(r'$V_x$ (mm/s)'); plt.xlabel('time since loom (s)')
                
                time_all.append(time_i[pos])
                turn_all.append(theta_i[pos] - 180 - 0)#*circular_mean_deg_complex(theta_i[pos_pre])) #0*np.nanmean(theta_i[pos]))
                cnt += 1
        
        ### flipping to align
        if direction < 0:
                time_vec = np.arange(0, len(pos))/60 - pre_time
                theta_vec = (-theta_i[pos] - 0)
                plt.plot(time_vec, theta_vec, 'k', alpha=0.1)
                time_all.append(time_i[pos])
                turn_all.append( (theta_vec - 0))#*circular_mean_deg_complex(-theta_i[pos_pre]-180)) )
                
                cnt+=1

time_all = np.concatenate(time_all)
turn_all = np.concatenate(turn_all)
print(cnt)

# %%
nbins = 70#60*2
at,bt = np.histogram(time_all, nbins)
mean_speed = np.zeros(nbins)
std_speed = np.zeros(nbins)
for ii in range(1,nbins):
    pos = np.where((time_all>bt[ii-1]) & (time_all<bt[ii]))[0]
    if len(pos)>0:
        # mean_speed[ii-1] = np.nanmean((turn_all[pos]))
        # std_speed[ii-1] = np.nanstd((turn_all[pos]))/len(pos)**0.5
        
        aa,bb = circular_mean_deg(turn_all[pos])
        mean_speed[ii-1] = aa  #np.nanmean(turn_all[pos])
        std_speed[ii-1] = bb/len(pos)**0.5

tt, mean, error = bt[:-2] - bt[0] - pre_time*1, unwrap_angles_deg(mean_speed[:-1]-360), std_speed[:-1]
plt.figure()
# plt.plot(bt[:-2], mean_speed[:-1], '-o')
plt.plot(tt, mean)
plt.fill_between(tt, mean - error, mean + error, color='blue', alpha=0.3, label='± Error')
# x = [0, 0.5,  0.5, 0]  # x-coordinates of corners
# y = [-5, -5, 17, 17]  # y-coordinates of corners
# plt.fill(x, y, color='gray', alpha=0.5); plt.ylim([-5,17])
plt.xlabel('time since loom (s)'); plt.ylabel('heading (degrees)'); #plt.ylim([5,15])
plt.title('aligned tracks with loom on the right')

# plt.savefig("heading.pdf", bbox_inches='tight')