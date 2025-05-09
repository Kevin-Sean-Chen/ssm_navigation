# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:08:38 2025

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

# %% vx projection across sensory environments

# %% for perturbed data
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-3-13'  ### V+O exp
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-3-17'  ### O or empty exp
root_dir = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/2025-4-4'  ### O or empty exp
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\perturb_ribbon\2025-4-17'  ### wider ribbons
 
target_file = "exp_matrix.pklz"
exp_type = "gaussianribbon_vial"

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

# pkl_files = pkl_files[10:20]
print(pkl_files) 
pkl_files = sorted(pkl_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

# %% filing
### 3/13
ff = np.arange(40,59) ### ribbon
# ff = np.arange(0,10)  ### edge

### 3/17
ff = np.arange(20,40) ### ribbon
# ff = np.arange(0,15)  ### edge

### 4/4
ff = np.arange(10,20)  ### landscape
# ff = np.arange(20,30)

### 4/17
ff = np.arange(0,25)

threshold_track_l = 60 * 2*1 #2 
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
                # theta = data['theta'][pos]
                temp_v = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
                temp_x = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                times.append(data['t'][pos])
                
                signal = data['signal'][pos]  ### if there is signal
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
vec_signal = np.concatenate(signals)  # odor signal
vec_msd_x = np.concatenate(msd_x)

vec_speed = np.linalg.norm(vec_vxy,axis=1)

# %% plot tracks
plt.figure()
for ii in range(len(tracks)):
    xy_i = tracks[ii]
    time_i = times[ii]
    # pos = np.where((times[ii]>30) & (times[ii]<90))[0]  ### pos stim
    pos = np.where((times[ii]<30))[0]  ### during stim
    # pos = np.where((times[ii]>0))[0]
    plt.plot(xy_i[pos,0], xy_i[pos,1],'k',alpha=.5)
    
# %% compute projection
speed_threshold = 5
pos_boundary = np.where((vec_xy[:,0]<260) & (vec_xy[:,0]>15) & (vec_xy[:,1]>15) & (vec_xy[:,1]<160))[0]
pos_boundary = np.where((vec_xy[:,0]<260) & (vec_xy[:,0]>15) & (vec_xy[:,1]>70) & (vec_xy[:,1]<130) & (vec_signal>=0))[0]
# pos_boundary = np.where((vec_xy[:,0]<260) & (vec_xy[:,0]>15) & (vec_xy[:,1]>15) & (vec_xy[:,1]<80) & (vec_signal>1))[0]
pos_time = np.where((vec_time<30) & (vec_speed>speed_threshold))[0]
# pos_time = np.where((vec_time>30) & (vec_speed>speed_threshold))[0]
pos = np.intersect1d(pos_boundary, pos_time)
def compute_angle(v, reference=np.array([-1,0])):
    vx, vy = v[:, 0], v[:, 1]  # Extract vx and vy components
    rx, ry = reference  # Extract reference components
    angles = np.degrees(np.arctan2(vx * ry - vy * rx, vx * rx + vy * ry))
    return angles
vx_proj = np.abs(vec_vxy[pos,0]) / vec_speed[pos]
vx_angs = np.abs(compute_angle(vec_vxy[pos,:]))

plt.figure()
plt.hist(vx_angs,bins=50, density=True)
plt.xlabel('angle to x axis'); plt.ylim([0,0.025])

# %% make independent model
# import numpy as np
# import matplotlib.pyplot as plt

# # Example data
# np.random.seed(0)
# data1 = vx_angs_odor*1
# data2 = vx_angs_vision*1
# data3 = vx_angs_combined

# # Create histograms
# bins = np.linspace(0, 180, 25)

# h1, edges = np.histogram(data1, bins=bins, density=True)
# h2, edges = np.histogram(data2, bins=bins, density=True)
# # h3, _ = np.histogram(data3, bins=bins, density=True)

# # Combine densities by summing
# # h_combined = h1 + h2

# # Normalize combined histogram to be a valid density
# bin_width = edges[1] - edges[0]
# # h_combined /= np.sum(h_combined) * bin_width

# # Plotting
# bin_centers = (edges[:-1] + edges[1:]) / 2

# plt.figure(figsize=(8,5))
# plt.plot(bin_centers, h1, label='odor')
# plt.plot(bin_centers, h2, label='vsion')
# plt.plot(bin_centers, h3, label='V+O data')
# # plt.plot(bin_centers, h1, label='middle')
# # plt.plot(bin_centers, h_combined, label='V+O sum', linestyle='--', color='k')
# plt.xlabel('angle to x axis')
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True);  plt.ylim([0,0.027])

# %%
def linear_regression_two_predictors(a, b, c):
    # Stack predictors into a (N, 2) matrix
    A = np.column_stack((a, b))
    
    # Solve least squares: minimize ||A @ [x, y] - c||
    coeffs, residuals, rank, s = np.linalg.lstsq(A, c, rcond=None)
    
    return coeffs

coeff = linear_regression_two_predictors(h1, h2, h3)
plt.figure(figsize=(8,5))
plt.plot(bin_centers, h3, label='V+O data')
plt.plot(bin_centers, coeff[0]*h1+coeff[1]*h2, label='regression')
plt.xlabel('angle to x axis')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

# %% show tracks
###############################################################################
# %% check signal
sigs = np.zeros(len(tracks))
for ii in range(len(tracks)):
    sigs[ii] = np.nanvar(signals[ii])

interesting_tracks = np.where(sigs>3000)[0] #15000

# %% visualize
for ii in range(len(interesting_tracks)):
    plt.figure()
    xyi = tracks[interesting_tracks[ii]]
    plt.plot(xyi[:,0], xyi[:,1])
    sigi = signals[interesting_tracks[ii]]
    pos = np.where(sigi>0)[0]
    plt.plot(xyi[pos,0], xyi[pos,1],'ro')
    plt.plot(xyi[0,0], xyi[0,1],'k*')
    plt.xlim([0,300]); plt.ylim([0,190])
    
# %% compare projx, projy, condition on stim
proj_up = []
proj_down = []

for ii in range(len(tracks)):
    xy_i = tracks[ii]
    vxy_i = vxys[ii]
    time_i = times[ii]
    stim_i = signals[ii]
    pos_time = np.where((time_i<30))[0]
    pos_time = np.where((time_i>30))[0]
    pos_signal = np.where(stim_i>1)[0]
    pos_boundary = np.where((xy_i[:,0]<260) & (xy_i[:,0]>15) & (xy_i[:,1]>15) & (xy_i[:,1]<160))[0]
    pos = np.intersect1d(pos_boundary, pos_signal)
    pos = np.intersect1d(pos_boundary, pos_time)
    # if len(pos)>0 and len(pos_signal)>0 and np.mean(speeds[ii])>1:
    if len(pos)>0 and np.mean(speeds[ii])>1:
        vyi = vxy_i[pos,1]  ### velocity along y
        
        dx = xy_i[pos_time[0],0] - xy_i[pos_time[-1],0]
        
        if dx>0:
            proj_up.append(np.var(vyi))
        elif dx<=0:
            proj_down.append(np.var(vyi))
            
## %%
from scipy.stats import sem  # for standard error of mean

group1 = proj_up
group2 = proj_down

# ----------------------------------------------------------
# Compute means and SEMs
means = [np.mean(group1), np.mean(group2)]
sems = [sem(group1), sem(group2)]

# ----------------------------------------------------------
# Plot
fig, ax = plt.subplots(figsize=(8,6))

x_positions = np.arange(2)

# Bar plot (means)
ax.bar(x_positions, means, yerr=sems, capsize=5, color=['skyblue', 'salmon'], edgecolor='black', width=0.6)

# Overlay raw data points
jitter_strength = 0.08  # to spread dots horizontally
x1_jittered = np.random.normal(0, jitter_strength, size=len(group1)) + x_positions[0]
x2_jittered = np.random.normal(0, jitter_strength, size=len(group2)) + x_positions[1]

ax.plot(x1_jittered, group1, 'o', color='blue', alpha=0.7, label='Group 1')
ax.plot(x2_jittered, group2, 'o', color='red', alpha=0.7, label='Group 2')

# Customize
ax.set_xticks(x_positions)
ax.set_xticklabels(['up wind', 'down wind'])
ax.set_ylabel('displacement (mm/s)^2')
ax.grid(True, axis='y')

plt.tight_layout()

# %% compute velocity distribution
speed_up = []
speed_down = []

for ii in range(len(tracks)):
    xy_i = tracks[ii]
    vxy_i = vxys[ii]
    time_i = times[ii]
    stim_i = signals[ii]
    pos_time = np.where((time_i<30))[0]
    # pos_time = np.where((time_i>30))[0]
    pos_signal = np.where(stim_i>1)[0]
    pos_boundary = np.where((xy_i[:,0]<260) & (xy_i[:,0]>15) & (xy_i[:,1]>15) & (xy_i[:,1]<160))[0]
    pos = np.intersect1d(pos_boundary, pos_signal)
    # pos = np.intersect1d(pos_boundary, pos_time)
    if len(pos)>0 and len(pos_signal)>0 and np.mean(speeds[ii])>1:
    # if len(pos)>0 and np.mean(speeds[ii])>1:
        speed_i = np.sum(vxy_i[pos,:]**2,1)**0.5
        
        ### track based
        # dx = xy_i[pos_time[0],0] - xy_i[pos_time[-1],0] 
        # if dx>0:
        #     speed_up.append(speed_i)
        # elif dx<=0:
        #     speed_down.append(speed_i)
            
        ### vector based
        pos_up = np.where(vxy_i[pos,0]<0)[0]
        pos_down = np.where(vxy_i[pos,0]>0)[0]
        speed_up.append(speed_i[pos_up])
        speed_down.append(speed_i[pos_down])
# %%
bins = np.linspace(0,30,30)
plt.figure()
plt.hist(np.concatenate(speed_up), bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='up')
plt.hist(np.concatenate(speed_down), bins=bins, density=True, alpha=0.5, color='k', edgecolor='black', label='down')
plt.xlim([-.5, 35]); plt.ylim([0,0.6]); plt.legend()
