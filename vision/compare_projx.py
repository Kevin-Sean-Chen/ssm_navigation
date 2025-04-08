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
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-3-13'  ### V+O exp
# root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\opto_rig\odor_vision\2025-3-17'  ### O or empty exp
 
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
### 3/13
ff = np.arange(10,20) ### ribbon
# ff = np.arange(0,10)  ### edge

### 3/17
ff = np.arange(20,40) ### ribbon
# ff = np.arange(0,15)  ### edge

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
vec_signal = np.concatenate(signal)  # odor signal
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
speed_threshold = 2
pos_boundary = np.where((vec_xy[:,0]<260) & (vec_xy[:,0]>15) & (vec_xy[:,1]>15) & (vec_xy[:,1]<160))[0]
pos_time = np.where((vec_time<30) & (vec_speed>speed_threshold))[0]
pos_time = np.where((vec_time>30) & (vec_speed>speed_threshold))[0]
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
plt.xlabel('angle to x axis')

# %% make independent model
import numpy as np
import matplotlib.pyplot as plt

# Example data
np.random.seed(0)
data1 = vx_angs_odor*1
data2 = vx_angs_vision*1
data3 = vx_angs_combined

# Create histograms
bins = np.linspace(0, 180, 100)

h1, edges = np.histogram(data1, bins=bins, density=True)
h2, _ = np.histogram(data2, bins=bins, density=True)
h3, _ = np.histogram(data3, bins=bins, density=True)

# Combine densities by summing
h_combined = h1 + h2

# Normalize combined histogram to be a valid density
bin_width = edges[1] - edges[0]
h_combined /= np.sum(h_combined) * bin_width

# Plotting
bin_centers = (edges[:-1] + edges[1:]) / 2

plt.figure(figsize=(8,5))
plt.plot(bin_centers, h1, label='olfaction data')
plt.plot(bin_centers, h2, label='vision data')
plt.plot(bin_centers, h3, label='V+O data')
plt.plot(bin_centers, h_combined, label='V+O sum', linestyle='--', color='k')
plt.xlabel('angle to x axis')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

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

interesting_tracks = np.where(sigs>15000)[0]

# %% visualize
for ii in range(len(interesting_tracks)):
    plt.figure()
    xyi = tracks[interesting_tracks[ii]]
    plt.plot(xyi[:,0], xyi[:,1])
    sigi = signals[interesting_tracks[ii]]
    pos = np.where(sigi>0)[0]
    plt.plot(xyi[pos,0], xyi[pos,1],'ro')
    plt.plot(xyi[0,0], xyi[0,1],'k*')