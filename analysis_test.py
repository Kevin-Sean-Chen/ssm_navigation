# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:34:58 2024

@author: ksc75
"""

import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt

import glob
import os

# %% analysis conditions
### cutoff for short tracks
threshold_track_l = 60 * 20  # look at long-enough tracks

# %% load folder batch
# Define the folder path
folder_path = 'C:/Users/ksc75/Downloads/ribbon_data_kc/'

# Use glob to search for all .pkl files in the folder
pkl_files = glob.glob(os.path.join(folder_path, '*.pklz'))

# Print the list of .pkl files
for file in pkl_files:
    print(file)

# %% load single file
directory = 'C:/Users/ksc75/Downloads/ribbon_data_kc/'
filename = 'exp_matrix_091724_3.pklz'
with gzip.open(directory+filename, 'rb') as f:
    data = pickle.load(f)

# Now, 'data' contains the loaded Python object
print(data)

# %% concatenate a file
n_tracks = np.unique(data['trjn'])
stats = []

for ii in n_tracks:
    pos = np.where(data['trjn']==ii)[0] # find track elements
    if sum(data['behaving'][pos]):  # check if behaving
        if len(pos) > threshold_track_l:
            stats.append( data['signal'][pos] )
            # stats.append( data['vx_smooth'][pos] )
            # stats.append(np.array([len(pos)]))
stats = np.concatenate(stats)

# %%
def angle_between_vectors(v1, v2):
    # Compute the dot product between the vectors
    dot_product = np.dot(v1, v2)
    
    # Compute the magnitudes (norms) of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Compute the cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Ensure the value is within the valid range for arccos due to floating point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Compute the angle in radians
    angle = np.arccos(cos_theta)
    
    # Optionally convert the angle to degrees
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

def ang_t(vecs):
    angs = np.zeros(vecs.shape[1])
    for tt in range(len(angs)):
        angs[tt] = angle_between_vectors(vecs[:,tt], np.array([0,1]))
    return angs

# %% concatenate across files in a folder
stats = []
stats_signal = []
nf = len(pkl_files)

for ff in range(nf):
    ### load file
    with gzip.open(pkl_files[ff], 'rb') as f:
        data = pickle.load(f)
        
    ### extract tacks
    n_tracks = np.unique(data['trjn'])
    
    for ii in n_tracks:
        pos = np.where(data['trjn']==ii)[0] # find track elements
        if sum(data['behaving'][pos]):  # check if behaving
            if len(pos) > threshold_track_l:
                
                ### raw traces
                stats_signal.append( data['signal'][pos] )   # raw signal
                # stats.append( data['signal'][pos] )
                # stats.append( data['vx_smooth'][pos] )  # smoothed speed
                # stats.append(np.array([len(pos)]))  # checking the track lengths
                # stats.append(data['theta_smooth'][pos] - 180)  # theta angle to wind
                
                ### compute velocity
                v_xy = np.sqrt(data['vx_smooth'][pos]**2 + data['vy_smooth'][pos]**2)
                # stats.append(np.array([np.nanmean(v_xy)]))  # average speed
                # stats.append(v_xy)
                # stats.append( data['theta'][pos] )   # raw signal
                
                ### compute angles
                vec_xy = np.diff(np.vstack((data['x_smooth'][pos],data['y_smooth'][pos])),0)
                ang_ = ang_t(vec_xy)
                stats.append(ang_)
stats = np.concatenate(stats)
stats_signal = np.concatenate(stats_signal)

# %%
#####################
### checking the theta in video...
#####################
# %%
plt.figure()
plt.hist(stats, 100)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Define a function to compute autocorrelation for each lag
def compute_autocorrelation(data, max_lag):
    """
    Compute the autocorrelation for each lag from 1 to max_lag.

    Parameters:
    - data: 1D numpy array or list of time series data
    - max_lag: Maximum number of lags to compute

    Returns:
    - lags: Array of lags (from 1 to max_lag)
    - autocorr_values: Autocorrelation values corresponding to each lag
    """
    n = len(data)
    mean = np.nanmean(data)
    autocorr_values = []
    
    # Compute autocorrelation for each lag
    for lag in range(1, max_lag + 1):
        numerator = np.nansum((data[:-lag] - mean) * (data[lag:] - mean))
        denominator = np.nansum((data - mean) ** 2)
        autocorrelation = numerator / denominator
        autocorr_values.append(autocorrelation)
    
    return np.arange(1, max_lag + 1), autocorr_values

# Compute the autocorrelation for up to 20 lags
lags, autocorr_values = compute_autocorrelation(stats, max_lag=threshold_track_l)

# Plot the autocorrelation function
time_lags = np.arange(0,len(lags))* 1/60  # frames to seconds
plt.figure(figsize=(7, 5))
plt.plot(time_lags, autocorr_values, linewidth=9)
plt.xlabel('time lag (s)')
plt.ylabel('autocorrelation')
plt.title('theta')
plt.axhline(0, color='gray', linestyle='--')
plt.grid()
plt.show()

# %% wihtihn plume analysis
###############################################################################
threshold_within = 5
pos = np.where(stats_signal > threshold_within)[0]  ### set stats_signal for concatenating the full signal vector
win_stats = stats[pos]
lags, autocorr_values = compute_autocorrelation(win_stats, max_lag=threshold_track_l)

# Plot the autocorrelation function
time_lags = np.arange(0,len(lags))* 1/60  # frames to seconds
plt.figure(figsize=(7, 5))
plt.plot(time_lags, autocorr_values, linewidth=9)
plt.xlabel('time lag (s)')
plt.ylabel('autocorrelation')
plt.title('theta')
plt.axhline(0, color='gray', linestyle='--')
plt.grid()
plt.show()

# %% density
plt.figure()
plt.hist(stats, density=True, alpha=0.5)
plt.hist(win_stats, density=True, alpha=0.5)
