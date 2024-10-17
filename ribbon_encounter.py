# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:31:36 2024

@author: ksc75
"""


import numpy as np
import matplotlib.pyplot as plt

import pickle
import gzip
import glob
import os

import ssm
import numpy.random as npr
import seaborn as sns

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% crossing analysis
# without model, analyze ribbon encountering dynamics
###############################################################################
# %% for Kiri's data
### cutoff for short tracks
threshold_track_l = 60 * 10  # 20 # look at long-enough tracks

# Define the folder path
folder_path = 'C:/Users/ksc75/Downloads/ribbon_data_kc/'

# Use glob to search for all .pkl files in the folder
pkl_files = glob.glob(os.path.join(folder_path, '*.pklz'))

# Print the list of .pkl files
for file in pkl_files:
    print(file)

# %% for perturbed data
root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/100424_new/'
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

pkl_files = pkl_files[9:]

# %% concatenate across files in a folder
data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
nf = len(pkl_files)
masks = []   # where there is nan
track_id = []  # record track id (file and track)
rec_tracks = []  # record the full track x,y
rec_signal = []  # record opto signal
times = []   # record time in epoch
encount_sig = []
cond_id = 0

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
                
                ### make per track data
                # temp = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos] , \
                                        # data['theta_smooth'][pos] , data['signal'][pos]))
                thetas = data['theta'][pos]
                temp = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
                # temp = np.stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]),1)#######
                
                temp_xy = np.column_stack((data['x'][pos] , data['y'][pos]))
                
                
                ### criteria
                mask_i = np.where(np.isnan(temp), 0, 1)
                mask_j = np.where(np.isnan(thetas), 0, 1)
                mean_v = np.nanmean(np.sum(temp**2,1)**0.5)
                max_v = np.max(np.sum(temp**2,1)**0.5)
                # print(mean_v)
                if np.prod(mask_i)==1 and np.prod(mask_j)==1 and mean_v>1 and max_v<30:  ###################################### removing nan for now
                    data4fit.append(temp)  # get data for ssm fit
                    rec_tracks.append(temp_xy)  # get raw tracks
                    track_id.append(np.array([ff,ii]))  # get track id
                    rec_signal.append(data['signal'][pos])
                    cond_id += 1
                    masks.append(thetas)
                    
                    time_vec_i = data['t'][pos]
                    times.append(time_vec_i)
                    pos_enc = np.where((time_vec_i>45) & (time_vec_i<45+30))[0]
                    encount_sig.append(data['signal'][pos][pos_enc])
                # masks.append(mask_i)
                
# %% analyze per track
num_cross = []
inter_cross = []
threshold_sig = 5
window_size = 30*2  # .5 window

for ii in range(len(encount_sig)):
    time_series = encount_sig[ii].reshape(-1)
    if len(time_series) > 0:
        time_series = np.convolve(time_series, np.ones(window_size) / window_size, mode='valid')
        crossings = np.where((time_series[:-1] < threshold_sig) & (time_series[1:] >= threshold_sig))[0] + 1
        intervals = np.diff(crossings)/60
        num_cross.append(np.arange(len(crossings)-1))
        inter_cross.append(intervals)

# %%
plt.figure()
plt.plot(np.concatenate(num_cross), np.concatenate(inter_cross), 'o') 

# %% stats
analysis_threshold = 2  # minumum incidents
bins = np.unique(np.concatenate(num_cross))
y = np.concatenate(inter_cross)*1
x = np.concatenate(num_cross)*1
bin_indices = np.digitize(x, bins)
y_means = np.array([y[bin_indices == i].mean() for i in range(1, len(bins))])
y_stds = np.array([y[bin_indices == i].std() for i in range(1, len(bins))])
x_count = np.array([len(y[bin_indices == i]) for i in range(1, len(bins))])
pos_ana = np.where(x_count > analysis_threshold)[0]

# Plot the results
plt.figure() 
plt.errorbar(bins[pos_ana+1], y_means[pos_ana], yerr=y_stds[pos_ana]/x_count[pos_ana], fmt='o', capsize=5, label="Mean with Std Error")
# plt.errorbar(bins[pos_ana+1], y_means[pos_ana], yerr=y_stds[pos_ana], fmt='o', capsize=5, label="Mean with Std Error")
plt.xlabel("encounter number")
plt.ylabel("counterturn duration (s)")

# %% stop-walk analysis
# without model, analyze ribbon encountering dynamics
###############################################################################
# %%

