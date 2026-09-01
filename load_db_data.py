# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:13:37 2026

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import joblib
from natsort import natsorted

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import matplotlib
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)

# %% load joblib data and show data
target_file = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/projects/optogui/data/kevin_2026_july_FC2_gap_ribbon_source.joblib'
# target_file = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/projects/optogui/data/kevin_2026_july_FC2_flash.joblib'
# target_file = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/projects/optogui/data/kevin_2026_july_117_flash.joblib'
# target_file = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/projects/optogui/data/kevin_2026_july_117_gap_ribbon_source.joblib'

# %% find target file

data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
masks = []   # where there is nan
track_id = []  # record track id (file and track)
rec_tracks = []  # record the full track x,y
rec_signal = []  # record opto signal
times = []   # record time in epoch
thetas = []
dthetas = []
speeds = []
cond_id = 0
threshold_track_l = 60*1


data_all = joblib.load(target_file)
for ff in range(0,len(data_all),1):
    ### load file
    data = data_all[ff]['data']
        
    ### extract tacks
    n_tracks = np.unique(data['trjn'])
    
    if "vx_smooth" in data and "signal" in data:
        print(ff)
        for ii in n_tracks:
            pos = np.where(data['trjn']==ii)[0] # find track elements
            # if sum(data['behaving'][pos]):  # check if behaving
            if 1==1: 
                if len(pos) > threshold_track_l:
                    
                    ### make per track data
                    # temp = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos] , \
                                            # data['theta_smooth'][pos] , data['signal'][pos]))
                    theta = data['theta'][pos]
                    dtheta = data['dtheta_smooth'][pos]
                    temp = np.stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]),1)#######
                    temp_xy = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                    temp_xy = np.column_stack((data['headx_smooth'][pos] , data['heady_smooth'][pos]))
                                    
                    ### criteria
                    mask_i = np.where(np.isnan(temp), 0, 1)
                    mask_j = np.where(np.isnan(theta), 0, 1)
                    mean_v = np.nanmean(np.sum(temp**2,1)**0.5)
                    max_v = np.max(np.sum(temp**2,1)**0.5)
                    # print(mean_v)
                    # if np.prod(mask_i)==1 and np.prod(mask_j)==1: 
                    if np.prod(mask_i)==1 and mean_v>.1 and max_v<50: #max_v<20:  ###################################### removing nan for now
                        data4fit.append(temp)  # get data for ssm fit
                        rec_tracks.append(temp_xy)  # get raw tracks
                        track_id.append(np.zeros(len(pos))+ii) 
                        rec_signal.append(data['signal'][pos].squeeze())
                        # rec_signal.append(np.ones((len(pos),1)))   ########################## hacking if needed
                        cond_id += 1
                        times.append(data['t'][pos])
                        thetas.append(theta)
                        dthetas.append(dtheta)
                        speeds.append(data['spd_smooth'][pos].squeeze())

# %% vectorize for simpliciy
vec_signal = np.concatenate(rec_signal)  # odor signal
vec_time = np.concatenate(times)  # time in trial
vec_vxy = np.concatenate(data4fit)  # velocity
vec_xy = np.concatenate(rec_tracks)  # position
vec_ids = np.concatenate(track_id)  # track ID
vec_theta = np.concatenate(thetas)
vec_dth = np.concatenate(dthetas)
vec_spd = np.concatenate(speeds)

# %% visualization with signal
pos = np.where(vec_signal>0)[0]
plt.figure()
plt.plot(vec_xy[:,0], vec_xy[:,1],'k,')
plt.plot(vec_xy[pos,0], vec_xy[pos,1],'r,')

# %% density
plt.figure()
plt.hist(vec_xy[:,1], 50)
plt.xlabel('cross wind')
plt.xlim([20, 170])

# %% average time trace
bin_width = 0.5
bin_edges = np.arange(
        np.floor(vec_time.min() / bin_width) * bin_width,
        np.ceil(vec_time.max() / bin_width) * bin_width + bin_width,)
       
bin_index = np.digitize(vec_time, bin_edges) - 1
n_bins = len(bin_edges) - 1

mean_time = np.full(n_bins, np.nan)
mean_response = np.full(n_bins, np.nan)
std_response = np.full(n_bins, np.nan)
counts = np.zeros(n_bins, dtype=int)

for i in range(n_bins):
    in_bin = bin_index == i
    counts[i] = np.sum(in_bin)

    if counts[i] > 0:
        mean_time[i] = np.mean(vec_time[in_bin])
        responsei = vec_vxy[in_bin,0] #vec_spd[in_bin] #
        mean_response[i] = np.mean(responsei)
        std_response[i] = np.std(responsei)/np.sqrt(len(responsei))

# %% plot it
plt.figure()
plt.plot(mean_time, mean_response, linewidth=2, label="Mean response")

plt.fill_between(
    mean_time,
    mean_response - std_response,
    mean_response + std_response,
    alpha=0.3,
    label="Mean ± SD",
)

plt.xlabel("Time (s)")
plt.ylabel("Upwind (mm/s)")
# plt.legend()
plt.tight_layout()
plt.show()

# %%
###############################################################################
# %% raw navigational kinetics
###############################################################################
# %% basic stats
track_len = np.array([len(dd)/60 for dd in rec_signal])
print('mean length in seconds:', np.mean(track_len))
print('median length in seconds:', np.median(track_len))

# %% overall speed
bins = np.arange(0,50,2)
plt.figure()
plt.hist(vec_spd, bins, density=True)
pos = np.where(vec_signal>0)
plt.hist(vec_spd[pos], bins=bins, alpha=0.5, density=True)
plt.yscale('log')
plt.xlim([0,45]); plt.ylim([10e-6, 1])
plt.ylabel('pdf'); plt.xlabel('speed (mm/s)')
plt.show()

# %% wind-axis movement
bins = np.arange(-50,50,2)
plt.figure()
plt.hist(vec_vxy[:,0], bins, density=True)
pos = np.where(vec_signal>0)
plt.hist(vec_vxy[:,0][pos], bins=bins, alpha=0.5, density=True)
plt.yscale('log')
plt.xlim([-37,37]); plt.ylim([10e-6, 0.1])
plt.ylabel('pdf'); plt.xlabel('wind-axis velocity (mm/s)')
plt.show()

# %% d-theta analysis
bins = np.arange(-150,150,10)
plt.figure()
plt.hist(vec_dth, bins, density=True)
pos = np.where(vec_signal>0)
plt.hist(vec_dth[pos], bins=bins, alpha=0.5, density=True)
plt.yscale('log')
plt.xlim([-190,190]); plt.ylim([10e-6, 0.1])
plt.ylabel('pdf'); plt.xlabel('dtheta (deg/s)')
plt.show()

# %% ### polor
theta = np.linspace(0, 2 * np.pi, 100)
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

angles = np.mod(np.deg2rad(vec_theta), 2 * np.pi)
pos = vec_signal > 0

ax.hist(
    angles,
    bins=60,
    range=(0, 2 * np.pi),
    edgecolor="black",
    density=True,
)

ax.hist(
    angles[pos],
    bins=60,
    range=(0, 2 * np.pi),
    alpha=0.5,
    edgecolor="black",
    density=True,
)

plt.show()