# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:21:29 2025

@author: ksc75
"""

import numpy as np
import scipy as sp
from scipy import stats
from scipy import ndimage
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import h5py

# %% load and prpocess navigation data
# %% load mat file for the data structure
file_dir = r'C:\Users\ksc75\Yale University Dropbox\users\mahmut_demir\data\Smoke Navigation Paper Data\ComplexPlumeNavigationPaperData.mat'
# Open the .mat file
with h5py.File(file_dir, 'r') as file:
    # Access the structure
    your_struct = file['ComplexPlume']

    # Access fields within the structure
    expmat = your_struct['Smoke']['expmat'][:]  # Load the dataset as a numpy array
    col_k = list(your_struct['Smoke']['col'].keys())
    col_v = list(your_struct['Smoke']['col'].values())
    # print(col.keys())

### for straight plume
with h5py.File(file_dir, 'r') as file:
    # Access the structure
    your_struct = file['StraightPlume']

    # Access fields within the structure
    expmat_str = your_struct['Smoke']['Dose4']['expmat'][:]  # Load the dataset as a numpy array
    print(expmat.shape)
    col_k_str = list(your_struct['Smoke']['Dose4']['col'].keys())
    col_v_str = list(your_struct['Smoke']['Dose4']['col'].values())
    
# %% now extract track data
chop = 3000000
down_samp = 3
trjNum = expmat[0,:][::down_samp][:chop]
signal = expmat[12,:][::down_samp][:chop]
stops = expmat[38,:][::down_samp][:chop]
turns = expmat[39,:][::down_samp][:chop]
vx_smooth = expmat[28,:][::down_samp][:chop]
vy_smooth = expmat[29,:][::down_samp][:chop]
x_smooth = expmat[31,:][::down_samp][:chop]
y_smooth = expmat[32,:][::down_samp][:chop]
speed_smooth = expmat[30,:][::down_samp][:chop]  #11 31
dtheta_smooth = expmat[34,:][::down_samp][:chop]  #14 35

trjNum_str = expmat_str[0,:][::down_samp][:chop]
signal_str = expmat_str[12,:][::down_samp][:chop]
vx_str = expmat_str[8,:][::down_samp][:chop]
vy_str = expmat_str[9,:][::down_samp][:chop]
x_str = expmat_str[6,:][::down_samp][:chop]
y_str = expmat_str[7,:][::down_samp][:chop]
speed_str = np.sqrt(vx_str**2 + vy_str**2)  ### speed calculation
stops_str = speed_str*0 + 0
stops_str[speed_str<1] = 1 

# %% switching environments
###############################################################################
vx_smooth, vy_smooth, stops, signal, trjNum = vx_str*1, vy_str*1, stops_str*1, signal_str*1, trjNum_str*1
x_smooth, y_smooth = x_str*1, y_str*1 

###############################################################################
# %%
plt.figure()
plt.plot(x_smooth, y_smooth,'k,')
pos = signal>5
plt.plot(x_smooth[pos], y_smooth[pos],'r,')

# %% plot track
n_tracks = np.unique(trjNum).shape[0]
trk = 1
pos = np.where(trjNum==trk)[0][10:-10]
pos_stop = np.where((stops==1))[0]# | (turns>0))[0]
pos_stop = np.intersect1d(pos, pos_stop)
plt.figure()
plt.plot(x_smooth[pos], y_smooth[pos],'k')
plt.plot(x_smooth[pos_stop], y_smooth[pos_stop],'r.')

# %% some pre-processing
v_threshold = 30
vx_smooth[np.abs(vx_smooth)>v_threshold] = v_threshold
vy_smooth[np.abs(vy_smooth)>v_threshold] = v_threshold
### make vxy concatenated
vxy_smooth = np.concatenate((vx_smooth[:,np.newaxis], vy_smooth[:,np.newaxis]), axis=1)
signal[np.isnan(signal)] = 0

dtheta_threshold = 360
dtheta_smooth[np.abs(dtheta_smooth)>dtheta_threshold] = dtheta_threshold
dtheta_smooth[np.isnan(dtheta_smooth)] = 0

# %% discretization for now
thre = 3
bin_signal = signal*1
bin_signal[signal<thre] = 0
bin_signal[signal>=thre] = 1

def discretize_time_series(series, thresholds):
    thresholds = np.sort(thresholds)
    states = np.digitize(series, bins=thresholds)
    return states

### list of bins
bin_stops = stops*1
bin_stops[stops>0] = 1  #### stops
bin_turns = turns*1
bin_turns[turns>0] = 1  ##### turns
bin_vi = discretize_time_series(speed_smooth*1,  [5,15])  #### try more continuous variables

# %% compute the error as a function of time lags
###############################################################################
# %% scaling in time
Ts = np.array([1,5,10,20,40,80,160, 320])
# Ts = np.array([1, 2,4,8,16,32,64,128])
repeat_sampling = 100
remove_pre = 10
# use a dict so each time-lag can hold a variable-length list of errors
errt = {tau: [] for tau in range(len(Ts))}

for rr in range(repeat_sampling):
    print(rr)
    ### randomly select two tracks and pre-process
    trk_ids = np.random.choice(n_tracks, size=2, replace=False)
    pos_a,pos_b = np.where(trjNum==trk_ids[0])[0], np.where(trjNum==trk_ids[1])[0]
    if len(pos_a)<=remove_pre or len(pos_b)<=remove_pre:
        continue
    else:
        vxy_a, vxy_b = vxy_smooth[pos_a,:][remove_pre:], vxy_smooth[pos_b,:][remove_pre:]
        signal_a, signal_b = bin_signal[pos_a][remove_pre:], bin_signal[pos_b][remove_pre:]
        stops_a, stops_b = bin_stops[pos_a][remove_pre:], bin_stops[pos_b][remove_pre:]
        turns_a, turns_b = bin_turns[pos_a][remove_pre:], bin_turns[pos_b][remove_pre:]
    
    ### CONDITIONS: find locations of the selected tracks, where it is stopping
        pos_twin = np.intersect1d(np.where(stops_a==1)[0], np.where(stops_b==1)[0])
        if len(pos_twin)>0:
            ### find pairs of pos_twin in nested for loops
            pos_st_a = np.intersect1d(np.where(stops_a==1)[0], np.where(signal_a==0)[0]-5) #np.where(stops_a == 1)[0]  #
            pos_st_b = np.intersect1d(np.where(stops_b==1)[0], np.where(signal_b==0)[0]-5) #np.where(stops_b == 1)[0]  #
            # pos_st_a = np.where(stops_a == 1)[0]  #
            # pos_st_b = np.where(stops_b == 1)[0]  #
            n_pairs = min(len(pos_st_a), len(pos_st_b))
            for pp in range(n_pairs):
                pos_ai, pos_bi = pos_st_a[pp], pos_st_b[pp]
                for tau in range(len(Ts)):
                    ### measure MSE of vxy at time lag Ts[tau]
                    lag = int(Ts[tau])
                    # ensure arrays are long enough to apply the lag
                    if lag <= 0:
                        continue
                    if lag+pos_ai >= len(vxy_a) or lag+pos_bi >= len(vxy_b):
                        # not enough samples in one or both tracks for this lag
                        continue
                    valid_pos = pos_twin[pos_twin + lag < len(vxy_a)]
                    if valid_pos.size == 0:
                        continue
                    diffs = vxy_a[pos_ai + lag, :] - vxy_b[pos_bi + lag, :]
                    errs = np.linalg.norm(diffs)#, axis=1)
                    errt[tau].append(errs)#.tolist())

# %% plot scaling results
# plot error vs time with raw points and std
plt.figure(figsize=(6,4))

# Convert lag bins to time (assume 60 Hz sampling). Change divisor if different.
times = Ts / 20.0

# compute mean and std for each lag
means = np.array([np.mean(errt[i]) if len(errt[i])>0 else np.nan for i in range(len(Ts))])
stds  = np.array([np.std(errt[i])  if len(errt[i])>0 else np.nan for i in range(len(Ts))])

# plot raw points (jitter x for visibility)
for i, t in enumerate(times):
    vals = errt[i]
    if len(vals) == 0:
        continue
    jitter = np.random.normal(0, (times.max()-times.min())*0.005, size=len(vals))
    plt.scatter(np.full(len(vals), t) + jitter, vals, color='gray', alpha=0.4, s=10, edgecolors='none')

# plot mean with std error bars
plt.errorbar(times, means, yerr=stds, fmt='-o', color='k', capsize=4, lw=1.5)

plt.xlabel('time lag (s)')
plt.ylabel('speed difference (mm/s)')
plt.ylim([-2,30])
plt.grid(True, alpha=0.3)
plt.tight_layout()

# %% compare
plt.figure()
plt.errorbar(times, means_0, yerr=stds_0, fmt='-o', capsize=4, lw=1.5, label='w/o signal')
plt.errorbar(times, means, yerr=stds, fmt='-o', capsize=4, lw=1.5, label='w/ signal')
plt.xlabel('time lag (s)')
plt.ylabel('speed difference (mm/s)')
# plt.ylim([-2,30]); 
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout();