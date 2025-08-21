# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 17:02:56 2025

@author: ksc75
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from collections import defaultdict

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import h5py
import matplotlib.cm as cm

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

# %% some pre-processing
v_threshold = 30
vx_smooth[np.abs(vx_smooth)>v_threshold] = v_threshold
vy_smooth[np.abs(vy_smooth)>v_threshold] = v_threshold
signal[np.isnan(signal)] = 0

dtheta_threshold = 360
dtheta_smooth[np.abs(dtheta_smooth)>dtheta_threshold] = dtheta_threshold
dtheta_smooth[np.isnan(dtheta_smooth)] = 0

# %% build track-based data
track_id = np.unique(trjNum)
n_tracks = len(track_id)
data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
masks = []   # where there is nan
track_ids = []  # record track id (file and track)
rec_tracks = []  # record the full track x,y
rec_signal = []  # record opto signal
times = []   # record time in epoch
speeds = []
rec_stops = []
rec_turns = []

for tr in range(n_tracks):
    print(tr)
    ### extract features
    pos = np.where(trjNum==track_id[tr])[0]  # position of this track
    temp_xy = np.column_stack((x_smooth[pos] , y_smooth[pos]))
    temp_vxy = np.column_stack((vx_smooth[pos] , vy_smooth[pos]))
    # temp_vxy = np.column_stack((vx_smooth[pos] , vy_smooth[pos], dtheta_smooth[pos]))   ### test with dtheta feature!
    
    ### recording
    data4fit.append(temp_vxy)  # get data for ssm fit
    rec_tracks.append(temp_xy)  # get raw tracksd
    track_ids.append(np.zeros(len(pos))+tr) 
    rec_signal.append(signal[pos])
    speeds.append(speed_smooth[pos])
    rec_stops.append(stops[pos])
    rec_turns.append(turns[pos])

# %% information function
def surprisal_kth_order(binary_series, k):
    counts = defaultdict(lambda: np.array([1, 1]))  # Laplace smoothing

    for t in range(k, len(binary_series) - 1):
        context = tuple(binary_series[t-k:t])
        next_val = binary_series[t]
        counts[context][next_val] += 1

    surprisals = []
    for t in range(k, len(binary_series)):
        context = tuple(binary_series[t-k:t])
        next_val = binary_series[t]
        prob = counts[context][next_val] / counts[context].sum()
        surprisals.append(-np.log2(prob))

    return np.array(surprisals)

def discretize_time_series(series, thresholds):
    thresholds = np.sort(thresholds)
    states = np.digitize(series, bins=thresholds)
    return states

### list of bins
bin_vi = discretize_time_series(speed_smooth*1,  [5,15]) 

# Example data
# np.random.seed(0)
binary_series = np.random.choice([0, 1], size=100)

# Compute surprisal
k = 5
surp = surprisal_kth_order(binary_series, k)

# Define a threshold to identify "surprising" events
threshold = 1.0  # in bits
surprising_mask = surp > threshold  # boolean vector

# Print example output
print("Surprisal values:", surp[:10])
print("Surprising events mask:", surprising_mask[:10])

plt.figure()
plt.subplot(211)
plt.plot(surp[:30]); plt.ylabel('surprisal'); plt.xticks([])
# plt.plot(surplll[:30]); plt.ylabel('surprisal'); plt.xticks([])
plt.subplot(212)
plt.plot(binary_series[1:31],'k'); plt.ylabel('signal')

# %% explore doubel conditions
def surprisal_joint_kth_order(x_series, y_series, k):
    assert len(x_series) == len(y_series), "x and y must be the same length"
    counts = defaultdict(lambda: np.array([1, 1]))  # Laplace smoothing for binary x_t

    # Build counts for (x_past, y_past) -> next x
    for t in range(k, len(x_series) - 1):
        x_context = tuple(x_series[t-k:t])
        y_context = tuple(y_series[t-k:t])
        context = x_context + y_context
        next_val = x_series[t]
        counts[context][next_val] += 1

    # Compute surprisal -log2(P(x_t | x_past, y_past))
    surprisals = []
    for t in range(k, len(x_series)):
        x_context = tuple(x_series[t-k:t])
        y_context = tuple(y_series[t-k:t])
        context = x_context + y_context
        next_val = x_series[t]
        prob = counts[context][next_val] / counts[context].sum()
        surprisals.append(-np.log2(prob))

    return np.array(surprisals)

# %% analysis loop
signal_suprisal = []
behavior_response = []
thre = 5
k_back = 5

for ii in range(n_tracks):
    
    #### compute signal suprisal
    signal_i = rec_signal[ii]
    bin_signal = signal_i*1
    bin_signal[signal_i<thre] = 0
    bin_signal[signal_i>=thre] = 1
    # sup_i = surprisal_kth_order(bin_signal.astype(int), k_back)
    # signal_suprisal.append(sup_i)
    
    ### record binarize behavior, or not...
    stops_i, turns_i = rec_stops[ii], rec_turns[ii]
    bin_stops = stops_i*1
    bin_stops[stops_i>0] = 1  #### stops
    bin_turns = turns_i*1
    bin_turns[turns_i>0] = 1  ##### turns
    behavior_response.append(bin_stops[k_back:])  ###[:-k]
    # behavior_response.append(bin_stops[:-k_back])
    
    ### testing double condition ###
    sup_i = surprisal_joint_kth_order(bin_signal.astype(int), (bin_stops).astype(int), k_back)
    signal_suprisal.append(sup_i)

# %% batch analysis
full_sup = np.concatenate(signal_suprisal)
full_response = np.concatenate(behavior_response)

### fix bin
# n_bins = 7
# bins = np.linspace(0,2,n_bins)
# bin_indices = np.digitize(full_sup, bins=bins, right=False)

### adaptive bin
n_bins = 7
bins = np.quantile(full_sup, q=np.linspace(0, 1, n_bins + 1))[:-1]
bin_indices = np.digitize(full_sup, bins=bins, right=True)

p_beh_sup = np.zeros(n_bins)
for bb in range(n_bins):
    pos = np.where(bin_indices==bb+1)[0]
    bin_response = full_response[pos]
    p_beh = len(np.where(bin_response==1)[0])/len(bin_response)
    p_beh_sup[bb] = p_beh

plt.figure()
plt.plot(bins, p_beh_sup, '-o')
plt.xlabel('mean surprisal (sensory and behavior)'); plt.ylabel('P(turn)')

# %% add sampling
import numpy as np
import matplotlib.pyplot as plt

n_bins = 7
n_bootstrap = 10

# Bin the data
bins = np.quantile(full_sup, q=np.linspace(0, 1, n_bins + 1))[:-1]
bin_indices = np.digitize(full_sup, bins=bins, right=True)

p_beh_sup = np.zeros(n_bins)
p_beh_err = np.zeros(n_bins)

for bb in range(n_bins):
    pos = np.where(bin_indices == bb + 1)[0]
    bin_response = full_response[pos]

    # Compute observed proportion
    if len(bin_response) > 0:
        p_beh_sup[bb] = np.mean(bin_response == 1)

        # Bootstrap error estimation
        bootstrap_vals = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(bin_response, size=len(bin_response), replace=True)
            bootstrap_vals.append(np.mean(sample == 1))

        p_beh_err[bb] = np.std(bootstrap_vals)
    else:
        p_beh_sup[bb] = np.nan
        p_beh_err[bb] = np.nan

# Plot with error bars
plt.figure()
plt.errorbar(bins, p_beh_sup, yerr=p_beh_err, fmt='-o', capsize=4)
plt.xlabel(r'mean surprisal (bits, $<-logP>_t$)')
plt.ylabel('P(stop)')
plt.title('Behavior vs. Surprisal with Error Bars')
plt.grid(True)
plt.show()

# %% scan K... later
# plt.figure()
# plt.errorbar(bins1, p_beh_sup1, yerr=p_beh_err1, fmt='-o', label='k=50ms')
# plt.errorbar(bins5, p_beh_sup5, yerr=p_beh_err5, fmt='-o', label='250ms')
# plt.errorbar(bins10, p_beh_sup10, yerr=p_beh_err10, fmt='-o',label='500ms')
# plt.errorbar(bins20, p_beh_sup20, yerr=p_beh_err20, fmt='-o', label='1s')

# plt.errorbar(bins, p_beh_sup, yerr=p_beh_err, fmt='-*')
# plt.xlim([-0.01, 0.15])
# plt.xlabel(r'mean surprisal (bits, $<-logP>_t$)')
# plt.ylabel('P(stop)')
# plt.grid(True)
# plt.legend()

# %% hits
def hit_rate_calc(tracks):
    hits = []
    for track in tracks:
        hit = 0
        end_coords = track[len(track)-5:]
        for coord in end_coords:
            if (0 < coord[0] < 50) & (75 < coord[1] < 125):
                hit = 1
                break
        hits.append(hit)
    return np.array(hits)

def angle_between(u, v, degrees=False):
    # Normalize dot product by magnitudes
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    cosine = dot_product / (norm_u * norm_v)

    # Clip to avoid numerical issues outside [-1, 1]
    cosine = np.clip(cosine, -1.0, 1.0)

    angle_rad = np.arccos(cosine)
    return np.degrees(angle_rad) if degrees else angle_rad

def nav_eff_calc(tracks):
    hits = []
    target = np.array([0, 90])
    for track in tracks:
        hit = 0
        end_coords = track[len(track)-5:]
        for coord in end_coords:
            if (0 < coord[0] < 50) & (75 < coord[1] < 125):
                vec = track[5,:] - track[-5,:]
                target_proj = target - track[5,:]
                hit = angle_between(target, vec, True) #/ len(track)
                break
        hits.append(hit)
    return np.array(hits)

def sig_calc(signals, thre=5):
    fracs, sfs = [], []
    for signal in signals:
        signal = signal[:len(signal)//3] #### remove later in time ###
        bin_signal = signal*1  
        bin_signal[signal<thre] = 0
        bin_signal[signal>=thre] = 1
        frac = len(np.where(bin_signal>0)[0])/len(bin_signal)
        temp = np.abs(np.diff(bin_signal))
        sf = len(np.where(temp>0)[0])/len(temp)
        fracs.append(frac)
        sfs.append(sf)
    return np.array(fracs), np.array(sfs)
    
k = 5  ### surprisal window
thre = 5  ### odor detection threshold
signals = rec_signal
tracks = rec_tracks
surps = []
hits = hit_rate_calc(tracks) ### hit rate
# hits = nav_eff_calc(tracks)  ### projection
fracs, sfs = sig_calc(signals)
print(np.where(hits==1))
for ii in np.arange(n_tracks):
    signal = signals[ii]
    bin_signal = signal*1
    bin_signal[signal<thre] = 0
    bin_signal[signal>=thre] = 1
    sup_i = surprisal_kth_order(bin_signal.astype(int), k)
    mean_surp = np.mean(sup_i[:len(sup_i)//3])  ### remove later in time ###
    surps.append(mean_surp / fracs[ii])
### adaptive bin
n_bins = 7 #3
bins = np.quantile(surps, q=np.linspace(0, 1, n_bins + 1))[:-1]
bin_indices = np.digitize(surps, bins=bins, right=True)
hit_surp = np.zeros(n_bins)
for bb in range(n_bins):
    pos = np.where(bin_indices==bb+0)
    
    ### as a function of hit rate, fraction on, or switching frequency ###
    bin_hit = np.mean(hits[pos])  ### hit rate
    # bin_hit = np.mean(fracs[pos])  ### fraction on
    # bin_hit = np.mean(sfs[pos])   ### switching rate
    
    hit_surp[bb] = bin_hit
plt.figure()
plt.plot(bins, hit_surp, "-o")
plt.xlabel(r'mean surprisal per track (bits, $<-logP>_t$)')
plt.xlabel('surprisal per encounter (surprise/frequency)')
plt.ylabel('navigational hit rate')
# plt.ylabel('switching frequency')
# plt.ylabel('fraction on')

# %% visualize in space
plt.figure()
unique_labels = np.unique(bin_indices)
color_map = {label: color for label, color in zip(unique_labels, cm.tab10.colors)}

# Plot in a loop
plt.figure()
for i in range(200):#len(tracks)):
    track = tracks[i]
    label = bin_indices[i]
    plt.plot(track[:,0], track[:,1], '.', markersize=.7,alpha=0.5, color=color_map[label], label=label)

# %% show track with later time removed
plt.figure()
for ii in range(500):
    track = tracks[ii]
    plt.plot(track[:len(track)//3,0], track[:len(track)//3,1],'.')
    
# %% testing how behavior changes the NEXT surprisal signal
# %% analysis loop
signal_suprisal = []
behavior_response = []
thre = 5
k_back = 5
window = 150
lags = 125


def behavior_rate(behavior, window=window):
    # rate = np.convolve(np.ones(window)/window, behavior,mode='same')
    rate = np.convolve(behavior, np.ones(window)/window, mode='full')[:len(behavior)] 
    return rate

plt.figure()
lagss = np.array([100, 200, 400, 800, 1600, 3200])
for ll in range(len(lagss)):
    lags = lagss[ll]
    print(ll)
    
    for ii in range(n_tracks):
        #### compute signal suprisal
        signal_i = rec_signal[ii]
        bin_signal = signal_i*1
        bin_signal[signal_i<thre] = 0
        bin_signal[signal_i>=thre] = 1
        sup_i = surprisal_kth_order(bin_signal.astype(int), k_back)
        signal_suprisal.append(sup_i[:-lags])
        
        ### record binarize behavior, or not...
        stops_i, turns_i = rec_stops[ii], rec_turns[ii]
        bin_stops = stops_i*1
        bin_stops[stops_i>0] = 1  #### stops
        bin_turns = turns_i*1
        bin_turns[turns_i>0] = 1  ##### turns
        behavior_response.append(behavior_rate(bin_turns)[k_back:][lags:])  ### test turns or stops
    
    ## %% batch analysis
    full_sup = np.concatenate(signal_suprisal)
    full_response = np.concatenate(behavior_response)
    
    ### adaptive bin
    n_bins = 9
    bins = np.quantile(full_response, q=np.linspace(0, 1, n_bins + 1))[:-1]
    bin_indices = np.digitize(full_response, bins=bins, right=True)
    
    p_sup_beh = np.zeros(n_bins)
    for bb in range(n_bins):
        pos = np.where(bin_indices==bb+1)[0]
        bin_sup = full_sup[pos]
        p_sup_beh[bb] = np.nanmean(bin_sup)
    
    # plt.figure()
    plt.plot(bins, p_sup_beh, '-o', label= f"lag= {int(round(lags/30))}")
    plt.xlabel('turn rate'); plt.ylabel('next surprisal'); plt.legend()

# %% checking autocorr
def compute_autocorrelation(data, max_lag):
    n = len(data)
    mean = np.nanmean(data)
    autocorr_values = []
    for lag in range(1, max_lag + 1):
        numerator = np.nansum((data[:-lag] - mean) * (data[lag:] - mean))
        denominator = np.nansum((data - mean) ** 2)
        autocorrelation = numerator / denominator
        autocorr_values.append(autocorrelation)
    return np.arange(1, max_lag + 1), np.array(autocorr_values)/max(autocorr_values)

aa,bb = compute_autocorrelation(full_response, 300)
plt.figure()
plt.plot(aa,bb)
