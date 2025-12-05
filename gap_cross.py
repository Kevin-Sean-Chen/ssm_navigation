# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 11:48:30 2025

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

# %% meta analysis
### gap crossing analysis
### also the first test from new pipeline with stimuli

# %% find target files
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-09-06\kevin' ### gap crossing data
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-09-09\kevin' ### gap crossing data
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-10-11\kevin' ### gap crossing data
root_dir = r'C:\Users\ksc75\Yale University Dropbox\users\kevin_chen\data\gap_cross\2025-10-30\kevin' ### gap crossing data

target_file = "exp_matrix.joblib"
exp_type = 'increasing gap 60s Kir_EPG'
# exp_type = 'short to gap 60s wind15'
# forbidden_subs = ['Kir', 'TNT']
forbidden_subs = []

subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
target_files = []
# Loop through each subfolder to search for the target file
for subfolder in subfolders:
    # Get files directly in this subfolder (no recursion)
    files = os.listdir(subfolder)
    if target_file in files and exp_type in subfolder and not any(bad in subfolder for bad in forbidden_subs):
        full_path = os.path.join(subfolder, target_file)
        target_files.append(full_path)
        print(full_path)
        
# %% load data
target_files = pkl_files*1   ### from batch ###################################

target_files_sorted = natsorted(target_files)
data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
nf = len(target_files)
masks = []   # where there is nan
track_id = []  # record track id (file and track)
rec_tracks = []  # record the full track x,y
rec_signal = []  # record opto signal
times = []   # record time in epoch
thetas = []
cond_id = 0
threshold_track_l = 60*2

for ff in range(nf):
    ### load file
    print(ff)
    data = joblib.load(target_files_sorted[ff])
        
    ### extract tacks
    n_tracks = np.unique(data['trjn'])
    
    for ii in n_tracks:
        pos = np.where(data['trjn']==ii)[0] # find track elements
        # if sum(data['behaving'][pos]):  # check if behaving
        if 1==1: 
            if len(pos) > threshold_track_l:
                
                ### make per track data
                # temp = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos] , \
                                        # data['theta_smooth'][pos] , data['signal'][pos]))
                theta = data['theta'][pos]
                temp = np.stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]),1)#######
                temp_xy = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                                
                ### criteria
                mask_i = np.where(np.isnan(temp), 0, 1)
                mask_j = np.where(np.isnan(theta), 0, 1)
                mean_v = np.nanmean(np.sum(temp**2,1)**0.5)
                max_v = np.max(np.sum(temp**2,1)**0.5)
                # print(mean_v)
                # if np.prod(mask_i)==1 and np.prod(mask_j)==1 
                if np.prod(mask_i)==1 and mean_v>.1 and max_v<30: #max_v<20:  ###################################### removing nan for now
                    data4fit.append(temp)  # get data for ssm fit
                    rec_tracks.append(temp_xy)  # get raw tracks
                    track_id.append(np.zeros(len(pos))+ii) 
                    rec_signal.append(data['signal'][pos].squeeze())
                    # rec_signal.append(np.ones((len(pos),1)))   ########################## hacking if needed
                    cond_id += 1
                    times.append(data['t'][pos])
                    thetas.append(theta)

# %% vectorize for simpliciy
vec_signal = np.concatenate(rec_signal)  # odor signal
vec_time = np.concatenate(times)  # time in trial
vec_vxy = np.concatenate(data4fit)  # velocity
vec_xy = np.concatenate(rec_tracks)  # position
vec_ids = np.concatenate(track_id)  # track ID
vec_theta = np.concatenate(thetas)

# %% visualization++
pos = np.where(vec_signal>0)[0]
plt.figure()
plt.plot(vec_xy[:,0], vec_xy[:,1],'k,')
plt.plot(vec_xy[pos,0], vec_xy[pos,1],'r,')

# %% upwind when in signal
ntracks = len(rec_tracks)
upwindx = []
thre_signalt = 60*3

for ii in range(ntracks):
    tracki = rec_tracks[ii]
    signali = rec_signal[ii]
    pos = np.where(signali>0)[0]
    if len(pos)>thre_signalt:
        dx = tracki[pos[0],0] - tracki[pos[-1],0]
        upwindx.append(dx)
        
# %% compare MSD
# plt.figure()
# plt.violinplot([upwindx, dec_dx], positions=[1.2, 2], showmeans=True)
# # Formatting
# plt.xticks([1.2, 2], ["increase", "decrease"])
# plt.ylabel("upwind via tracking (mm)")

# %% search during crossing
window = 60*2  # window size in frames
lossx = np.array([75, 131, 183, 233])-1  ### for increasing
lossx = np.array([45, 105, 167, 232])-1  ### for decreasing
crossing_indices = {i: [] for i in range(len(lossx))}  # Dictionary to store indices for each condition
crossing_segments = {i: [] for i in range(len(lossx))}  # Dictionary to store track segments

for ii in range(ntracks):
    ### load track
    tracki = rec_tracks[ii]
    signali = rec_signal[ii]
    pos = np.where(signali>0)[0]
    ### if crossing
    if len(pos)>thre_signalt:
        for ll in range(len(lossx)):
            xi = tracki[:,0]
            cross_idx = np.where((xi[:-1] > lossx[ll]) & (xi[1:] <= lossx[ll]))[0]
            ### if crossed this one
            if len(cross_idx) > 0:
                # Found crossing point(s)
                for idx in cross_idx:
                    # Store crossing indices
                    crossing_indices[ll].append((ii, idx))
                    # Store track segment after crossing
                    if idx + window <= len(tracki):
                        segment = tracki[idx:idx+window]
                        seg_signal = signali[idx:idx+window]
                        pos_signal = np.where(seg_signal!=0)[0]
                        segment[pos_signal,:] = np.nan
                        crossing_segments[ll].append(segment)

# %% plots
plt.figure(figsize=(15,5))
mean_dx, std_dx = [],[]
mean_dy, std_dy = [],[]
kk = 0
for ii in range(len(lossx)):#-1, -1, -1):  # Changed to iterate in reverse
    plt.subplot(1,4,kk+1)
    displaceix = np.zeros(len(crossing_segments[ii]))
    displaceiy = np.zeros(len(crossing_segments[ii]))
    for jj in range(len(crossing_segments[ii])):
        trackj = crossing_segments[ii][jj]
        plt.plot(trackj[:,0]-trackj[0,0], trackj[:,1]-trackj[0,1],'k-', alpha=0.1)
        displaceix[jj] = np.nanmean((trackj[:,0]-trackj[0,0])**2)
        displaceiy[jj] = np.nanmean((trackj[:,1]-trackj[0,1])**2)
    kk += 1
    plt.title(f'crossing at {lossx[ii]}mm')
    # plt.xlim([lossx[ii]-20, lossx[ii]+100])
    plt.ylim([-50, 50])
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

    mean_dx.append(np.nanmean(displaceix))
    std_dx.append(np.nanstd(displaceix)/jj**0.5)
    mean_dy.append(np.nanmean(displaceiy))
    std_dy.append(np.nanstd(displaceiy)/jj**0.5)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.errorbar([1,2,3,4], mean_dx, yerr=std_dx, fmt='o')
plt.xlabel('gap order'); plt.ylabel('up wind displacement (x)'); plt.ylim([50,150])
plt.subplot(1,2,2)
plt.errorbar([1,2,3,4], mean_dy, yerr=std_dy, fmt='o')
plt.xlabel('gap order'); plt.ylabel('cross wind displacement (y)'); plt.ylim([10,100])

# %% checking odor reponse
mask_pos = (vec_xy[:,0] > 25) & (vec_xy[:,0] < 170) & (vec_xy[:,1] > 20) & (vec_xy[:,1] < 295)
vec_speed = np.mean(vec_vxy**2,1)**0.5
pos_ws = vec_signal>1
pos_wo = vec_signal<1
pos_wo, pos_ws = np.where(mask_pos | pos_wo)[0], np.where(mask_pos | pos_ws)[0]

valid_speed = vec_speed[np.isfinite(vec_speed)]
bins = np.histogram_bin_edges(valid_speed, bins=30)
plt.figure()
plt.hist(vec_speed[pos_wo], bins=bins, density=True, alpha=0.7, label='odor')
plt.hist(vec_speed[pos_ws], bins=bins, density=True, alpha=0.5, label='w/o')
plt.xlabel('speed'); plt.ylabel('density'); plt.legend(); plt.yscale('log')
plt.tight_layout(); plt.show()

valid_speed = vec_theta[np.isfinite(vec_theta)]
bins = np.histogram_bin_edges(valid_speed, bins=20)
plt.figure()
plt.hist(vec_theta[pos_wo], bins=bins, density=True, alpha=0.7, label='odor')
plt.hist(vec_theta[pos_ws], bins=bins, density=True, alpha=0.5, label='w/o')
plt.xlabel('angle'); plt.ylabel('density'); plt.legend(); #plt.yscale('log')
plt.tight_layout(); plt.show()


# %% visualization
###############################################################################
# %% measure pre, post
window = 60*2  # window size in frames
wind_past = 60*6 # window prior to loss
min_spd = 0
# lossx = np.array([75, 131, 183, 233])  ### for increasing
# lossx = np.array([45, 105, 167, 232])-1   ### for decreasing
crossing_indices = {i: [] for i in range(len(lossx))}  # Dictionary to store indices for each condition
crossing_segments = {i: [] for i in range(len(lossx))}  # Dictionary to store track segments
history_signal = {i: [] for i in range(len(lossx))} # Dictionary to store past history or behavior
cross_events = {i: [] for i in range(len(lossx))}  # Dictionary to store crossing events
cross_action = {i: [] for i in range(len(lossx))}  ### record the action for clustering
raw_hist_sig = {i: [] for i in range(len(lossx))} ### measure raw history trace for classification
hist_features = {i: [] for i in range(len(lossx))}  ### place some a priori chosen factors to make crossing prediction later

for ii in range(ntracks):  ### loop for tracks
    ### load track
    tracki = rec_tracks[ii]  # track location
    signali = rec_signal[ii]  # signal
    vxyi = data4fit[ii]  # velocity
    meani = np.mean(data4fit[ii]**2,1)
    pos = np.where(signali>0)[0]
    ### if crossing
    if len(pos)>thre_signalt:  ### check for signal
        for ll in range(len(lossx)):
            xi = tracki[:,0]
            cross_idx = np.where((xi[:-1] > lossx[ll]) & (xi[1:] <= lossx[ll]))[0]
            ### if crossed this one
            if len(cross_idx) > 0:  ### check for crossing
                # Found crossing point(s)
                for idx in cross_idx:  ### looop for tracks that cross
                    # Store crossing indices
                    crossing_indices[ll].append((ii, idx))
                    # Store track segment after crossing
                    if idx + window <= len(tracki) and len(tracki[:idx])>wind_past:  ### check for history  ### can be relaxed if not clustering for same-size!
                        
                        if np.nanmean(meani[:idx])>min_spd:  ### conditional of past behavior
                            segment = tracki[idx:idx+window]
                            seg_signal = signali[idx:idx+window]
                            pos_signal = np.where(seg_signal!=0)[0]
                            # segment[pos_signal,:] = np.nan
                            crossing_segments[ll].append(segment)
                            pre_wind = np.min([len(signali[:idx]), wind_past])
                            hist_signal = signali[idx-pre_wind:idx]
                            hist_signal[np.isnan(hist_signal)] = 0
                            raw_hist_sig[ll].append(hist_signal)
                            ### z-score
                            # history_signal[ll].append(np.nanstd(hist_signal)/np.nanmean(hist_signal))
                            ### mean (intermittency)
                            # temp = hist_signal*0
                            # temp[hist_signal>0] = 1 
                            # history_signal[ll].append(np.nanmean(temp))
                            ### encounters
                            temp = hist_signal*0
                            temp[hist_signal>0] = 1 
                            history_signal[ll].append(len(np.where(np.diff(temp)>0)[0]))
                            
                            v_temp = np.array([[vxyi[idx:idx+window,0]],[(vxyi[idx:idx+window,1])]])
                            cross_action[ll].append(v_temp.reshape(-1))

                            ### Check if there are future positions beyond current loss point with signal (aks crossing)
                            future_positions = tracki[idx:, 0] < lossx[ll]  # positions beyond current loss point  #### UP (<) or DOWN (>) ###
                            future_signals = signali[idx:] > 0  # non-zero signals in future positions
                            if np.any(future_positions & future_signals):  # if any positions match both conditions
                                cross_events[ll].append(1)  # mark as crossing event
                            else:
                                cross_events[ll].append(0)  # mark as non-crossing event
                                
                            ### build features ###
                            mean_sig = np.nanmean(hist_signal)
                            std_sig = np.nanstd(hist_signal)
                            freq_sig = len(np.where(np.diff(temp)>0)[0]) / (pre_wind/60)  # encounters per second
                            ### compute past speed
                            past_speed = np.nanmean(meani[idx-pre_wind:idx]**0.5)
                            past_spd_std = np.nanstd(meani[idx-pre_wind:idx]**0.5)
                            # Temporal features
                            duration_in_signal = np.nansum(temp) * (1/60)  # seconds in signal
                            # Path features
                            path_positions = tracki[idx-pre_wind:idx, :]
                            path_length = np.nansum(np.sqrt(np.sum(np.diff(path_positions, axis=0)**2, axis=1)))
                            net_displacement = np.sqrt(np.nansum((path_positions[-1] - path_positions[0])**2))
                            path_tortuosity = path_length / (net_displacement + 1e-6)  # add small value to avoid division by zero
                            hist_features[ll].append([mean_sig, std_sig, freq_sig, past_speed, past_spd_std, path_tortuosity, duration_in_signal])

# %% simple logistic prediction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Concatenate features and events across all loss points
X_all = np.concatenate([np.array(hist_features[ll]) for ll in range(1,len(lossx))])
y_all = np.concatenate([np.array(cross_events[ll]) for ll in range(1,len(lossx))])

# Feature names for plotting
feature_names = ['Mean Signal', 'Std Signal', 'Encounter Freq', 
                'Past Speed', 'Speed Std', 'Path Tortuosity', 
                'Duration in Signal']

# K-fold sampling and training
K = 30
accuracies = []
all_weights = []
all_confusion = []
all_probas = []

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

for k in range(K):
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all, test_size=0.2, stratify=y_all)
    model = LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=5000)
    model.fit(X_train, y_train)
    
    # Record metrics
    accuracies.append(model.score(X_test, y_test))
    all_weights.append(model.coef_[0])
    
    # Get confusion matrix
    y_pred = model.predict(X_test)
    all_confusion.append(confusion_matrix(y_test, y_pred))
    
    # Get probabilities
    all_probas.append((model.predict_proba(X_test), y_test))

# Convert to numpy arrays
accuracies = np.array(accuracies)
all_weights = np.array(all_weights)
mean_confusion = np.mean(np.array(all_confusion), axis=0)

# Plot results
fig = plt.figure(figsize=(15, 5))

# 1. Feature importance
plt.subplot(131)
mean_weights = np.mean(all_weights, axis=0)
std_weights = np.std(all_weights, axis=0)
plt.errorbar(range(len(feature_names)), mean_weights, yerr=std_weights, fmt='o')
plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel('Weight')
plt.title(f'Feature Importance\nAccuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}')

# 2. Confusion matrix
plt.subplot(132)
disp = ConfusionMatrixDisplay(mean_confusion)
disp.plot(ax=plt.gca(), cmap='Blues', colorbar=False)
plt.title("Mean Confusion Matrix")

# 3. Probability validation
plt.subplot(133)
# Aggregate probabilities from all folds
all_proba_true = []
all_proba_false = []
for proba, y_true in all_probas:
    p_c = proba[:, 1]  # probability of crossing
    mask_true = (y_true == 1)
    all_proba_true.extend(p_c[mask_true])
    all_proba_false.extend(p_c[~mask_true])

plt.hist(all_proba_true, bins=20, alpha=0.7, label="true crossing", density=True)
plt.hist(all_proba_false, bins=20, alpha=0.4, label="no crossing", density=True)
plt.xlabel("P(crossing)")
plt.ylabel("density")
plt.legend(fontsize=8)
plt.title("Prediction Confidence")

plt.tight_layout()
plt.show()
    
# %% sorted by history
plt.figure(figsize=(15,5))
mean_dx, std_dx = [],[]
mean_dy, std_dy = [],[]
kk = 0
for ii in range(len(lossx)):
    # plt.subplot(1,4,kk+1)
    displaceix = np.zeros(len(crossing_segments[ii]))
    displaceiy = np.zeros(len(crossing_segments[ii]))
    
    # Create color map based on history signal values
    history_vals = history_signal[ii]
    norm = plt.Normalize(vmin=min(history_vals), vmax=max(history_vals))
    
    for jj in range(len(crossing_segments[ii])):
        trackj = crossing_segments[ii][jj]
        # Color each line based on its history signal value
        color = plt.cm.viridis(norm(history_vals[jj]))
        plt.plot(trackj[:,0]-trackj[0,0], trackj[:,1]-trackj[0,1], '-', color=color, alpha=0.5)
        # plt.plot(trackj[:,0]-trackj[0,0], trackj[:,1]-trackj[0,1], 'k-', alpha=0.5)
        displaceix[jj] = np.nanmean((trackj[:,0]-trackj[0,0])**2)
        displaceiy[jj] = np.nanmean((trackj[:,1]-trackj[0,1])**2)
    
    # Add colorbar
ax = plt.gca()  # get current axes
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)  ### viridis, plasma, inferno, magma, cividis, turbo
plt.colorbar(sm, ax=ax, label='History signal')

kk += 1
plt.title(f'crossing at {lossx[ii]}mm')
plt.ylim([-50, 50])
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')

mean_dx.append(np.nanmean(displaceix))
std_dx.append(np.nanstd(displaceix)/jj**0.5)
mean_dy.append(np.nanmean(displaceiy))
std_dy.append(np.nanstd(displaceiy)/jj**0.5)

# %% P(crossing analysis)
reps = 100
n_samps = 50
p_cross = {i: [] for i in range(4)}  # Dictionary for each loss point
mean_signal = {i: [] for i in range(4)}  # Dictionary for each loss point
colors = ['r', 'g', 'b', 'k']  # Different color for each loss point

for rr in range(reps):
    print(rr)
    for ll in range(0,4):
        n_crosses = len(cross_events[ll])
        ### sample reps without replacement from n_crosses
        sampled_indices = np.random.choice(n_crosses, size=min(n_samps, n_crosses), replace=False)
        mean_signal[ll].append(np.nanmean(np.array(history_signal[ll])[sampled_indices]))
        p_cross[ll].append(np.mean(np.array(cross_events[ll])[sampled_indices]))

plt.figure()
for ll in range(4):
    plt.plot(mean_signal[ll], p_cross[ll], 'o', color=colors[ll], 
             label=f'Loss point {lossx[ll]}mm', alpha=0.5)
plt.xlabel('Mean History Signal')
plt.xlabel('Fano(history signal)')
plt.ylabel('Probability of Crossing')
# plt.ylabel('Probability of Refind')
plt.xlabel('mean # encounter')
# plt.xlabel('intermittency')
plt.title('Crossing Probability vs. History Signal')
plt.title('Refinding Probability vs. History Signal')
plt.legend(fontsize=15)
plt.show()

# %% same-gap comparison
reps = 150
n_samps = 100
dec_pcross, dec_sig = cross_events_dec[1], history_signal_dec[1]
inc_pcross, inc_sig = cross_events_inc[0], history_signal_inc[0]
colors = ['r', 'k']  # Different color for each loss point

dec_pcross, dec_sig = cross_events_dec[0], history_signal_dec[0]
inc_pcross, inc_sig = cross_events_inc[1], history_signal_inc[1]
colors = ['b', 'g']

for rr in range(reps):
    print(rr)
    n_crosses = len(dec_pcross)
    sampled_indices = np.random.choice(n_crosses, size=min(n_samps, n_crosses), replace=False)
    sigi = np.nanmean(np.array(dec_sig)[sampled_indices])
    pcrosi = np.mean(np.array(dec_pcross)[sampled_indices])
    plt.plot(sigi, pcrosi, 'o', color=colors[0], alpha=0.5)
    
    n_crosses = len(inc_pcross)
    sampled_indices = np.random.choice(n_crosses, size=min(n_samps, n_crosses), replace=False)
    sigi = np.nanmean(np.array(inc_sig)[sampled_indices])
    pcrosi = np.mean(np.array(inc_pcross)[sampled_indices])
    plt.plot(sigi, pcrosi, 'o', color=colors[1], alpha=0.5)

plt.xlabel('Mean History Signal')
plt.xlabel('Fano(history signal)')
plt.ylabel('Probability of Refinding')
plt.title('Refinding Probability vs. History Signal')
plt.show()

# %% test clusering
gapi = 0
X_cross = np.array(cross_action[gapi])
X_sig = np.array(raw_hist_sig[gapi])

# for ii in range(1,3):
#     X_cross = np.concatenate(( np.array(cross_action[ii]), X_cross))
#     X_sig = np.concatenate((np.array(raw_hist_sig[ii]), X_sig))

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import plotly.io as pio
# pio.renderers.default = 'browser'   # opens in your default web browser
pio.renderers.default = 'png'     # static fallback if needed
# pio.renderers.default = 'svg'     # another static option

def kmeans_central_ids(X, n_clusters, random_state=0, return_reps=False):
    """
    X: (n_samples, n_features) array
    n_clusters: int
    return_reps: if True, also return representative sample index per cluster

    Returns
    -------
    labels : (n_samples,) int array of cluster IDs in [0, n_clusters-1]
    reps   : (n_clusters,) int array of representative sample indices (optional)
    """
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)  # length = n_samples

    if not return_reps:
        return labels

    # representative index per cluster: closest sample to the centroid
    reps = np.empty(n_clusters, dtype=int)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            reps[c] = -1  # empty cluster (rare with k-means)
            continue
        d = pairwise_distances(X[idx], km.cluster_centers_[c:c+1], metric="euclidean").ravel()
        reps[c] = idx[np.argmin(d)]
    return labels, reps

labels = kmeans_central_ids(X_cross, n_clusters=4)                 # length = n_samples
cols = ['r','g','b','k','c']
plt.figure()
for ii in range(len(labels)):
    pos = np.where(labels[:200]==ii)[0]
    for jj in range(len(pos)):
        trackj = crossing_segments[gapi][pos[jj]]
        plt.plot(trackj[:,0]-trackj[0,0], trackj[:,1]-trackj[0,1], color=cols[ii], alpha=0.5)

labels_sig = kmeans_central_ids(X_sig, n_clusters=2)
temp = np.nanmean(X_sig,1) #np.std(X_sig,1)/(np.mean(X_sig,1)+1)
labels_sig = temp*0
labels_sig[temp>50*1.2] = 1

labels = np.array(labels, dtype=int)
labels_sig = np.array(labels_sig, dtype=int)

# %% turn into names
names  = ["small", "large"]  # names by index
labels_sig = np.array(names, dtype=object)[labels_sig]
names  = ["right", "left","dwell","cross"]  # names by index
labels = np.array(names, dtype=object)[labels]

# %% cool catagory plots
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

def alluvial_from_categories(cats_t, cats_tp1, stage_labels=("t", "t+1"),
                             renderer="browser"):
    """
    cats_t   : iterable of categories at time t      (length N(t))
    cats_tp1 : iterable of categories at time t+1    (length N(t+1))
               (If they represent the same cohort, lengths should match.
                If not, we still show flows between observed categories.)
    stage_labels : names for the two stages
    renderer     : plotly renderer (e.g., 'browser', 'notebook', 'png')
    """
    pio.renderers.default = renderer

    # Build a flow table (counts) between categories of t and t+1
    df = pd.DataFrame({"left": cats_t, "right": cats_tp1})
    flow = df.groupby(["left", "right"]).size().reset_index(name="value")

    # Label nodes with stage prefixes to keep ordering clear
    flow["left_lab"]  = stage_labels[0] + ":" + flow["left"].astype(str)
    flow["right_lab"] = stage_labels[1] + ":" + flow["right"].astype(str)

    # Create node list and mapping
    labels = pd.Index(pd.concat([flow["left_lab"], flow["right_lab"]]).unique()).tolist()
    lab2idx = {lab: i for i, lab in enumerate(labels)}

    # Build sankey source/target/value arrays
    sources = flow["left_lab"].map(lab2idx).to_list()
    targets = flow["right_lab"].map(lab2idx).to_list()
    values  = flow["value"].to_list()

    # Plot
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=20, thickness=14),
        link=dict(source=sources, target=targets, value=values)
    ))
    fig.update_layout(title=f"Alluvial-style flow: {stage_labels[0]} → {stage_labels[1]}",
                      font_size=12)
    fig.show()
    return fig

# --- Example ---
cats_t   = labels_sig
cats_tp1 = labels
alluvial_from_categories(cats_t, cats_tp1, stage_labels=("signal","action"))
