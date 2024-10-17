# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:33:22 2024

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

import pickle
import gzip
import glob
import os

import seaborn as sns

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% test embedding method for trajectories
# start with vx,vy
# find dimension and clusters
# tirgger on odor removal for comparison, then see evolution of clusters

# %% for Kiri's data
### cutoff for short tracks
threshold_track_l = 60 * 20  # 20 # look at long-enough tracks

# Define the folder path
folder_path = 'C:/Users/ksc75/Downloads/ribbon_data_kc/'

# Use glob to search for all .pkl files in the folder
pkl_files = glob.glob(os.path.join(folder_path, '*.pklz'))

# Print the list of .pkl files
for file in pkl_files:
    print(file)

# %% for perturbed data
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/100424_new/'
# target_file = "exp_matrix.pklz"

# # List all subfolders in the root directory
# subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
# pkl_files = []

# # Loop through each subfolder to search for the target file
# for subfolder in subfolders:
#     for dirpath, dirnames, filenames in os.walk(subfolder):
#         if target_file in filenames:
#             full_path = os.path.join(dirpath, target_file)
#             pkl_files.append(full_path)
#             print(full_path)

# pkl_files = pkl_files[:8]

# %% concatenate across files in a folder
data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
nf = len(pkl_files)
masks = []   # where there is nan
track_id = []  # record track id (file and track)
rec_tracks = []  # record the full track x,y
rec_signal = []  # record opto signal
times = []   # record time in epoch
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
                if np.prod(mask_i)==1 and np.prod(mask_j)==1 and mean_v>1 and max_v<20:  ###################################### removing nan for now
                    data4fit.append(temp)  # get data for ssm fit
                    rec_tracks.append(temp_xy)  # get raw tracks
                    track_id.append(np.array([ff,ii]))  # get track id
                    rec_signal.append(data['signal'][pos])
                    cond_id += 1
                    masks.append(thetas)
                    times.append(data['t'][pos])
                # masks.append(mask_i)

# %% vectorize for simpliciy
vec_signal = np.concatenate(rec_signal)
vec_time = np.concatenate(times)
vec_vxy = np.concatenate(data4fit)
vec_xy = np.concatenate(rec_tracks)

# %% build features
window = 60*5
def build_features(data, window=window):
    T = len(data)
    samp_vec = data[:-np.mod(T, window),:]
    # features_tens = samp_vec.reshape(T // window, window, 2)
    # features = features_tens.reshape(T // window, window*2 )
   
    # features = samp_vec.reshape(-1, window * 2)
    
    vx = samp_vec[:, 0]
    vy = samp_vec[:, 1]
    vx_windowed = vx.reshape(-1, window)
    vy_windowed = vy.reshape(-1, window)
    features = np.hstack((vx_windowed, vy_windowed))
    return features

feature_vxy = build_features(vec_vxy)
feature_pos = build_features(vec_xy)

# %% test clustering
###############################################################################
# %%
from sklearn.cluster import AgglomerativeClustering
cmap = plt.get_cmap('tab10')

agg_clust = AgglomerativeClustering(n_clusters=5)
labels = agg_clust.fit_predict(feature_vxy)

# %%
plt.figure()
for ii in range(5):
    plt.subplot(1,5,ii+1)
    pos = np.where(labels==ii)[0]
    for tr in range(len(pos)):
        xy_i = feature_pos[pos[tr],:]
        xy_i = np.vstack((xy_i[:window], xy_i[window:])).T
        xy_i = xy_i - xy_i[0,:]
        plt.plot(xy_i[:,0], xy_i[:,1], '.', color=cmap(ii), alpha=.01)
    plt.xlim([-50,50])
    plt.ylim([-50,50])
    plt.grid(True)
    
# %% use U-map to visualize
import umap

# Step 1: Generate clustered data (T x feature) and labels (T x 1)
T = 500  # Number of data points
features = 10  # Number of features

# Generate random data with 5 clusters/classes
data, labels = feature_vxy, labels

# Step 2: Apply UMAP to reduce dimensionality to 2D
reducer = umap.UMAP(n_components=2, random_state=42)
data_2d = reducer.fit_transform(data)

# Step 3: Plot the 2D data and color-code according to the labels
unique_labels = np.unique(labels)
# cmap = plt.get_cmap('tab10', len(unique_labels))

# Step 4: Plot the 2D data and color-code according to the labels
plt.figure(figsize=(10, 6))

# Use the `cmap` to color each class
# scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap=cmap, s=50, alpha=0.8)

for ii in range(5):
    pos = np.where(labels==ii)[0]
    plt.scatter(data_2d[pos, 0], data_2d[pos, 1], cmap=cmap(ii), s=50, alpha=0.8)

# Add a color bar for the classes
# cbar = plt.colorbar(scatter, ticks=unique_labels)
# cbar.set_label('Classes')

# Add plot title and labels
plt.title("2D UMAP projection of clustered tracks")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)

# %% conditioning on odor on or off!
def build_signal(data, window=window):
    T = len(data)
    samp_vec = data[:-np.mod(T, window)]
    signal_windowed = samp_vec.reshape(-1, window)
    return signal_windowed

feat_time = build_signal(vec_time)
feat_odor = build_signal(vec_signal)

# %%
odor_on = np.where(np.sum(feat_odor,1)>0)[0]
odor_off = np.where((feat_time[:,0]>45+30) & (feat_time[:,-1]<45+30+20))[0]

plt.figure()
plt.scatter(data_2d[:, 0], data_2d[:, 1], color='k', s=5, alpha=0.8)
plt.scatter(data_2d[odor_on, 0], data_2d[odor_on, 1], color='b', s=50, alpha=0.5)
plt.scatter(data_2d[odor_off, 0], data_2d[odor_off, 1], color='r', s=50, alpha=0.5)
