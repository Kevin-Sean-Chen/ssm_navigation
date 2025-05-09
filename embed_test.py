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
# folder_path = 'C:/Users/ksc75/Downloads/ribbon_data_kc/'

# # Use glob to search for all .pkl files in the folder
# pkl_files = glob.glob(os.path.join(folder_path, '*.pklz'))

# # Print the list of .pkl files
# for file in pkl_files:
#     print(file)

# %% for perturbed data
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/100424_new/'
root_dir = 'C:/Users/kevin/Yale University Dropbox/users/kiri_choi/data/ribbon_sleap/2024-9-17/'
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
                
                ### symmetrize!
                temp = np.column_stack((data['vx_smooth'][pos] , np.abs(data['vy_smooth'][pos])))
                # temp = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
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
                    # track_id.append(np.array([ff,ii]))  # get track id
                    track_id.append(np.zeros(len(pos))+ii) 
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
vec_ids = np.concatenate(track_id)

# %% build features
window = int(60*2.)
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
action_id = np.zeros(feature_vxy.shape[0])  # keep track of the index for action

# %% test clustering
###############################################################################
# %%
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
cmap = plt.get_cmap('tab10')

agg_clust = AgglomerativeClustering(n_clusters=5)
labels = agg_clust.fit_predict(feature_vxy)

kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(feature_vxy)
centroids = kmeans.cluster_centers_

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
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate clustered data (T x feature) and labels (T x 1)
T = 500  # Number of data points
features = 10  # Number of features

# Generate random data with 5 clusters/classes
data, labels = feature_vxy, labels

# Step 2: Apply UMAP to reduce dimensionality to 2D
n_dim = 10
reducer = umap.UMAP(n_components=n_dim, random_state=42) #42
data_2d = reducer.fit_transform(data)

# Step 3: Plot the 2D data and color-code according to the labels
unique_labels = np.unique(labels)
# cmap = plt.get_cmap('tab10', len(unique_labels))

# Step 4: Plot the 2D data and color-code according to the labels
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Use the `cmap` to color each class
# scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap=cmap, s=50, alpha=0.8)

for ii in range(5):
    pos = np.where(labels==ii)[0]
    # plt.scatter(data_2d[pos, 0], data_2d[pos, 1], cmap=cmap(ii), s=50, alpha=0.8)
    ax.scatter(data_2d[pos, 0], data_2d[pos, 1], data_2d[pos, 2], cmap=cmap(ii), s=10, alpha=0.8)

# Add a color bar for the classes
# cbar = plt.colorbar(scatter, ticks=unique_labels)
# cbar.set_label('Classes')

# Add plot title and labels
ax.set_title("3D UMAP projection of clustered tracks")
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.set_zlabel("UMAP Dimension 2")
plt.grid(True)

# %% conditioning on odor on or off!
def build_signal(data, window=window):
    T = len(data)
    samp_vec = data[:-np.mod(T, window)]
    signal_windowed = samp_vec.reshape(-1, window)
    return signal_windowed

feat_time = build_signal(vec_time)
feat_odor = build_signal(vec_signal)
feat_ids = build_signal(vec_ids)

# %% condition on odor
odor_on = np.where(np.sum(feat_odor,1)>0)[0]
odor_off = np.where((feat_time[:,0]>45+30+1) & (feat_time[:,-1]<45+30+20))[0]

plt.figure()
plt.scatter(data_2d[:, 0], data_2d[:, 1], color='k', s=5, alpha=0.8)
plt.scatter(data_2d[odor_on, 0], data_2d[odor_on, 1], color='b', label='odor-on' ,s=50, alpha=0.5)
plt.scatter(data_2d[odor_off, 0], data_2d[odor_off, 1], color='r',label='odor-off', s=50, alpha=0.5)
plt.legend()

# %% LDA test
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

plt.figure()
group1 = data_2d[odor_on,:]
group2 = data_2d[odor_off,:]

# Combine the two groups into one dataset
X = np.vstack((group1, group2))
y = np.hstack((np.zeros(group1.shape[0]), np.ones(group2.shape[0])))
center = np.mean(X, 0)
X = X - center[None,:]
# Step 1: Fit Linear Discriminant Analysis (LDA)
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X, y)  # Project the points onto the LDA axis

# Step 2: Plot the original 2D scatter points
plt.scatter(group1[:, 0], group1[:, 1], color='blue', label='odor-on', alpha=0.7)
plt.scatter(group2[:, 0], group2[:, 1], color='red', label='odor-off', alpha=0.7)

# Step 3: Plot the LDA axis (green line)
# Calculate the slope and intercept of the LDA axis
slope = -lda.coef_[0, 0] / lda.coef_[0, 1]
intercept = -lda.intercept_ / lda.coef_[0, 1]
x_vals = np.array([X[:, 0].min(), X[:, 0].max()])
y_vals = slope * x_vals + intercept
# plt.plot(x_vals, y_vals, color='green', label='LDA Axis')

# Step 4: Project points onto the LDA axis
lda_direction = lda.coef_ / np.linalg.norm(lda.coef_)  # Normalize the LDA direction
X_proj = np.dot(X, lda_direction.T) * lda_direction  # Project each point onto the LDA direction


# Plot the projected points (along the LDA axis)
# plt.scatter(X_proj[:100, 0]+center[0], X_proj[:100, 1]+center[1], color='blue', marker='x', alpha=0.7)
# plt.scatter(X_proj[100:, 0]+center[0], X_proj[100:, 1]+center[1], color='red', marker='x', alpha=0.7)

# Step 5: Add labels and legend
plt.legend()
plt.title("Linear Discriminant Analysis and Projection")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

# %% projecting onto LDA
proj_value = np.dot(data_2d[odor_on,:]-center[None,:], lda_direction.T).reshape(-1)
prev_odor = np.sum(feat_odor[odor_on],1)/window
plt.figure()
plt.plot(prev_odor, proj_value,'b.')
plt.xlabel('experienced odor')
plt.ylabel('proejction')
# plt.ylim([-6,6])

# %% projections conditioned on past odor!
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def last_argmin(arr):
    # Find the minimum value
    min_val = np.min(arr)
    
    # Find the last index of this minimum value
    return len(arr) - np.flip(arr).argmin() - 1
proj_val_off = []
prev_odor_off = []
for tt in range(len(odor_off)):
    pos = odor_off[tt]
    vectorized_id = feat_ids[:pos,:].reshape(-1)
    vectorized_time = feat_time[:pos,:].reshape(-1)
    close_time_pos1 = last_argmin(np.abs(vectorized_time - 45+0))
    close_time_pos2 = last_argmin(np.abs(vectorized_time - 45+30))
    if (vectorized_id[close_time_pos1] == vectorized_id[-1]) or (vectorized_id[close_time_pos2] == vectorized_id[-1]):  # of the same track
        vectorized_signal = feat_odor[:pos,:].reshape(-1)  ### might want to figure out when it ends
        proj_val_off.append(np.dot(data_2d[pos,:]-center, lda_direction.T))
        prev_odor_off.append(np.mean(vectorized_signal[close_time_pos1:pos*window])/50)# / len(np.where(vectorized_signal[close_time_pos1:pos*window]>0)[0]) )
        # print(pos)

# %%
### fitting
model = LinearRegression()
x_,y_ = np.array(prev_odor_off), np.array(proj_val_off).squeeze()
x_[np.isnan(x_)] = 0
pos = x_ != 0
x_, y_ = x_[pos], y_[pos]
x_ = x_.reshape(-1, 1)
model.fit(x_, y_)
y_pred = model.predict(x_)
r2 = r2_score(y_, y_pred)
## %%
plt.figure()
plt.plot(x_, y_, '.')
plt.plot(x_, y_pred, color='k', label="Fitted")
plt.xlabel('odor experience (normalized)', fontsize=20)
plt.ylabel('projected off response', fontsize=20)
# plt.ylim([-6,6])

# %% relaxation upon odor-off
proj_val_offt = []
time_since_off = []
back2track_id = []
for tt in range(feat_time.shape[0]):
    timei = feat_time[tt,0]
    if (timei >= 1) & (timei<45+30+90):
        projtt = np.dot(data_2d[tt,:]-center, lda_direction.T)
        time_since_off.append(timei - (45+30))
        proj_val_offt.append(projtt)
        back2track_id.append(tt)
        
plt.figure()
plt.plot(time_since_off, proj_val_offt, 'k.',alpha=.5)
plt.axvline(x=-30, color='r', linestyle='--'); plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('since odor off (s)'); plt.ylabel('projected action')

# %% visualization
import seaborn as sns

# Sample scatter data
x = np.array(time_since_off)
y = np.array(proj_val_offt)[:,0]

# pos = np.where(np.abs(y)>0.01)[0]
# x,y = x[pos], y[pos]
down_samp = 1
x,y = x[::down_samp], y[::down_samp]

plt.figure(figsize=(8, 6))
sns.kdeplot(x=x, y=y, cmap="viridis", fill=True, thresh=0, bw_method='silverman')
plt.scatter(x, y, s=10, color="black", alpha=0.1)  # Optional: overlay scatter points
plt.colorbar(label="Density")
plt.xlabel('since odor off (s)'); plt.ylabel('projected action')
plt.axvline(x=-30, color='r', linestyle='--'); plt.axvline(x=0, color='r', linestyle='--')
plt.title("Smoothed Density Plot")
plt.xlim([x.min(), x.max()]); plt.ylim([y.min(), y.max()])

# %% sample in time for typical tracks
n_bins = 150
hist, bin_edges = np.histogram(time_since_off, bins=n_bins)
bin_indices = np.digitize(time_since_off, bins=bin_edges) - 1  # "-1" to match 0-based indexing
elements_in_bins = [np.where(bin_indices == i)[0] for i in range(n_bins)]

v_samps = []
construct_vxy = []
timing = []
for nn in range(int(len(elements_in_bins)*1/1)):  #3/4
    bini = elements_in_bins[nn]
    this_bin = np.random.choice(bini,1)[0]
    v_samps.append(proj_val_offt[this_bin])
    idi = back2track_id[this_bin]
    
    vxy = feature_vxy[idi,:]
    construct_vxy.append(np.vstack((vxy[:window], vxy[window:])).T/60)
    if (bin_edges[nn]>-30) & (bin_edges[nn]<0):
        timing.append(np.ones(window))
    elif bin_edges[nn]>0:
        timing.append(np.ones(window)+1)
    else:
        timing.append(np.ones(window)*0)
        
construct_vxy = np.concatenate(construct_vxy)
construct_xy = np.cumsum(construct_vxy, axis=0)
timing = np.concatenate(timing)
    
plt.figure()
plt.plot(v_samps)
plt.figure()
plt.plot(construct_xy[:,0], construct_xy[:,1],'k')
on_window = np.where(timing==1)[0]
plt.plot(construct_xy[on_window,0], construct_xy[on_window,1],'b')
off_window = np.where(timing==2)[0]
plt.plot(construct_xy[off_window,0], construct_xy[off_window,1],'r')

plt.figure()
plt.plot(construct_vxy[:,0], construct_vxy[:,1],'k.')
plt.plot(construct_vxy[on_window,0], construct_vxy[on_window,1],'b.',alpha=.2)
plt.plot(construct_vxy[off_window,0], construct_vxy[off_window,1],'r.',alpha=.1)
plt.xlabel(r'$v_x$'); plt.ylabel(r'$v_y$')

# %% by fine time points...
# # data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
# # rec_tracks = []  # record the full track x,y
# # rec_signal = []  # record opto signal
# # times = []   # record time in epoch

# prev_odor_off = []
# proj_val_off = []
# for ii in range(len(data4fit)):
#     time_i = times[ii]
#     signal_i = rec_signal[ii]
#     vxy_i = data4fit[ii]
#     pos_pre = np.where((time_i>45)&(time_i<45+30))[0]
#     pos_pos = np.where((time_i>45+30)&(time_i<45+30+20))[0]
#     if len(pos_pre)>1 and len(pos_pos)>window:
#         prev_odor_off.append(np.nanmean(signal_i[pos_pre])+np.zeros(3))#len(pos_pos)-window))
#         for tt in range(3):#len(pos_pos)-window):
#             latent = np.concatenate((vxy_i[pos_pos[tt]:pos_pos[tt]+window,0] , vxy_i[pos_pos[tt]:pos_pos[tt]+window,1]))
#             # z = reducer.fit_transform(latent[None,:])
#             z = reducer.transform(latent[None,:])
#             proj_val_off.append((z@lda_direction.T).reshape(-1))
#             print(tt)
            
# plt.figure()
# plt.plot(np.concatenate(prev_odor_off), np.concatenate(proj_val_off), 'r.')
# plt.xlabel('experienced odor before off')
# plt.ylabel('proejction of off response')
# plt.ylim([-6,4])

# %% idea: study state diversity
def compute_dispersion(X_new, centroids):
    # Assign each new data point to the nearest centroid
    distances = np.linalg.norm(X_new[:, np.newaxis] - centroids, axis=2)
    closest_centroids = np.argmin(distances, axis=1)
    
    # Calculate the dispersion (MSE) for each assigned cluster
    dispersion = []
    for i in range(len(centroids)):
        # Get points assigned to the i-th centroid
        cluster_points = X_new[closest_centroids == i]
        
        if cluster_points.size > 0:
            # Calculate the distances from the points to the centroid
            centroid = centroids[i]
            # Calculate squared distances
            squared_distances = np.linalg.norm(cluster_points - centroid, axis=1) ** 2
            mse = np.mean(squared_distances)  # Mean Squared Error
            dispersion.append(mse)
        else:
            dispersion.append(0)  # No points in this cluster

    # Total dispersion
    total_dispersion = np.sum(dispersion)
    
    return dispersion, total_dispersion

from sklearn.metrics import silhouette_score
time_window = np.arange(5,45+30+60, 4)
reps = 10
diversity = np.zeros((reps, len(time_window)))
for rr in range(reps):
    for tt in range(len(time_window)-1):
        pos = np.where((feat_time[:,0]>time_window[tt]) & (feat_time[:,2]<time_window[tt+1]))[0]
        sub_feat = feature_vxy[pos,:]
        # agg_clustering_test = AgglomerativeClustering(n_clusters=5)
        # agg_clustering_test.fit(sub_feat)
        
        kmeans = KMeans(n_clusters=5)
        labels = kmeans.fit_predict(sub_feat)
        centroids = kmeans.cluster_centers_
        
        dispersion, total_dispersion = compute_dispersion(sub_feat, centroids)
        
        # closest_centroids = kmeans.predict(sub_feat)
        # dispersion = silhouette_score(sub_feat, closest_centroids)
        diversity[rr,tt] = total_dispersion 
    print(rr)
# %%
plt.figure()
plt.plot(time_window[:-1], diversity[:,:-1].T, 'k-o',alpha=0.5)
plt.axvline(x=45, color='r', linestyle='--'); plt.axvline(x=45+30, color='r', linestyle='--')
plt.xlabel('time in trial (s)'); plt.ylabel('dispersion from centroids')
