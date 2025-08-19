# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:37:42 2025

@author: ksc75
"""

import numpy as np
# for plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# for clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy.ma as ma
from sklearn.cluster import MiniBatchKMeans
import umap
from statsmodels.tsa.stattools import acf
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

# for data IO
from scipy.io import loadmat

# %% A stand-alone script to investigate behavioral states in bacteria
###############################################################################
# load data from .mat file
# plot tracks and compute basic stats
# build feature vectors for clustering
# if cluster works, build transition models
# sample from the model and see if it matches with data
###############################################################################

# %% load mat file for the data structure
file_dir = r'D:/github/behavior_state_space/data/PAK_1.rad_swimtracker.mat'
# file_dir = r'D:/github/behavior_state_space/data/delY_1.rad_swimtracker.mat'
data = loadmat(file_dir) # Open the .mat file
samp_rate = 1/data['tracks'][0]['dt'][0][0][0]  ### about 30 Hz

# %% extract variables
x_list, y_list, speed_list, ang_list = [], [], [], []
n_tracks = len(data['tracks'])

for ii in range(n_tracks):
    x_list.append(data['tracks'][ii]['x'][0].squeeze())
    y_list.append(data['tracks'][ii]['y'][0].squeeze())
    speed_list.append(data['tracks'][ii]['speed'][0].squeeze())
    ang_list.append(data['tracks'][ii]['angvelocity'][0].squeeze())

# %% simple visualization
ith_cell = 1
plt.figure()
plt.plot(x_list[ith_cell], y_list[ith_cell])
plt.xlabel('x'); plt.ylabel('y')

sample_n = 1000
rand_id = np.random.randint(0, n_tracks, sample_n)
samp_speed = [speed_list[ii] for ii in rand_id]
samp_ang = [ang_list[ii] for ii in rand_id]
plt.figure()
plt.plot(np.concatenate(samp_speed), np.concatenate(samp_ang), 'k.', alpha=0.1)
plt.xlabel('speed'); plt.ylabel('angle velocity')

# %% compute acf
nlags = 100
population_acf_sp = []
population_acf_dth = []

for ii in rand_id:
    speed = speed_list[ii]  
    ac_sp = acf(speed, nlags=nlags, fft=True, missing='conservative')
    ac_dth = acf(ang_list[ii], nlags=nlags, fft=True, missing='conservative')
    if len(ac_sp) > nlags:
        population_acf_sp.append(ac_sp[:nlags])
        population_acf_dth.append(ac_dth[:nlags])

population_acf_sp = np.array(population_acf_sp)
population_acf_dth = np.array(population_acf_dth)

plt.figure()
plt.plot(np.mean(population_acf_sp, axis=0), label='speed')
plt.plot(np.mean(population_acf_dth, axis=0), label='angle')
plt.xlabel('Lag')
plt.ylabel('Mean ACF')
plt.title('population acf')
plt.legend(); plt.grid()

# %% simple clustering of behavioral features
###############################################################################
# %% functional
def build_X(behavior, K, return_id=False):
    """
    with behavior list, currently list of speed and angle
    we output the Hankel matrix, which is the delay embedded features
    one can also output the track id to analyze individual tracks later
    """
    K = int(K)
    sp, dth = behavior
    features = []
    ids = []
    n_tracks = len(sp)
    for tr in range(n_tracks):
        sp_i, dth_i = sp[tr], dth[tr]
        T = len(sp_i)
        if T>K+1:
            for tt in range(T-K):
                vx = sp_i[tt:tt+K]
                vy = dth_i[tt:tt+K]
                vx_windowed = vx.reshape(-1, K)
                vy_windowed = vy.reshape(-1, K)
                features.append(np.hstack((vx_windowed, vy_windowed)))   ### might have problems here across tracks!!!
                ids.append(tr)
    if return_id:
        feats = np.concatenate(features) 
        ids = np.array(ids)
        jumps = ids[1:]==ids[:-1]
        pos = np.where(jumps==False)[0]
        mask = feats*0
        mask[pos, :] = True
        masked_data = np.ma.masked_array(feats, mask)
        return masked_data , ids
    else:
        return np.concatenate(features)
    
def discretize(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    return cluster_labels

def kmeans_knn_partition(tseries,n_seeds,batchsize=None,return_centers=False):
    if batchsize==None:
        batchsize = n_seeds*5
    if ma.count_masked(tseries)>0:
        labels = ma.zeros(tseries.shape[0],dtype=int)
        labels.mask = np.any(tseries.mask,axis=1)
        kmeans = MiniBatchKMeans(batch_size=batchsize,n_clusters=n_seeds).fit(ma.compress_rows(tseries))
        labels[~np.any(tseries.mask,axis=1)] = kmeans.labels_
    else:
        kmeans = MiniBatchKMeans(batch_size=batchsize,n_clusters=n_seeds).fit(tseries)
        labels=kmeans.labels_
    if return_centers:
        return labels,kmeans.cluster_centers_
    return labels

# %% quick test with clusters
data4cluster = (speed_list, ang_list)
K = 30
X,ids = build_X(data4cluster, K, return_id=True)

n_states = 4
test_label = kmeans_knn_partition(X, n_states)
col_list = ['r', 'g', 'b', 'k', 'y']
# col_list = plt.cm.tab10.colors[:n_states]  # returns 5 RGB tuples

# %% build same dimension in real space
rec_tracks = (x_list, y_list)
X_xy, track_id = build_X(rec_tracks, K, return_id=True)

# %% plot state in space
window = np.arange(0,100000)
x_samp, y_samp = X_xy[window,0], X_xy[window, K]
state_samp = test_label[window]
plt.figure()
for ss in range(n_states):
    pos = np.where(state_samp==ss)[0]
    plt.plot(x_samp[pos], y_samp[pos], '.', markersize=1)
    
# %%
sample_n = 500
zero_center = 1
rand_id = np.random.randint(0, n_tracks, sample_n)

plt.figure()
for ii in range(sample_n):
    ith = rand_id[ii]
    pos_track = np.where(ids==ith)[0]
    x_samp, y_samp = X_xy[pos_track,0], X_xy[pos_track, K]
    state_samp = test_label[pos_track]
    for ss in range(n_states):
        pos_state = np.where(state_samp==ss)[0]
        if len(pos_state)>0:
            plt.plot(x_samp[pos_state] - zero_center*x_samp[0], y_samp[pos_state] - zero_center*y_samp[0], '.', markersize=1, color=col_list[ss])
plt.xlabel('x'); plt.ylabel('y'); plt.title('zero-centered example tracks')

# %% plot state in phase space
plt.figure()
for ii in range(sample_n):
    ith = rand_id[ii]
    pos_track = np.where(ids==ith)[0]
    sp_samp, dth_samp = X[pos_track,0], X[pos_track, K]
    state_samp = test_label[pos_track]
    for ss in range(n_states):
        pos_state = np.where(state_samp==ss)[0]
        if len(pos_state)>0:
            plt.plot(sp_samp[pos_state], dth_samp[pos_state], '.', markersize=3, alpha=0.5, color=col_list[ss])
plt.xlabel('speed'); plt.ylabel('angle velocity')

# %% fancier stuff for next steps
###############################################################################
# %% visualization with U-map
X_unmasked = X[~X.mask.any(axis=1)]
X_filled = X_unmasked.data  # convert masked array to plain array
# Subsample
sub_samp = np.random.choice(X_filled.shape[0], 10000, replace=False)
data_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(X_filled[sub_samp])

# %% the 3D plot
phi2 = X[:,1] #test_label*1 ### reploace with mode later
fig=plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
color_abs = np.max(np.abs(phi2[sub_samp]))
# sc = plt.scatter(data_2d[:,0], data_2d[:,1], c=phi2[sub_samp], cmap='coolwarm', s=.1, vmin=-color_abs, vmax=color_abs)
sc = ax.scatter(data_3d[:,0], data_3d[:,1],data_3d[:,2], c=phi2[sub_samp], cmap='coolwarm', s=.5, vmin=0, vmax=color_abs)
plt.colorbar(sc)

# %% NEXT
# next step is to build the transition matrix and do mode decompoisiton

# %% functional for Markov model
def compute_transition_matrix(time_series, track_id, n_states):
    """
    modified from previous function to handle track id and not compute those transitions
    """
    # Initialize the transition matrix (n x n)
    transition_matrix = np.zeros((n_states, n_states))
    # find valid transition that is not across tracks
    valid_transitions = track_id[:-1] == track_id[1:]
    # Get the current and next state only for valid transitions
    current_states = time_series[:-1][valid_transitions]
    next_states = time_series[1:][valid_transitions]
    # Use np.add.at to efficiently accumulate transitions
    np.add.at(transition_matrix, (current_states, next_states), 1)
    
    # Normalize the counts by dividing each row by its sum to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
    return transition_matrix 

def get_steady_state(transition_matrix):
    # Find the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    # Find the index of the eigenvalue that is approximately 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    # The corresponding eigenvector (steady-state)
    steady_state = np.real(eigenvectors[:, idx])
    # Normalize the eigenvector so that its elements sum to 1
    steady_state = steady_state / np.sum(steady_state)
    return steady_state

def trans_entropy(M):
    pi = get_steady_state(M)
    h = 0
    for ii in range(M.shape[0]):
        for jj in range(M.shape[0]):
            h += pi[ii]*M[ii,jj]*np.log(M[ii,jj] + 1e-10)
    return -h

def get_reversible_transition_matrix(P):
    probs = get_steady_state(P)
    P_hat = diags(1/probs)*P.transpose()*diags(probs)
    R=(P+P_hat)/2
    return R

def sorted_spectrum(R,k=5,which='LR'):
    eigvals,eigvecs = eigs(R,k=k,which=which)
    sorted_indices = np.argsort(eigvals.real)[::-1]
    return eigvals[sorted_indices],eigvecs[:,sorted_indices]

# %% test
Pij = compute_transition_matrix(test_label, ids, n_states)
h_est = trans_entropy(Pij)

# %% scanning
# tau = 1
# Ks = np.array([2, 4, 8, 16, 32, 64])
# Ns = np.array([10, 100, 250, 500, 1000, 2000])
# nats = np.zeros((len(Ks), len(Ns)))

# for kk in range(len(Ks)):
#     for nn in range(len(Ns)):
#         ### build delay embedding
#         Xi,idsi = build_X(data4cluster, Ks[kk], return_id=True)
#         Xi,idsi = Xi[::tau, :], idsi[::tau]
#         ### cluster
#         time_series = kmeans_knn_partition(Xi, Ns[nn])  ### mask the transition ones too!
#         ### build matrix
#         Pij = compute_transition_matrix(time_series, idsi, Ns[nn])
#         ### compute entropy
#         nati = trans_entropy(Pij)
        
#         nats[kk,nn] = nati / (tau/samp_rate)  ### nats per second for transitions
#         print(kk,nn)

# # %%
# plt.figure()
# colors_K = plt.cm.viridis(np.linspace(0,1,len(Ks)))
# for k,K in enumerate(Ks):
#     temp = round(K/samp_rate,3)
#     plt.plot(Ns, nats[k]/1,c=colors_K[k],marker='o',label=f'K={temp} s')
# plt.xlabel('N')
# plt.ylabel('nats/s')
# plt.legend()

# %% build delay embedded Markov model
K_star = 20
N_star = 1000
X_traj, track_id = build_X(data4cluster, K_star, return_id=True)
labels, centrals = kmeans_knn_partition(X_traj, N_star, return_centers=True)
P = compute_transition_matrix(labels, track_id, N_star)

# %% analyze spectrum
R = get_reversible_transition_matrix(P)
eigvals, eigvecs = sorted_spectrum(R, k=10)  # choose the top k modes
phi2 = eigvecs[labels,1].real
X_unmasked = X_traj[~X_traj.mask.any(axis=1)]
phi2 = phi2[~X_traj.mask.any(axis=1)]
X_filled = X_unmasked.data 
u,s,v = np.linalg.svd(X_filled, full_matrices=False)

plt.figure(figsize=(10,7))
color_abs = np.max(np.abs(phi2))
plt.scatter(u[:,0],u[:,1],c=phi2,cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.show()

# %% try U-MAP
sub_samp = np.random.choice(X_filled.shape[0], 20000, replace=False)
reducer = umap.UMAP(n_components=3, random_state=42)
data_2d = reducer.fit_transform(X_filled[sub_samp,:])

# %%
fig=plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
color_abs = np.max(np.abs(phi2[sub_samp]))
sc = ax.scatter(data_2d[:,0], data_2d[:,1],data_2d[:,2], c=phi2[sub_samp], cmap='coolwarm', s=.1, vmin=-color_abs, vmax=color_abs)
plt.colorbar(sc)

# %% spectral analysis
# P_shuff = compute_transition_matrix(np.random.permutation(labels),track_id, N_star)
# uu,vv = np.linalg.eig(P_shuff)  #P_shuff
uu,vv = np.linalg.eig(P)
idx = uu.argsort()[::-1]  # Get indices to sort eigenvalues
sorted_eigenvalues = np.real(uu[idx])
plt.figure()
plt.plot((-1/samp_rate)/np.log(sorted_eigenvalues[1:20]),'-o')
plt.ylabel('relaxation time (s)')
plt.xlabel('eigenvalue index')
# plt.yscale('log')
# plt.ylim([0.001, 20])

# %% color code tracks]
imode = 4
phi2 = eigvecs[labels,imode].real
window_show = np.arange(1,200000)
X_xy, track_id = build_X(rec_tracks, K_star, return_id=True)
xy_back = X_xy[:, [0,K_star]]
plt.figure()
plt.scatter(xy_back[window_show, 0],xy_back[window_show, 1],c=phi2[window_show],cmap='coolwarm',s=.5,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}')

