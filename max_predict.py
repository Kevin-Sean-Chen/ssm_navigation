# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:23:49 2024

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

import numpy.ma as ma
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

# %% test maximum Markov prediction method to identify tracks motifs
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

# pkl_files = pkl_files[8:]

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

# %% build features with delayed time series
window = int(60*2.)
# def build_X(data, K=window):
#     features = []
#     T = len(data)
#     samp_vec = data[:-np.mod(T,K),:]
#     X = np.zeros((T-K, K))
#     for tt in range(len(samp_vec)-K):
#         vx = samp_vec[tt:tt+K, 0]
#         vy = samp_vec[tt:tt+K, 1]
#         vx_windowed = vx.reshape(-1, K)
#         vy_windowed = vy.reshape(-1, K)
#         features.append(np.hstack((vx_windowed, vy_windowed)))
#     return np.concatenate(features)
# X = build_X(vec_vxy)

def build_X(data, K=window):
    K = int(K)
    features = []
    n_tracks = len(data)
    for tr in range(n_tracks):
        datai = data[tr]
        T = len(datai)
        samp_vec = datai[:-np.mod(T,K),:]
        X = np.zeros((T-K, K))
        for tt in range(len(samp_vec)-K):
            vx = samp_vec[tt:tt+K, 0]
            vy = samp_vec[tt:tt+K, 1]
            vx_windowed = vx.reshape(-1, K)
            vy_windowed = vy.reshape(-1, K)
            features.append(np.hstack((vx_windowed, vy_windowed)))
    return np.concatenate(features)

X = build_X(data4fit)

# %% clustering and assigning states
from sklearn.cluster import KMeans
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

n_states = 10
test_label = kmeans_knn_partition(X, n_states)

# %% compute transition and measure entropy
def compute_transition_matrix(time_series, n_states):
    # Initialize the transition matrix (n x n)
    transition_matrix = np.zeros((n_states, n_states))
    # Loop through the time series and count transitions
    for (i, j) in zip(time_series[:-1], time_series[1:]):
        transition_matrix[i, j] += 1
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

Pij = compute_transition_matrix(test_label, n_states)
h_est = trans_entropy(Pij)

# %%
###############################################################################
# %% scanning
Ks = np.array([.5, 1, 2, 4, 6, 8])*60
Ns = np.array([10, 100, 250, 500, 1000,1500,2000])
nats = np.zeros((len(Ks), len(Ns)))

for kk in range(len(Ks)):
    for nn in range(len(Ns)):
        ### build delay embedding
        Xi = build_X(data4fit,Ks[kk])
        ### cluster
        time_series = kmeans_knn_partition(Xi, Ns[nn])
        ### build matrix
        Pij = compute_transition_matrix(time_series, Ns[nn])
        ### compute entropy
        nati = trans_entropy(Pij)
        
        nats[kk,nn] = nati
        print(kk,nn)

# %%
plt.figure()
colors_K = plt.cm.viridis(np.linspace(0,1,len(Ks)))
for k,K in enumerate(Ks):
    temp = K/60
    plt.plot(Ns, nats[k],c=colors_K[k],marker='o',label=f'K={temp} s')
# plt.plot(Ns, nats.T)
plt.xlabel('N')
plt.ylabel('nats')
plt.legend()

###############################################################################
# %% check representation
# now with fixed N and K, study the transition structure!!
# visualization and transition rates
### spead up by using spase matrices!

# %%
def get_reversible_transition_matrix(P):
    probs = get_steady_state(P)
    P_hat = diags(1/probs)*P.transpose()*diags(probs)
    R=(P+P_hat)/2
    return R

def sorted_spectrum(R,k=5,which='LR'):
    eigvals,eigvecs = eigs(R,k=k,which=which)
    sorted_indices = np.argsort(eigvals.real)[::-1]
    return eigvals[sorted_indices],eigvecs[:,sorted_indices]

# %% fix param now
N = 1000
K = 5*60
X_traj = build_X(data4fit,K)
labels = kmeans_knn_partition(X_traj, N)
P = compute_transition_matrix(labels, N)

# %%
R = get_reversible_transition_matrix(P)
eigvals,eigvecs = sorted_spectrum(R,k=3)
phi2=eigvecs[labels,1].real
u,s,v = np.linalg.svd(X_traj,full_matrices=False)

plt.figure(figsize=(10,7))
color_abs = np.max(np.abs(phi2))
plt.scatter(u[:,0],u[:,1],c=phi2,cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.show()

# %% try U-MAP
import umap

sub_samp = np.random.choice(X_traj.shape[0], 20000, replace=False)
reducer = umap.UMAP(n_components=2, random_state=42)
data_2d = reducer.fit_transform(X_traj[sub_samp,:])

# %%
plt.figure(figsize=(10,7))
color_abs = np.max(np.abs(phi2[sub_samp]))
plt.scatter(data_2d[:,0], data_2d[:,1],c=phi2[sub_samp],cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.show()