# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:59:50 2024

@author: ksc75
"""

import ssm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import numpy.ma as ma

from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% Max-predict protocol

### embedding

### clustering

### transition matrix

### entropy

### other matrix calculations
    # steady-state
    # reversibility
    # spectrum
    # reconstruction
    # sampling

### meta-stable state analysis...

### validations
    # autocorrelation
    # transition rates
    
# %% simulate HMM
num_states = 5 # K states
obs_dim = 2  # D dimsional observation
test_hmm = ssm.HMM(num_states, obs_dim, M=0, observations="gaussian",  transitions="sticky")
test_hmm.observations.mus *= 10
true_states, observed_vxy = test_hmm.sample(100000)

# %% delay embedding
###############################################################################
# %% functions
def build_X(data, K):
    features = []
    T = len(data)
    samp_vec = data[:-np.mod(T,K)-1,:]
    X = np.zeros((T-K, K))
    for tt in range(len(samp_vec)-K):
        vx = samp_vec[tt:tt+K, 0]
        vy = samp_vec[tt:tt+K, 1]
        vx_windowed = vx.reshape(-1, K)
        vy_windowed = vy.reshape(-1, K)
        # print(tt)
        features.append(np.hstack((vx_windowed, vy_windowed)))
    
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

def get_reversible_transition_matrix(P):
    probs = get_steady_state(P)
    P_hat = diags(1/probs)*P.transpose()*diags(probs)
    R=(P+P_hat)/2
    return R

def sorted_spectrum(R,k=5,which='LR'):
    eigvals,eigvecs = eigs(R,k=k,which=which)
    sorted_indices = np.argsort(eigvals.real)[::-1]
    return eigvals[sorted_indices],eigvecs[:,sorted_indices]

# %%
tau = 1
Ks = np.array([2, 4, 6, 8, 10, 12])
Ns = np.array([10, 100, 250, 500, 1000, 2000])
nats = np.zeros((len(Ks), len(Ns)))

for kk in range(len(Ks)):
    for nn in range(len(Ns)):
        ### build delay embedding
        Xi = build_X(observed_vxy, K=Ks[kk])
        ### cluster
        time_series = kmeans_knn_partition(Xi, Ns[nn])  ### mask the transition ones too!
        ### build matrix
        Pij = compute_transition_matrix(time_series, Ns[nn])
        ### compute entropy
        nati = trans_entropy(Pij)
        
        nats[kk,nn] = nati / (tau/60)  ### nats per second for transitions
        print(kk,nn)

# %%
plt.figure()
colors_K = plt.cm.viridis(np.linspace(0,1,len(Ks)))
for k,K in enumerate(Ks):
    temp = K/1
    plt.plot(Ns, nats[k]/1,c=colors_K[k],marker='o',label=f'K={temp} s')
# plt.plot(Ns, nats.T)
plt.xlabel('N')
plt.ylabel('nats/s')
plt.legend()

# %% fix param now
N = 1000  # number of states
K = 10  # delay window
X_traj = build_X(observed_vxy, K)
labels, centrals = kmeans_knn_partition(X_traj, N, return_centers=True)
P = compute_transition_matrix(labels, N)

# %%
R = get_reversible_transition_matrix(P)
eigvals,eigvecs = sorted_spectrum(R,k=7)  # choose the top k modes
phi2=eigvecs[labels,1].real
u,s,v = np.linalg.svd(X_traj,full_matrices=False)

plt.figure(figsize=(10,7))
color_abs = np.max(np.abs(phi2))
plt.scatter(u[:,0],u[:,1],c=phi2,cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.show()

# %% plotting, compared to ground truth!
import umap

sub_samp = np.random.choice(X_traj.shape[0], 20000, replace=False)
reducer = umap.UMAP(n_components=3, random_state=42)
data_2d = reducer.fit_transform(X_traj[sub_samp,:])

# %%
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
color_abs = np.max(np.abs(phi2[sub_samp]))
# sc = plt.scatter(data_2d[:,0], data_2d[:,1], c=phi2[sub_samp], cmap='coolwarm', s=.1, vmin=-color_abs, vmax=color_abs)
sc = ax.scatter(data_2d[:,0], data_2d[:,1],data_2d[:,2], c=phi2[sub_samp], cmap='coolwarm', s=.1, vmin=-color_abs, vmax=color_abs)
plt.colorbar(sc)

# %% spectral analysis
# P_shuff = compute_transition_matrix(np.random.permutation(labels), N)
uu,vv = np.linalg.eig(P)  #P_shuff
idx = uu.argsort()[::-1]  # Get indices to sort eigenvalues
sorted_eigenvalues = uu[idx]
plt.plot((-1/1*tau)/np.log(sorted_eigenvalues[1:30]),'-o')
plt.ylabel('relaxation time (s)')
plt.xlabel('eigenvalue index')

# %% color code tracks]
imode = 3
phi2 = eigvecs[labels,imode].real
window_show = np.arange(50000,52000)
plt.figure()
plt.scatter(observed_vxy[window_show, 0],observed_vxy[window_show, 1],c=phi2[window_show],cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}')

plt.figure()
plt.plot(window_show, observed_vxy[window_show, 0],'k--',alpha=.4)
plt.scatter(window_show, observed_vxy[window_show, 0],c=phi2[window_show],cmap='coolwarm',s=1)#,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}')

# %% detecting transitions....

