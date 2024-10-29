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
from scipy.sparse import diags,identity,coo_matrix,csr_matrix

from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from scipy.sparse import csr_matrix
from deeptime.markov.tools.estimation import largest_connected_set, transition_matrix, count_matrix
from scipy.signal import find_peaks
import scipy.stats as stats

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
test_hmm = ssm.HMM(num_states, obs_dim, M=0, observations="gaussian",  transitions="standard")
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

def build_signal(data, K):
    features = []
    T = len(data)
    samp_vec = data[:-np.mod(T,K)-1]
    for tt in range(len(samp_vec)-K):
        vx = samp_vec[tt:tt+K]
        features.append(vx)
    return np.vstack(features)

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

def compute_transition_matrix(time_series, n_states, return_count=False):
    # Initialize the transition matrix (n x n)
    count_matrix = np.zeros((n_states, n_states))
    # Loop through the time series and count transitions
    for (i, j) in zip(time_series[:-1], time_series[1:]):
        count_matrix[i, j] += 1
    # Normalize the counts by dividing each row by its sum to get probabilities
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(count_matrix, row_sums, where=row_sums!=0)
    if return_count:
        return transition_matrix, count_matrix
    else:
        return transition_matrix
    
def compute_transition_matrix(time_series, n_states, return_count=False):
    # Initialize the transition matrix (n x n)
    count_matrix = np.zeros((n_states, n_states))
    
    # Check if input is a masked array, and handle accordingly
    if np.ma.is_masked(time_series):
        time_series = np.ma.compressed(time_series)  # Remove masked values
    
    # Loop through the time series and count transitions
    for (i, j) in zip(time_series[:-1], time_series[1:]):
        if i is not np.ma.masked and j is not np.ma.masked:  # Skip masked entries
            count_matrix[i, j] += 1
    
    # Normalize the counts by dividing each row by its sum to get probabilities
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(count_matrix, row_sums, where=row_sums != 0)
    
    if return_count:
        return transition_matrix, count_matrix
    else:
        return transition_matrix
    
def get_count_matrix(labels,lag,nstates):
    observable_seqs = ma.compress_rows(ma.vstack([labels[:-lag],labels[lag:]]).T)

    row = observable_seqs[:,0]
    col = observable_seqs[:,1]

    data = np.ones(row.size)
    C = coo_matrix((data, (row, col)), shape=(nstates, nstates))
    # export to output format
    count_matrix = C.tocsr()
    
    return count_matrix
def transition_matrix_lcs(labels,lag,return_connected=False):
    nstates = np.max(labels)+1
    count_matrix = get_count_matrix(labels,lag,nstates)
    # connected_count_matrix = msm_estimation.connected_cmatrix(count_matrix)
    P = transition_matrix(count_matrix)
    if return_connected:
        lcs = largest_connected_set(count_matrix)
        return lcs,P
    else:
        return P

def get_steady_state(transition_matrix):
    epsilon = 1e-6
    perturbation = epsilon * np.ones(transition_matrix.shape) / transition_matrix.shape[0]
    matrix = (1 - epsilon) * transition_matrix + perturbation

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
    probs = get_steady_state(P) + 1e-10
    probs /= np.sum(probs)  # temporary fix
    P_hat = diags(1/probs)*P.transpose()*diags(probs)
    R=(P+P_hat)/2
    return R

def sorted_spectrum(R,k=5,which='LR'):
    eigvals,eigvecs = eigs(R,k=k,which=which)
    sorted_indices = np.argsort(eigvals.real)[::-1]
    return eigvals[sorted_indices],eigvecs[:,sorted_indices]

# %%
tau = 1
# Ks = np.array([2, 4, 6, 8, 10, 12])
# Ns = np.array([10, 100, 250, 500, 1000, 2000])
# nats = np.zeros((len(Ks), len(Ns)))

# for kk in range(len(Ks)):
#     for nn in range(len(Ns)):
#         ### build delay embedding
#         Xi = build_X(observed_vxy, K=Ks[kk])
#         ### cluster
#         time_series = kmeans_knn_partition(Xi, Ns[nn])  ### mask the transition ones too!
#         ### build matrix
#         Pij = compute_transition_matrix(time_series, Ns[nn])
#         ### compute entropy
#         nati = trans_entropy(Pij)
        
#         nats[kk,nn] = nati / (tau/60)  ### nats per second for transitions
#         print(kk,nn)

# %%
# plt.figure()
# colors_K = plt.cm.viridis(np.linspace(0,1,len(Ks)))
# for k,K in enumerate(Ks):
#     temp = K/1
#     plt.plot(Ns, nats[k]/1,c=colors_K[k],marker='o',label=f'K={temp} s')
# # plt.plot(Ns, nats.T)
# plt.xlabel('N')
# plt.ylabel('nats/s')
# plt.legend()

# %% fix param now
N = 1000  # number of states
K = 10  # delay window
X_traj = build_X(observed_vxy, K)
labels, centrals = kmeans_knn_partition(X_traj, N, return_centers=True)
P = compute_transition_matrix(labels, N)
true_state_embed = build_signal(true_states, K)
true_state_embed = true_state_embed[:,0]

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
plt.figure()
plt.plot((-1/1*tau)/np.log(sorted_eigenvalues[1:30]),'-o')
plt.ylabel('relaxation time (s)')
plt.xlabel('eigenvalue index')

# %% color code tracks]
imode = 1
phi2 = eigvecs[labels,imode].real
window_show = np.arange(40000,50000)
plt.figure()
plt.scatter(observed_vxy[window_show, 0],observed_vxy[window_show, 1],c=phi2[window_show],cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}')

plt.figure()
plt.plot(window_show, observed_vxy[window_show, 0],'k--',alpha=.4)
plt.scatter(window_show, observed_vxy[window_show, 0],c=phi2[window_show],cmap='coolwarm',s=1)#,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}')

# %%
###############################################################################
# %% detecting transitions.... test with functions...
def optimal_partition(phi2,inv_measure,P,return_rho = True):
    """
    input eigenfunction phi2, inv_measure is pi, P is transition, and return_pho is logic
    """
    #make sure P is a sparse matrix!
    X = phi2
    c_range = np.sort(phi2)[1:-1]
    rho_c = np.zeros(len(c_range))
    rho_sets = np.zeros((len(c_range),2))
    for kc,c in enumerate(c_range):
        labels = np.zeros(len(X),dtype=int)
        labels[X<=c] = 1
        rho_sets[kc] = [(inv_measure[labels==idx]*(P[labels==idx,:][:,labels==idx])).sum()/inv_measure[labels==idx].sum()
                      for idx in range(2)]  ### eqn.5  choherent sets chi = pi*P/pi  # porbablity of staying in set
    rho_c = np.min(rho_sets,axis=1)  ### eqn.6 across S+,S-
    peaks, heights = find_peaks(rho_c, height=0.5)  ### maximize this measure of choerence
    if len(peaks)==0:
        print('No prominent coherent set')
        return None
    else:
        idx = peaks[np.argmax(heights['peak_heights'])]

        c_opt = c_range[idx]
        kmeans_labels = np.zeros(len(X),dtype=int)
        kmeans_labels[X<c_opt] = 1

        if return_rho:
            return c_range,rho_sets,idx,kmeans_labels
        else:
            return kmeans_labels

def subdivide_state_optimal(state_to_split,phi2,kmeans_labels,inv_measure,P,indices):
    # print(kmeans_labels.shape)
    # print(indices.shape)
    # print((optimal_partition(phi2,inv_measure,P,return_rho=False)+np.max(kmeans_labels)+10).shape)
    # print(P.shape)
    kmeans_labels[indices] = (optimal_partition(phi2,inv_measure,P,return_rho=False)+np.max(kmeans_labels)+10)[indices]
    #check if there were no labels left behind because they were not connected components
    mask = np.ones(kmeans_labels.shape[0],dtype=bool)
    mask[indices] = False
    if np.any(kmeans_labels[mask]==state_to_split):
        idx = np.arange(len(kmeans_labels))[mask][np.where(kmeans_labels[mask]==state_to_split)[0]]
        kmeans_labels[idx] = np.nan

    final_kmeans_labels = np.zeros(kmeans_labels.shape)
    sel = ~np.isnan(kmeans_labels)
    final_kmeans_labels[np.isnan(kmeans_labels)] = np.nan
    for new_idx,label in enumerate(np.sort(np.unique(kmeans_labels[sel]))):
        final_kmeans_labels[kmeans_labels==label]=new_idx
    return final_kmeans_labels
    
def recursive_partitioning_optimal(final_labels,delay,phi2,inv_measure,P,n_final_states,save=False):
    c_range,rho_sets,c_opt,kmeans_labels =  optimal_partition(phi2,inv_measure,P,return_rho=True)

    labels_tree=np.zeros((n_final_states,len(kmeans_labels)))
    labels_tree[0,:] = kmeans_labels
    k=1
    measures_iter = []
    for k in range(1,n_final_states):  # iterate meta-states
        print('k=',k)
        eigfunctions_states=[]
        indices_states=[]
        im_states=[]
        P_states=[]
        for state in np.unique(kmeans_labels):   # iterate unique states
            cluster_traj = ma.zeros(final_labels.shape,dtype=int)
            cluster_traj[~final_labels.mask] = np.array(kmeans_labels)[final_labels[~final_labels.mask]]
            cluster_traj[final_labels.mask] = ma.masked
            labels_here = ma.zeros(final_labels.shape,dtype=int)
            sel = cluster_traj==state
            labels_here[sel] = final_labels[sel]
            labels_here[~sel] = ma.masked
            
            ### custom function for P and lcs
            nstates = N*1 #np.max(labels_here)+1
            P, count_matrix = compute_transition_matrix(labels_here, nstates, return_count=True) #(labels_here,delay,return_connected=True)
            # P, count_matrix = transition_matrix_lcs(labels_here, 1, return_connected=True)
            lcs = largest_connected_set(count_matrix)
            
            R = get_reversible_transition_matrix(P)
            im = get_steady_state(P)
            eigvals,eigvecs = sorted_spectrum(R, k=2)
            indices = np.zeros(N, dtype=bool)#np.zeros(len(np.unique(final_labels.compressed())),dtype=bool)
            indices[lcs] = True
            print('lcs:', lcs.shape,np.unique(labels_here.compressed()).shape)

            eigfunctions_states.append((eigvecs.real/np.linalg.norm(eigvecs.real,axis=0))[:,1])
            indices_states.append(indices)
            # P = csr_matrix(P)
            P_states.append(P)
            im_states.append(im)

        measures = [(inv_measure[kmeans_labels==state]).sum() for state in np.unique(kmeans_labels)]
        measures_iter.append(measures)

        state_to_split = np.argmax(measures)
        print('splitting', state_to_split,measures)
        kmeans_labels = subdivide_state_optimal(state_to_split,eigfunctions_states[state_to_split],
                                               kmeans_labels,im_states[state_to_split],P_states[state_to_split],
                                               indices_states[state_to_split])
        labels_tree[k,:] = np.copy(kmeans_labels)
        k+=1
    sel = ~np.isnan(kmeans_labels)
    measures = [(inv_measure[sel][kmeans_labels[sel]==state]).sum() for state in np.unique(kmeans_labels[sel])]
    measures_iter.append(measures)
    return labels_tree,measures_iter

# %%
def get_connected_labels(labels,lcs):
    final_labels = ma.zeros(labels.shape,dtype=int)
    for key in np.argsort(lcs):
        final_labels[labels==lcs[key]]=key+1
    final_labels[final_labels==0] = ma.masked
    final_labels-=1
    return final_labels

P, count_matrix = compute_transition_matrix(labels, N, return_count=True)
lcs = largest_connected_set(count_matrix)
final_labels = get_connected_labels(labels,lcs)
delay = 1  # not used...
n_final_states = 4
imode = 1
inv_measure = get_steady_state(P)
phi2_vec = eigvecs[:, imode].real
labels_tree,measures = recursive_partitioning_optimal(final_labels,delay,phi2_vec,inv_measure,P,n_final_states)

# %% now reconstruct matrix!?
kmeans_labels = labels_tree[-1,:]
cluster_traj = ma.copy(final_labels)
cluster_traj[~final_labels.mask] = ma.masked_invalid(kmeans_labels)[final_labels[~final_labels.mask]]
cluster_traj[final_labels.mask] = ma.masked

compressed_traj=[]
cluster_traj_nan = ma.filled(cluster_traj,10)
for k in range(len(cluster_traj)-1):
    if cluster_traj_nan[k+1]!=cluster_traj_nan[k]:
        compressed_traj.append(cluster_traj_nan[k])
compressed_traj = ma.hstack(compressed_traj)
compressed_traj[compressed_traj==10] = ma.masked
Pc = compute_transition_matrix(compressed_traj, 6)
# Pc = op_calc.transition_matrix(compressed_traj,lag=1)
plt.figure(figsize=(7,7))
plt.imshow(Pc.T,vmax=.38)
plt.colorbar()

# %% alignment!
from scipy.optimize import linear_sum_assignment

def find_best_permutation(series1, series2):
    """
    Finds the permutation of indices that best matches series1 to series2.
    
    Parameters:
    series1 (np.ndarray): First integer time series of shape (T,)
    series2 (np.ndarray): Second integer time series of shape (T,)

    Returns:
    np.ndarray: Permutation of indices to reorder series1 to best match series2
    """
    # Ensure both series have the same length
    if len(series1) != len(series2):
        raise ValueError("Both time series must have the same length")
    
    # Create cost matrix based on absolute differences
    cost_matrix = np.abs(series1[:, None] - series2[None, :])
    
    # Apply Hungarian algorithm to minimize total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Return the optimal permutation that reorders series1 to match series2
    return col_ind

# Example usage:
wind = np.arange(10,1000)
series_state = true_state_embed[wind+5]
series_embed = cluster_traj_nan[wind]

best_permutation = find_best_permutation(series_embed, series_state)
series_embed = series_embed[best_permutation]

# %%
plt.figure()
plt.plot(series_state, label='true HMM')
plt.plot(series_embed,'--', label='clusters')
plt.legend()

plt.figure()
plt.plot(observed_vxy[wind+5, 0],'k--',alpha=.4)

# %%
from collections import Counter

wind = np.arange(10,5000)
series_state = true_state_embed[wind+5]
series_embed = cluster_traj_nan[wind]

best_permutation = find_best_permutation(series_embed, series_state)
series_embed = series_embed[best_permutation]

# Count occurrences of each (integer1, integer2) pair
pair_counts = Counter(zip(series_state, series_embed))

# Separate the unique pairs and their counts
pairs, counts = zip(*pair_counts.items())
x, y = zip(*pairs)

# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(x, y, s=np.array(counts) * 10, c=counts, cmap='viridis', alpha=0.7)
plt.colorbar(label="Count")
plt.xlabel("true HMM")
plt.ylabel("embedded cluster")
plt.show()
