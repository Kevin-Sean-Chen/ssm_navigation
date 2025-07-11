# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:00:54 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import schur, eigvals
from scipy.linalg import hankel
import scipy as sp
from sklearn.cluster import SpectralClustering

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import h5py

# %% load large matrix Pij and the corresponsding centroids
P = P*1
labels, centrals = labels*1, centrals*1
vx_smooth, vy_smooth, acf_data = vx_smooth, vy_smooth*1, acf_data*1

K_star = centrals.shape[1]//2
tau_star = 3
down_samp = 3
kmean_seed = 37 #42

# %% reduced model exploration... ############################################
from scipy.linalg import eig

def reduce_and_sample_markov(P, num_clusters=10, num_steps=1000):
    """
    Reduce a Markov transition matrix using spectral clustering and simulate a discrete time series.
    
    Parameters:
        P (numpy.ndarray): NxN Markov transition matrix.
        num_clusters (int): Number of reduced states (metastable clusters).
        num_steps (int): Length of the sampled time series.

    Returns:
        reduced_P (numpy.ndarray): Coarse-grained transition matrix.
        state_sequence (numpy.ndarray): Simulated discrete time series from the reduced model.
        cluster_labels (numpy.ndarray): Cluster assignments for original states.
        mapping_matrix (numpy.ndarray): Mapping from original to reduced states.
    """
    N = P.shape[0]  # Number of original states

    # Step 1: Compute eigenvalues & eigenvectors
    eigenvalues, eigenvectors = eig(P.T)  # Left eigenvectors give coherent structures
    # eigenvectors,eigenvalues,_ = schur(P.T)

    # Step 2: Sort eigenvalues by magnitude (slowest modes closest to 1)
    sorted_indices = np.argsort(-np.real(eigenvalues))
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 3: Cluster states using K-means on slowest eigenvectors (excluding stationary mode)
    num_modes = min(num_clusters, N - 1)
    # kmeans = KMeans(n_clusters=num_clusters, random_state=kmean_seed, n_init=10)
    # cluster_labels = kmeans.fit_predict(np.real(eigenvectors[:, 1:num_modes + 1]))
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels="kmeans", random_state=kmean_seed)
    cluster_labels = clustering.fit_predict(P)  # Directly cluster the transition matrix

    # Step 4: Construct reduced transition matrix
    reduced_P = np.zeros((num_clusters, num_clusters))
    for i in range(N):
        for j in range(N):
            reduced_P[cluster_labels[i], cluster_labels[j]] += P[i, j]

    # Normalize to keep row-stochastic property
    reduced_P /= reduced_P.sum(axis=1, keepdims=True)

    # Step 5: Sample from the reduced Markov model
    state_sequence = np.zeros(num_steps, dtype=int)
    state_sequence[0] = np.random.choice(num_clusters)  # Start from a random reduced state

    for t in range(1, num_steps):
        state_sequence[t] = np.random.choice(num_clusters, p=reduced_P[state_sequence[t-1]])

    # Step 6: Compute the mapping matrix
    mapping_matrix = np.zeros((num_clusters, N))  # Rows: reduced states, Cols: original states
    for i, cluster in enumerate(cluster_labels):
        mapping_matrix[cluster, i] = 1  # Mark assignment

    return reduced_P, state_sequence, cluster_labels, mapping_matrix

# Example: Generate a random Markov matrix (100 states)
# N = 100
# P = np.random.rand(N, N)
# P /= P.sum(axis=1, keepdims=True)  # Normalize rows to make it stochastic

# Reduce to 10 states and simulate 1000 time steps
num_clusters = 5
reduced_P, state_sequence, cluster_labels, mapping_matrix = reduce_and_sample_markov(P, num_clusters=num_clusters, num_steps=1000)

# Plot the sampled time series
plt.figure(figsize=(10, 4))
plt.plot(state_sequence, '.', markersize=3, alpha=0.7)
plt.xlabel("Time Step")
plt.ylabel("Reduced State")
plt.title("Simulated Time Series from Reduced Markov Model")
plt.show()

# Plot the state mapping
plt.figure(figsize=(8, 6))
plt.imshow(mapping_matrix, aspect='auto', cmap="viridis")
plt.colorbar(label="Cluster Assignment")
plt.xlabel("Original States (N)")
plt.ylabel("Reduced States (d)")
plt.title("Mapping of Original States to Reduced States")
plt.show()

# %% generative from reduced model
def gen_data_from_redused_states(reduced_P, lt):
    # Step 5: Sample from the reduced Markov model
    state_sequence = np.zeros(lt, dtype=int)
    num_clusters = reduced_P.shape[0]
    state_sequence[0] = np.random.choice(num_clusters)  # Start from a random reduced state

    for t in range(1, lt):
        state_sequence[t] = np.random.choice(num_clusters, p=reduced_P[state_sequence[t-1]])
        
    subsample_vxy = []
    for tt in range(lt):
        ### choice from reduced
        vec_map = np.where(mapping_matrix[state_sequence[tt],:]==1)[0]  ### find the mapping
        this_state = np.random.choice(vec_map)  ### randomly choose for now
        ### reconstruction
        vxy_i = np.vstack((np.random.choice(centrals[this_state,:int(K_star)],1), np.random.choice(centrals[this_state,int(K_star):],1))).T  # take sample
        subsample_vxy.append(vxy_i*(tau_star/90*down_samp))
    subsample_vxy = np.concatenate(subsample_vxy)
    ### tracks!
    subsample_xy = np.cumsum(subsample_vxy, axis=0)
    
    return subsample_vxy, subsample_xy
    
# %% analyze gen tracks
samp_vxy, samp_xy = gen_data_from_redused_states(reduced_P, 50000)

# %%
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

xy_id = 1
lags, acf_mark = compute_autocorrelation(samp_vxy[:, xy_id], 1000)
plt.figure()
plt.plot(np.arange(len(lags))*tau_star/90*1, acf_data, label='data')
plt.plot(np.arange(len(lags))*tau_star/90*1, acf_mark, label='reduced')
plt.legend(); plt.xlabel(r'$\tau$ (s)');  plt.ylabel(r'$<v(t),v(t+\tau)>$')

# %% distribution
bins = np.arange(-15,15,0.5)
plt.figure()
# count_data,_ = np.histogram(vx_smooth, bins)
count_data,_ = np.histogram(vy_smooth, bins)
count_simu,_ = np.histogram(samp_vxy[:, xy_id] /(tau_star/90*down_samp), bins)
plt.plot(bins[:-1], count_data/count_data.sum(), label='data')
plt.plot(bins[:-1], count_simu/count_simu.sum(), label='reduced Markov')
plt.xlabel(r'$v_x$'); plt.ylabel('count'); plt.legend(); plt.yscale('log')

# %% test with decompositions
###############################################################################
### compute pi
eigenvalues, eigenvectors = np.linalg.eig(P.T) #.T
idx = np.argmin(np.abs(eigenvalues - 1))
steady_state = np.real(eigenvectors[:, idx])
pi = steady_state / np.sum(steady_state)
### compute Pij
Pij = P*0
for ii in range(Pij.shape[0]):
    Pij[:,ii] = pi[ii]*P[:,ii]
### compute traffic
tau_ij = Pij + Pij.T
### compute edige flux
J_ij = Pij - Pij.T

# %% three terms
### stickyness
stick = 0
for ii in range(len(pi)):
    stick += pi[ii]*P[ii,ii]*np.log(P[ii,ii]+1e-10)
    
### traffic
traffic = 0
for ii in range(len(pi)):
    for jj in range(len(pi)):
        if jj>ii:
            traffic += 0.5*tau_ij[ii,jj]* np.log(P[ii,jj] * P[jj,ii] + 1e-10)
            
### irrevseribility
irr = 0
for ii in range(len(pi)):
    for jj in range(len(pi)):
        if jj>ii:
            irr += J_ij[ii,jj]* (np.log(P[jj,ii] + 1e-10) - np.log(P[ii,jj] + 1e-10))
# %%
plt.figure()
plt.bar(['sticky', 'traffic','irreversible'], [-stick, -traffic, -irr])
plt.ylabel('entropy')

# %%
###############################################################################
# %% two types of analysis for reduced Markov model
### how many clusters?
### what are the clusters?
# %%
n_reduced = np.array([2,3,4, 5,7,11,20])
errs = np.zeros((len(n_reduced), 4))  ### n-states by the metrics

for ns in range(len(n_reduced)):
    print('n-states=', n_reduced[ns])
    reduced_P, state_sequence, cluster_labels, mapping_matrix = reduce_and_sample_markov(P, num_clusters=n_reduced[ns], num_steps=1000)
    samp_vxy, samp_xy = gen_data_from_redused_states(reduced_P, 70000)
    ### x-stats
    xy_id = 0
    lags, acf_mark = compute_autocorrelation(samp_vxy[:, xy_id], 1000)
    count_simu,_ = np.histogram(samp_vxy[:, xy_id] /(tau_star/90*down_samp), bins)
    errs[ns, 0] = np.sum((acf_data - acf_mark)**2)**0.5
    errs[ns, 1] = np.sum((count_data - count_simu)**2)**0.5
    ### y-stats
    xy_id = 1
    lags, acf_mark = compute_autocorrelation(samp_vxy[:, xy_id], 1000)
    count_simu,_ = np.histogram(samp_vxy[:, xy_id] /(tau_star/90*down_samp), bins)
    errs[ns, 2] = np.sum((acf_data - acf_mark)**2)**0.5
    errs[ns, 3] = np.sum((count_data - count_simu)**2)**0.5
    
    #### add fraction of successs???? #########################################

# %%
plt.figure()
plt.plot(n_reduced, errs[:, 0], '-o', label='Vx')
plt.plot(n_reduced, errs[:, 2], '-o', label='Vy')
plt.yscale('log')
plt.xlabel('reduced states'); plt.ylabel('acf error'); plt.legend()

plt.figure()
plt.plot(n_reduced, errs[:, 1], '-o')
plt.plot(n_reduced, errs[:, 3], '-o')
plt.xlabel('reduced states'); plt.ylabel('density error')

# %% test with simulated performances
def sim_performance(red_P, sim_len=1500, reps=50,  x_range=(-300, -200), y_range=(-50, 50)):
    n_success = 0
    for rr in range(reps):
        _, sim_xy = gen_data_from_redused_states(reduced_P, sim_len)
        xi, yi = sim_xy[:,0], sim_xy[:,1]
        if np.any( (xi > x_range[0]) & (xi < x_range[1]) & (yi > y_range[0]) & (yi < y_range[1]) ):
            n_success+=1
    return n_success/reps

rep_sim = 5
hit_rate = np.zeros((len(n_reduced), rep_sim))  ### n-states by the metrics

for rr in range(rep_sim):
    print(rr)
    for ns in range(len(n_reduced)):
        print('n-states=', n_reduced[ns])
        reduced_P, state_sequence, cluster_labels, mapping_matrix = reduce_and_sample_markov(P, num_clusters=n_reduced[ns], num_steps=1000)
        hit_rate[ns,rr] = sim_performance(reduced_P)

# %%
plt.figure()
# plt.plot(n_reduced, hit_rate, '-o')
plt.errorbar(n_reduced, np.mean(hit_rate,1), yerr=np.std(hit_rate,1),color='k')
plt.xlabel('reduced states'); plt.ylabel('navigation hit rate')

# %% visualize states
for st in range(mapping_matrix.shape[0]):
    plt.figure()
    pos = np.where(mapping_matrix[st,:]==1)[0]
    vxyi = centrals[pos, :]
    plt.plot(vxyi[:,:120], vxyi[:,120:],'k.', alpha=0.1)
    plt.xlim([-30, 30]); plt.ylim([-30, 30])

# %% compare markovness through n-reduction
###############################################################################
# %%
def autocorrelation_discrete(time_series, max_lag=None):
    time_series = np.asarray(time_series)
    n = len(time_series)
    
    if max_lag is None:
        max_lag = n // 2  # default

    # One-hot encoding
    unique_states = np.unique(time_series)
    state_to_index = {state: idx for idx, state in enumerate(unique_states)}
    encoded = np.zeros((n, len(unique_states)))
    for t, state in enumerate(time_series):
        encoded[t, state_to_index[state]] = 1.0

    # Mean-center
    encoded = encoded - np.mean(encoded, axis=0)

    # Compute variance (normalizer)
    var = np.mean(np.sum(encoded**2, axis=1))

    # Autocorrelation
    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        prod = np.sum(encoded[lag:] * encoded[:n-lag], axis=1)
        autocorr[lag] = np.mean(prod) / var  # normalize by variance

    lags = np.arange(max_lag)
    return lags, autocorr

def map_time_series(mapping_matrix, time_series):
    mapping_matrix = np.asarray(mapping_matrix)
    time_series = np.asarray(time_series)

    # Find for each original state its mapped group
    state_map = np.argmax(mapping_matrix, axis=0)  # shape: (1000,)
    
    # Map the whole time series
    mapped_series = state_map[time_series]
    return mapped_series

# %% scanning
n_reduced = np.array([2,3,5,10,20])

for ns in range(len(n_reduced)):
    plt.figure()
    reduced_P, state_sequence, cluster_labels, mapping_matrix = reduce_and_sample_markov(P, num_clusters=n_reduced[ns], num_steps=2000)
    reduced_states = map_time_series(mapping_matrix, labels)
    lags, acf_data = autocorrelation_discrete(reduced_states, max_lag=1000)
    lags, acf_mark = autocorrelation_discrete(state_sequence, max_lag=1000)
    plt.loglog(lags, acf_data)
    plt.loglog(lags, acf_mark)
    plt.xlabel('Lag'); plt.ylabel('Autocorrelation'); plt.title('Autocorrelation of Discrete States')
    plt.grid(True)
    