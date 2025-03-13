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
centrals = centrals*1

K_star = centrals.shape[1]//2
tau_star = 3
down_samp = 3
kmean_seed = 42

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
num_clusters = 4
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
def gen_data_from_redused_states(P_reduced, lt):
    # Step 5: Sample from the reduced Markov model
    state_sequence = np.zeros(lt, dtype=int)
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
plt.bar(['sticky', 'traffic','irreversible'], [stick, traffic, irr])
