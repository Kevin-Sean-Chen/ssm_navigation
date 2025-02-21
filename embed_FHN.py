# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:38:25 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import hankel
import random
from scipy.optimize import minimize

# %%
# FitzHugh-Nagumo model parameters
a = 0.7
b = 0.8
epsilon = 0.08
dt = 0.1  # Time step for Euler method
T = 50000  # Total time
num_steps = int(T / dt)  # Number of steps

# Define input current I(t) (time-dependent input)
def input_current(t):
    current = .0 * (np.sin(0.9 * t) + 1) + np.random.randn()*.5
    if t>10000:
        current = 0. * (np.sin(0.2 * t) + 1) + np.random.randn()*.5
        # current = 0.5
        
    # if current>0:
    #     current=1
    # else:
    #     current=-1
    return current  # Example: oscillating input

# Initialize v and w
v = np.zeros(num_steps)
w = np.zeros(num_steps)
iptt = np.zeros(num_steps)

# Initial conditions (v0, w0)
v[0] = -1.0  # Initial membrane potential
w[0] = -1.0  # Initial recovery variable

# Euler method loop
for t in range(1, num_steps):
    # Current time
    time = t * dt
    
    # FitzHugh-Nagumo equations
    iptt[t] = input_current(time)*1
    dvdt = v[t-1] - (v[t-1]**3) / 3 - w[t-1] + iptt[t]
    dwdt = epsilon * (v[t-1] + a - b * w[t-1])
    
    ### test bifurcation
    if time>10000 and time<12000:
        dvdt += 0.2
    if time>20000 and time<22000:
        dvdt += 0.2
    if time>30000 and time<32000:
        dvdt += 0.2
        
    # Euler update for v and w
    v[t] = v[t-1] + dvdt * dt
    w[t] = w[t-1] + dwdt * dt

# Plot the results
time = np.linspace(0, T, num_steps)
stim_vec = iptt  #np.array([input_current(t) for t in time])

plt.figure(figsize=(10, 6))

# Plot v(t) - membrane potential
plt.subplot(2, 1, 1)
plt.plot(time, v, label='Membrane Potential v(t)')
plt.xlabel('time')
plt.ylabel('v(t)')
plt.title('FitzHugh-Nagumo Model')
plt.grid(True)

# Plot w(t) - recovery variable
plt.subplot(2, 1, 2)
plt.plot(time, stim_vec, label='stochastic drive', color='orange')
plt.xlabel('Time')
plt.ylabel('w(t)')
plt.title('stochastic input')
plt.grid(True)

plt.tight_layout()
plt.show()

v_data = v*1

# %% IDEAS
# recover FHN state space
# embed with stimuli
# find modes of bifurcation

# %% embedding Markov
N_star = 1000
K_star = 150
tau_star = 2

# %% functional
def build_signal(data, K=K_star):
    features = []
    T = len(data)
    samp_vec = data[:-np.mod(T,K)-1]
    for tt in range(len(samp_vec)-K):
        vx = samp_vec[tt:tt+K]
        features.append(vx)
    return np.vstack(features)

X_volt = build_signal(v_data)#, use_dtheta=True)
X_stim = build_signal(stim_vec)
X = X_volt*1 #
# X = np.concatenate((X_volt, X_stim), 1)

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
    kmeans = MiniBatchKMeans(batch_size=batchsize,n_clusters=n_seeds).fit(tseries)
    labels=kmeans.labels_
    if return_centers:
        return labels,kmeans.cluster_centers_
    return labels

n_states = 10
test_label = kmeans_knn_partition(X, N_star)

# %% compute transition and measure entropy
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

def get_steady_state(transition_matrix):
    # Find the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T) #.T
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
    probs = probs/np.sum(probs)
    P_hat = diags(1/probs)*P.transpose()*diags(probs)
    R=(P+P_hat)/2
    return R

def sorted_spectrum(R,k=5,which='LR'):
    eigvals,eigvecs = eigs(R,k=k,which=which)
    sorted_indices = np.argsort(eigvals.real)[::-1]
    return eigvals[sorted_indices],eigvecs[:,sorted_indices]

Pij = compute_transition_matrix(test_label, N_star)
h_est = trans_entropy(Pij)
print(h_est)

# %%
labels, centrals = kmeans_knn_partition(X, N_star, return_centers=True)
P = compute_transition_matrix(labels, N_star)

# %%
R = get_reversible_transition_matrix(P)
eigvals,eigvecs = sorted_spectrum(R,k=7)  # choose the top k modes
phi2=eigvecs[labels,1].real
u,s,v = np.linalg.svd(X,full_matrices=False)

plt.figure(figsize=(10,7))
color_abs = np.max(np.abs(phi2))
plt.scatter(u[:,0],u[:,1],c=phi2,cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.show()

# %% plotting, compared to ground truth!
import umap

sub_samp = np.random.choice(X.shape[0], 10000, replace=False)
reducer = umap.UMAP(n_components=3, random_state=42)
data_2d = reducer.fit_transform(X[sub_samp,:])

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
idx = np.real(uu).argsort()[::-1]  # Get indices to sort eigenvalues
sorted_eigenvalues = uu[idx]
plt.figure()
plt.plot((-dt/1*tau_star)/np.log(sorted_eigenvalues[1:30]),'-o')
plt.ylabel('relaxation time (s)')
plt.xlabel('eigenvalue index')

# %% color code tracks]
imode = 1
phi2 = eigvecs[labels,imode].real
color_abs = np.max(np.abs(phi2))
window_show = np.arange(50000,250000)
plt.figure()
plt.scatter(window_show, v_data[window_show],c=phi2[window_show],cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}')

# %% color code stims
imode = 1
phi2 = eigvecs[labels,imode].real
color_abs = np.max(np.abs(phi2))
window_show = np.arange(50000,150000)
plt.figure()
plt.scatter(window_show, stim_vec[window_show],c=phi2[window_show],cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}')

# %% now, analyze driving part!!!
###############################################################################
# %%
### compute P(x'|x)
### compute P(x'|x, u)
### infer weights on the input
### compute TE
# %% stim avarage
###
# deal with threshold crossing here!!!
###
# IDEA: it has to be delayed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###
def find_threshold_crossings(time_series, threshold, direction="down"):
    above_threshold = time_series > threshold
    changes = np.diff(above_threshold.astype(int))  # Compute changes

    if direction == "up":
        crossings = np.where(changes == 1)[0]  # Crossing from below to above
    elif direction == "down":
        crossings = np.where(changes == -1)[0]  # Crossing from above to below
    else:
        raise ValueError("Direction must be 'up' or 'down'")
    return crossings
pos = np.where(phi2>-0.01)[0]
# pos = find_threshold_crossings(phi2, -0.01,'down')
plt.figure()
plt.plot(np.mean(X_stim[pos-0,:],0)) ### <---- this result should depent on the FHN parameter!!! ex, oscillation vs. excited!
plt.xlabel('time steps'); plt.ylabel('mode-triggered average'); #plt.ylim([-0.009,0.006])
