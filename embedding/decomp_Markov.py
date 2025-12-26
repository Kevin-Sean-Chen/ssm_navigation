# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 10:51:20 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import hankel
import scipy as sp
from collections import defaultdict

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import h5py

# %% load mat file for the data structure
file_dir = r'C:\Users\ksc75\Yale University Dropbox\users\mahmut_demir\data\Smoke Navigation Paper Data\ComplexPlumeNavigationPaperData.mat'
# Open the .mat file
with h5py.File(file_dir, 'r') as file:
    # Access the structure
    your_struct = file['ComplexPlume']

    # Access fields within the structure
    expmat = your_struct['Smoke']['expmat'][:]  # Load the dataset as a numpy array
    col_k = list(your_struct['Smoke']['col'].keys())
    col_v = list(your_struct['Smoke']['col'].values())
    # print(col.keys())

# %% now extract track data
chop = 1000000 #500000 #4331205
down_samp = 3
trjNum = expmat[0,:][::down_samp][:chop]
signal = expmat[12,:][::down_samp][:chop]
stops = expmat[38,:][::down_samp][:chop]
turns = expmat[39,:][::down_samp][:chop]
vx_smooth = expmat[28,:][::down_samp][:chop]
vy_smooth = expmat[29,:][::down_samp][:chop]
x_smooth = expmat[31,:][::down_samp][:chop]
y_smooth = expmat[32,:][::down_samp][:chop]
speed_smooth = expmat[30,:][::down_samp][:chop]  #11 31
dtheta_smooth = expmat[34,:][::down_samp][:chop]  #14 35

# %% some pre-processing
v_threshold = 30
vx_smooth[np.abs(vx_smooth)>v_threshold] = v_threshold
vy_smooth[np.abs(vy_smooth)>v_threshold] = v_threshold
signal[np.isnan(signal)] = 0

dtheta_threshold = 360
dtheta_smooth[np.abs(dtheta_smooth)>dtheta_threshold] = dtheta_threshold
dtheta_smooth[np.isnan(dtheta_smooth)] = 0

# %% build data
track_id = np.unique(trjNum)
n_tracks = len(track_id)
data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
masks = []   # where there is nan
track_ids = []  # record track id (file and track)
rec_tracks = []  # record the full track x,y
rec_signal = []  # record opto signal
times = []   # record time in epoch
speeds = []

for tr in range(n_tracks):
    print(tr)
    ### extract features
    pos = np.where(trjNum==track_id[tr])[0]  # position of this track
    temp_xy = np.column_stack((x_smooth[pos] , y_smooth[pos]))
    temp_vxy = np.column_stack((vx_smooth[pos] , vy_smooth[pos]))
    # temp_vxy = np.column_stack((vx_smooth[pos] , vy_smooth[pos], dtheta_smooth[pos]))   ### test with dtheta feature!
    
    ### recording
    data4fit.append(temp_vxy)  # get data for ssm fit
    rec_tracks.append(temp_xy)  # get raw tracksd
    track_ids.append(np.zeros(len(pos))+tr) 
    rec_signal.append(signal[pos])
    speeds.append(speed_smooth[pos])
    # masks.append(thetas)
    # times.append(data['t'][pos])

# %% set parameters (for now, need to scan later...)
K_star = int(4* (90/down_samp))
N_star = 1200
tau_star = 2  ### data is at 90 Hz
    
# %% functionals
def build_signal(data, K=K_star, tau=tau_star, return_id=False):
    K = int(K)
    features = []
    ids = []
    n_tracks = len(data)
    for tr in range(n_tracks):
        datai = data[tr]
        T = len(datai)
        samp_vec = datai[:-np.mod(T,K)-1]
        for tt in range(0, len(samp_vec)-K, tau):
            vs = samp_vec[tt:tt+K] #[::tau]????????????
            vs_windowed = vs.reshape(-1, K)
            features.append(vs_windowed)   ### might have problems here across tracks!!!
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

def build_X(data, return_id=False, K=K_star , tau=tau_star, use_dtheta=False):
    K = int(K)
    features = []
    ids = []
    n_tracks = len(data)
    for tr in range(n_tracks):
        datai = data[tr]
        T = len(datai)
        samp_vec = datai[:-np.mod(T,K)-1,:]
        for tt in range(0, len(samp_vec)-K, tau):
            vx = samp_vec[tt:tt+K, 0]
            vy = samp_vec[tt:tt+K, 1]
            vx_windowed = vx.reshape(-1, K)
            vy_windowed = vy.reshape(-1, K)
            if use_dtheta:
                dtheta = samp_vec[tt:tt+K,2]
                dtheta_windowed = dtheta.reshape(-1, K)
                dtheta = samp_vec[tt:tt+K,2]
                features.append(np.hstack((vx_windowed, vy_windowed, dtheta_windowed)))   ### might have problems here across tracks!!!    
            else:
                features.append(np.hstack((vx_windowed, vy_windowed)))   ### might have problems here across tracks!!!
            ids.append(tr)
    if return_id:
        feats = np.concatenate(features) 
        ids = np.array(ids)
        jumps = ids[1:]==ids[:-1]
        pos = np.where(jumps==False)[0]
        # feats[pos,:] = ma.masked
        mask = feats*0
        mask[pos, :] = True
        masked_data = np.ma.masked_array(feats, mask)
        return masked_data , ids
    else:
        return np.concatenate(features)

X,ids = build_X(data4fit, return_id=True)#, use_dtheta=True)

# %% clustering and assigning states
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

def null_Markov_entropy(M):
    N = M.shape[0]
    Pnull = np.full((N, N), 1.0 / N)
    return Pnull, trans_entropy(Pnull)
    
Pij = compute_transition_matrix(test_label, ids, N_star)
h_est = trans_entropy(Pij)
P_null, h_null = null_Markov_entropy(Pij)
print('EP of data: ', h_est)
print('EP of null: ', h_null)

# %% entropy decomposition
def EP_decomposition(P):
    ### compute pi
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
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

    ### compute the three terms ###
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

    return -stick, -traffic, -irr

# %% decomposition and scaling
###############################################################################
# %% scanning
tau = 2
reps = 3
Ks = np.array([.5, 1, 2, 4, 6, 8])*int(5* (90/down_samp))/tau
Ns = np.array([10, 100, 250, 500, 1000, 2000])
ep_decomps = np.zeros((reps, len(Ns), 4))
ep_null = ep_decomps*1

# for kk in range(len(Ks)):

for nn in range(len(Ns)):
    
    for rr in range(reps):
        print('repeat', rr)
        
        ### build delay embedding
        Xi,idsi = build_signal(rec_signal, return_id=True, K=K_star,tau=tau)   ##### for signal
        # Xi,idsi = build_X(data4fit, return_id=True, K=K_star)  ###### for behavior
        ### cluster
        time_series = kmeans_knn_partition(Xi, Ns[nn])  ### mask the transition ones too!
        ### build matrix
        Pij = compute_transition_matrix(time_series, idsi, Ns[nn])
        ### compute entropy
        nati = trans_entropy(Pij)
        
        ### compute decomposition
        s_, t_, i_ = EP_decomposition(Pij)
        
        ### null model
        P_null, h_null = null_Markov_entropy(Pij)
        s_n, t_n, i_n = EP_decomposition(P_null)
        
        ### record
        ep_decomps[rr, nn,0], ep_decomps[rr, nn,1], ep_decomps[rr, nn,2], ep_decomps[rr, nn,3] = s_, t_, i_, nati
        ep_null[rr, nn,0], ep_null[rr, nn,1], ep_null[rr, nn,2], ep_null[rr, nn,3] = s_n, t_n, i_n, h_null
        print('states: ', nn)

# %% plotting
plt.figure()
lines_data = plt.plot(Ns, ep_decomps[:, :, -1].T, '-o')
lines_null = plt.plot(Ns, ep_null[:, :, -1].T, '--o')
plt.legend([lines_data[-1], lines_null[-1]], ['data', 'null'])
plt.xlabel('N'); plt.ylabel('nats'); plt.title('full embedded Markov')

# %%
plt.figure()
parts = ['s', 't', 'i']
for ii in range(3):
    plt.subplot(3,1,ii+1); plt.title(parts[ii])
    plt.plot(Ns, ep_decomps[:,:,ii].T, '-o')
    plt.plot(Ns, ep_null[:,:,ii].T, '--o')
plt.xlabel('N'); plt.ylabel('nats')

# %% checking Markov-ness!
###############################################################################
# %% discrietize
X_traj, track_id = build_X(data4fit, return_id=True)
labels, centrals = kmeans_knn_partition(X_traj, N_star, return_centers=True)

# %% conditionals
def cond_prob_first_order_chosen_state(states, chosen_state, n_states=None):
    """
    Compute P(x_t = chosen_state | x_{t-1}) as a vector of shape (n_states,).

    Returns:
        np.ndarray: Vector where [i] = P(x_t = chosen_state | x_{t-1} = i)
    """
    states = np.asarray(states)
    if n_states is None:
        n_states = np.max(states) + 1

    total_counts = np.zeros(n_states)
    match_counts = np.zeros(n_states)

    for t in range(1, len(states)):
        prev, curr = states[t-1], states[t]
        total_counts[prev] += 1
        if curr == chosen_state:
            match_counts[prev] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        probs = match_counts / total_counts
        probs[np.isnan(probs)] = 0.0

    return probs

def cond_prob_second_order_chosen_state(states, chosen_state, n_states=None):
    """
    Compute P(x_t = chosen_state | x_{t-2}, x_{t-1}) as a matrix of shape (n_states, n_states).

    Returns:
        np.ndarray: Matrix where [i, j] = P(x_t = chosen_state | x_{t-2} = i, x_{t-1} = j)
    """
    states = np.asarray(states)
    if n_states is None:
        n_states = np.max(states) + 1

    total_counts = np.zeros((n_states, n_states))
    match_counts = np.zeros((n_states, n_states))

    for t in range(2, len(states)):
        i, j, k = states[t-2], states[t-1], states[t]
        total_counts[i, j] += 1
        if k == chosen_state:
            match_counts[i, j] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        probs = match_counts / total_counts
        probs[np.isnan(probs)] = 0.0

    return probs


np.random.seed(0)
states = np.random.choice(5, size=1000)
chosen_state = 2

P1_vec = cond_prob_first_order_chosen_state(states, chosen_state)
P2_mat = cond_prob_second_order_chosen_state(states, chosen_state)

print("P(x_t = 2 | x_{t-1}):", P1_vec)
print("P(x_t = 2 | x_{t-2}, x_{t-1}):")
print(P2_mat)


# %% conditional entropy
def entropy_chosen_state_first_order(states, chosen_state, n_states=None):
    states = np.asarray(states)
    if n_states is None:
        n_states = np.max(states) + 1

    match_counts = np.zeros(n_states)
    context_counts = np.zeros(n_states)

    for t in range(1, len(states)):
        prev, curr = states[t-1], states[t]
        context_counts[prev] += 1
        if curr == chosen_state:
            match_counts[prev] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        p_cond = match_counts / context_counts
        p_joint = match_counts / np.sum(match_counts)
        entropy = -np.nansum(p_joint * np.log2(p_cond + 1e-12))

    return entropy

def entropy_chosen_state_second_order(states, chosen_state, n_states=None):
    states = np.asarray(states)
    if n_states is None:
        n_states = np.max(states) + 1

    match_counts = np.zeros((n_states, n_states))
    context_counts = np.zeros((n_states, n_states))

    for t in range(2, len(states)):
        i, j, k = states[t-2], states[t-1], states[t]
        context_counts[i, j] += 1
        if k == chosen_state:
            match_counts[i, j] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        p_cond = match_counts / context_counts
        p_joint = match_counts / np.sum(match_counts)
        entropy = -np.nansum(p_joint * np.log2(p_cond + 1e-12))

    return entropy

np.random.seed(0)
states = np.random.choice(5, size=1000)
chosen_state = 2

H1_k = entropy_chosen_state_first_order(states, chosen_state)
H2_k = entropy_chosen_state_second_order(states, chosen_state)

print(f"H(x_t={chosen_state} | x_t-1): {H1_k:.4f} bits")
print(f"H(x_t={chosen_state} | x_t-1, x_t-2): {H2_k:.4f} bits")

# %% test with data
reps = 7
Ns = np.array([10, 100, 250, 500, 1000, 2000])
ep_conditions = np.zeros((reps, len(Ns), 2))

for nn in range(len(Ns)):
    
    for rr in range(reps):
        chosen_state = np.random.randint(0, Ns[nn])
        print('repeat', rr)
        
        ### build delay embedding
        Xi,idsi = build_signal(rec_signal, return_id=True, K=K_star,tau=tau)   ### for signal
        # Xi,idsi = build_X(data4fit, return_id=True, K=K_star, tau=tau_star)  #### for behavior
        ### cluster
        time_series = kmeans_knn_partition(Xi, Ns[nn])  ### mask the transition ones too!
        ### build matrix
        Pij = compute_transition_matrix(time_series, idsi, Ns[nn])
        ### compute entropy
        nati = trans_entropy(Pij)
        
        ### compute conditional EPs
        H1_k = entropy_chosen_state_first_order(time_series, chosen_state)
        H2_k = entropy_chosen_state_second_order(time_series, chosen_state)
        
        ### record
        ep_conditions[rr,nn,0] = H1_k
        ep_conditions[rr,nn,1] = H2_k
        print('states: ', nn)

# %% plotting
plt.figure()
# plt.plot(Ns, ep_conditions[:,:,0].T, '-o')
# plt.plot(Ns, ep_conditions[:,:,1].T, '--o')
plt.errorbar(Ns, np.mean(ep_conditions[:,:,0].T,1), np.std(ep_conditions[:,:,0].T,1))
plt.errorbar(Ns, np.mean(ep_conditions[:,:,1].T,1), np.std(ep_conditions[:,:,1].T,1))
plt.xlabel('N'); plt.ylabel('nats'); plt.title('first vs. seoncd order entropy, for stimuli!')

# %% for reduced model
###############################################################################
# %% entropy as a function of coarse grained dynamics!
from scipy.linalg import eig
from sklearn.cluster import SpectralClustering
kmean_seed = 1 #37 #1

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

def map_time_series(mapping_matrix, time_series):
    mapping_matrix = np.asarray(mapping_matrix)
    time_series = np.asarray(time_series)

    # Find for each original state its mapped group
    state_map = np.argmax(mapping_matrix, axis=0)  # shape: (1000,)
    
    # Map the whole time series
    mapped_series = state_map[time_series]
    return mapped_series

def first_order_conditional_entropy(states):
    """
    Compute H(x_t | x_{t-1}) from a discrete time series.
    """
    states = np.asarray(states)
    n_states = np.max(states) + 1
    counts = np.zeros((n_states, n_states))

    for t in range(1, len(states)):
        counts[states[t-1], states[t]] += 1

    # Normalize
    row_sums = counts.sum(axis=1, keepdims=True)
    P_cond = np.divide(counts, row_sums, where=row_sums != 0)

    # Compute joint probability
    total = counts.sum()
    P_joint = counts / total

    with np.errstate(divide='ignore', invalid='ignore'):
        H = -np.nansum(P_joint * np.log2(P_cond))

    return H

def second_order_conditional_entropy(states):
    """
    Compute H(x_t | x_{t-2}, x_{t-1}) from a discrete time series.
    """
    states = np.asarray(states)
    n_states = np.max(states) + 1
    counts = np.zeros((n_states, n_states, n_states))

    for t in range(2, len(states)):
        counts[states[t-2], states[t-1], states[t]] += 1

    # Normalize
    slice_sums = counts.sum(axis=2, keepdims=True)
    P_cond = np.divide(counts, slice_sums, where=slice_sums != 0)

    # Compute joint probability
    total = counts.sum()
    P_joint = counts / total

    with np.errstate(divide='ignore', invalid='ignore'):
        H = -np.nansum(P_joint * np.log2(P_cond))

    return H

# %% pick a full model
# Xi,idsi = build_X(data4fit, return_id=True, K=K_star, tau=tau_star)  #### for behavior
Xi,idsi = build_signal(rec_signal, return_id=True, K=K_star,tau=tau)  ### for signal!=
time_series = kmeans_knn_partition(Xi, N_star) 
Pij = compute_transition_matrix(time_series, idsi, N_star)

# %% scanning
reps = 1
n_reduced = np.array([2, 3, 4, 5, 7, 10, 20])
n_reduced = np.array([2,5,10,20,40,80])
errs = np.zeros((len(n_reduced), 4))  ### n-states by the metrics
ep_reduced = np.zeros((reps, len(n_reduced), 4))
ep_cond_reduced = np.zeros((reps, len(n_reduced), 2))

for rr in range(reps):
    print('repeat', rr)
    for nn in range(len(n_reduced)):
        chosen_state = np.random.randint(0,n_reduced[nn])
        print('n-states=', n_reduced[nn])
        ### coarse graining
        reduced_P, state_sequence, cluster_labels, mapping_matrix = reduce_and_sample_markov(Pij, num_clusters=n_reduced[nn], num_steps=1000)
        reduced_sequence = map_time_series(mapping_matrix, time_series)
        
        ### entropy decomposition
        s_, t_, i_ = EP_decomposition(reduced_P)
        nati = trans_entropy(reduced_P)
        ep_reduced[rr, nn,0], ep_reduced[rr, nn,1], ep_reduced[rr, nn,2], ep_reduced[rr, nn,3] = s_, t_, i_, nati
        
        ### Markov-ness
        H1_k = first_order_conditional_entropy(reduced_sequence)
        H2_k = second_order_conditional_entropy(reduced_sequence)
        ep_cond_reduced[rr,nn,0] = H1_k
        ep_cond_reduced[rr,nn,1] = H2_k

# %% plotting
plt.figure()
plt.plot(n_reduced, ep_reduced[:,:,0].T, '-o')
plt.plot(n_reduced, ep_reduced[:,:,1].T, '--o')
# plt.errorbar(Ns, np.mean(ep_reduced[:,:,0].T,1), np.std(ep_conditions[:,:,0].T,1))
# plt.errorbar(Ns, np.mean(ep_reduced[:,:,1].T,1), np.std(ep_conditions[:,:,1].T,1))
plt.xlabel('reduced N'); plt.ylabel('nats'); plt.title('first vs. seoncd order entropy')

# %%
plt.figure()
parts = ['s', 't', 'i']
for ii in range(3):
    plt.subplot(3,1,ii+1); plt.title(parts[ii])
    plt.plot(n_reduced, ep_reduced[:,:,ii].T / 1, '-o')
    # plt.plot(n_reduced, ep_reduced[:,:,ii].T / np.log(n_reduced)[:,None], '-o')
plt.xlabel('reduced N'); plt.ylabel('nats')

# %% analyze reduced model
###############################################################################
# %% load time sereis of x and s
np.random.seed(42) #42
Xi_behavior, idsi = build_X(data4fit, return_id=True, K=K_star, tau=tau_star)  #### for behavior
Xi_stim,_ = build_signal(rec_signal, return_id=True, K=K_star,tau=tau_star)  ### for signal!

# %% reduce dim for both
base_odor = 5
odor_threshold = 2.
reduced_n_state = 5
stim_bin_ = np.mean(Xi_stim,1)  ### avarage of window
stim_bin_ = Xi_stim[:,0]  ### just the recent index
stim_bin = stim_bin_*0
stim_bin[stim_bin_ < base_odor] = 0
stim_bin[stim_bin_ > odor_threshold] = 1

time_series, centrals = kmeans_knn_partition(Xi_behavior, N_star, return_centers=True)
Pij = compute_transition_matrix(time_series, idsi, N_star)
reduced_P, state_sequence, cluster_labels, mapping_matrix = reduce_and_sample_markov(Pij, num_clusters=reduced_n_state, num_steps=1000)
reduced_behavior = map_time_series(mapping_matrix, time_series)

# %% reduced conditionals
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# pos = np.where(stim_bin==0)[0]
# P_conditional0 = compute_transition_matrix(reduced_behavior[pos], idsi[pos], reduced_n_state)
# im1 = axs[0].imshow(np.log(P_conditional0+1e-9), aspect='auto')
# axs[0].set_title('non-stim')

# pos = np.where(stim_bin==1)[0]
# P_conditional1 = compute_transition_matrix(reduced_behavior[pos], idsi[pos], reduced_n_state)
# im2 = axs[1].imshow(np.log(P_conditional1+1e-9), aspect='auto')
# axs[1].set_title('stim')

# divider = make_axes_locatable(axs[1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(im2, cax=cax)

# plt.tight_layout()

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
lag_ = 5
stim_bin_, reduced_behavior_, idsi_ = stim_bin[:-lag_], reduced_behavior[lag_:], idsi[lag_:]
pos = np.where(stim_bin_ < 200)[0]
A = compute_transition_matrix(reduced_behavior_[pos], idsi_[pos], reduced_n_state)
pos = np.where(stim_bin_==1)[0]
B = compute_transition_matrix(reduced_behavior_[pos], idsi_[pos], reduced_n_state)
A,B = np.log(A+1e-9), np.log(B+1e-9)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Set the same color scale for both
vmin = min(A.min(), B.min())
vmax = max(A.max(), B.max())

# Plot first matrix
im1 = axs[0].imshow(A, vmin=vmin, vmax=vmax, aspect='auto')
axs[0].set_title('non-stim')

# Plot second matrix
im2 = axs[1].imshow(B, vmin=vmin, vmax=vmax, aspect='auto')
axs[1].set_title('stim')

# Create a divider for the last axis and append a colorbar
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax)

plt.tight_layout()
plt.show()

# %% decomposition
s_a, t_a, i_a = EP_decomposition(np.exp(A))
s_b, t_b, i_b = EP_decomposition(np.exp(B))

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.bar(['sticky', 'traffic','irreversible'], [s_a, t_a, i_a]); plt.ylabel('entropy')
plt.subplot(122)
plt.bar(['sticky', 'traffic','irreversible'], [s_b, t_b, i_b]);

# %% visualize states
for st in range(mapping_matrix.shape[0]):
    plt.figure()
    pos = np.where(mapping_matrix[st,:]==1)[0]
    vxyi = centrals[pos, :]
    plt.plot(vxyi[:,:K_star], vxyi[:,K_star:],'k.', alpha=0.1)
    plt.xlim([-30, 30]); plt.ylim([-30, 30])

# %% scan threshold and history effects
delay_tau = 1
threshold = 1

from scipy.linalg import eig

def kl_divergence_markov(P, Q):
    eigvals, eigvecs = eig(P.T)
    stat_dist = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    stat_dist = stat_dist[:,0]
    stat_dist = stat_dist / stat_dist.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where((P > 0) & (Q > 0), P / Q, 1)
        local_kl = np.where(P > 0, P * np.log(ratio), 0)
        kl_per_state = np.sum(local_kl, axis=1)

    D_kl = np.sum(stat_dist * kl_per_state)
    return max(D_kl, 0.0)  # Clip tiny negatives to 0

# %%
thres = np.array([1.5, 2, 3, 4, 5, 7.5, 10,20])
lags = np.array([1,5,10,15,20])
stim_kl = np.zeros( (len(thres), len(lags)) )

for tt in range(len(thres)):
    for ll in range(len(lags)):
        odor_threshold = thres[tt]
        time_lag = lags[ll]
        stim_bin_ = np.mean(Xi_stim,1)
        stim_bin = stim_bin_*0
        stim_bin[stim_bin_<base_odor] = 0
        stim_bin[stim_bin_>odor_threshold] = 1
        stim_bin, reduced_behavior_, idsi_ = stim_bin[:-time_lag], reduced_behavior[time_lag:], idsi[time_lag:]
        pos = np.where(stim_bin==0)[0]
        A = compute_transition_matrix(reduced_behavior_[pos], idsi_[pos], reduced_n_state)
        pos = np.where(stim_bin==1)[0]
        B = compute_transition_matrix(reduced_behavior_[pos], idsi_[pos], reduced_n_state)
        stim_kl[tt, ll] = kl_divergence_markov(A, B)
    
# %%
plt.figure()
# plt.plot(thres, stim_kl,'-o')
for i in range(stim_kl.shape[1]):
    l = lags[i]
    plt.plot(thres, stim_kl[:,i], '-o',label=f"lag= {l}")
plt.xlabel('odor threshold'); plt.ylabel('KL(stim|non-stim)'); plt.legend()

# %% test input-driven state model!
###############################################################################
# %%
### try SSM? pytorch?
### just regression?
# %% simple looping
window_back = 400
filts = np.zeros((reduced_n_state, reduced_n_state, window_back))
trans_counts = np.zeros((reduced_n_state, reduced_n_state))
shifts = np.where(idsi[:-1] != idsi[1:])[0]+0 ### remoe transition from tracks
for ii in range(reduced_n_state):
    print(ii)
    for jj in range(reduced_n_state):
        if ii is not jj:
            indices = np.where((reduced_behavior[:-1] == ii) & (reduced_behavior[1:] == jj))[0] + 0
            indices = indices[~np.isin(indices, shifts)]
            indices = indices[indices >= window_back]
            for tt in range(len(indices)):
                filts[ii,jj,:] += stim_bin[indices[tt]-window_back: indices[tt]]
            filts[ii,jj,:] = filts[ii,jj,:]/tt ### triggered average
            trans_counts[ii,jj] = tt
# %%
plt.figure(figsize=(15, 15))

for i in range(reduced_n_state):
    for j in range(reduced_n_state):
        # Subplot index needs to be (i * 5 + j + 1)
        plt.subplot(reduced_n_state, reduced_n_state, i * reduced_n_state + j + 1)
        plt.plot(filts[i, j, :])
        plt.title(f'({i},{j})', fontsize=8)
        plt.xticks([])  # remove x ticks
        plt.yticks([])  # remove y ticks

plt.tight_layout()
plt.show()
