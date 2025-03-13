# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:27:03 2024

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
chop = 500000
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
# def compute_transition_matrix(time_series, n_states):
#     # Initialize the transition matrix (n x n)
#     transition_matrix = np.zeros((n_states, n_states))
#     # Loop through the time series and count transitions
#     for (i, j) in zip(time_series[:-1], time_series[1:]):
#         transition_matrix[i, j] += 1
#     # Normalize the counts by dividing each row by its sum to get probabilities
#     row_sums = transition_matrix.sum(axis=1, keepdims=True)
#     transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
#     return transition_matrix

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

Pij = compute_transition_matrix(test_label, ids, N_star)
h_est = trans_entropy(Pij)
print(h_est)

# %%
###############################################################################
# %% scanning
# tau = 2
# Ks = np.array([.5, 1, 2, 4, 6, 8])*int(5* (90/down_samp))/tau
# Ns = np.array([10, 100, 250, 500, 1000, 2000, 3000])
# nats = np.zeros((len(Ks), len(Ns)))

# for kk in range(len(Ks)):
#     for nn in range(len(Ns)):
#         ### build delay embedding
#         # Xi,idsi = build_signal(rec_signal, return_id=True, K=Ks[kk],tau=tau)   ### for signal
#         Xi,idsi = build_X(data4fit, return_id=True, K=Ks[kk])  #### for behavior
#         ### cluster
#         time_series = kmeans_knn_partition(Xi, Ns[nn])  ### mask the transition ones too!
#         ### build matrix
#         Pij = compute_transition_matrix(time_series, idsi, Ns[nn])
#         ### compute entropy
#         nati = trans_entropy(Pij)
        
#         nats[kk,nn] = nati / (tau/90)  ### nats per second for transitions
#         print(kk,nn)

# %% plot entropy rates
# plt.figure()
# colors_K = plt.cm.viridis(np.linspace(0,1,len(Ks)))
# for k,K in enumerate(Ks):
#     temp = K/90*tau
#     plt.plot(Ns, nats[k]/1,c=colors_K[k],marker='o',label=f'K={temp:.1f} s')
# # plt.plot(Ns, nats.T)
# plt.xlabel('N')
# plt.ylabel('nats/s')
# plt.legend()
# plt.title('behavior embedding')

# %% bulding Markov model
###############################################################################
# %% spectrum
X_traj, track_id = build_X(data4fit, return_id=True)
labels, centrals = kmeans_knn_partition(X_traj, N_star, return_centers=True)
P = compute_transition_matrix(labels, track_id, N_star)

# %% get reverisble matrix
top_n_values = 25
R = get_reversible_transition_matrix(P)
eigvals,eigvecs = sorted_spectrum(R,k=top_n_values)  # choose the top k modes
phi2=eigvecs[labels, 1].real
u,s,v = np.linalg.svd(X_traj,full_matrices=False)

plt.figure(figsize=(10,7))
color_abs = np.max(np.abs(phi2))
plt.scatter(u[:,0],u[:,1],c=phi2,cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.show()

# %% try U-MAP (takes longer time...)
import umap

sub_samp = np.random.choice(X_traj.shape[0], 10000, replace=False)
reducer = umap.UMAP(n_components=3, random_state=42)
data_2d = reducer.fit_transform(X_traj[sub_samp,:])

# %% show in 3D
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
color_abs = np.max(np.abs(phi2[sub_samp]))
# sc = plt.scatter(data_2d[:,0], data_2d[:,1], c=phi2[sub_samp], cmap='coolwarm', s=.1, vmin=-color_abs, vmax=color_abs)
sc = ax.scatter(data_2d[:,0], data_2d[:,1],data_2d[:,2], c=phi2[sub_samp], cmap='coolwarm', s=.2, vmin=-color_abs, vmax=color_abs)
plt.colorbar(sc)

# %% spectral analysis
P_shuff = compute_transition_matrix(np.random.permutation(labels),track_id, N_star)
# uu,vv = np.linalg.eig(P_shuff)  #P_shuff
uu,vv = np.linalg.eig(P)
idx = uu.argsort()[::-1]  # Get indices to sort eigenvalues
sorted_eigenvalues = np.real(uu[idx])
plt.figure()
plt.plot((-1/60*down_samp*tau_star)/np.log(sorted_eigenvalues[1:30]),'-o')
plt.ylabel('relaxation time (s)')
plt.xlabel('eigenvalue index')
# plt.yscale('log')
# plt.ylim([0.001, 20])

# %% color code tracks]
imode = 1
phi2 = eigvecs[labels,imode].real
window_show = np.arange(1,50000,3)
X_xy, track_id = build_X(rec_tracks, return_id=True)
xy_back = X_xy[:, [0,int(K_star)]]
plt.figure()
plt.scatter(xy_back[window_show, 0],xy_back[window_show, 1],c=phi2[window_show],cmap='coolwarm',s=.5,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}')

# %% modes back to velocity
X_speed = build_signal(speeds, K_star)
plt.figure()
plt.scatter(X_traj[window_show, 1+K_star*1], phi2[window_show],c=phi2[window_show],cmap='coolwarm',s=.5,vmin=-color_abs,vmax=color_abs) # veloctiy
plt.scatter(X_traj[window_show, 1+K_star*0], X_traj[window_show, 1+K_star*1],c=phi2[window_show],cmap='coolwarm',s=.5,vmin=-color_abs,vmax=color_abs)
# plt.scatter(X_speed[window_show,0], phi2[window_show],c=phi2[window_show],cmap='coolwarm',s=.5,vmin=-color_abs,vmax=color_abs)  # speed
plt.xlabel('vx'); plt.ylabel('vy')

# %% traj vs. mode -> functional analysis #####################################
window_show = np.arange(1,100000,3)
xys_samp = xy_back[window_show, :]
vxy_samp = X_traj[:, [0, K_star*1]][window_show]
dists = np.zeros(len(window_show))
for tt in range(len(window_show)):
    dists[tt] = np.linalg.norm(xys_samp[tt,:] - np.array([0,90]))  ### change this to projection towards the goal!!
    vec_xy = np.array([0,90]) - xys_samp[tt,:]
    vec_xy = vec_xy/np.linalg.norm(vec_xy)
    dists[tt] = np.dot(vec_xy, vxy_samp[tt,:])  ### projection onto the goal direction
### remove jumps
pos = np.where(np.abs(dists)<50)[0]
phi_samp = -phi2[window_show]
plt.figure()
plt.plot(phi_samp[pos], dists[pos], '.',alpha=0.1) 
plt.ylabel('projection to goal'); plt.xlabel('behavioral mode')

# %% test LDS fitting!
import numpy as np
from scipy.linalg import eig
from scipy.ndimage import gaussian_filter1d

def estimate_jacobian(x, y, dx_dt, dy_dt):
    """
    Estimate the Jacobian matrix using least squares fit:
    dx/dt ≈ J * [x, y]^T
    dy/dt ≈ J * [x, y]^T
    """
    A = np.vstack([x, y]).T  # Construct feature matrix
    J11, J12 = np.linalg.lstsq(A, dx_dt, rcond=None)[0]
    J21, J22 = np.linalg.lstsq(A, dy_dt, rcond=None)[0]
    
    return np.array([[J11, J12], [J21, J22]])

# Example: Simulated time series (replace with real data)
t =window_show*1/60  ### time vector
pos = np.where(np.abs(dists)<50)[0]
x,y = phi_samp[pos], dists[pos]

# Estimate time derivatives numerically
dx_dt = np.gradient(x, t)
dy_dt = np.gradient(y, t)

# Smooth derivatives to reduce noise
dx_dt = gaussian_filter1d(dx_dt, sigma=10)
dy_dt = gaussian_filter1d(dy_dt, sigma=10)

# Estimate Jacobian
J = estimate_jacobian(x, y, dx_dt, dy_dt)

# Compute eigenvalues to analyze stability
eigenvalues, _ = eig(J)
print("Estimated Jacobian matrix:\n", J)
print("Eigenvalues:", eigenvalues)

# %% transient grouth curve
t_values = np.arange(1,1000,3)*1/60
growth = [np.linalg.norm(sp.linalg.expm(J*t)) for t in t_values]

plt.plot(t_values, growth)
plt.xlabel("Time")
plt.ylabel("||exp(Jt)|| (Transient Growth)")
plt.title("Transient Growth of Non-Normal System")
plt.grid()
plt.show()

# %% sample traj
xx = -gaussian_filter1d(x, sigma=1)
yy = gaussian_filter1d(y, sigma=1)
from matplotlib.collections import LineCollection
# Generate sample data (Replace with your real data)
init_t = 600 #600 #2300  #600 #
window_show = np.arange(init_t, init_t+650, 1)
colors = np.linspace(0, 1, len(window_show))  # Color progression
# Convert points into line segments
points = np.array([xx[window_show], yy[window_show]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)  # Create line segments
# Create a LineCollection with the color progression
fig, ax = plt.subplots(figsize=(6, 5))
lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, 1))
lc.set_array(colors[:-1])  # Color based on progression
# Add line collection to plot
ax.add_collection(lc)
ax.scatter(xx[window_show[0]], yy[window_show[0]], color="red", edgecolors="black", s=100, label="Start Point", zorder=3)
ax.autoscale()
ax.set_xlim([-0.22,.1])
ax.set_ylim([-21,20])
# Add colorbar
cbar = plt.colorbar(lc)
# Labels
ax.set_xlabel("mode #2")
ax.set_ylabel("projected goal")
ax.set_title("trajectory across 30s")

# %% symmetry measurement
################################################################################################################
################################################################################################################
A = 0.5*(P-R)
# Reversibility Score
# Detailed Balance Violation: Quantifies how far the matrix deviates from detailed balance.
# Non-Reversible Flux: Total flux arising from the non-reversible component.
# Eigenvalue Gap
score = np.linalg.norm(R)**2/np.linalg.norm(P)**2 ### compare to shuffle?  ### random is around .85?
print(score)
### random control
testP = np.random.rand(N_star,N_star)  # random transition
testM = P*0
testM[P>0] = 1
testP = testP*testM  # masking for sparsity
row_sums = testP.sum(axis=1, keepdims=True)
testP = np.divide(testP, row_sums, where=row_sums!=0)  # normalization
testR = get_reversible_transition_matrix(testP)
score_control = np.linalg.norm(testR)**2/np.linalg.norm(testP)**2 ### compare to shuffle?  ### random is around .85?
print(score_control)

# %% study driven-modes
###############################################################################
# %% y=beta X
lags = 401
n_top_modes = 7
threshold = 2.
phi_top = eigvecs[labels,1:n_top_modes].real
X_odor = build_signal(rec_signal, K_star)
X_odor = X_odor[:, 0]
X_odor[X_odor<threshold] = 0
X_odor[X_odor>threshold] = 1
# y_ = np.diff(X_odor)
# y_ = np.append(y_, 0)
# X_odor[y_<0] = 0

X = (hankel(X_odor[:lags], X_odor[lags-1:]).T)
X = np.concatenate((X, np.ones((X.shape[0],1))), 1)

time_vec = np.arange(lags)*0.011*down_samp
plt.figure()
for ii in range(n_top_modes-1):
    # y = phi_top[lags-1:,ii]*1  ### future
    y = phi_top[:-lags+1,ii]*1  ### past
    
    xxt = X.T @ X #+ lamb*D.T @ D + theta*np.eye(D.shape[1])
    invxx = np.linalg.inv(xxt)
    beta = invxx @ X.T @ y
    beta_phi = beta[:-1] #.reshape(n_top_modes-1, lags)
    plt.plot(time_vec, beta_phi,  label=f'mode: {ii+1}')

plt.legend()
plt.xlabel('time (s)'); plt.ylabel('weights')
plt.xlim([-0.051,4])

# %% information rank
### make the full dist vector
dists = np.zeros(len(phi2))
for tt in range(len(dists)):
    xys_i =  X_traj[tt, [0, K_star*1]]
    vxy_i = X_traj[tt, [0, K_star*1]]
    dists[tt] = np.linalg.norm(xys_i - np.array([0,90]))  ### change this to projection towards the goal!!
    vec_xy = np.array([0,90]) - xys_i
    vec_xy = vec_xy/np.linalg.norm(vec_xy)
    dists[tt] = np.dot(vec_xy, vxy_i)

pos = np.where(np.abs(dists)<50)[0]
dists = dists[pos]
# %% loop through modes
from sklearn.metrics import mutual_info_score
def compute_mutual_information(x, y, bins=10):
    x_binned = np.digitize(x, bins=np.histogram_bin_edges(x, bins=bins))
    y_binned = np.digitize(y, bins=np.histogram_bin_edges(y, bins=bins))
    mi = mutual_info_score(x_binned, y_binned)
    return mi
n_top_modes = 25
reps = 10
segment = 60000
nbins = 5
modes_k = eigvecs[labels,1:n_top_modes].real
MIs = np.zeros((n_top_modes, reps))
signal = dists*1  ### againts projection to source
signal = X_odor[pos]*1 ### against odor

for nn in range(n_top_modes):
    if nn == n_top_modes-1:
        # phi_samp = np.random.permutation(modes_k[pos, -1])
        phi_samp = np.roll(modes_k[pos, 0], shift=np.random.randint(0, len(modes_k)))
    else:
        phi_samp = modes_k[pos, nn]
    
    # MIs[nn] = compute_mutual_information(phi_samp, signal, bins=10)
    for rr in range(reps):
        rand_int = np.random.randint(0, len(dists)- segment)
        MIs[nn, rr] = compute_mutual_information(phi_samp[rand_int:rand_int+segment], signal[rand_int:rand_int+segment], bins=nbins)

plt.figure()
# plt.plot(MIs,'-o')
plt.errorbar(np.arange(n_top_modes), np.mean(MIs,1), np.std(MIs,1))#/reps**0.5)
plt.xlabel('ranked modes')
plt.ylabel('I(odor; mode)')

# %% start with simple regression
# X_odor = build_signal(rec_signal, K_star)
# n_top_modes = 5
# phi_top = eigvecs[labels,1:n_top_modes].real
# lags = 71
# matrices = []
# ### create design matrix and find weights at one time point
# for ii in range(n_top_modes-1):
#     Xi = hankel(phi_top[:lags, ii], phi_top[lags-1:, ii]).T  ### time by lag
#     matrices.append(Xi)

# X = np.hstack(matrices)  ### put all together
# X = np.concatenate((X, np.ones((X.shape[0],1))), 1)  ### add offset

# ## %% processs signal
# # y = X_odor[lags-1:,0]  ### pre
# y = X_odor[:-lags+1,0]*1  ### post
# # y = X[:,0]*1  ################# test
# y[y<2] = 0
# y[y>2] = 1
# y_ = np.diff(y)
# y_ = np.append(y_, 0)
# y[y_<0] = 0

# # %% analytic OSL, adding ridge...
# def make_difference_matrix(n):
#     D = np.zeros((n - 1, n))
#     for i in range(n - 1):
#         D[i, i] = -1
#         D[i, i + 1] = 1
#     return D
# time_vec = np.arange(lags)*0.011*down_samp
# lamb = 10
# theta = 10
# D = make_difference_matrix(X.shape[1])
# D[-1, :] = 0
# xxt = X.T @ X + lamb*D.T @ D + theta*np.eye(D.shape[1])
# invxx = np.linalg.inv(xxt)
# beta = invxx @ X.T @ y
# beta_phi = beta[:-1].reshape(n_top_modes-1, lags)
# plt.figure()
# for ii in range(0,n_top_modes-1):
#     plt.plot(time_vec, beta_phi[ii, :], label=f'mode: {ii+1}')
# plt.legend()
# plt.xlabel('time (s)'); plt.ylabel('weights')

# %% recovering stats
def sub_transition_matrix(P, S):
    sub_P = P[np.ix_(S, S)]  # extract submatrix
    row_sums = sub_P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero for rows that sum to zero
    normalized_sub_P = sub_P / row_sums
    return normalized_sub_P

### symbolic sequence
import random
def sample_markov_chain(P, n_steps, init=None):
    """
    Sample a sequence from a Markov transition matrix, starting from a random initial state.

    Parameters:
    P (np.ndarray): Transition matrix (size N x N).
    n_steps (int): Number of steps to sample.

    Returns:
    np.ndarray: Sequence of sampled states.
    """
    n_states = P.shape[0]  # Number of states
    states = np.zeros(n_steps, dtype=int)  # To store the sequence of states
    if init is None:
        states[0] = np.random.choice(n_states) # Randomly choose the initial state
    else:
        states[0] = init
    # Sample the next state based on the current state and transition matrix
    for t in range(1, n_steps):
        current_state = states[t - 1]
        next_state = np.random.choice(n_states, p=P[current_state])
        states[t] = next_state
    return states

def gen_tracks_given_substates(subid, n_steps, return_v=False, init=False):
    ### get submtrix and symbolic sequence
    subP = sub_transition_matrix(P, subid)
    sub_state = sample_markov_chain(subP, n_steps, init)
    ### back to veclicity, then construct tracks!
    sub_centrals = centrals[subid,:]
    subsample_vxy = []
    for tt in range(len(sub_state)):
        # vxy_i = np.vstack((sub_centrals[sub_state[tt],:K].mean(), sub_centrals[sub_state[tt],K:].mean())).T  # take mean
        vxy_i = np.vstack((np.random.choice(sub_centrals[sub_state[tt],:int(K_star)],1), np.random.choice(sub_centrals[sub_state[tt],int(K_star):],1))).T  # take sample
        subsample_vxy.append(vxy_i*(tau_star/90*down_samp))
    subsample_vxy = np.concatenate(subsample_vxy)
    ### tracks!
    subsample_xy = np.cumsum(subsample_vxy, axis=0)
    if return_v:
        return subsample_xy, subsample_vxy
    else:
        return subsample_xy

### all ids
subid = np.arange(N_star)
### sub-strategies ###
imode = 1
# subid = np.where(eigvecs[:,imode].real>0)[0]
subsample_xy = gen_tracks_given_substates(subid, 500)
plt.plot(subsample_xy[:,0], subsample_xy[:,1])

# %% validating full Makov model...
###############################################################################
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
samp_xy, samp_vxy = gen_tracks_given_substates(subid, 100000, return_v=True)

# %% autocorr
xy_id = 0
lags, acf_data = compute_autocorrelation(vx_smooth[::tau_star], 1000)
# lags, acf_data = compute_autocorrelation(vy_smooth[::tau_star], 1000)
lags, acf_mark = compute_autocorrelation(samp_vxy[:, xy_id], 1000)
plt.figure()
plt.plot(np.arange(len(lags))*tau_star/90*1, acf_data, label='data')
plt.plot(np.arange(len(lags))*tau_star/90*1, acf_mark, label='delayed Markov')
plt.legend(); plt.xlabel(r'$\tau$ (s)');  plt.ylabel(r'$<v(t),v(t+\tau)>$')

# %% distribution
bins = np.arange(-15,15,0.5)
plt.figure()
count_data,_ = np.histogram(vx_smooth, bins)
# count_data,_ = np.histogram(vy_smooth, bins)
count_simu,_ = np.histogram(samp_vxy[:, xy_id] /(tau_star/90*down_samp), bins)
plt.plot(bins[:-1], count_data/count_data.sum(), label='data')
plt.plot(bins[:-1], count_simu/count_simu.sum(), label='delayed Markov')
plt.xlabel(r'$v_x$'); plt.ylabel('count'); plt.legend(); plt.yscale('log')


# %% NEXT
### scanning
### autcorr
### navigation sim
### use these as RL basis of action!!

# %% test with simple metastable state clustering (not the coherent set method)
# eigenvalues, left_eigenvectors = np.linalg.eig(P.T)
# idx = eigenvalues.argsort()[::-1]  # Get indices to sort eigenvalues
# sorted_eigenvalues_left = np.real(eigenvalues[idx])
top_n_values_left = 100
eigvals_left,eigvecs_left = sorted_spectrum(P.T, k=top_n_values_left)

# %% computing stimulus conditional transitions
def compute_transition_matrix_stim(behavior_series, stim_series, track_id, n_states):
    """
    modified from previous function to handle track id and not compute those transitions
    """
    # Initialize the transition matrix (n x n)
    transition_matrix = np.zeros((n_states, n_states))
    # find valid transition that is not across tracks
    valid_transitions = (track_id[:-1] == track_id[1:]) & (stim_series[:-1] == 1)
    # Get the current and next state only for valid transitions
    current_states = behavior_series[:-1][valid_transitions]
    next_states = behavior_series[1:][valid_transitions]
    # Use np.add.at to efficiently accumulate transitions
    np.add.at(transition_matrix, (current_states, next_states), 1)
    
    # Normalize the counts by dividing each row by its sum to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
    return transition_matrix 

threshold = 2.
X_odor = build_signal(rec_signal, K_star)
X_odor = np.mean(X_odor,1)
X_odor[X_odor<threshold] = 0
X_odor[X_odor>threshold] = 1
P_cond = compute_transition_matrix_stim(test_label, X_odor, ids, N_star)

keep_rows = P_cond.sum(1) == 1
P_cond = P_cond[np.ix_(keep_rows, keep_rows)]
row_sums = P_cond.sum(axis=1, keepdims=True)
P_cond = np.divide(P_cond, row_sums)#, where=row_sums!=0)

# %% get reverisble matrix
top_n_values = 5
R_cond = get_reversible_transition_matrix(P_cond)
eigvals,eigvecs = sorted_spectrum(R_cond, k=top_n_values)  # choose the top k modes
phi2_cond = eigvecs[labels, 1].real
u,s,v = np.linalg.svd(X_traj,full_matrices=False)

plt.figure(figsize=(10,7))
color_abs = np.max(np.abs(phi2_cond))
plt.scatter(u[:,0],u[:,1],c=phi2_cond,cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.show()

# %%
uu,vv = np.linalg.eig(P_cond)
idx = uu.argsort()[::-1]  # Get indices to sort eigenvalues
sorted_eigenvalues = np.real(uu[idx])
plt.figure()
plt.plot((-1/90*down_samp*tau_star)/np.log(sorted_eigenvalues[1:30]),'-o')
plt.ylabel('relaxation time (s)')
plt.xlabel('eigenvalue index')

# %%
score = np.linalg.norm(R_cond)**2/np.linalg.norm(P_cond)**2 ### compare to shuffle?  ### random is around .85?
print(score)
# %% sample for Markovianity
def irreversibility(Pij):
    pi = get_steady_state(Pij)
    irr = 0
    for ii in range(Pij.shape[0]):
        for jj in range(Pij.shape[1]):
            # if ii<jj:
            if Pij[ii, jj] > 0 and Pij[jj, ii] > 0:
                irr += 0.5* pi[ii]*Pij[ii,jj]*np.log((pi[ii]*Pij[ii,jj])/(pi[jj]*Pij[jj,ii] + 1e-10))
    return irr

print(irreversibility(P))
print(irreversibility(P_cond))

# %% fit w weight on input u? joint embedding???

# %% coarse graining?? link back to TE??

