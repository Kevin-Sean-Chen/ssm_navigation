# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:23:49 2024

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

import pickle
import gzip
import glob
import os

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
threshold_track_l = 60 * 30  # 20 # look at long-enough tracks

# Define the folder path
# folder_path = 'C:/Users/ksc75/Downloads/ribbon_data_kc/'

# # Use glob to search for all .pkl files in the folder
# pkl_files = glob.glob(os.path.join(folder_path, '*.pklz'))

# # Print the list of .pkl files
# for file in pkl_files:
#     print(file)

# %% for perturbed data
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/100424_new/'  ### for OU-ribbons
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kiri_choi/data/ribbon_sleap/2024-9-17/'  ### for lots of ribbon data
root_dir = 'C:/Users/kevin/Yale University Dropbox/users/kiri_choi/data/ribbon_sleap/2024-9-17/'
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/odor_vision/2024-11-5'
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

# pkl_files = pkl_files[8:]
# pkl_files = pkl_files[4:]
print(pkl_files) 

# %%
# plt.figure()
# nf = len(pkl_files)
# for ff in range(nf):
#     ### load file
#     with gzip.open(pkl_files[ff], 'rb') as f:
#         data = pickle.load(f)
#     for dd in range(len(data)):
#         x,y = data['x_smooth'], data['y_smooth']
#         plt.plot(x,y,'k.')
    
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
                
                temp_xy = np.column_stack((data['x_smooth'][pos] , data['y_smooth'][pos]))
                                
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
def build_signal(data, K=window):
    K = int(K)
    features = []
    ids = []
    n_tracks = len(data)
    for tr in range(n_tracks):
        datai = data[tr]
        T = len(datai)
        samp_vec = datai[:-np.mod(T,K)-1]
        X = np.zeros((T-K, K))
        for tt in range(len(samp_vec)-K):
            vs = samp_vec[tt:tt+K]
            vs_windowed = vs.reshape(-1, K)
            features.append(vs_windowed)   ### might have problems here across tracks!!!
            ids.append(tr)
    return np.concatenate(features)

def build_X(data, return_id=False, K=window):
    K = int(K)
    features = []
    ids = []
    n_tracks = len(data)
    for tr in range(n_tracks):
        datai = data[tr]
        T = len(datai)
        samp_vec = datai[:-np.mod(T,K)-1,:]
        X = np.zeros((T-K, K))
        for tt in range(len(samp_vec)-K):
            vx = samp_vec[tt:tt+K, 0]
            vy = samp_vec[tt:tt+K, 1]
            vx_windowed = vx.reshape(-1, K)
            vy_windowed = vy.reshape(-1, K)
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

X,ids = build_X(data4fit, return_id=True)

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

Pij = compute_transition_matrix(test_label, ids, n_states)
h_est = trans_entropy(Pij)

# %%
###############################################################################
# %% scanning
# tau = 3
# Ks = np.array([.5, 1, 2, 4, 6, 8])*(60/tau)
# Ns = np.array([10, 100, 250, 500, 1000, 2000, 3000])
# nats = np.zeros((len(Ks), len(Ns)))

# for kk in range(len(Ks)):
#     for nn in range(len(Ns)):
#         ### build delay embedding
#         Xi,idsi = build_X(data4fit, return_id=True, K=Ks[kk])
#         Xi,idsi = Xi[::tau, :], idsi[::tau]
#         ### cluster
#         time_series = kmeans_knn_partition(Xi, Ns[nn])  ### mask the transition ones too!
#         ### build matrix
#         Pij = compute_transition_matrix(time_series, idsi, Ns[nn])
#         ### compute entropy
#         nati = trans_entropy(Pij)
        
#         nats[kk,nn] = nati / (tau/60)  ### nats per second for transitions
#         print(kk,nn)

# # %%
# plt.figure()
# colors_K = plt.cm.viridis(np.linspace(0,1,len(Ks)))
# for k,K in enumerate(Ks):
#     temp = K/60*3
#     plt.plot(Ns, nats[k]/1,c=colors_K[k],marker='o',label=f'K={temp} s')
# # plt.plot(Ns, nats.T)
# plt.xlabel('N')
# plt.ylabel('nats/s')
# plt.legend()

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

# %% scan tau
N = 1000  # number of states
K = 2*60  # delay window
X_traj, track_id = build_X(data4fit, return_id=True, K=K)
X_traj, track_id = build_X(data4fit, return_id=True, K=K)
labels, centrals = kmeans_knn_partition(X_traj, N, return_centers=True)

taus = np.array([1,5,10,30,30,60,120,180])
specs_tau = np.zeros((len(taus), N-1))
for tt in range(len(taus)):
    taui = taus[tt]
    P = compute_transition_matrix(labels[::taui], track_id[::taui], N)
    uu,vv = np.linalg.eig(P)
    idx = uu.argsort()[::-1]  # Get indices to sort eigenvalues
    sorted_eigenvalues = uu[idx]
    specs_tau[tt,:] = (-1/60*taui)/np.log(sorted_eigenvalues[1:])
    print(tt)

# %% 
plt.figure()
plt.plot(taus/60, specs_tau[:,:5],'-o')
# plt.xscale('log')
plt.xlabel(r'step $\tau$ (s)'); plt.ylabel('relaxation time (s)')

# %% fix param now
N = 1000  # number of states
K = 2*60  # delay window
tau = 10   # transition steps
X_traj, track_id = build_X(data4fit, return_id=True, K=K)
X_traj, track_id = X_traj[::tau, :], track_id[::tau]
labels, centrals = kmeans_knn_partition(X_traj, N, return_centers=True)
# labels = labels[::tau]
P = compute_transition_matrix(labels, track_id, N)

# %%
R = get_reversible_transition_matrix(P)
eigvals,eigvecs = sorted_spectrum(R,k=7)  # choose the top k modes
phi2=eigvecs[labels,2].real
u,s,v = np.linalg.svd(X_traj,full_matrices=False)

plt.figure(figsize=(10,7))
color_abs = np.max(np.abs(phi2))
plt.scatter(u[:,0],u[:,1],c=phi2,cmap='coolwarm',s=.1,vmin=-color_abs,vmax=color_abs)
plt.show()

# %% try U-MAP
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
P_shuff = compute_transition_matrix(np.random.permutation(labels),track_id, N)
uu,vv = np.linalg.eig(P_shuff)  #P_shuff
uu,vv = np.linalg.eig(P)
idx = uu.argsort()[::-1]  # Get indices to sort eigenvalues
sorted_eigenvalues = np.real(uu[idx])
plt.figure()
plt.plot((-1/60*tau)/np.log(sorted_eigenvalues[1:1000]),'-o')
plt.ylabel('relaxation time (s)')
plt.xlabel('eigenvalue index')
plt.yscale('log')
plt.ylim([0.001, 20])

# %% color code tracks]
imode = 2
tau = 10
phi2 = eigvecs[labels,imode].real
# phi2 = np.ma.masked_array(phi2, mask=np.isin(labels, idx[600:]))
window_show = np.arange(10000,20000)
# X_xy = build_X(rec_tracks, K)[::tau, :]
X_xy, track_id = build_X(rec_tracks, return_id=True, K=K)
X_xy, track_id = X_xy[::tau, :], track_id[::tau]
xy_back = X_xy[:, [0,K]]
plt.figure()
plt.scatter(xy_back[window_show, 0],xy_back[window_show, 1],c=phi2[window_show],cmap='coolwarm',s=.5,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}')

# %% analyze in the task!
X_time = build_signal(times, K)[::tau]
X_odor = build_signal(rec_signal, K)[::tau]
# subsamp = np.arange(0,X_time.shape[0],60)
proj_val_offt = []
time_since_off = []
back2track_id = []
for tt in range(X_time.shape[0]):
    timei = X_time[tt,0]
    signali = X_odor[tt,:]
    if (timei >= 1) & (timei<45+30+90): # & (np.sum(signali)>0):
        time_since_off.append(timei - (45+30))
        proj_val_offt.append(phi2[tt])
        back2track_id.append(tt)
        
plt.figure()
plt.plot(time_since_off, proj_val_offt, 'k.',alpha=.01)
plt.axvline(x=-30, color='r', linestyle='--'); plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('since odor off (s)'); plt.ylabel(r'$\phi$')

# %% visualization
import seaborn as sns

# Sample scatter data
x = np.array(time_since_off)
y = np.array(proj_val_offt)

# pos = np.where(np.abs(y)>0.01)[0]
# x,y = x[pos], y[pos]
down_samp = 3
x,y = x[::down_samp], y[::down_samp]

plt.figure(figsize=(8, 6))
sns.kdeplot(x=x, y=y, cmap="viridis", fill=True, thresh=0, bw_method='silverman')
plt.scatter(x, y, s=10, color="black", alpha=0.1)  # Optional: overlay scatter points
plt.colorbar(label="Density")
plt.xlabel('since odor off (s)'); plt.ylabel(r'$\phi$')
plt.axvline(x=-30, color='r', linestyle='--'); plt.axvline(x=0, color='r', linestyle='--')
plt.title("Smoothed Density Plot")
plt.xlim([x.min(), x.max()]); plt.ylim([y.min(), y.max()])

# %% sorting raw track by phi2 during odor!
track_sort_id = []
mean_phi = []
for nn in range(len(data4fit)):
    pos_track = np.where(track_id==nn)[0]
    pos_post = np.where((X_time[:,0]>45+30) & (X_time[:,0]<45+30+30))[0]
    pos_stim = np.where((X_time[:,0]>45) & (X_time[:,0]<45+30))[0]
    # pos = np.intersect1d(pos_stim,pos_track)
    pos = np.intersect1d(pos_post,pos_track)
    if len(pos)>0:
        track_sort_id.append(nn)
        mean_phi.append(np.mean((phi2[pos])))
        # mean_phi.append(np.std(X_odor[pos,0])*np.mean(X_odor[pos,0]))
        # mean_phi.append(np.mean(X_odor[pos,0]))

# %% plotting
dispy = 50
offset = 1
post_window = 90*60
rec_tracks_sub = [rec_tracks[ii] for ii in track_sort_id]
times_sub = [times[ii] for ii in track_sort_id]

import matplotlib.cm as cm
colors = cm.viridis(np.linspace(0, 1, len(mean_phi)))

plt.figure()
sortt_id = np.argsort(mean_phi)[::-1]
kk = 0
for ii in range(0,len(sortt_id),1):
    dd = sortt_id[ii]
    traji = rec_tracks_sub[dd]
    timei = times_sub[dd]
    anchor_pos = np.argmin(np.abs(timei-(45+30)))
    phi2i = phi2[np.where(track_id==track_sort_id[ii])[0]]
    
    ### remove boundary conditions
    if traji[anchor_pos:,1].max()<180 and traji[anchor_pos:,1].min()>8 and traji[anchor_pos:,0].max()<270:
    # if traji[anchor_pos,1].max()<180 and traji[anchor_pos,1].min()>10 and traji[anchor_pos:,0].max()<270 and len(traji[anchor_pos:,0])>1:
    # if traji[anchor_pos:,0].max()<270:
        
        ### plotting tracks
        if len(traji[anchor_pos:,:])<post_window:
            plt.plot(traji[anchor_pos:,0] - traji[anchor_pos,0]*offset, kk*dispy + traji[anchor_pos:,1]-traji[anchor_pos,1]*offset, color=colors[ii])
        else:
            wind_ = anchor_pos+post_window
            plt.plot(traji[anchor_pos:wind_,0] - traji[anchor_pos,0]*offset, kk*dispy + traji[anchor_pos:wind_,1]-traji[anchor_pos,1]*offset, color=colors[ii])
        plt.plot(traji[anchor_pos,0]  - traji[anchor_pos,0]*offset, kk*dispy +traji[anchor_pos,1]-traji[anchor_pos,1]*offset,'r.')
        
        #### TRY REWRITE WITH X_xy !!! ###########
        
        # plt.scatter(traji[anchor_pos:,0] - traji[anchor_pos,0]*offset, kk*dispy + traji[anchor_pos:,1]-traji[anchor_pos,1]*offset,c=phi2i[anchor_pos:],cmap='coolwarm',s=.5,vmin=-color_abs,vmax=color_abs)
        kk = kk+1

plt.xlabel('displacement in x since off (mm)')
plt.ylabel('sorted by projection')
# plt.ylabel('sorted by odor encounter')

# %% with matrix
dispy = 50
offset = 1
post_window = 15*60
rec_tracks_sub = [rec_tracks[ii] for ii in track_sort_id]
times_sub = [times[ii] for ii in track_sort_id]

import matplotlib.cm as cm
colors = cm.viridis(np.linspace(0, 1, len(mean_phi)))

plt.figure()
sortt_id = np.argsort(mean_phi)[::-1]
kk = 0
for ii in range(0,len(sortt_id),1):
    dd = sortt_id[ii]
    traji = track_sort_id[dd]
    
    pos = np.where(track_id==traji)[0]
    traji = np.array([X_xy[pos,0], X_xy[pos,K+1]]).T
    timei = X_time[pos,0]
    anchor_pos = np.argmin(np.abs(timei-(45+30)))
    phi2i = phi2[pos]
    
    ### remove boundary conditions
    if traji[anchor_pos:,1].max()<180 and traji[anchor_pos:,1].min()>8 and traji[anchor_pos:,0].max()<270:
    # if traji[anchor_pos,1].max()<180 and traji[anchor_pos,1].min()>10 and traji[anchor_pos:,0].max()<270 and len(traji[anchor_pos:,0])>1:
    # if traji[anchor_pos:,0].max()<270:
        #### TRY REWRITE WITH X_xy !!! ###########
        if len(traji[anchor_pos:,:])<post_window:
            plt.scatter(traji[anchor_pos:,0] - traji[anchor_pos,0]*offset, kk*dispy + traji[anchor_pos:,1]-traji[anchor_pos,1]*offset,c=phi2i[anchor_pos:],cmap='coolwarm',s=.5,vmin=-color_abs,vmax=color_abs)
            # plt.scatter(traji[anchor_pos:,0] - traji[anchor_pos,0]*offset, kk*dispy + traji[anchor_pos:,1]-traji[anchor_pos,1]*offset,c=np.abs(phi2i[anchor_pos:]),cmap='coolwarm',s=.5,vmin=0,vmax=color_abs)
        else:
            wind_ = anchor_pos+post_window
            plt.scatter(traji[anchor_pos:wind_,0] - traji[anchor_pos,0]*offset, kk*dispy + traji[anchor_pos:wind_,1]-traji[anchor_pos,1]*offset,c=phi2i[anchor_pos:wind_],cmap='coolwarm',s=.5,vmin=-color_abs,vmax=color_abs)
            
        plt.plot(traji[anchor_pos,0]  - traji[anchor_pos,0]*offset, kk*dispy +traji[anchor_pos,1]-traji[anchor_pos,1]*offset,'r.')
        kk = kk+1

plt.xlabel('displacement in x since off (mm)')
# plt.ylabel('sorted by projection')
plt.ylabel('sorted by odor encounter')


# %% sample from most-likely phi states to reconstruct tracks
n_bins = 100
hist, bin_edges = np.histogram(time_since_off, bins=n_bins)
bin_indices = np.digitize(time_since_off, bins=bin_edges) - 1  # "-1" to match 0-based indexing
elements_in_bins = [np.where(bin_indices == i)[0] for i in range(n_bins)]

proj_val_offt = np.array(proj_val_offt)
v_samps = []
construct_vxy = []
timing = []
for nn in range(int(len(elements_in_bins)*1/1)):  #3/4
    bini = elements_in_bins[nn]
    this_bin = np.random.choice(bini,1)[0]
    v_samps.append(proj_val_offt[this_bin])
    idi = back2track_id[this_bin]
    
    ### find median
    # median_value = np.mean(proj_val_offt[bini])
    # idi = np.abs(proj_val_offt[bini] - median_value).argmin()
    # print(idi)
    
    vxy = X_traj[idi,:]
    construct_vxy.append(np.vstack((vxy[:K], vxy[K:])).T*1/60)
    if (bin_edges[nn]>-30) & (bin_edges[nn]<0):
        timing.append(np.ones(K))
    elif bin_edges[nn]>0:
        timing.append(np.ones(K)+1)
    else:
        timing.append(np.ones(K)*0)
        
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


# %% test metastable state
from scipy.signal import find_peaks
def optimal_partition(phi2,inv_measure,P,return_rho = True):
    """
    input eigen mode phi2, inv_measure is pi, P is transition, and return_pho is logic
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
                      for idx in range(2)]
    rho_c = np.min(rho_sets,axis=1)
    peaks, heights = find_peaks(rho_c, height=0.5)
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

# %%
imode = 1
inv_measure = get_steady_state(P)
phi2_vec = eigvecs[:, imode].real
c_range,rho_sets,idx,kmeans_labels = optimal_partition(phi2_vec, inv_measure[:,None], P)

# %%
plt.figure()
plt.plot(c_range,rho_sets,lw=2)
rho_c = np.min(rho_sets,axis=1)
plt.plot(c_range,rho_c,c='k',ls='--')
plt.axvline(c_range[idx],c='r',ls='--')
plt.ylim(.85,1)
plt.xlim(-0.04,0.04)
plt.xlabel(r'$\phi_2$',fontsize=15)
plt.ylabel(r'$\rho$',fontsize=15)
plt.xticks(fontsize=12)

# %% metastable in tracks
def indices_in_s(S, X):
    mask = np.isin(X, S)
    return np.where(mask)[0]

window_show = np.arange(0,10000)  # choose window
meta_pos = indices_in_s(np.where(kmeans_labels==1)[0], labels[window_show])
# X_xy = build_X(rec_tracks, K)[::tau, :]
X_xy, track_id = build_X(rec_tracks, return_id=True, K=K)
X_xy, track_id = X_xy[::tau, :], track_id[::tau]
xy_back = X_xy[:, [0,K]]
plt.figure()
plt.scatter(xy_back[window_show, 0],xy_back[window_show, 1],color='k',s=.1,vmin=-color_abs,vmax=color_abs)
plt.scatter(xy_back[meta_pos, 0],xy_back[meta_pos, 1],color='r',s=.1,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}')

# %%
###############################################################################
# %% sampling from state!!
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
        vxy_i = np.vstack((np.random.choice(sub_centrals[sub_state[tt],:K],1), np.random.choice(sub_centrals[sub_state[tt],K:],1))).T  # take sample
        subsample_vxy.append(vxy_i*(tau/60))
    subsample_vxy = np.concatenate(subsample_vxy)
    ### tracks!
    subsample_xy = np.cumsum(subsample_vxy, axis=0)
    if return_v:
        return subsample_xy, subsample_vxy
    else:
        return subsample_xy

subid = np.where(kmeans_labels==1)[0]
subsample_xy = gen_tracks_given_substates(subid, 500)
plt.plot(subsample_xy[:,0], subsample_xy[:,1])

# %% sim for groups
reps = 30
for ii in range(reps):
    subid = np.where(kmeans_labels==1)[0]
    subsample_xy = gen_tracks_given_substates(subid, 500)
    plt.plot(subsample_xy[:,0], subsample_xy[:,1], 'b',alpha=.7)
plt.title('sampling from metastable state')

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
samp_xy, samp_vxy = gen_tracks_given_substates(np.arange(N), 70000, return_v=True)

###
xy_id = 1
lags, acf_data = compute_autocorrelation(vec_vxy[::tau, xy_id], 1000)
lags, acf_mark = compute_autocorrelation(samp_vxy[:, xy_id], 1000)
plt.figure()
plt.plot(np.arange(len(lags))*tau/60, acf_data, label='data')
plt.plot(np.arange(len(lags))*tau/60, acf_mark, label='delayed Markov')
plt.legend(); plt.xlabel(r'$\tau$ (s)');  plt.ylabel(r'$<v(t),v(t+\tau)>$')

# %%
bins = np.arange(-15,15,0.5)
plt.figure()
count_data,_ = np.histogram(vec_vxy[:, xy_id], bins)
count_simu,_ = np.histogram(samp_vxy[:, xy_id] /(tau/60), bins)
plt.plot(bins[:-1], count_data/count_data.sum(), label='data')
plt.plot(bins[:-1], count_simu/count_simu.sum(), label='delayed Markov')
plt.xlabel(r'$v_x$'); plt.ylabel('count'); plt.legend(); plt.yscale('log')

# %% compute mixing
Ts = np.array([1,5,10,20,40,80,160])#, 320])
# Ts = np.array([1, 2,4,8,16,32,64,128])
init_state = np.argmin(phi2_vec)
reps = 50
errt = np.zeros((len(Ts), reps))
for rr in range(reps):
    print(rr)
    for tau in range(len(Ts)):
        samp_xy1, samp_v1 = gen_tracks_given_substates(np.arange(N), Ts[tau], return_v=True, init=init_state)
        samp_xy2, samp_v2 = gen_tracks_given_substates(np.arange(N), Ts[tau], return_v=True, init=init_state)
        # erri = (np.sum((samp_v1[-1,:] - samp_v2[-1,:])**2))**0.5   # v diference
        # erri = (np.sum((samp_xy1[-1,:] - samp_xy2[-1,:])**2))**0.5  # x difference
        erri = (np.sum((samp_xy1[-1,:] - np.array([0,0]))**2))**0.5  # displacement
        errt[tau,rr] = erri

# %%
plt.figure()
plt.plot(Ts*(tau/60), errt,'ko');
plt.errorbar(Ts*(tau/60), np.mean(errt,1), np.std(errt,1), fmt='-o')
plt.xlabel('time lag (s)')
# plt.ylabel('velocity difference')
plt.ylabel('displacement (mm)')
# plt.xscale('log')

# %% check stop to speed-up time
vec_speed = np.sum(vec_vxy**2,1)**.5
threshold = 5
stop_or_not = vec_speed*0
stop_or_not[vec_speed>threshold] = 1
stop2run = np.diff(stop_or_not)
stops = np.where(stop2run==-1)[0]
runs = np.where(stop2run==1)[0]
min_length = min(len(stops), len(runs))
vec1_trimmed = stops[:min_length]
vec2_trimmed = runs[:min_length]
dur = -(vec1_trimmed - vec2_trimmed)
plt.figure()
plt.hist(np.abs(dur)*1/60, 500)
plt.xlim([0,5])
plt.xlabel('stop time (s)')

# %% scoring the matrix
###############################################################################
# Generate a random Markov transition matrix
def random_markov_matrix(N):
    P = np.random.rand(N, N)
    P /= P.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1
    return P

# Compute stationary distribution
def stationary_distribution(P):
    evals, evecs = np.linalg.eig(P.T)  # Eigen decomposition of P^T
    stationary = np.real(evecs[:, np.isclose(evals, 1)])  # Select eigenvector for eigenvalue ~1
    stationary = stationary[:, 0]
    stationary /= stationary.sum()  # Normalize to sum to 1
    return stationary

# Compute reversible component
def reversible_component(P):
    pi = stationary_distribution(P)
    N = P.shape[0]
    P_rev = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P_rev[i, j] = (pi[j] * P[i, j] + pi[i] * P[j, i]) / (2 * pi[i])
    return P_rev

# Quantify reversibility
def reversibility_score(P):
    P_rev = reversible_component(P)
    frobenius_norm = np.linalg.norm(P_rev - P, 'fro')
    return frobenius_norm, P_rev

# Example
N = 1000
P = random_markov_matrix(N)
score, P_rev = reversibility_score(P)
