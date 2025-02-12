# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:38:47 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from autograd import grad
# import autograd.numpy as np
from scipy.linalg import hankel

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import hankel

### get transfer entropy function
import sys
import os

# Get the absolute path of the "util" folder
utilities_path = os.path.abspath("utils")
sys.path.append(utilities_path)
from transfer_entropy import transfer_entropy

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% minimal model for navigation (run-and-tumble); then test mode discovery methods
### try it with Markov model
### try it with DMD with control
### later try with RNN for dynamic modes

# %% parameter setup
dt = 1   # characteristic time scale
t_M = 5  # memory
N = 1  # receptor gain
H = 1  # motor gain
F0 = 0.  # adapted internal state
v0 = 1  # run speed
vn = .1

target = np.array([100,100])  ### location of source
tau_env = 10   ### fluctuation time scale of the environment
C0 = 100
sigC = 2

lt = 5000   #max time lapse
eps = 5  # criteria to reach sourc

# %% kinematics
def r_F(F):
    return 1/(1+np.exp(-H*F))+.0  # run rate

def tumble(r):
    p = np.random.rand()
    if r>p:
        angle = np.random.randn()*.1  # run
        tumb = 0
    else:
        angle = (np.random.rand()*2-1)*np.pi  # tumble
        tumb = 1
    return angle, tumb

def theta2state(pos, theta):
    """
    take continuous angle dth and put back to discrete states
    """
    dv = v0 + np.random.randn()*vn  # draw speed
    vec = np.array([np.cos(theta)*dv , np.sin(theta)*dv] )
    pos = pos + vec #np.array([dx, dy])
    return pos, vec
    
# %% factorized environment
def temporal_ou_process(lt, tau, A, dt=0.01):
    n_steps = lt*1  # Number of steps
    x = np.ones(n_steps)  # OU process values
    
    # Variance of the noise
    sigma = A * np.sqrt(2 / tau)
    
    for i in range(1, n_steps):
        # Update step for OU process
        x[i] = x[i-1] - ((x[i-1]-1) / tau) * dt + sigma * np.sqrt(dt) * np.random.randn()
    
    return x

# Parameters
tau = 10.0      # Correlation time
A = 0.5        # Amplitude
# Simulate and plot the process
fluctuation_t = temporal_ou_process(lt, tau, A)#*.1 + 1
plt.figure()
plt.plot(fluctuation_t)

def dist2source(x):
    # dist = np.sum( (x-target)**2 )**0.5  ### for point source
    dist = np.sum( (x[0]-target[0])**2 )**0.5  ### for one direction
    return dist
    
def env_space(x, tt=-1):
    # C = np.exp(-np.sum((x - target)**2)/sigC**2)*C0 + np.random.randn()*0.1  ### point source
    C = C0*np.exp(x[0]/sigC) + np.random.randn()*0.1  ### exp graident
    if tt==-1:
        return np.max([C,0.1])
    else:
        return np.max([C*fluctuation_t[tt],0.1])

def env_space_xy(x,y):
    # C = np.exp(-np.sum((x - target)**2)/sigC**2)*C0
    # C = -((x - target[0])**2 / (sigC**2) + (y - target[1])**2 / (sigC**2))
    C = C0*np.exp(x[0]/sigC) + y*0
    return C

def env_time(x):
    return 1   ### let's not have environmental dynamics for now

def gaussian(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1):
    return np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))

gradient = grad(env_space, argnum=0)
def grad_env(x, u_x, tt=-1):
    grad_x = gradient(x, tt)
    if np.linalg.norm(grad_x)==0:
        percieved = C0
    else:
        percieved = np.dot(grad_x/np.linalg.norm(grad_x), u_x/(1))  # dot product between motion and local gradient
    # print(np.linalg.norm(grad_x))
    return percieved

# # Example point
# inputs = np.array([1.0, 1.5])
# grad_values = gradient(inputs)

# %% setup timee
xys = []
cs = []
vecs = []
Fs = []
tumbles = []
pos_t = np.random.randn(2)  # random init location
vec_t = np.random.randn(2)  # random init direction
df_dt = np.random.randn()     # init internal state
theta = 0
tt = 0

while tt<lt and dist2source(pos_t)>eps:
    ### compute impulse
    # d_phi = grad_env(pos_t, vec_t, tt)
    d_phi = np.log(env_space(pos_t+vec_t)) - np.log(env_space(pos_t))  ### for simple log sensing
    ### internal dynamics
    df_dt = df_dt + dt*(-1/t_M*(df_dt - F0) + d_phi)
    ### draw actions
    r_t = r_F(df_dt)*dt
    dth,tumb_t = tumble(r_t)
    ### make movements
    theta = theta + dth
    new_pos, new_vec = theta2state(pos_t, theta)
    ### record
    xys.append(new_pos)
    # cs.append(env_space(pos_t))
    # cs.append(np.log(env_space(pos_t)))
    cs.append(d_phi)
    # cs.append(env_space(new_pos) - env_space(pos_t))
    vecs.append(new_vec)
    Fs.append(df_dt)
    tumbles.append(tumb_t)
    ### update
    pos_t, vec_t = new_pos*1, new_vec*1
    tt += 1

def gen_tracks():
    eps = 1  # criteria to reach source
    xys = []
    cs = []
    vecs = []
    Fs = []
    tumbles = []
    pos_t = np.random.randn(2)  # random init location
    vec_t = np.random.randn(2)  # random init direction
    df_dt = np.random.randn()     # init internal state
    theta = 0
    tt = 0

    while tt<lt and dist2source(pos_t)>eps:
        ### compute impulse
        # d_phi = grad_env(pos_t, vec_t)
        d_phi = np.log(env_space(pos_t+vec_t)) - np.log(env_space(pos_t))  ### for simple log sensing
        ### internal dynamics
        df_dt = df_dt + dt*(-1/t_M*(df_dt - F0) + d_phi)
        ### draw actions
        r_t = r_F(df_dt)*dt
        dth,tumb_t = tumble(r_t)
        ### make movements
        theta = theta + dth
        new_pos, new_vec = theta2state(pos_t, theta)
        ### record
        xys.append(new_pos) ### old or new ####
        ### choose input
        # cs.append(env_space(pos_t))
        # cs.append(np.log(env_space(pos_t)))
        cs.append(d_phi)
        # cs.append(env_space(new_pos) - env_space(pos_t))
        ####
        vecs.append(new_vec)
        Fs.append(df_dt)
        tumbles.append(tumb_t)
        ### update
        pos_t, vec_t = new_pos*1, new_vec*1
        tt += 1
    ### vectorize
    vec_xy = np.array(xys)
    vec_cs = np.array(cs)
    vec_Fs = np.array(Fs)
    vec_vxy = np.array(vecs)
    vec_tumb = np.array(tumbles)
    return vec_xy, vec_cs, vec_Fs, vec_tumb, vec_vxy

### simple test
plt.figure()
plt.plot(np.array(xys)[:,0], np.array(xys)[:,1])

# %% measure track statistics
reps = 100
tracks_dic = {}
behavior = []
stimuli = []
ct = 0
for rr in range(reps):
    vec_xy, vec_cs, vec_Fs, vec_tumb, vec_vxy = gen_tracks()
    if len(vec_xy)<lt:  ### if reached goal...
        tracks_dic[ct] = {'xy':vec_xy, 'cs':vec_cs, 'Fs':vec_Fs, 'tumb': vec_tumb, 'vxy': vec_vxy}
        behavior.append(vec_vxy)
        stimuli.append(vec_cs)
        ct += 1
    print(rr)
behavior = np.concatenate((behavior),0)
stimuli = np.concatenate((stimuli))

# %% plot tracks
plt.figure()
x_range = np.linspace(-10, 140, 150)  # 100 points along the x-axis
y_range = np.linspace(-160, 140, 300)  # 100 points along the y-axis

# Create meshgrid for the grid points
X, Y = np.meshgrid(x_range, y_range)

# Compute Z values for each grid point
Z = env_space_xy(X, Y)
plt.imshow(np.log(Z), origin='lower', extent=[-10, 150, -150, 150])
for ii in range(len(tracks_dic)):
    temp = tracks_dic[ii]['xy']
    plt.plot(temp[:,0], temp[:,1])
    
    
# %%
###############################################################################
# %% let's measure TE first!
print('S->B', transfer_entropy(behavior[:,0], stimuli[:], 1))
print('B->S', transfer_entropy(stimuli[:], behavior[:,0], 1))

# %% Max-pred Markov here!!!
K_star = 30
N_star = 500
tau_star = 1

# %%
def build_X(data, return_id=False, K=K_star , tau=tau_star, joint_stim=False, dic='vxy'):
    K = int(K)
    features = []
    ids = []
    n_tracks = len(data)
    for tr in range(n_tracks):
        datai = data[tr][dic]
        T = len(datai)
        samp_vec = datai[1:-np.mod(T,K)-1,:]
        if joint_stim is not False:
            stimi = data[tr]['cs']
            samp_stim = stimi[1:-np.mod(T,K)-1]
        for tt in range(0, len(samp_vec)-K, tau):
            vx = samp_vec[tt:tt+K, 0]
            vy = samp_vec[tt:tt+K, 1]
            vx_windowed = vx.reshape(-1, K)
            vy_windowed = vy.reshape(-1, K)
            if joint_stim is not False:
                csi = samp_stim[tt:tt+K][None,:]
                features.append(np.hstack((vx_windowed, vy_windowed, csi)))   ### might have problems here across tracks!!!    
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

X,ids = build_X(tracks_dic, return_id=True, joint_stim=True)#, use_dtheta=True)

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

# %% spectrum
X_traj, track_id = build_X(tracks_dic, return_id=True, joint_stim=True)  ### try joint stim!
labels, centrals = kmeans_knn_partition(X_traj, N_star, return_centers=True)
P = compute_transition_matrix(labels, track_id, N_star)

# %% get reverisble matrix
top_n_values = 10
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

sub_samp = np.random.choice(X_traj.shape[0], 2000, replace=False)
reducer = umap.UMAP(n_components=3, random_state=42)
data_2d = reducer.fit_transform(X_traj[sub_samp,:])

# %% show in 3D
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
color_abs = np.max(np.abs(phi2[sub_samp]))
# sc = plt.scatter(data_2d[:,0], data_2d[:,1], c=phi2[sub_samp], cmap='coolwarm', s=.1, vmin=-color_abs, vmax=color_abs)
sc = ax.scatter(data_2d[:,0], data_2d[:,1],data_2d[:,2], c=phi2[sub_samp], cmap='coolwarm', s=12, vmin=-color_abs, vmax=color_abs)
plt.colorbar(sc)

# %% spectral analysis
P_shuff = compute_transition_matrix(np.random.permutation(labels),track_id, N_star)
uu,vv = np.linalg.eig(P_shuff)  #P_shuff
uu,vv = np.linalg.eig(P)
idx = uu.argsort()[::-1]  # Get indices to sort eigenvalues
sorted_eigenvalues = np.real(uu[idx])
plt.figure()
plt.plot((-1/1*tau_star)/np.log(sorted_eigenvalues[1:30]),'-o')
plt.ylabel('relaxation time (s)')
plt.xlabel('eigenvalue index')

# %% color code tracks]
imode = 1
phi2 = -eigvecs[labels,imode].real
window_show = np.arange(1,len(X_traj)//7,1)
X_xy, track_id = build_X(tracks_dic, return_id=True, dic='xy')
xy_back = X_xy[:, [0,int(K_star)]]
plt.figure()
plt.scatter(xy_back[window_show, 0],xy_back[window_show, 1],c=phi2[window_show],cmap='coolwarm',s=.9,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}')

plt.figure()
plt.scatter(window_show, X_traj[window_show, -K_star],c=phi2[window_show],cmap='coolwarm',s=.9,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}'); plt.xlabel('time steps'); plt.ylabel('log(c\'/c)')

# %% analyze modes
pos = np.where(phi2>.0)[0]
plt.figure()
plt.plot(np.mean(X_traj[pos, -K_star:],0))
plt.xlabel('time steps'); plt.ylabel('average input'); plt.title(r'input | ($\phi$)<0')
