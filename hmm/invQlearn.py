# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:11:59 2025

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

import h5py

# %% from plume navigation data, define states, then compute inverse Q-learning
### then simulate data constrained RL agent
### then replace states with experimental modes
### then extend to time varying states...
### does it say something about expectations?

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
rec_vxy = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
masks = []   # where there is nan
track_ids = []  # record track id (file and track)
rec_tracks = []  # record the full track x,y
rec_signal = []  # record opto signal
times = []   # record time in epoch
speeds = []
thetas = []

for tr in range(n_tracks):
    print(tr)
    ### extract features
    pos = np.where(trjNum==track_id[tr])[0]  # position of this track
    temp_xy = np.column_stack((x_smooth[pos] , y_smooth[pos]))
    temp_vxy = np.column_stack((vx_smooth[pos] , vy_smooth[pos]))
    
    ### recording
    rec_vxy.append(temp_vxy)  # get data for ssm fit
    rec_tracks.append(temp_xy)  # get raw tracksd
    track_ids.append(np.zeros(len(pos))+tr) 
    rec_signal.append(signal[pos])
    speeds.append(speed_smooth[pos])
    thetas.append(dtheta_smooth[pos])


# %% build track-based state and actions
states = []
actions = []
odor_threshold = 3
angle_threshold = 70
speed_threshold = 2

for tr in range(n_tracks):
    
    # load track
    odori = rec_signal[tr]
    speedi = speeds[tr]
    thetai = thetas[tr]
    vxyi = rec_vxy[tr]
    
    statei = np.zeros(len(odori))
    actioni = statei*1
    
    ### assign states
    pos_odor = np.where(odori>odor_threshold)[0]
    pos_no_odor = np.where(odori<odor_threshold)[0]
    pos_up = np.where(vxyi[:,0]<0)[0]
    pos_down = np.where(vxyi[:,0]>=0)[0]
    
    statei[np.intersect1d(pos_odor, pos_up)] = 1
    statei[np.intersect1d(pos_odor, pos_down)] = 2
    statei[np.intersect1d(pos_no_odor, pos_up)] = 3
    statei[np.intersect1d(pos_no_odor, pos_down)] = 4
    
    ### assign actions
    pos_stop = np.where(speedi<speed_threshold)[0]
    pos_walk = np.where(speedi>=speed_threshold)[0]
    pos_turn_left = np.where(thetai>angle_threshold)[0]
    pos_turn_right = np.where(thetai<-angle_threshold)[0]
    
    actioni[pos_stop] = 1
    actioni[pos_walk] = 2
    actioni[pos_turn_left] = 3
    actioni[pos_turn_right] = 4
    
    ### record
    states.append(statei)
    actions.append(actioni)

# %% tabular inverse Q-learning!
nS, nA = 4,4
epochs = 100
n_tracks = len(states)
n_samples = 30
### learning and discount parameters
alpha_r = 0.001 #0.0001
alpha_q = 0.1 #0.01
alpha_sh = 0.1 #0.01
gamma = 0.9
epsilon = 1e-6
# initialize tables for reward function, value functions and state-action visitation counter.
r = np.zeros((nS, nA))
q = np.zeros((nS, nA))
q_sh = np.zeros((nS, nA))
state_action_visitation = np.zeros((nS, nA))

for ep in range(epochs):
    if ep%10 == 0:
        print("Epoch %s/%s" %(ep+1, epochs))
   
    samples = np.random.choice(n_tracks, size=n_samples, replace=False)
    for kk in range(n_samples):
        traj = samples[kk]
        s_i = states[traj][:-1]
        a_i = actions[traj][:-1]
        ns_i = states[traj][1:]  # next states
        lt_i = len(s_i)
        
        for ii in range(lt_i):
            s,a,ns = int(s_i[ii]-1), int(a_i[ii]-1), int(ns_i[ii]-1)  ### states to index
            
            state_action_visitation[s][a] += 1
            d = False   # no terminal state

            # compute shifted q-function.
            q_sh[s, a] = (1-alpha_sh) * q_sh[s, a] + alpha_sh * (gamma * (1-d) * np.max(q[ns]))
            
            # compute log probabilities.
            sum_of_state_visitations = np.sum(state_action_visitation[s])
            log_prob = np.log((state_action_visitation[s]/sum_of_state_visitations) + epsilon)
            
            # compute eta_a and eta_b for Eq. (9).
            eta_a = log_prob[a] - q_sh[s][a]
            other_actions = [oa for oa in range(nA) if oa != a]
            eta_b = log_prob[other_actions] - q_sh[s][other_actions]
            sum_oa = (1/(nA-1)) * np.sum(r[s][other_actions] - eta_b)

            # update reward-function.
            r[s][a] = (1-alpha_r) * r[s][a] + alpha_r * (eta_a + sum_oa)

            # update value-function.
            q[s, a] = (1-alpha_q) * q[s, a] + alpha_q * (r[s, a] + gamma * (1-d) * np.max(q[ns]))
            s = ns

# %%
# compute Boltzmann distribution.
boltzman_distribution = []
for s in range(nS):
    boltzman_distribution.append([])
    for a in range(nA):
        boltzman_distribution[-1].append(np.exp(q[s][a]))
boltzman_distribution = np.array(boltzman_distribution)
boltzman_distribution /= np.sum(boltzman_distribution, axis=1).reshape(-1, 1)

# %% analyze the inferred values
plt.figure(figsize=(6, 5))
plt.imshow(boltzman_distribution, cmap='viridis')

x_labels = ['stop', 'walk', 'turn left', 'turn right']
y_labels = ['odor upwind', 'odor downwind', 'non upwind', 'non downwind']

# Set custom tick labels
plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)
plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)

# Optionally show the values
plt.colorbar(label="P(a|s)")