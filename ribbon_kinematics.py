# -*- coding: utf-8 -*-
"""
Created on Thu May 22 17:35:02 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

import pickle
import gzip
import glob
import os
import matplotlib.cm as cm

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% data files
files = ['D:/github/ssm_navigation/saved_data/jit_off_tracks3.pkl',
         'D:/github/ssm_navigation/saved_data/str_off_tracks3.pkl',
         'D:/github/ssm_navigation/saved_data/OU_off_tracks3.pkl']

# data = {'post_xy': post_xy, 'post_vxy': post_vxy, 'track_xy': track_xy, 'track_vxy': track_vxy, 'track_signal': track_signal}
# %% load data
post_xys = []
post_vxys = []
track_vxys = []
track_xys = []
track_sigs = []
for ii in range(3):
    with open(files[ii], 'rb') as f:
        data = pickle.load(f)
    post_xys.append(data['post_xy'])
    post_vxys.append(data['post_vxy'])
    track_xys.append(data['track_xy'])
    track_vxys.append(data['track_vxy'])
    track_sigs.append(data['track_signal'])
    
# %% PLOTS
###############################################################################
# example trajectories in and out of the ribbon
# average speed and heading, in and out of odor

# %% speed distribution
track_vxy = track_vxys[1]
track_sig = track_sigs[1]
post_vxy = post_vxys[1]

cond_speed_wi, cond_speed_wo = [],[]
post_speed = []

for ii in range(len(track_vxy)):
    pos_odor = np.where(track_sig[ii]>0)[0]
    speed_i = (np.sum(track_vxy[ii][pos_odor,:]**2,1))**0.5
    cond_speed_wi.append(speed_i)
    pos_no = np.where(track_sig[ii]==0)[0]
    speed_i = (np.sum(track_vxy[ii][pos_no,:]**2,1))**0.5
    cond_speed_wo.append(speed_i)
    
    post_speed.append((np.sum(post_vxy[ii][:,:]**2,1))**0.5)

# %%
plt.figure()
plt.hist(np.concatenate(cond_speed_wi),30, alpha=0.5,density=True)
plt.hist(np.concatenate(cond_speed_wo),30, alpha=0.5,density=True)
# plt.hist(np.concatenate(post_speed),30, alpha=0.5,density=True)
plt.yscale('log')

# %% do this for angles aligned to wind direction
# remove boundary and try polar plots
