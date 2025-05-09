# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 23:36:52 2025

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


# %% chemotaxis relation to SNR
### simple 1D run-and-tumble chemotaxis
### tune SNR of the slope to find transition
### add memory to see improvement

# %% settings
T = 1000
N = 100
v = 1
vs = .5
alpha = 100.1
sigma = 1.
pt_max = 0.15
pt_min = 0.01
beta = 1
h = 1
tau = 1

def prob_turn(x):
    p = (pt_max - pt_min)/(1+np.exp(-x*beta)) + pt_min
    return p

def environment(x, alpha=alpha):
    c = alpha*x + np.random.randn()*sigma
    return c

# %% for correlated space
spatial_corr_scale = 10
# Parameters
x_grid = np.linspace(-T, T, 10000)  # spatial domain
raw_noise = np.random.randn(len(x_grid)) * sigma
correlated_noise = gaussian_filter1d(raw_noise, sigma=spatial_corr_scale)
signal = alpha * x_grid + correlated_noise
# Interpolation function to define the environment
temp_func = interp1d(x_grid, signal, kind='linear', fill_value='extrapolate')

# Total signal: slope + spatially correlated noise
def environment_corr(x,):
    c = temp_func(x) 
    return c

# %% simulation
tracks = np.zeros((N, T))
for nn in range(N):
    track = np.zeros(T)
    track[:h] = np.random.randn()
    dx_prev = 0
    for tt in range(h, T-1):
        # hi = min(tt, 10)  # filtering window (or fix to a constant like 10)
        history = track[tt-h:tt+1]  # past h positions (inclusive of current)
        conc = np.array([environment(x) for x in history])
        # Filtered signal: simple difference (could also use np.convolve)
        if h==1:
            dc = conc[-1] - conc[0] 
        # dc = np.mean(np.diff(conc))
        # w = np.exp(-np.arange(h)[::-1] / tau)
        else:
            w = np.arange(h+1); w = w-np.mean(w)
            dc = np.dot(conc, w)
        # dc = environment(track[tt]) - environment(track[tt-1])
        pt = prob_turn(dc)
        dirr = np.sign(track[tt] - track[tt-1])
        if pt < np.random.rand():
            dx = np.random.choice([-1, 0, 1])* (v + np.random.randn()*vs)
        else:
            dx = dirr*(v + np.random.randn()*vs)
        
        dx = dx_prev + -dx_prev/tau + dx
        dx_prev = dx*1
        track[tt+1] = track[tt] + dx
    tracks[nn,:] = track

plt.figure()
plt.plot(tracks.T, 'k', alpha=.1);

# %%
def sim_RT(alpha, h=1, tau=1):
    tracks = np.zeros((N, T))
    for nn in range(N):
        track = np.zeros(T)
        track[:h] = np.random.randn()
        dx_prev = 0
        for tt in range(h, T-1):
            # hi = min(tt, 10)  # filtering window (or fix to a constant like 10)
            history = track[tt-h:tt+1]  # past h positions (inclusive of current)
            conc = np.array([environment(x, alpha) for x in history])
            # Filtered signal: simple difference (could also use np.convolve)
            if h==1:
                dc = conc[-1] - conc[0] 
            # dc = np.mean(np.diff(conc))
            # w = np.exp(-np.arange(h)[::-1] / tau)
            else:
                w = np.arange(h+1); w = w-np.mean(w)
                dc = np.dot(conc, w)
            # dc = environment(track[tt], alpha) - environment(track[tt-1], alpha)
            pt = prob_turn(dc)
            dirr = np.sign(track[tt] - track[tt-1])
            if pt < np.random.rand():
                dx = np.random.choice([-1, 0, 1])* (v + np.random.randn()*vs)
            else:
                dx = dirr*(v + np.random.randn()*vs)
            
            dx = dx_prev + -dx_prev/tau + dx
            dx_prev = dx*1
            track[tt+1] = track[tt] + dx
        tracks[nn,:] = track
    return tracks

# %% scanning
alphas = np.array([0.01, 0.1, 0.5, 1., 5, 10, 50, 100])
cis = np.zeros((len(alphas), 2))

for ii in range(len(cis)):
    print(ii)
    tracks_wo = sim_RT(alphas[ii], h=1, tau=1)
    pos_wo = np.where(tracks_wo[:,-1]>0)[0]
    cis[ii,0] = len(pos_wo)/N
    tracks_w = sim_RT(alphas[ii], h=1, tau=5)
    pos_w = np.where(tracks_w[:,-1]>0)[0]
    cis[ii,1] = len(pos_w)/N

# %%
plt.figure()
plt.semilogx(alphas, cis[:,0], '-o', label='w/o')
plt.semilogx(alphas, cis[:,1], '-o', label='memory')
plt.semilogx([alphas[0], alphas[-1]], [.5, .5], 'k--')
plt.xlabel('SNR of gradient'); plt.ylabel('chemotaxis index'); plt.legend()

# %%
plt.figure()
plt.semilogx(alphas, cis[:,1] - cis[:,0], '-o', label='benifit')
plt.xlabel('SNR of gradient'); plt.ylabel('benefit');