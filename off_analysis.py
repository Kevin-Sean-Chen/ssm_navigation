# -*- coding: utf-8 -*-
"""
Created on Thu May  1 12:25:23 2025

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

import numpy.ma as ma
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %% data files
files = ['D:/github/ssm_navigation/saved_data/jit_off_tracks.pkl',
         'D:/github/ssm_navigation/saved_data/str_off_tracks.pkl',
         'D:/github/ssm_navigation/saved_data/OU_off_tracks.pkl']

# %% load data
post_xys = []
post_vxys = []
for ii in range(3):
    with open(files[ii], 'rb') as f:
        data = pickle.load(f)
    post_xys.append(data['post_xy'])
    post_vxys.append(data['post_vxy'])

# %% functional
def MSD_scaling(track_set):
    max_lag = max(len(track) for track in track_set)
    msd = np.zeros(max_lag)
    counts = np.zeros(max_lag)
    msd_std = msd*0

    # Compute MSD
    valid_track = 0
    for track in track_set:
        pos_boundary = np.where((track[:,0]<270) & (track[:,0]>15) & (track[:,1]>10) & (track[:,1]<170))[0] 
        pos_boundary = np.where((track[:,0]<275) & (track[:,0]>5) & (track[:,1]>10) & (track[:,1]<175))[0]  #### checking this!!! #####
        pos_out = np.where((track[:,0]>275) | (track[:,0]<5) | (track[:,1]<10) | (track[:,1]>175))[0]  #### checking this!!! #####
        ### removal
        # if len(pos_out)>0:
        #     track = track[:pos_out[0],:]
        # else:
        #     track = track[pos_boundary,:]
        
        ### track-based
        # if len(pos_boundary)==len(track):  #### only use tracks within
        #     valid_track += 1
        
        n_points = len(track)//1### truncation here
        for lag in range(1, n_points):
            displacements = track[lag:,:] - track[:-lag,:]  # Displacements for this lag
            squared_displacement = np.sum(displacements**2)#, axis=1)  # (dx^2 + dy^2)
            msd[lag] += np.sum(squared_displacement)  # Sum displacements
            counts[lag] += len(displacements)  # Count valid pairs
            msd_std[lag] += np.sum(squared_displacement**2)
            
    # Normalize to get the average MSD for each lag
    print(valid_track)
    msd_mean = msd / counts
    variance_msd = (msd_std / counts) - (msd_mean**2)
    sem_msd = np.sqrt(variance_msd) / counts**0.5 * 1
    lag_times = np.arange(max_lag)*1/60  # Lag times
    return lag_times, msd_mean, counts #sem_msd

def MSD_compute(track_set, tau):
    msd = 0
    counts = 0
    msd_std = msd*0

    # Compute MSD
    for track in track_set:
        n_points = len(track)//2 ### truncation here
        if n_points>=tau:
            lag = tau*1
            displacements = track[lag:,:] - track[:-lag,:]  # Displacements for this lag
            squared_displacement = np.sum(displacements**2)#, axis=1)  # (dx^2 + dy^2)
            msd += np.sum(squared_displacement)  # Sum displacements
            counts += len(displacements)  # Count valid pairs
            msd_std += np.sum(squared_displacement**2)
    # print(counts)
    # Normalize to get the average MSD for each lag
    msd_mean = msd / counts
    return  msd_mean

# %% sample MSD
pert_types = ['space', 'static', 'time']
cols = ['k','r', 'b']
plt.figure(figsize=(8,6))
for ii in range(3):
    ll,mm,cc = MSD_scaling(post_xys[ii])
    # plt.plot(cc)
    plt.plot(ll, mm, 'o', color=cols[ii], label=pert_types[ii])
plt.ylabel(r'MSD (mm$^2$)'); plt.xlabel('lag time (s)'); plt.legend(); plt.grid(True)

# %% sampling
reps = 50
n_samps = 100
msds = np.zeros((reps, 3))
for rr in range(reps):
    for ii in range(3):
        samples = np.random.choice(len(post_xys[ii]), n_samps, replace=False)
        samp_xys = post_xys[ii]
        subset = [samp_xys[i] for i in samples]
        msds[rr,ii] = MSD_compute(subset, 60*31)

plt.figure()
plt.bar(pert_types, np.mean(msds,0))
plt.errorbar(pert_types, np.mean(msds,0), yerr=np.std(msds,0), fmt='o', capsize=5, color='black') 
plt.ylabel(r'MSD (mm$^2$)')

# %% visualization
import seaborn as sns

# Sample scatter data
post_xy = post_xys[0]
exclude_t = 1
x = np.concatenate(([xy[60*exclude_t:,0]-xy[0,0] for xy in post_xy]))
y = np.concatenate(([xy[60*exclude_t:,1]-xy[0,1] for xy in post_xy]))

down_samp = 30
x,y = x[::down_samp], y[::down_samp]

# plt.figure(figsize=(8, 6))
# sns.kdeplot(x=x, y=y, cmap="viridis", fill=True, thresh=0, bw_method='silverman')
# plt.scatter(x, y, s=10, color="black", alpha=0.1)  # Optional: overlay scatter points
# plt.colorbar(label="Density")
fig, ax = plt.subplots(figsize=(8, 6))
# Create 2D KDE with fill and color mapping
kde = sns.kdeplot(
    x=x, y=y,
    fill=True, 
    cmap="viridis",
    ax=ax
)
# Attach colorbar to the contour set
mappable = kde.collections[0]
plt.colorbar(mappable, ax=ax)
plt.xlabel('x since off (mm)'); plt.ylabel('y since off (mm)')
plt.title("Smoothed Density Plot")
# plt.xlim([x.min(), x.max()]); plt.ylim([y.min(), y.max()])
plt.xlim([-100, 100]); plt.ylim([-100, 100])

# %% test log density
from scipy.stats import gaussian_kde
xy = np.vstack([x, y])

# Estimate density
kde = gaussian_kde(xy)

# Evaluate on a grid
xmin, xmax = -250, 250 #x.min()-1, x.max()+1
ymin, ymax = -250, 250 #y.min()-1, y.max()+1
X, Y = np.meshgrid(np.linspace(xmin, xmax, 100),
                   np.linspace(ymin, ymax, 100))
positions = np.vstack([X.ravel(), Y.ravel()])
Z = kde(positions).reshape(X.shape)

# Convert to log-density (avoid log(0))
Z_log = np.log10(Z + 1e-10)

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
cset = ax.contourf(X, Y, Z_log, cmap='viridis')
cb = plt.colorbar(cset, ax=ax)
plt.plot(0,0,'r*')
cb.set_label("Log Density")
# ax.set_title("2D KDE Log-Density")
ax.set_xlabel('x since off (mm)'); ax.set_ylabel('y since off (mm)')
plt.tight_layout()
plt.show()


# %% along one dimension
bins = 10 #
exclude_t = 20
wind = 1
bins = np.arange(-150,150,20)
# bins = np.arange(-100,100,5)
# bins = 14
for ii in range(3):
    x,y = [],[]
    www = []
    post_xy = post_xys[ii]
    
    for jj in range(len(post_xy)):
        track_i = post_xy[jj]
        #### exclude boundary touching!!! ####
        pos_boundary = np.where((track_i[:,0]<260) & (track_i[:,0]>15) & (track_i[:,1]>15) & (track_i[:,1]<160))[0]
        # track_i = track_i[pos_boundary,:]
        pos_boundary = np.where((track_i[:,0]<270) & (track_i[:,0]>15) & (track_i[:,1]>10) & (track_i[:,1]<170))[0] 
        pos_boundary = np.where((track_i[:,0]<275) & (track_i[:,0]>5) & (track_i[:,1]>10) & (track_i[:,1]<175))[0]  #### checking this!!! #####
        pos_out = np.where((track_i[:,0]>275) | (track_i[:,0]<5) | (track_i[:,1]<10) | (track_i[:,1]>175))[0]  #### checking this!!! #####
        
        ### removal
        if len(pos_out)>0:
            track_i = track_i[:pos_out[0],:]
        else:
            track_i = track_i[pos_boundary,:]
            
        ### track-based
        # if len(pos_boundary)==len(track_i):  #### only use tracks within
        # if len(track_i)>exclude_t:
            x.append(track_i[60*exclude_t:,0]-track_i[0,0])  ##### RETURNING??? #####
            y.append(track_i[60*exclude_t:,1]-track_i[0,1])
            www.append(np.ones(len(track_i[60*exclude_t:,0]))*len(track_i[60*exclude_t:,0]))
            # x.append(track_i[:60*exclude_t,0]-track_i[0,0])
            # y.append(track_i[:60*exclude_t,1]-track_i[0,1])
    
    x = np.concatenate(x)
    y = np.concatenate(y)
    www = np.concatenate(www)
    # x = np.concatenate(([xy[60*exclude_t:,0]-xy[0,0] for xy in post_xy]))  ##### change to be AFTER a while #####
    # y = np.concatenate(([xy[60*exclude_t:,1]-xy[0,1] for xy in post_xy]))
    
    # plt.figure()
    dim = y*1 # y*1, x*1
    aa,bb = np.histogram(dim, bins, weights=www)
    print(np.std(dim))
    plt.plot((bins[1:]+bins[:-1])/2, aa/np.sum(aa), label=pert_types[ii])
    # plt.plot((bb[1:]+bb[:-1])/2, aa/np.sum(aa), label=pert_types[ii])
    # plt.yscale('log')
    plt.legend()
    plt.xlabel('position since off (mm) along x')
    
    
# %% diffusion plots
bin_t = np.array([ 0.5, 1, 2, 4, 8, 16, 32])*60
bin_x = np.arange(-150,150,20)
density_xt = np.zeros((len(bin_t), len(bin_x)))  # space-time density since off

post_xy = post_xys[1]
xy = 1 ### x:0, y:1
for ii in range(len(post_xy)):
    track_i = post_xy[ii]
    ## exclude boundary touching!!! ####
    # pos_boundary = np.where((track_i[:,0]<275) & (track_i[:,0]>5) & (track_i[:,1]>10) & (track_i[:,1]<175))[0]  #### checking this!!! #####
    # pos_out = np.where((track_i[:,0]>275) | (track_i[:,0]<5) | (track_i[:,1]<10) | (track_i[:,1]>175))[0]  #### checking this!!! #####
    # if len(pos_out)>0:
    #     track_i = track_i[:pos_out[0],:]
    # else:
    #     track_i = track_i[pos_boundary,:]
        
    ### space-time condition
    for tt in range(len(bin_t)):
        t_till = int(bin_t[tt])
        if len(track_i)>t_till:
            pos_end = track_i[t_till, xy] - track_i[0, xy]
            idx = np.searchsorted(bin_x, pos_end, side='right') - 1
            density_xt[tt, idx] += 1

density_xt = density_xt/np.sum(density_xt,1)[:,None]

## %%
x,y = bin_x, bin_t/60
plt.figure()
# plt.imshow((density_xt), extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', origin='lower')
plt.imshow(np.log(density_xt), extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', origin='lower', interpolation='bilinear',cmap='viridis')
# plt.plot(np.log(density_xt.T))
# contour = plt.contour(x,y, np.log(density_xt), colors='b', linewidths=1)
# plt.clabel(contour, inline=True, fontsize=8)
plt.colorbar(label='log P(x|t)')
plt.xlabel("x")
plt.ylabel("t")          

## %% color diffusion plot
matrix = np.log(density_xt*1)

# Get a colormap (viridis) and map row indices to colors
cmap = cm.get_cmap('viridis', matrix.shape[0])  # N colors
colors = cmap(np.linspace(0, 1, matrix.shape[0]))

plt.figure()
# Plot each row with a different color
for i in range(matrix.shape[0]):
    plt.plot(bin_x, matrix[i], color=colors[i])

plt.ylim([-6,0.3])
plt.xlabel("y")
plt.ylabel("P(y|t)")
plt.show()


# %%
###############################################################################
# %% actions
spd_bin = np.linspace(0, 30, 50)
process_post = []
process_pre = []
window = 10*60
for ii in range(len(post_vxy)):
    if len(post_vxy[ii])<window:
        process_post.append(post_vxy[ii])
    else:
        process_post.append(post_vxy[ii][:window, :])
for ii in range(len(pre_vxy)):
    process_pre.append(pre_vxy[ii])
    
post_action = np.concatenate(process_post)#[:,0]
post_action = np.sum(post_action**2,1)**0.5
pre_action = np.concatenate(process_pre)#[:,0]
pre_action = np.sum(pre_action**2,1)**0.5
plt.figure()
plt.hist(post_action, bins=spd_bin, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.hist(pre_action, bins=spd_bin, density=True, alpha=0.7, color='r', edgecolor='black')

full_action = np.sum(vec_vxy**2,1)**0.5
# plt.hist(full_action, bins=spd_bin, density=True, alpha=0.5, color='k', edgecolor='black')
plt.xlim([-.5, 20]); plt.ylim([0,0.9])

# %% MSD analysis!

# sortt_id = np.array(sortt_id, dtype=int)
# track_set = post_xy[sortt_id[:len(sortt_id)//2]]  ## compare sorted
# track_set = [post_xy[i] for i in sortt_id[:len(sortt_id)//1]]
track_set = post_xys[2]
# track_set = [post_xy[i] for i in sortt_id[len(sortt_id)//3:-len(sortt_id)//3]]
# track_set = [post_xy[i] for i in sortt_id[-len(sortt_id)//3:]]
max_lag = max(len(track) for track in track_set)
# max_lag = int(10*1/(1/60))
# Initialize arrays for MSD and counts
msd = np.zeros(max_lag)
counts = np.zeros(max_lag)
msd_std = msd*0

# Compute MSD
for track in track_set:
    n_points = len(track)//2 ### truncation here
    for lag in range(1, n_points):
        displacements = track[lag:,:] - track[:-lag,:]  # Displacements for this lag
        squared_displacement = np.sum(displacements**2)#, axis=1)  # (dx^2 + dy^2)
        ##############################################################3 TRY <X^2> vs. <R^2>!!!
        msd[lag] += np.sum(squared_displacement)  # Sum displacements
        counts[lag] += len(displacements)  # Count valid pairs
        msd_std[lag] += np.sum(squared_displacement**2)
# Normalize to get the average MSD for each lag

# %%
msd_mean = msd / counts
variance_msd = (msd_std / counts) - (msd_mean**2)
sem_msd = np.sqrt(variance_msd) / counts**0.5 * 1
lag_times = np.arange(max_lag)*1/60  # Lag times

# %%
# Plot MSD
plt.figure(figsize=(8, 6))
plt.plot(lag_times, msd_mean, marker='o', linestyle='-', color='r', label='fluc')
# plt.loglog(lag_times_mid, msd_mean_mid, marker='o', linestyle='-.', color='r', label='middle')
# plt.loglog(lag_times_last, msd_mean_last, marker='o', linestyle='--', color='b', label='short')
# plt.loglog(lag_times, lag_times**2 + msd_mean[1], marker='o', linestyle='-', color='g')
# plt.fill_between(
#     lag_times,
#     msd_mean - sem_msd,  # Lower bound of shaded region
#     msd_mean + sem_msd,  # Upper bound of shaded region
#     color='red',
#     alpha=0.5,)
#     # label='1 Std Dev')
plt.xlabel("lag time (s)")
plt.ylabel(r"MSD (mm$^2$)")
plt.title("MSD scaling")
plt.grid(True)
# plt.xlim([0,90]); plt.ylim([0.001, 20000])
# plt.show()
plt.legend()