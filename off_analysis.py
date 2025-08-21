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
files = ['D:/github/ssm_navigation/saved_data/jit_off_tracks3.pkl',
         'D:/github/ssm_navigation/saved_data/space_jitter_2_ribbons2.pkl',
         'D:/github/ssm_navigation/saved_data/time_flicker_2_ribbons2.pkl']
         # 'D:/github/ssm_navigation/saved_data/str_off_tracks3.pkl',
         # 'D:/github/ssm_navigation/saved_data/OU_off_tracks3.pkl']

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
        pos_boundary = np.where((track[:,0]<275) & (track[:,0]>15) & (track[:,1]>15) & (track[:,1]<175))[0]  #### checking this!!! #####
        pos_out = np.where((track[:,0]>275) | (track[:,0]<15) | (track[:,1]<15) | (track[:,1]>175))[0]  #### checking this!!! #####
        ## removal
        # if len(pos_out)>0:
        #     track = track[:pos_out[0],:]
        # else:
        #     track = track[pos_boundary,:]
        
        # ## track-based
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
    return lag_times, msd_mean, counts, sem_msd

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
lls, mms, ccs, sss = [], [], [], []
plt.figure(figsize=(8,6))
for ii in range(3):
    ll,mm,cc,ss = MSD_scaling(post_xys[ii])
    lls.append(ll)
    mms.append(mm)
    ccs.append(cc)
    sss.append(ss)
    # plt.plot(cc)
    plt.plot(ll, mm, '-', color=cols[ii], label=pert_types[ii])
plt.ylabel(r'MSD (mm$^2$)'); plt.xlabel('lag time (s)'); plt.legend(); plt.grid(True)
plt.xlim([0,50]); plt.ylim([0,14000])

# %% plotting
plt.figure()
for ii in range(3):
    plt.plot(lls[ii], mms[ii], '-', color=cols[ii], label=pert_types[ii])
    std_vals = sss[ii] / 1 #np.sqrt(10) #(ccs[ii])
    plt.fill_between(lls[ii], mms[ii] - std_vals, mms[ii] + std_vals, alpha=0.3)
    
plt.ylabel(r'MSD (mm$^2$)'); plt.xlabel('lag time (s)'); plt.legend(); plt.grid(True)
plt.xlim([1,50]); plt.ylim([0,14000]); 
# plt.yscale('log')
# plt.savefig("MSD2.pdf", bbox_inches='tight')

# %% sampling for error bars
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

post_xy = post_xys[2]
xy = 0 ### x:0, y:1
for ii in range(len(post_xy)):
    track_i = post_xy[ii]
    ## exclude boundary touching!!! ####
    pos_boundary = np.where((track_i[:,0]<275) & (track_i[:,0]>5) & (track_i[:,1]>10) & (track_i[:,1]<175))[0]  #### checking this!!! #####
    pos_out = np.where((track_i[:,0]>275) | (track_i[:,0]<5) | (track_i[:,1]<10) | (track_i[:,1]>175))[0]  #### checking this!!! #####
    if len(pos_out)>0:
        track_i = track_i[:pos_out[0],:]
    else:
        track_i = track_i[pos_boundary,:]
        
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

# %% return analysis
time_window = 10*60
time_winds = np.array([1,5,10,15,20,25,30])*60
space_scale = 30 #25*1.5 
p_return = np.zeros((3, len(time_winds)))


for tt in range(len(time_winds)):
    time_window = time_winds[tt]
    for ii in range(1,3):
        n_back, n_tracks = 0,0
        post_xy = post_xys[ii]
        
        for jj in range(len(post_xy)):
            track_i = post_xy[jj]
            #### exclude boundary touching!!! ####
            pos_boundary = np.where((track_i[:,0]<260) & (track_i[:,0]>15) & (track_i[:,1]>15) & (track_i[:,1]<160))[0]
            # track_i = track_i[pos_boundary,:]
            pos_boundary = np.where((track_i[:,0]<270) & (track_i[:,0]>15) & (track_i[:,1]>10) & (track_i[:,1]<170))[0] 
            # pos_boundary = np.where((track_i[:,0]<275) & (track_i[:,0]>5) & (track_i[:,1]>10) & (track_i[:,1]<175))[0]  #### checking this!!! #####
            pos_out = np.where((track_i[:,0]>275) | (track_i[:,0]<5) | (track_i[:,1]<10) | (track_i[:,1]>175))[0]  #### checking this!!! #####
            
            ### removal
            if len(pos_out)>0:
                track_i = track_i[:pos_out[0],:]
            else:
                track_i = track_i[pos_boundary,:]
            
            
            if len(track_i)>time_window:
                xy_final = track_i[time_window,:]
                dists = np.linalg.norm(track_i - xy_final, axis=1)
                exit_check = np.where(dists>space_scale)[0]  ### have to exit
                if len(exit_check)>0:  ### exit or not
                # if True:
                    n_tracks += 1
                    if np.linalg.norm( track_i[0,:] - xy_final ) < space_scale:
                        n_back += 1
        
        p_return[ii, tt] = n_back/n_tracks
        # print(n_tracks)

plt.figure()
plt.plot(time_winds/60, p_return.T)
plt.xlabel('time since last event (s)'); plt.ylabel('P(remain close)')

# plt.savefig("p_return.pdf", bbox_inches='tight')

# %% return analysis, with sampling!
reps = 20
n_samps = 400
time_window = 10*60
time_winds = np.array([1,5,10,15,20,25,30])*60
space_scale = 20 #25*1.5 
p_return = np.zeros((3, len(time_winds), reps))


for rr in range(reps):
    for tt in range(len(time_winds)):
        time_window = time_winds[tt]
        for ii in range(3):
            n_back, n_tracks = 0,0
            post_xy = post_xys[ii]
            
            ### sub-sample
            samp_id = np.random.randint(0,len(post_xy), n_samps)
            post_xy_ = [post_xy[kk] for kk in samp_id]
            
            for jj in range(n_samps):
                track_i = post_xy_[jj]
                #### exclude boundary touching!!! ####
                pos_boundary = np.where((track_i[:,0]<260) & (track_i[:,0]>15) & (track_i[:,1]>15) & (track_i[:,1]<160))[0]
                # track_i = track_i[pos_boundary,:]
                pos_boundary = np.where((track_i[:,0]<250) & (track_i[:,0]>50) & (track_i[:,1]>50) & (track_i[:,1]<150))[0] 
                pos_boundary = np.where((track_i[:,0]<275) & (track_i[:,0]>15) & (track_i[:,1]>15) & (track_i[:,1]<175))[0]  #### checking this!!! #####
                pos_out = np.where((track_i[:,0]>275) | (track_i[:,0]<15) | (track_i[:,1]<15) | (track_i[:,1]>175))[0]  #### checking this!!! #####
                
                ### removal
                if len(pos_out)>0:
                    track_i = track_i[:pos_out[0],:]
                else:
                    track_i = track_i[pos_boundary,:]
                
                
                if len(track_i)>time_window:
                    xy_final = track_i[time_window,:]
                    dists = np.linalg.norm(track_i - xy_final, axis=1)
                    exit_check = np.where(dists>space_scale)[0]  ### have to exit
                    if len(exit_check)>0:  ### exit or not
                    # if True:
                        n_tracks += 1
                        if np.linalg.norm( track_i[0,:] - xy_final ) < space_scale:
                            n_back += 1
            
            p_return[ii, tt, rr] = n_back/n_tracks
            # print(n_tracks)

# %%
# plt.figure()
# for ii in range(3):
#     temp = p_return[ii,:,:]
#     plt.errorbar(time_winds/60, np.mean(temp,1), np.std(temp,1))
plt.figure()
for ii in range(1,3):
    temp = p_return[ii, :, :]
    mean_vals = np.mean(temp, axis=1)
    std_vals = np.std(temp, axis=1)/1 #np.sqrt(n_samps)

    plt.plot(time_winds / 60, mean_vals, label=f'Condition {ii+1}')
    plt.fill_between(time_winds / 60, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3)

plt.legend()
plt.xlabel('time since last event (s)'); plt.ylabel('P(remain close)')

# plt.savefig("p_return2.pdf", bbox_inches='tight')

# %%
###############################################################################
# %% functional
def mean_dwell_time(binary_vector, value=1):
    binary_vector = np.asarray(binary_vector)
    # Find changes
    changes = np.diff(binary_vector.astype(int))
    # Start and end indices of dwell periods
    starts = np.where(changes == 1)[0] + 1 if value == 1 else np.where(changes == -1)[0] + 1
    ends = np.where(changes == -1)[0] + 1 if value == 1 else np.where(changes == 1)[0] + 1

    # Handle case where it starts with value
    if len(binary_vector)>2:
        if binary_vector[0] == value:
            starts = np.insert(starts, 0, 0)
        # Handle case where it ends with value
        if binary_vector[-1] == value:
            ends = np.append(ends, len(binary_vector))
    
        dwell_times = ends - starts
        return np.mean(dwell_times) if len(dwell_times) > 0 else 0
    else:
        return np.nan

def x_persistence(track_xy):
    """
    Compute a scalar measure of persistence along the x-axis.

    Parameters:
        track_xy: (T, 2) array of [x, y] positions over time

    Returns:
        persistence: float — higher means more persistence along x
    """
    if len(track_xy)>0:
        track_xy = np.asarray(track_xy)
        x = track_xy[:, 0]
        y = track_xy[:, 1]
    
        dx = np.abs(x[-1] - x[0])  # end-to-end distance along x
        y_var = np.mean((y - np.mean(y))**2)  # mean square y fluctuation
    
        if y_var == 0:
            return np.inf  # perfectly straight
        else:
            return dx / np.sqrt(y_var)
    else:
        return np.nan
    
def tortuosity(track):
    """
    Compute tortuosity of a trajectory.
    
    Parameters:
        track: (T, 2) or (T, 3) numpy array of positions over time

    Returns:
        tortuosity: float (>= 1), higher = more winding path
    """
    if len(track)>0:
        track = np.asarray(track)
        displacement = np.linalg.norm(track[-1] - track[0])
        segment_lengths = np.linalg.norm(np.diff(track, axis=0), axis=1)
        path_length = np.sum(segment_lengths)
        
        if displacement == 0:
            return np.inf  # completely closed or stationary
        return path_length / displacement
    else:
        return np.nan
    
# %% relating pre to post
data_id = 0
odor_threshold = 50
window = 10*60
search, signal = [], []

# plt.figure()
# for kk in range(3):
#     data_id = kk*1
#     plt.subplot([1,3,kk])
for ii in range(len(post_xys[data_id])):
    track_i = post_xys[data_id][ii]
    track_v = post_vxys[data_id][ii]
    sig_i = track_sigs[data_id][ii]
    pre_vxy_i = track_vxys[data_id][ii]
    pre_xy_i = track_xys[data_id][ii]
    
    ## exclude boundary touching!!! ####
    # pos_boundary = np.where((track_i[:,0]<275) & (track_i[:,0]>5) & (track_i[:,1]>10) & (track_i[:,1]<175))[0]  #### checking this!!! #####
    # pos_out = np.where((track_i[:,0]>275) | (track_i[:,0]<5) | (track_i[:,1]<10) | (track_i[:,1]>175))[0]  #### checking this!!! #####
    # if len(pos_out)>0:
    #     track_i = track_i[:pos_out[0],:]
    # else:
    #     track_i = track_i[pos_boundary,:]
        
    ### process signal
    sig_ii = np.zeros_like(sig_i)
    sig_ii[sig_i<odor_threshold] = 0
    sig_ii[sig_i>0] = 1
    diff = np.diff(sig_ii)
    n_events = len(np.where(diff>0)[0]) + 1
    
    ### process tracks
    
    
    ### simple measurements
    search.append(np.mean(np.sum(track_v[:window, :]**2,1))**0.5)  ### speed
    # search.append(np.mean(track_v[:window, 1]**2)**.5)  ### along one direction
    # search.append(tortuosity(track_i[:window, :]))
    
    # signal.append(mean_dwell_time(sig_ii)/60)  ### dwell time
    # signal.append(np.sum(sig_i)/1)  ### sum odor
    # signal.append(np.mean(np.sum(pre_vxy_i[-window:, :]**2,1)**0.5* 1))  ### prior speed
    # signal.append(np.mean(pre_vxy_i[-window:, 1]**2)**0.5)
    # signal.append(n_events)  ### number of events
    # signal.append(np.sum(sig_ii)/60)  ### time in odor
    signal.append(np.mean(pre_xy_i[:,1]))  ### prior location
    # signal.append(tortuosity(pre_xy_i))
    
    
plt.figure()
plt.plot(signal, search, '.', alpha=0.9);# plt.ylim([0,15])
# plt.xlabel('pre mean speed'); plt.ylabel('post mean speed')
# plt.xlabel('total time in odor (s)'); plt.ylabel('post mean speed')
# plt.xlabel('mean location in y during tracking (mm)'); plt.ylabel('post mean speed')
# plt.xlabel('mean location in y during tracking (mm)'); plt.ylabel('post mean speed'); 
# plt.xlabel('odor encount'); plt.ylabel('tortuosity'); plt.yscale('log'); #plt.xscale('log'); 


# %% raw off-tracks
min_t = 30*60

# for ii in range(3):
ii = 2
plt.figure(ii)
post_xy = post_xys[ii]
for jj in range(0,500,1): #range(len(post_xy)):
    track_i = post_xy[jj]
    #### exclude boundary touching!!! ####
    pos_boundary = np.where((track_i[:,0]<260) & (track_i[:,0]>15) & (track_i[:,1]>15) & (track_i[:,1]<160))[0]
    pos_out = np.where((track_i[:,0]>275) | (track_i[:,0]<5) | (track_i[:,1]<10) | (track_i[:,1]>175))[0]  #### checking this!!! #####
    
    ### removal
    if len(pos_out)>0:
        track_i = track_i[:pos_out[0],:]
    else:
        track_i = track_i[pos_boundary,:]
    if len(track_i)>min_t:   
        track_i = track_i - track_i[0,:][None,:]
        plt.plot(track_i[:,0], track_i[:,1],'k', alpha=0.5)
        plt.plot(track_i[-1,0], track_i[-1,1],'ro', markersize=5)
        plt.xlim([-210,210]); plt.ylim([-140, 140])
        
# plt.savefig("track_time.pdf", bbox_inches='tight')

