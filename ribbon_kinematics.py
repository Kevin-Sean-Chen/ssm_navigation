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
files = ['D:/github/ssm_navigation/saved_data/straight_ribbon_data.pkl',
         'D:/github/ssm_navigation/saved_data/jit_off_tracks3.pkl',
         'D:/github/ssm_navigation/saved_data/str_off_tracks3.pkl',
         'D:/github/ssm_navigation/saved_data/OU_off_tracks3.pkl']

# data = {'post_xy': post_xy, 'post_vxy': post_vxy, 'track_xy': track_xy, 'track_vxy': track_vxy, 'track_signal': track_signal \
#         'rec_signal': rec_signal, 'times': times, 'data4fit':data4fit, 'thetas': thetas}

# %% load data
post_xys = []
post_vxys = []
track_vxys = []
track_xys = []
track_sigs = []
for ii in range(1): #3
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
condition = 0
track_vxy = track_vxys[condition]
track_sig = track_sigs[condition]
post_vxy = post_vxys[condition]

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
nbin = 50
plt.figure()
# plt.hist(np.concatenate(cond_speed_wi),nbin, alpha=0.5,density=True, color='r', label='odor')
plt.hist(np.concatenate(cond_speed_wo),nbin, alpha=0.5,density=True, color='g', label='w/o odor')
plt.hist(np.concatenate(post_speed),nbin, alpha=0.5,density=True, color='b', label='odor-loss')
plt.yscale('log'); plt.legend()

# %% do this for angles aligned to wind direction
# remove boundary and try polar plots
vec_signal = np.concatenate(data['rec_signal'])  # odor signal
vec_time = np.concatenate(data['times'])  # time in trial
vec_theta = np.concatenate(data['thetas'])  # angles
vec_xys = np.concatenate(data['rec_tracks'])  # tracks

# data = {'post_xy': post_xy, 'post_vxy': post_vxy, 'track_xy': track_xy, 'track_vxy': track_vxy, 'track_signal': track_signal \
#         'rec_signal': rec_signal, 'times': times, 'data4fit':data4fit, 'thetas': thetas}

# %% conditional analysis
dur = []
post = []
non = []
dur_ang = []
post_ang = []
non_ang = []
window = 60* 10
window_in = 60*2

for ii in range(len(data['data4fit'])):
    xyi = data['rec_tracks'][ii]
    vxyi = data['data4fit'][ii]
    sigi = data['rec_signal'][ii]
    timei = data['times'][ii]
    thetai = data['thetas'][ii]
    
    ### during odor
    pos_sig = np.where(sigi>0)[0]
    pos_time = np.where((timei>0) & (timei<30))[0]
    pos_bnd = np.where((xyi[:,0]>20) & (xyi[:,0]<270) & (xyi[:,1]>20) & (xyi[:,1]<160))[0]
    # pos = np.intersect1d(pos_sig, pos_bnd)
    pos = np.intersect1d(pos_sig, pos_time)
    if len(pos)>window_in:
        speed_i = (np.sum(vxyi[pos,:]**2,1))**0.5 ### speed
        dur.append(speed_i)
        dur_ang.append(thetai[pos])
    
    ### post odor
    if np.nansum(sigi)>0:  # some odor encounter
        pos_last = np.where(sigi>0)[0][-1]
        remain = len(xyi) - pos_last
        if remain > window:
            post_vxy = vxyi[pos_last:pos_last+window, :]
            post_theta = thetai[pos_last:pos_last+window]
        else:
            post_vxy = vxyi[pos_last:, :]
            post_theta = thetai[pos_last:]
        speed_i = (np.sum(post_vxy**2,1))**0.5 ### speed
        post.append(speed_i) #### condition on having experience! ###################################
        post_ang.append(post_theta)
        
    ### baseline
    pos_sig = np.where(sigi==0)[0]
    pos_time = np.where((timei>40) & (timei<60))[0]
    # pos = np.intersect1d(pos_sig, pos_bnd)
    pos = np.intersect1d(pos_sig, pos_time)
    speed_i = (np.sum(vxyi[pos,:]**2,1))**0.5 ### speed
    non.append(speed_i)
    # 
    # pos = np.where(speed_i>1)[0]
    # pos = np.intersect1d(pos, pos_bnd)
    non_ang.append(thetai[pos_bnd])
    
nbin = 40
# plt.figure()
# plt.hist(np.concatenate(dur),nbin, alpha=0.5,density=True, color='r', label='odor')
# plt.hist(np.concatenate(post),nbin, alpha=0.5,density=True, color='g', label='odor-loss')
# plt.hist(np.concatenate(non),nbin, alpha=0.5,density=True, color='b', label='w/o odor')
# plt.yscale('log'); plt.legend()


hist_dur, bins = np.histogram(np.concatenate(dur), bins=nbin, density=True)
hist_post, _ = np.histogram(np.concatenate(post), bins=bins, density=True)
hist_non, _ = np.histogram(np.concatenate(non), bins=bins, density=True)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
plt.figure()
plt.plot(bin_centers, hist_dur, color='r', label='odor')
plt.plot(bin_centers, hist_post, color='g', label='odor-loss')
plt.plot(bin_centers, hist_non, color='b', label='w/o odor')
plt.xlabel("speed (mm/s)")
plt.ylabel("density")
plt.yscale('log'); plt.legend()

# plt.savefig("speed.pdf", bbox_inches='tight')

# %% angles
nbin = 20
plt.figure()
plt.hist(np.concatenate(dur_ang),nbin, alpha=0.5,density=True, color='r', label='odor')
plt.hist(np.concatenate(post_ang),nbin, alpha=0.5,density=True, color='g', label='odor-loss')
plt.hist(np.concatenate(non_ang),nbin, alpha=0.5,density=True, color='b', label='w/o odor')

# %%
def symmetrize_angles(angles_deg):
    angles = np.asarray(angles_deg) % 360  # Ensure within [0, 360)
    return np.minimum(angles, 360 - angles)

data_ = [symmetrize_angles(np.concatenate(dur_ang)), symmetrize_angles(np.concatenate(post_ang)), symmetrize_angles(np.concatenate(non_ang))]

# Create violin plot
plt.figure(figsize=(6, 5))
plt.violinplot(data_, showmeans=False, showmedians=True, showextrema=False)

# Customize x-axis
plt.xticks([1, 2, 3], ['odor', 'odor-loss', 'w/o odor'])
plt.ylabel("oreientation (deg)")
plt.grid(True)

# plt.savefig("angle.pdf", bbox_inches='tight')

# %%
vec_dur_ang = np.concatenate(dur_ang)
vec_post_ang = np.concatenate(post_ang)
vec_non_ang = np.concatenate(non_ang)

def fraction_up_wind(vec):
    # return len(np.where((vec>150)&(vec<210))[0])/len(vec)
    return len(np.where((vec>90)&(vec<270))[0])/len(vec)

p_upwind = np.array([fraction_up_wind(vec_dur_ang), fraction_up_wind(vec_post_ang), fraction_up_wind(vec_non_ang)])

plt.figure()
plt.bar(['odor', 'odor-loss', 'w/o odor'], p_upwind)
plt.ylabel('P(up-wind)')

# %% example tracks
sig_sum = np.zeros(len(data['data4fit']))

for ii in range(len(data['data4fit'])):
    xyi = data['rec_tracks'][ii]
    vxyi = data['data4fit'][ii]
    sigi = data['rec_signal'][ii]
    timei = data['times'][ii]
    thetai = data['thetas'][ii]
    
    sig_sum[ii] = np.nansum(sigi)

# %%
candidates = np.argsort(sig_sum)[::-1][:100]
min_speed = 3
cnt = 0
# plt.figure()
for ii in range(len(candidates)):
    xyi = data['rec_tracks'][candidates[ii]]
    vxyi = data['data4fit'][candidates[ii]]
    sigi = data['rec_signal'][candidates[ii]]
    pos_out = np.where((xyi[:,0]>275) | (xyi[:,0]<5) | (xyi[:,1]<10) | (xyi[:,1]>175))[0]
    pos_sig = np.where(sigi>0)[0]
    ### removal
    # if len(pos_out)>0:
    #     xyi = xyi[:pos_out[0],:]
    #     vxyi = vxyi[:pos_out[0],:]
    # if np.mean((np.sum(vxyi**2,1))**0.5)  > min_speed:
    #     plt.figure()
    #     plt.title(ii)
    #     plt.plot(xyi[:pos_sig[-1],0], xyi[:pos_sig[-1],1])
    #     plt.plot(xyi[pos_sig[-1]:,0], xyi[pos_sig[-1]:,1],'--')
    #     cnt += 1

# %%
good_exp = np.array([24, 29, 34, 41, 79, 82, 87, 99])

plt.figure()
for ii in range(len(good_exp)):
    xyi = data['rec_tracks'][candidates[good_exp[ii]]]
    vxyi = data['data4fit'][candidates[good_exp[ii]]]
    sigi = data['rec_signal'][candidates[good_exp[ii]]]
    pos_out = np.where((xyi[:,0]>275) | (xyi[:,0]<5) | (xyi[:,1]<10) | (xyi[:,1]>175))[0]
    pos_sig = np.where(sigi>0)[0]
    ### removal
    if len(pos_out)>0:
        xyi = xyi[:pos_out[0],:]
        vxyi = vxyi[:pos_out[0],:]
    if np.mean((np.sum(vxyi**2,1))**0.5)  > min_speed:
        # plt.figure()
        plt.plot(xyi[:pos_sig[-1],0], xyi[:pos_sig[-1],1])
        plt.plot(xyi[pos_sig[-1]:,0], xyi[pos_sig[-1]:,1],'--')
# plt.savefig("track.pdf", bbox_inches='tight')
