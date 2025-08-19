# -*- coding: utf-8 -*-
"""
Created on Sat May 31 21:20:20 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

import pickle
import sys, os
import scipy.io as spio

import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)


# %% load Nirag's data and Gustavo's
pkl_file = r'C:\Users\ksc75\Yale University Dropbox\users\nirag_kadakia\data\ellipsoid-body-navigation\rec_videos\E-PG-kir\exp_mat.pkl'
with open(pkl_file, 'rb') as f:
    pkl_data = pickle.load(f)

# %%
mat_file = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/EPG_analysis/combined_EPG_EA10.mat'
mat_data_full = spio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)

### use this another time...
# from scipy.io import loadmat
# import scipy.io.matlab

# def _check_keys(dict_in):
#     """
#     Recursively convert mat_structs to nested dictionaries.
#     """
#     for key in dict_in:
#         if isinstance(dict_in[key], scipy.io.matlab.mio5_params.mat_struct):
#             dict_in[key] = _todict(dict_in[key])
#     return dict_in

# def _todict(matobj):
#     """
#     Convert mat_struct to dictionary.
#     """
#     d = {}
#     for strg in matobj._fieldnames:
#         elem = getattr(matobj, strg)
#         if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
#             d[strg] = _todict(elem)
#         elif isinstance(elem, np.ndarray):
#             d[strg] = _tolist(elem)
#         else:
#             d[strg] = elem
#     return d

# def _tolist(ndarray):
#     """
#     Recursively convert cell arrays (ndarrays) to lists of dicts.
#     """
#     elem_list = []
#     for sub_elem in ndarray:
#         if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
#             elem_list.append(_todict(sub_elem))
#         elif isinstance(sub_elem, np.ndarray):
#             elem_list.append(_tolist(sub_elem))
#         else:
#             elem_list.append(sub_elem)
#     return elem_list

# # --- Load and convert ---
# data = loadmat(mat_file, struct_as_record=False, squeeze_me=True)
# data = _check_keys(data)
####
# %% refactoring to extract trjNum, speed, theta, signal
expmat = mat_data_full['expmat']
select = np.array([0, 6,7, 10, 11, 15, 23])
mat_data = expmat[:, select] ### trkNum, x,y,speed, theta, signal

    #           trjNum: 1
    #          vialNum: 2
    #         trialNum: 3
    #        fileindex: 4
    #      trjNumVideo: 5
    #                t: 6
    #                x: 7
    #                y: 8
    #               vx: 9
    #               vy: 10
    #            speed: 11
    #            theta: 12
    #                a: 13
    #                b: 14
    #             area: 15
    #           signal: 16
    #           dtheta: 17
    #         frameNum: 18
    #           waldir: 19
    #          dwaldir: 20
    #              fps: 21
    #        mm_per_px: 22
    #               sx: 23
    #               sy: 24
    #         nofflies: 25
    #       starve_day: 26
    #          age_day: 27
    #        room_Temp: 28
    #         room_Hum: 29
    #       reflection: 30
    #  reflection_meas: 31
    #        collision: 32
    #         overpass: 33
    #             jump: 34
    #       fly_status: 35
    #      refOverLap1: 36
    #      refOverLap2: 37
    #           excent: 38
    #        perimeter: 39
    #        coligzone: 40
    #        periphery: 41
    #      signal_mask: 42
    #        onset_num: 43
    #         peak_num: 44
    #       offset_num: 45
    # signal_threshold: 46
    #               mu: 47
    #            sigma: 48
    #             nfac: 49
    #         wind_dir: 50
    #       wind_speed: 51
    #  ant_signal_grad: 52
    
# %% angle upon odor

### for mat data
# mat_data = expmat[:, select] ### trkNum, x,y,speed, theta, signal, sy
theta = mat_data[:,4]
spd = mat_data[:,3]
y = mat_data[:,2] - mat_data[:,-1]
x = mat_data[:,1]
xbnds = [200, 300]
ybnds = [0, 20]
num_bins = 30

### for pkl data
# theta = pkl_data['theta_smooth']
# spd = pkl_data['spd_smooth']
# y = pkl_data['y_smooth'] - pkl_data['sy']
# x = pkl_data['x_smooth']

for xbnds in [[5, 200], [200, 300]]:
    bin_hits = (x < xbnds[1])*(x > xbnds[0])*(abs(y) >= ybnds[0])*(abs(y) <= ybnds[1])
    bin_hits *= np.isfinite(theta)
    bin_hits *= spd > 3
    
    # fig = plt.figure(figsize=(12, 3))
    plt.subplot(111, polar=True)
    vals = theta[bin_hits]*np.pi/180
    hist, bins = np.histogram(vals, bins=np.linspace(0, 2*np.pi, num_bins), density=True)
    hist = np.hstack((hist, hist[0]))
    bins = (bins[1:] + bins[:-1])/2
    bins = np.hstack((bins, bins[0]))
    plt.plot(bins, hist, lw=1.5)
    plt.fill_between(bins, np.zeros(len(hist)), hist, alpha=0.2)

# plt.savefig("epg_wt.pdf", bbox_inches='tight')

# %%
# mat_data = expmat[:, select] ### trkNum, x,y,speed, theta, signal, sy
theta = mat_data[:,4]
spd = mat_data[:,3]
y = mat_data[:,2] - mat_data[:,-1]
x = mat_data[:,1]
xbnds = [200, 300]
ybnds = [0, 15]
num_bins = 30

### for pkl data
# theta = pkl_data['theta_smooth']
# spd = pkl_data['spd_smooth']
# y = pkl_data['y_smooth'] - pkl_data['sy']
# x = pkl_data['x_smooth']

for xbnds in [[250, 300]]:
    bin_hits = (x < xbnds[1])*(x > xbnds[0])*(abs(y) >= ybnds[0])*(abs(y) <= ybnds[1])
    bin_hits *= np.isfinite(theta)
    bin_hits *= spd > 2
    
    # fig = plt.figure(figsize=(12, 3))
    theta = mat_data[:,4]
    spd = mat_data[:,3]
    y = mat_data[:,2] - mat_data[:,-1]
    x = mat_data[:,1]
    
    plt.subplot(111, polar=True)
    vals = theta[bin_hits]*np.pi/180
    hist, bins = np.histogram(vals, bins=np.linspace(0, 2*np.pi, num_bins), density=True)
    hist = np.hstack((hist, hist[0]))
    bins = (bins[1:] + bins[:-1])/2
    bins = np.hstack((bins, bins[0]))
    plt.plot(bins, hist, lw=1.5)
    plt.fill_between(bins, np.zeros(len(hist)), hist, alpha=0.2)
    
    theta = pkl_data['theta_smooth']
    spd = pkl_data['spd_smooth']
    y = pkl_data['y_smooth'] - pkl_data['sy']
    x = pkl_data['x_smooth']
    
    bin_hits = (x < xbnds[1])*(x > xbnds[0])*(abs(y) >= ybnds[0])*(abs(y) <= ybnds[1])
    bin_hits *= np.isfinite(theta)
    bin_hits *= spd > 3
    
    plt.subplot(111, polar=True)
    vals = theta[bin_hits]*np.pi/180
    hist, bins = np.histogram(vals, bins=np.linspace(0, 2*np.pi, num_bins), density=True)
    hist = np.hstack((hist, hist[0]))
    bins = (bins[1:] + bins[:-1])/2
    bins = np.hstack((bins, bins[0]))
    plt.plot(bins, hist, lw=1.5)
    plt.fill_between(bins, np.zeros(len(hist)), hist, alpha=0.2)

# plt.savefig("epg_combined.pdf", bbox_inches='tight')
