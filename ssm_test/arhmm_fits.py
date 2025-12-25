# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:35:34 2024

@author: ksc75
"""

import ssm
from ssm.observations import AutoRegressiveObservations, IndependentAutoRegressiveObservations, AutoRegressiveDiagonalNoiseObservations, \
    RobustAutoRegressiveObservationsNoInput
import numpy as np
import matplotlib.pyplot as plt

import pickle
import gzip
import glob
import os

import numpy.random as npr

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% model comparison
# here we want to explore number of states
# also try exploring input, autoregressive, and mixture of output (with angle)

# %% for Kiri's data
### cutoff for short tracks
threshold_track_l = 60 * 20  # 20 # look at long-enough tracks

# Define the folder path
folder_path = 'C:/Users/ksc75/Downloads/ribbon_data_kc/'

# Use glob to search for all .pkl files in the folder
pkl_files = glob.glob(os.path.join(folder_path, '*.pklz'))

# Print the list of .pkl files
for file in pkl_files:
    print(file)

# %% concatenate across files in a folder
data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
nf = len(pkl_files)
masks = []   # where there is nan
track_id = []  # record track id (file and track)
rec_tracks = []  # record the full track x,y
rec_signal = []  # record opto signal
times = []   # record time in epoch
cond_id = 0

smoother = np.ones(100)/100

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
                
                # temp = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
                
                temp = np.stack((np.convolve(data['vx_smooth'][pos], smoother, 'same') , np.convolve(data['vy_smooth'][pos], smoother, 'same')), 1)####### testing this!!!
                
                temp_xy = np.column_stack((data['x'][pos] , data['y'][pos]))
                
                
                ### criteria
                mask_i = np.where(np.isnan(temp), 0, 1)
                mask_j = np.where(np.isnan(thetas), 0, 1)
                mean_v = np.nanmean(np.sum(temp**2,1)**0.5)
                max_v = np.max(np.sum(temp**2,1)**0.5)
                # print(mean_v)
                if np.prod(mask_i)==1 and np.prod(mask_j)==1 and mean_v>1 and max_v<30:  ###################################### removing nan for now
                    data4fit.append(temp)  # get data for ssm fit
                    rec_tracks.append(temp_xy)  # get raw tracks
                    track_id.append(np.array([ff,ii]))  # get track id
                    rec_signal.append(data['signal'][pos])
                    cond_id += 1
                    masks.append(thetas)
                    times.append(data['t'][pos])

# %% setup
num_states = 5 # K states
obs_dim = 2  # D dimsional observation
M = 0
lags = 3

"""
    (x_t | z_t = k, u_t) ~ N(A_k x_{t-1} + b_k + V_k u_t, S_k)
"""
test_hmm = ssm.HMM(num_states, obs_dim, M)#, observations="autoregressive",  transitions="sticky")

# test_hmm.observations = AutoRegressiveObservations(num_states, obs_dim, M, lags=lags)
test_hmm.observations = IndependentAutoRegressiveObservations(num_states, obs_dim, M, lags=lags)

# true_states, data4fit = test_hmm.sample(10000)

# %%
hmm_lls = test_hmm.fit(data4fit, method="em", num_iters=50, init_method="kmeans")
plt.figure()
plt.plot(hmm_lls, label="EM")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")

# %% analysis of the fit!
###############################################################################
# %%
from matplotlib.patches import Ellipse
cmap = plt.get_cmap('tab10')

def plot_gaussian_ellipse(mean, cov, ax=None, n_std=2.0, **kwargs):
    """
    Plots a 2D Gaussian as an ellipse.

    Parameters:
    - mean: The 2D mean of the distribution [x, y].
    - cov: 2x2 covariance matrix.
    - ax: Matplotlib axis object. If None, a new figure and axis are created.
    - n_std: Number of standard deviations to plot the ellipse at (default is 2).
    - kwargs: Additional arguments passed to the Ellipse patch.
    """
    if ax is None:
        ax = plt.gca()

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Calculate the angle of the ellipse's orientation
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    # Width and height of the ellipse are 2*sqrt(eigenvalues) scaled by the number of standard deviations
    width, height = 2 * n_std * np.sqrt(eigvals)

    # Create an Ellipse patch
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)

    # Add the ellipse to the plot
    ax.add_patch(ell)
    ax.set_aspect('equal')

    return ell

fig, ax = plt.subplots()
_, means_state, _, cov_states = test_hmm.observations.params
for nn in range(num_states):
    mean = means_state[nn,:]
    cov = cov_states[nn,:]

    temp = cmap(nn)
    plot_gaussian_ellipse(mean, cov, ax=ax, n_std=2, edgecolor=temp, facecolor='none')
    
    # Plot the center point
    plt.scatter(*mean, color=temp, label="Mean")
plt.xlabel(r'$V_x$')
plt.ylabel(r'$V_y$')
plt.title('Gaussian emissions')

# %%
weights = test_hmm.observations.params[0]
plt.figure()
for nn in range(num_states):
    temp = weights[nn,1,:]
    plt.plot(temp)

# %%
true_states, obs = test_hmm.sample(50000)
vec_data = np.concatenate(data4fit)
vec_signal = np.concatenate(rec_signal)
vec_time = np.concatenate(times)
vec_vxy = np.concatenate(data4fit)

# %%
def compute_autocorrelation(data, max_lag):
    """
    Compute the autocorrelation for each lag from 1 to max_lag.

    Parameters:
    - data: 1D numpy array or list of time series data
    - max_lag: Maximum number of lags to compute

    Returns:
    - lags: Array of lags (from 1 to max_lag)
    - autocorr_values: Autocorrelation values corresponding to each lag
    """
    n = len(data)
    mean = np.nanmean(data)
    autocorr_values = []
    
    # Compute autocorrelation for each lag
    for lag in range(1, max_lag + 1):
        numerator = np.nansum((data[:-lag] - mean) * (data[lag:] - mean))
        denominator = np.nansum((data - mean) ** 2)
        autocorrelation = numerator / denominator
        autocorr_values.append(autocorrelation)
    
    return np.arange(1, max_lag + 1), autocorr_values

# Compute the autocorrelation for up to 20 lags
lags, autocorr_values_model = compute_autocorrelation(obs[:,1], max_lag=threshold_track_l)
lags, autocorr_values_data = compute_autocorrelation(vec_data[:,1], max_lag=threshold_track_l)

# Plot the autocorrelation function
time_lags = np.arange(0,len(lags))* 1/60  # frames to seconds
# plt.figure(figsize=(7, 5))
plt.plot(time_lags, autocorr_values_data, linewidth=9, label='data')
plt.plot(time_lags, autocorr_values_model, linewidth=9, label='model')
plt.xlabel('time lag (s)')
plt.ylabel('autocorrelation')
plt.title('velocity')
plt.axhline(0, color='gray', linestyle='--')
plt.grid()
plt.legend()

# %% get states
ltr = len(data4fit)
post_z = []
for ll in range(ltr):
    most_likely_states = test_hmm.most_likely_states(data4fit[ll])
    post_z.append(most_likely_states[:,None])

vec_states = np.concatenate(post_z)

