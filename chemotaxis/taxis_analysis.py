# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:59:38 2025

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
import matplotlib.colors as mcolors

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import h5py

# %% from bacteria chemotaxis data, define states, then compute inverse Q-learning
### then simulate data constrained RL agent
### then replace states with experimental modes
### then extend to time varying states...
### does it say something about expectations?

# %% load mat file for the data structure
file_dir = r'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/chemotaxis/170222-RP437_gradient_4X_001.swimtracker.mat'
# Open the .mat file
with h5py.File(file_dir, 'r') as file:
    # Access the structure
    your_struct = file['tracks']
    col_k = list(your_struct.keys())
    print(col_k)
    
# %% load specific fields
extract_data = {}
keys_of_interest = ['RCD', 'x', 'y', 'dx', 'dy', 'tumble']
n_tracks = 500

with h5py.File(file_dir, 'r') as f:
    struct = f['tracks']
    keys = list(struct.keys())  # list of subfield names

    # Initialize empty lists for each subfield
    for key in keys_of_interest:
        extract_data[key] = []

    n_entries = len(struct[keys[0]])  # how many elements (assume consistent size)

    for i in range(n_tracks): #range(n_entries):
        print(i)
        for key in keys_of_interest:
            # Get the reference to the i-th object in the subfield
            ref = struct[key][i][0]  # [i][0] because it's stored in MATLAB HDF5 weirdly
            # Dereference and read data
            if isinstance(ref, h5py.Reference):
                obj = f[ref]
                value = obj[()]
                extract_data[key].append(value)
            else:
                extract_data[key].append(ref)

# %% visualize bacteria traces
plt.figure()
for ii in range(n_tracks):
    xi,yi = extract_data['x'][ii].squeeze(), extract_data['y'][ii].squeeze()
    plt.plot(xi, yi)

# %% visualize single
tracki = 13
plt.figure()
plt.plot(extract_data['x'][tracki].squeeze(),  extract_data['y'][tracki].squeeze())
pos = np.where(extract_data['tumble'][tracki].squeeze()==1)[0]
plt.plot(extract_data['x'][tracki].squeeze()[pos],  extract_data['y'][tracki].squeeze()[pos], 'r.')

# %%
###############################################################################
# %% funcational
def transfer_entropy(X,Y,delay=1,gaussian_sigma=None):
	'''
	TE implementation: asymmetric statistic measuring the reduction in uncertainty
	for a future value of X given the history of X and Y. Or the amount
	of information from Y to X. Calculated through the Kullback-Leibler divergence 
	with conditional probabilities

	author: Sebastiano Bontorin
	mail: sbontorin@fbk.eu

	args:
		- X (1D array):
			time series of scalars (1D array)
		- Y (1D array):
			time series of scalars (1D array)
	kwargs:
		- delay (int): 
			step in tuple (x_n, y_{n - delay}, x_(n - delay))
		- gaussian_sigma (int):
			sigma to be used
			default set at None: no gaussian filtering applied
	returns:
		- TE (float):
			transfer entropy between X and Y given the history of X
	'''

	if len(X)!=len(Y):
		raise ValueError('time series entries need to have same length')

	n = float(len(X[delay:]))

	# number of bins for X and Y using Freeman-Diaconis rule
	# histograms built with numpy.histogramdd
	binX = len(np.unique(X))
	binY = len(np.unique(Y))
    
	# Definition of arrays of shape (D,N) to be transposed in histogramdd()
	x3 = np.array([X[delay:],Y[:-delay],X[:-delay]])
	x2 = np.array([X[delay:],Y[:-delay]])
	x2_delay = np.array([X[delay:],X[:-delay]])

	p3,bin_p3 = np.histogramdd(
		sample = x3.T,
		bins = [binX,binY,binX])

	p2,bin_p2 = np.histogramdd(
		sample = x2.T,
		bins=[binX,binY])

	p2delay,bin_p2delay = np.histogramdd(
		sample = x2_delay.T,
		bins=[binX,binX])

	p1,bin_p1 = np.histogramdd(
		sample = np.array(X[delay:]),
		bins=binX)

	# Hists normalized to obtain densities
	p1 = p1/n
	p2 = p2/n
	p2delay = p2delay/n
	p3 = p3/n

	# Ranges of values in time series
	Xrange = bin_p3[0][:-1]
	Yrange = bin_p3[1][:-1]
	X2range = bin_p3[2][:-1]

	# Calculating elements in TE summation
	elements = []
	for i in range(len(Xrange)):
		px = p1[i]
		for j in range(len(Yrange)):
			pxy = p2[i][j]

			for k in range(len(X2range)):
				pxx2 = p2delay[i][k]
				pxyx2 = p3[i][j][k]

				arg1 = float(pxy*pxx2)
				arg2 = float(pxyx2*px)

				# Corrections avoding log(0)
				if arg1 == 0.0: arg1 = float(1e-8)
				if arg2 == 0.0: arg2 = float(1e-8)

				term = pxyx2*np.log2(arg2) - pxyx2*np.log2(arg1) 
				elements.append(term)

	# Transfer Entropy
	TE = np.sum(elements)
	return TE

def transfer_entropy_both(X, Y, delay=1):
    '''
    Compute transfer entropy both ways: TE(Y->X) and TE(X->Y),
    also entropy H(x'|x), H(y'|y), H(x',y'|x,y).

    Args:
        X, Y: 1D arrays
        delay: time lag

    Returns:
        TE_YX: Transfer entropy Y -> X
        TE_XY: Transfer entropy X -> Y
        Hx: H(x'|x)
        Hy: H(y'|y)
        Hxy: H(x',y'|x,y)
    '''

    if len(X) != len(Y):
        raise ValueError('X and Y must have the same length')

    # n = float(len(X[delay:]))

    binX = len(np.unique(X))
    binY = len(np.unique(Y))


    TE_XY = transfer_entropy(X, Y, delay=delay)
    TE_YX = transfer_entropy(Y, X, delay=delay)

    ## --- Entropies H(x'|x), H(y'|y)
    counts_xx = np.histogram2d(X[:-delay], X[delay:], bins=[binX, binX])[0]
    counts_xx /= np.sum(counts_xx)

    Hx = 0.0
    for i in range(binX):
        marginal_x = np.sum(counts_xx[i, :])
        for j in range(binX):
            if counts_xx[i, j] > 0 and marginal_x > 0:
                Hx -= counts_xx[i, j] * np.log2(counts_xx[i, j] / marginal_x)

    counts_yy = np.histogram2d(Y[:-delay], Y[delay:], bins=[binY, binY])[0]
    counts_yy /= np.sum(counts_yy)

    Hy = 0.0
    for i in range(binY):
        marginal_y = np.sum(counts_yy[i, :])
        for j in range(binY):
            if counts_yy[i, j] > 0 and marginal_y > 0:
                Hy -= counts_yy[i, j] * np.log2(counts_yy[i, j] / marginal_y)

    ## --- H(x',y'|x,y)
    stacked = np.vstack([X[:-delay], Y[:-delay], X[delay:], Y[delay:]]).T
    counts_xy_xyprime, _ = np.histogramdd(stacked, bins=[binX, binY, binX, binY])
    counts_xy_xyprime /= np.sum(counts_xy_xyprime)

    Hxy = 0.0
    for i in range(binX):
        for j in range(binY):
            marginal_xy = np.sum(counts_xy_xyprime[i, j, :, :])
            for k in range(binX):
                for l in range(binY):
                    p = counts_xy_xyprime[i, j, k, l]
                    if p > 0 and marginal_xy > 0:
                        Hxy -= p * np.log2(p / marginal_xy)

    return TE_YX, TE_XY, Hx, Hy, Hxy

def track2states(tracki):
    # tracki = 13
    xi, yi = extract_data['x'][tracki].squeeze(),  extract_data['y'][tracki]
    bi = np.zeros(len(xi))
    pos = np.where(extract_data['tumble'][tracki].squeeze()==1)[0]
    bi[pos] = 1
    dxi = np.diff(xi)
    dxi = np.append(0, dxi)
    si = dxi*0
    si[dxi>0] = 1
    return si, bi

# %% compute contributions
# entropy_components = np.zeros((5, n_tracks))
# v_drift = np.zeros(n_tracks)
entropy_components = [] #np.zeros((5, n_tracks))
v_drift = [] #np.zeros(n_tracks)

for ii in range(n_tracks):
    si,bi = track2states(ii)
    if np.sum(bi)>0:
        TE_YX, TE_XY, Hx, Hy, Hxy = transfer_entropy_both(bi, si, delay=10)
        # entropy_components[:, ii] = np.array([ TE_YX, TE_XY, Hx, Hy, Hxy ])
        entropy_components.append( np.array([ TE_YX, TE_XY, Hx, Hy, Hxy ]))
        print(ii)
        
        xi, yi = extract_data['x'][ii].squeeze(),  extract_data['y'][ii].squeeze()
        vv = (xi[-1] - xi[0])/len(xi)
        v_drift.append(vv)
    
# %% plotting
entropy_components_ = np.array(entropy_components).T
plt.figure()
# plt.hist(entropy_components[0,:] / entropy_components[4,:], 50)
plt.plot(entropy_components_[0,:] / entropy_components_[4,:], v_drift, 'k.', alpha=0.5); plt.xlim([0,0.07]); plt.ylim([-.5,2])
# plt.plot(entropy_components_[0,:] / entropy_components_[4,:], entropy_components_[3,:] / entropy_components_[4,:], 'k.', alpha=0.5);
# plt.xlim([0.,0.1]);plt.ylim([0.,0.1])

plt.xlabel('normed TE(b->s)'); plt.ylabel('drift velocity')

# %% visualize in space
cmap = cm.viridis
norm = mcolors.Normalize(vmin=0., vmax=0.05) 
norm = mcolors.Normalize(vmin=0.3, vmax=0.7) 
plt.figure(figsize=(6, 6))
for ii in range(300):
    xi, yi = extract_data['x'][ii].squeeze(),  extract_data['y'][ii].squeeze()
    color = cmap(norm(entropy_components_[2,ii] / entropy_components_[4,ii]))
    plt.plot(xi, yi, color=color)
# plt.xscale('log')

# %%
## %% simple binarization
# biniarize B and S
# compute information transfer
# compare to performance
###

