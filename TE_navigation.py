# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:30:31 2025

@author: ksc75
"""

import numpy as np
import scipy as sp
from scipy import stats
from scipy import ndimage
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import h5py

# %% load and prpocess navigation data
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
    # print(col.keys())

# %% now extract track data
chop = 3000000
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

# %% plot track
trk = 11
pos = np.where(trjNum==trk)[0][10:-10]
pos_stop = np.where((stops==1) | (turns>0))[0]
pos_stop = np.intersect1d(pos, pos_stop)
plt.figure()
plt.plot(x_smooth[pos], y_smooth[pos],'k')
plt.plot(x_smooth[pos_stop], y_smooth[pos_stop],'r.')

# %% some pre-processing
v_threshold = 30
vx_smooth[np.abs(vx_smooth)>v_threshold] = v_threshold
vy_smooth[np.abs(vy_smooth)>v_threshold] = v_threshold
signal[np.isnan(signal)] = 0

dtheta_threshold = 360
dtheta_smooth[np.abs(dtheta_smooth)>dtheta_threshold] = dtheta_threshold
dtheta_smooth[np.isnan(dtheta_smooth)] = 0

# %% discretization for now
thre = 5
bin_signal = signal*1
bin_signal[signal<thre] = 0
bin_signal[signal>=thre] = 1

def discretize_time_series(series, thresholds):
    thresholds = np.sort(thresholds)
    states = np.digitize(series, bins=thresholds)
    return states

### list of bins
bin_stops = stops*1
bin_stops[stops>0] = 1  #### stops
bin_turns = turns*1
bin_turns[turns>0] = 1  ##### turns
bin_vi = discretize_time_series(speed_smooth*1,  [5,15])  #### try more continuous variables
# bin_vi = discretize_time_series(vy_smooth*1,  [-15,-5,5,15])  ### test this

# %% compute TE
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

# %% 2D grid sampling
def coarse_grain_2d_scatter_indices(x, y, grid_size):
    x_bins, y_bins = grid_size

    # Define grid boundaries
    x_edges = np.linspace(10, max(x), x_bins + 1)
    y_edges = np.linspace(10, max(y), y_bins + 1)  ### min(y)

    # Digitize points to identify grid cells
    x_indices = np.digitize(x, x_edges) - 1
    y_indices = np.digitize(y, y_edges) - 1

    # Initialize a grid to store the indices of (x, y) points for each cell
    grid_positions = [[[] for _ in range(y_bins)] for _ in range(x_bins)]

    # Assign point indices to the corresponding grid cell
    for i in range(len(x)):
        x_cell = x_indices[i]
        y_cell = y_indices[i]
        if 0 <= x_cell < x_bins and 0 <= y_cell < y_bins:
            grid_positions[x_cell][y_cell].append(i)

    return grid_positions

### check x-corr for delay steps
X = bin_signal  # Random time series X
Y = bin_vi  # Random time series Y
lag_min = -100  # Minimum lag
lag_max = 100   # Maximum lag
cross_corr = sp.signal.correlate(X - np.mean(X), Y - np.mean(Y), mode='full')
lags = np.arange(-len(X) + 1, len(X))
lag_range = np.where((lags >= lag_min) & (lags <= lag_max))

plt.figure(figsize=(8, 6))
plt.plot(lags[lag_range], cross_corr[lag_range])
plt.xlabel("Lag"); plt.ylabel("Cross-Correlation"); plt.grid(True)

# sample for location
xy_grid = (17,9)
delay = 20  ### 10,20,30
grid_xy = coarse_grain_2d_scatter_indices(x_smooth, y_smooth, xy_grid)

# %%
###############################################################################
# %% compute TE!!
#### might need to remove transitions across tracks later... #########################
TE_s2b = np.zeros(xy_grid)
TE_b2s = np.zeros(xy_grid)
obs_num = np.zeros(xy_grid)
for xx in range(xy_grid[0]):
    print(xx)
    for yy in range(xy_grid[1]):
        pos = grid_xy[xx][yy]
        xi = bin_signal[pos]  ### sensory drive
        yi = bin_vi[pos]   ### behavior output
        TE_s2b[xx,yy] = transfer_entropy(xi,yi,delay)
        TE_b2s[xx,yy] = transfer_entropy(yi,xi,delay)
        obs_num[xx,yy] = len(pos)
TE_s2b = TE_s2b.T
TE_b2s = TE_b2s.T

# %% plotting
data1, data2, data3 = TE_s2b, TE_b2s, (TE_s2b - TE_b2s)
vmin = min(data1.min(), data2.min(), data3.min())
vmax = max(data1.max(), data2.max(), data3.max())

fig, axs = plt.subplots(3, 1, figsize=(18, 8))
for ax in axs:
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([]) 
cax1 = axs[0].imshow(data1, cmap='viridis',vmin=vmin, vmax=vmax)
axs[0].set_title("S -> B")
# axs[0].set_title("MI(S',S)")
cax2 = axs[1].imshow(data2, cmap='viridis',vmin=vmin, vmax=vmax)
axs[1].set_title("B -> S")
cax3 = axs[2].imshow(data3, cmap='viridis',vmin=vmin, vmax=vmax)
axs[2].set_title("difference")
fig.colorbar(cax3, ax=axs, orientation='horizontal', fraction=0.02, pad=0.1)



# %% quick test for mode-TE calculation
TEs = np.zeros(20)
TE2 = TEs*1
for ii in range(20):
    imode = ii+1
    phi2 = eigvecs[labels,imode].real
    phi_samp = -phi2[window_show]
    t = np.arange(1,50000,3)*1/60  ### time vector
    pos = np.where(np.abs(dists)<50)[0]
    x,y = phi_samp[pos], dists[pos]
    yy_,_ = discretize_vector(y[:], n_bins=5)
    xx_,_ = discretize_vector(x[:], n_bins=5)
    TEs[ii] = transfer_entropy(yy_,xx_,delay=20) / (3/60)
    TE2[ii] = transfer_entropy(xx_,yy_,delay=20) / (3/60)

plt.figure()
plt.plot(TEs,'-o', label='proj->mode')
plt.plot(TE2,'-o', label='mode->proj')
plt.xlabel('modes')
plt.ylabel('TE'); plt.legend()
