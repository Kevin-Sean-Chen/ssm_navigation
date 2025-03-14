# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:49:04 2024

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

import pickle
import gzip
import glob
import os

import ssm
import numpy.random as npr
import seaborn as sns

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% make a list of data
# list with tracks vx,vy
# future can include theta and the sensory input...

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

# %% for perturbed data
# root_dir = 'C:/Users/ksc75/Yale University Dropbox/users/kevin_chen/data/opto_rig/perturb_ribbon/100424_new/'
# target_file = "exp_matrix.pklz"

# # List all subfolders in the root directory
# subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
# pkl_files = []

# # Loop through each subfolder to search for the target file
# for subfolder in subfolders:
#     for dirpath, dirnames, filenames in os.walk(subfolder):
#         if target_file in filenames:
#             full_path = os.path.join(dirpath, target_file)
#             pkl_files.append(full_path)
#             print(full_path)

# pkl_files = pkl_files[:8]

# %% concatenate across files in a folder
data4fit = []  # list of tracks with its vx,vy,theta signal recorded;  conditioned on behavior and long-tracks
nf = len(pkl_files)
masks = []   # where there is nan
track_id = []  # record track id (file and track)
rec_tracks = []  # record the full track x,y
rec_signal = []  # record opto signal
times = []   # record time in epoch
cond_id = 0

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
                temp = np.column_stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]))
                # temp = np.stack((data['vx_smooth'][pos] , data['vy_smooth'][pos]),1)#######
                
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
                # masks.append(mask_i)

# %%
# pick_one = 7
# plt.plot(rec_tracks[pick_one][:,0], rec_tracks[pick_one][:,1])
vec_signal = np.concatenate(rec_signal)
plt.plot(vec_signal)

# %% study nans
###############################################################################
for tracki in range(10):#(len(data4fit)):
    pos_nan = np.where(np.isnan(rec_signal[tracki]))[0][::10]
    pos_tru = np.where(np.isnan(rec_signal[tracki])==0)[0][::10]
    temp_xy = rec_tracks[tracki]
    plt.plot(temp_xy[pos_nan,0], temp_xy[pos_nan,1],'.')
    plt.plot(temp_xy[pos_tru,0], temp_xy[pos_tru,1],'k-',markersize=1.5)
    # plt.plot(masks[tracki][pos_nan])

# %%
# %% quick ssm test
###############################################################################
# %% setup
num_states = 3
obs_dim = 2

# %%
# data = data4fit*1 # Treat observations generated above as synthetic data.
N_iters = 100

## testing the constrained transitions class
hmm = ssm.HMM(num_states, obs_dim, observations="gaussian",  transitions="sticky")

hmm_lls = hmm.fit(data4fit, method="em", num_iters=N_iters, init_method="kmeans")

plt.figure()
plt.plot(hmm_lls, label="EM")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")
plt.show()

# %% compute ll
measure_lp = hmm.log_probability(data4fit)

# %% filtering!
pick_id = 54  # 0,7
most_likely_states = hmm.most_likely_states(data4fit[pick_id])
track_i = rec_tracks[pick_id]

most_likely_states = most_likely_states[:] #:6
track_i = track_i[:] #:6

# Create a colormap for the two states
colors = ['red', 'blue']  # You can choose different colors for the two states
unique_states = np.unique(most_likely_states)
cmap = plt.get_cmap('tab10')

plt.figure(figsize=(8, 6))

# Loop over the unique states and plot the corresponding segments
# for i, state in enumerate(unique_states):
for ii in range(num_states): #(len(unique_states)):
    state_mask = np.where(most_likely_states==ii)[0]
    # Find where the trajectory is in the current state
    # state_mask = (state==most_likely_states)
    
    # Plot the trajectory segment with a different color
    plt.plot(track_i[state_mask,0], track_i[state_mask,1], 'o', color=cmap(ii), alpha=0.5)
    
# Add labels and legends
plt.title("state-code trajectories")
plt.xlabel("X")
plt.ylabel("Y")


# %% state of states??
###############################################################################
# %%
ltr = len(data4fit)
post_z = []
for ll in range(ltr):
    most_likely_states = hmm.most_likely_states(data4fit[ll])
    post_z.append(most_likely_states[:,None])

# %%
# # %% then cluster them
# n_state2 = 3
# hmm2 = ssm.HMM(n_state2, 1, observations="categorical", observation_kwargs=dict(C=num_states), transitions="sticky")

# hmm_lls = hmm2.fit(post_z, method="em", num_iters=N_iters, init_method="kmeans")

# # %% L2 filtering
# pick_id = 15
# most_likely_states2 = hmm2.most_likely_states(post_z[pick_id])
# track_i = rec_tracks[pick_id]

# most_likely_states2 = most_likely_states2[::6]
# track_i = track_i[::6]

# # Create a colormap for the two states
# colors = ['red', 'blue']  # You can choose different colors for the two states
# unique_states = np.unique(most_likely_states2)
# cmap = plt.get_cmap('tab10')

# plt.figure(figsize=(8, 6))

# # Loop over the unique states and plot the corresponding segments
# # for i, state in enumerate(unique_states):
# for ii in range(len(unique_states)):
#     state_mask = np.where(most_likely_states2==ii)[0]
#     # Find where the trajectory is in the current state
#     # state_mask = (state==most_likely_states)
    
#     # Plot the trajectory segment with a different color
#     plt.plot(track_i[state_mask,0], track_i[state_mask,1], 'o', color=cmap(ii), alpha=0.5)
    
# # Add labels and legends
# plt.title("state-code trajectories")
# plt.xlabel("X")
# plt.ylabel("Y")

# %%
###############################################################################
# %% state conditional analysis
###############################################################################
# ex: condition on stimuli, plot states evolution
# ex: condition on stimuli, plot post-stim states

# %% vectorize for simpliciy
vec_signal = np.concatenate(rec_signal)
vec_states = np.concatenate(post_z)
vec_time = np.concatenate(times)
vec_vxy = np.concatenate(data4fit)

# %% state occupency conditioned on signal
threshold_within = 5
pos = np.where(vec_signal > threshold_within)[0]  ### set stats_signal for concatenating the full signal vector
# pos = np.union1d(pos, np.where((vec_time>45) & (vec_time<45+30))[0])
win_stats = vec_states[pos]

counts, bin_edges = np.histogram(win_stats, bins=num_states)
bin_edge = np.arange(num_states)

plt.figure(figsize=(8, 6))

for i in range(len(counts)):
    plt.bar(bin_edge[i], counts[i], width=bin_edges[i+1] - bin_edges[i], color=cmap(i), edgecolor='black')
# plt.title('without signal detection', fontsize=20)
plt.title('during signal detection', fontsize=20)

# %% condition on past! signal
threshold_within = 5  # threhold for detection
threshold_cont = 60*30  #3  #20  # threshold for continuos detection
post_wind = 60*30  # post detection window

pos = np.where(vec_signal > threshold_within)[0]  ### set stats_signal for concatenating the full signal vector
bined_sig = vec_states*0
bined_sig[pos] = 1

###############################################################################
### figure out how to condition!!
# conved_signal = np.convolve(bined_sig[:,0], np.ones(int(threshold_cont)), mode='same')
# pos = np.where(conved_signal[post_wind//2:-post_wind//2] > 0.7*threshold_cont)[0]
# pos = np.where(conved_signal[:] > 0.7*threshold_cont)[0] + 0
# pos = np.union1d(pos, np.where((vec_time>45+30) & (vec_time<45+30+30))[0])
# bin_stats = vec_states[pos+post_wind]

### just post
bin_stats = vec_states[np.where((vec_time>45+30) & (vec_time<45+30+30))[0]]
###############################################################################

counts, bin_edges = np.histogram(bin_stats, bins=num_states)
bin_edge = np.arange(num_states)

plt.figure(figsize=(8, 6))

for i in range(len(counts)):
    plt.bar(bin_edge[i], counts[i], width=bin_edges[i+1] - bin_edges[i], color=cmap(i), edgecolor='black')
plt.title('post signal detection', fontsize=20)

# %% analyzing states
###############################################################################
from matplotlib.patches import Ellipse

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
means_state, cov_states = hmm.observations.params
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

# %% kernel densities
from scipy.stats import gaussian_kde

pos_state = np.where((vec_states!=2) & (vec_states!=1))[0]
pos_remove = np.where((np.abs(vec_vxy[:,0])>30) | (np.abs(vec_vxy[:,1])>30))[0]
pos =  np.setdiff1d(pos_state, pos_remove) #np.union1d(pos_state, ~pos_remove)
vec_move = vec_vxy[pos,:][::10]

# Generate random 2D data
x = vec_move[:,0]
y = vec_move[:,1]

# Perform Kernel Density Estimation (KDE)
data = np.vstack([x, y])  # Stack x and y into a 2D array
kde = gaussian_kde(data)

# Create a grid of points for plotting
xmin, xmax = x.min() - 1, x.max() + 1
ymin, ymax = y.min() - 1, y.max() + 1
x_grid, y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]  # Create grid points
grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])  # Flatten the grid for evaluation

# Evaluate the KDE on the grid
z = kde(grid_coords).reshape(x_grid.shape)

plt.figure()
# Plot scatter points
plt.scatter(x, y, s=5, color='blue', label='Data points')

# Plot KDE density as a contour plot
plt.contourf(x_grid, y_grid, z, levels=20, cmap='Blues')

# Show the plot
plt.colorbar(label='Density')
plt.xlabel('Vx')
plt.ylabel('Vy')
plt.title('KDE, without stopping')

# %% conditional
plt.figure()
for ii in range(num_states):
    pos = np.where(vec_states==ii)[0]
    temp_xy = vec_vxy[pos,:][::12]
    plt.plot(temp_xy[:,0], temp_xy[:,1], '.', color=cmap(ii), alpha=.1)
    
# %%
def plot_kde(x, y, label=None, cmap=None):
    # Perform Kernel Density Estimation (KDE)
    data = np.vstack([x, y])
    kde = gaussian_kde(data)
    # Create a grid for plotting
    xmin, xmax = x.min() - 1, x.max() + 1
    ymin, ymax = y.min() - 1, y.max() + 1
    x_grid, y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    # Evaluate the KDE on the grid
    z = kde(grid_coords).reshape(x_grid.shape)
    # Plot KDE density as a contour plot
    plt.contourf(x_grid, y_grid, z, levels=20, cmap=cmap, alpha=0.5)

color_kdes = ['Blues', 'Oranges', 'Greens','Reds','Purples']
plt.figure()
for nn in range(num_states):
    pos = np.where(vec_states==nn)[0]
    plot_kde(vec_vxy[pos,0][::12], vec_vxy[pos,1][::12], cmap=color_kdes[nn])
plt.title('data | states')

# %% Cross-Valdiation
###############################################################################
# %% setup
scan_n_states = np.arange(1,7)
obs_dim = 2
N_iters = 100
train_ll = np.zeros(len(scan_n_states))

train_data = data4fit[200:]
test_data = data4fit[:200]
n_samps = 50
n_samp_test = 20
test_vec = np.arange(len(test_data))
test_ll = np.zeros((len(scan_n_states), n_samp_test))

# %% iteration

for ii in range(len(scan_n_states)):
    print(ii)
    hmm_cv = ssm.HMM(scan_n_states[ii], obs_dim, observations="gaussian",  transitions="sticky")
    hmm_lls = hmm_cv.fit(train_data, method="em", num_iters=N_iters, init_method="kmeans")
    
    train_ll[ii] = hmm_cv.log_probability(train_data)
    
    for jj in range(n_samp_test):
        sampled_elements = np.random.choice(test_vec, size=n_samps, replace=False)
        test_data_i = [test_data[k] for k in sampled_elements]
        test_ll[ii,jj] = hmm_cv.log_probability(test_data_i)

# %% plotting LLs
from scipy.stats import ttest_ind

plt.figure()
# plt.plot(scan_n_states, train_ll,'k-o')
# plt.plot(scan_n_states, test_ll, 'ko')
plt.plot(np.tile(scan_n_states, (n_samp_test, 1)).T + np.random.randn(max(scan_n_states), n_samp_test)*0.05, test_ll, 'k.')
plt.xlabel('number of states')
plt.ylabel('test LL')
