# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 23:17:05 2025

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% double well example
# Simulation parameters
dt = 0.01       # time step
T = 100         # total time
N = int(T / dt) # number of steps
n_traj = 20     # number of trajectories
gamma = 1.0     # friction coefficient
kT = 1.0        # temperature (k_B * T)
sqrt_2kT = np.sqrt(2 * kT / gamma)

# Double well potential: U(x) = a*x^4 - b*x^2
a = 0.3
b = 1.9
tilt = 0.5

def force(x):
    return - (4*a*x**3 - 2*b*x + tilt)

# Initialize arrays
trajectories = np.zeros((n_traj, N))
x0 = np.random.randn(n_traj)  # initial positions
trajectories[:, 0] = x0

# Simulate Brownian dynamics
for t in range(1, N):
    x = trajectories[:, t-1]
    trajectories[:, t] = x + force(x)*dt/gamma + sqrt_2kT*np.sqrt(dt)*np.random.randn(n_traj)

# Plot trajectories
plt.figure(figsize=(10, 4))
for i in range(n_traj):
    plt.plot(np.linspace(0, T, N), trajectories[i], alpha=0.6)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Trajectories of particles in double well potential')
plt.grid(True)
plt.show()

# Plot histogram of all positions
plt.figure(figsize=(6, 4))
plt.hist(trajectories[:, int(N/2):].flatten(), bins=100, density=True)
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Histogram of particle positions')
plt.grid(True)
plt.show()

# %% reconstructio
# Reconstruct energy potential from histogram using Boltzmann distribution
# P(x) ~ exp(-U(x)/kT) => U(x) ~ -kT * log(P(x))
hist, bin_edges = np.histogram(trajectories[:, int(N/2):].flatten(), bins=200, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Avoid log(0)
hist[hist == 0] = 1e-10
U_reconstructed = -kT * np.log(hist)

# Ground truth potential
x_vals = np.linspace(-3, 3, 500)
U_true = a * x_vals**4 - b * x_vals**2 + tilt* x_vals

# Normalize both potentials to have min value 0
U_reconstructed -= np.min(U_reconstructed)
U_true -= np.min(U_true)

# Plot comparison
plt.figure(figsize=(8, 5))
plt.plot(bin_centers, U_reconstructed, label='Reconstructed U(x) (from histogram)', lw=2)
plt.plot(x_vals, U_true, label='True U(x)', lw=2, linestyle='--')
plt.xlabel('Position x')
plt.ylabel('Potential U(x)')
plt.title('Comparison of True and Reconstructed Energy Potential')
plt.legend()
plt.grid(True)
plt.show()

##### Kevin's comments ######
# this is assuming equlibrium to get the energy potential
# in the non-equilibrium case, we work with steady-state and recover the 'free-energy potential'
# the error for this reconstruction scales with finite data, so we have to make sure there are enough transitions in data
# one way to simplify the problem is to start with 1D (maybe along wind), and adaptively bin the location
# then work with transitions along these spatial bins
#####

# %% estimating forcing and diffusive terms
# Estimate drift and diffusion from the simulated trajectories
# Time step
dt = 0.01

# Flatten all trajectories (excluding initial transient)
x_vals = trajectories[:, :-1].flatten()
dx_vals = (trajectories[:, 1:] - trajectories[:, :-1]).flatten()

# Define bins for x
num_bins = 80
bin_edges = np.linspace(-3, 3, num_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Digitize x values into bins
bin_indices = np.digitize(x_vals, bin_edges) - 1

# Estimate drift and diffusion in each bin
drift_est = np.zeros(num_bins)
diff_est = np.zeros(num_bins)
counts = np.zeros(num_bins)

for i in range(num_bins):
    dx_bin = dx_vals[bin_indices == i]
    if len(dx_bin) > 10:  # avoid empty or underpopulated bins
        drift_est[i] = np.mean(dx_bin) / dt
        diff_est[i] = np.var(dx_bin) / (2 * dt)
        counts[i] = len(dx_bin)

# Ground truth force and diffusion
x_true = np.linspace(-3, 3, 500)
f_true = force(x_true)
D_true = np.full_like(x_true, kT / gamma)  # constant diffusion

# Plot drift (force) comparison
plt.figure(figsize=(8, 4))
plt.plot(x_true, f_true, label='True Drift (Force)', lw=2)
plt.plot(bin_centers, drift_est, 'o', label='Estimated Drift', markersize=4)
plt.xlabel('Position x')
plt.ylabel('Drift f(x)')
plt.title('Estimated vs True Drift (Force)')
plt.grid(True)
plt.legend()
plt.show()

# Plot diffusion comparison
plt.figure(figsize=(8, 4))
plt.plot(x_true, D_true, label='True Diffusion', lw=2)
plt.plot(bin_centers, diff_est, 'o', label='Estimated Diffusion', markersize=4)
plt.xlabel('Position x')
plt.ylabel('Diffusion D(x)')
plt.title('Estimated vs True Diffusion')
plt.grid(True)
plt.legend()
plt.show()

##### Kevin's note ######
# this is a demo for the drift-diffusion decomposition
# again the error scales with finite data
# if you explore correlated noise (filted and have correlation like OU proceess), you can see how the inference deviates
#########################
# %% fancier irreversibility analysis
# Now explicitly confirm that we are in a double-well potential setup
# and show irreversibility as a function of non-conservative driving strength
np.random.seed(42)
# Define double-well potential parameters
a = 1.0
b = 1.5

# Simulation parameters
dt = 0.001
T = 10
N = int(T / dt)
n_traj = 100
gamma = 1.0
kT = 1.0
mass = 1.0
sqrt_2gamma_kT = np.sqrt(2 * gamma * kT)

# Range of non-conservative forces to test
drive_strengths = np.linspace(0.0, 5.0, 10)
irreversibility = []

# Conservative force from double-well potential
def conservative_force(x):
    return - (4 * a * x**3 - 2 * b * x)

# Loop over drive strengths
for drive in drive_strengths:
    # Initialize position and velocity
    x = np.random.randn(n_traj)
    v = np.random.randn(n_traj) * np.sqrt(kT / mass)
    X = np.zeros((n_traj, N))
    V = np.zeros((n_traj, N))
    X[:, 0] = x
    V[:, 0] = v

    # Underdamped Langevin integration
    for t in range(1, N):
        F = conservative_force(x) + drive  # double-well + non-conservative drive
        v += dt * (-gamma * v + F) / mass + np.sqrt(dt) * sqrt_2gamma_kT / mass * np.random.randn(n_traj)
        x += dt * v
        X[:, t] = x
        V[:, t] = v

    # Irreversibility measure via squared momentum asymmetry
    x_vals = X[:, 1:-1].flatten()
    p_forward = V[:, 1:-1].flatten()
    p_backward = -V[:, :-2].flatten()
    asym = (p_forward - p_backward) / 2

    # Use normalized squared asymmetry as irreversibility measure
    irreversibility.append(np.mean(asym**2) / (kT / mass)**2)

# Plot: Irreversibility vs drive in double-well
plt.figure(figsize=(7, 4))
plt.plot(drive_strengths, irreversibility, marker='o')
plt.xlabel("Non-conservative Force Strength")
plt.ylabel("Irreversibility (Normalized Momentum Asymmetry)")
plt.title("Irreversibility vs Driving Strength in Double-Well Potential")
plt.grid(True)
plt.tight_layout()
plt.show()


##### Kevin's note #######
# you can see how irrevsersibility scales with the non-conservative force term
# can apply this to data and identify regions where flies are more reversible (close to equilibrium) and irrevserible (driven out of equ.)
##########################
