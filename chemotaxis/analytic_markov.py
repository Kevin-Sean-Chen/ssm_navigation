# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 22:39:33 2025

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

# Define fixed parameters
epsilon = 0.75       # persistance of running up
Px = 0.15            # Probability of spontaneious tumbling while going up
epsilon_t = 0.1      # Probability of dwelling in tumble
epsilon_d = 0.5      # Probability of diffusing up/down within tumble
epsilon_p = 0.2      # Probability of spontaneious tumbling while going down

# Parameter sweep ranges
Pr_vals = np.linspace(0.0, 0.5, 50)   # probability from (1,0) → (1,1)
Pt_vals = np.linspace(0.0, 0.5, 50)   # probability from (0,1) → (0,0)

# Create empty matrices to store results
drift_matrix = np.zeros((len(Pr_vals), len(Pt_vals)))
tumble_matrix = np.zeros((len(Pr_vals), len(Pt_vals)))

# State order: (1,1), (1,0), (0,1), (0,0)
for i, Pr in enumerate(Pr_vals):
    for j, Pt in enumerate(Pt_vals):
        # Construct custom transition matrix
        P = np.array([
            [epsilon,     Px,         1 - epsilon, 0          ],  # from (1,1)
            [Pr,          epsilon_t,  0,           epsilon_d  ],  # from (1,0)
            [1 - epsilon_p, 0,          epsilon_p,   Pt         ],  # from (0,1)
            [0,           epsilon_d,  Px,          epsilon_t  ]   # from (0,0)
        ])

        # Normalize each row to make stochastic
        P /= P.sum(axis=1, keepdims=True)

        # Compute steady-state
        eigvals, eigvecs = eig(P.T)
        i_stationary = np.argmin(np.abs(eigvals - 1))
        pi = np.real(eigvecs[:, i_stationary])
        pi /= np.sum(pi)

        # Metrics
        tumble_bias = pi[1] + pi[3]  # π₁₀ + π₀₀
        net_drift = pi[0] + pi[1] - pi[2] - pi[3]  # (1,1)+(1,0) - (0,1)-(0,0)

        # Store
        tumble_matrix[i, j] = tumble_bias
        drift_matrix[i, j] = net_drift

# Plot results
plt.figure(figsize=(10, 5))

# Tumble Bias plot
plt.subplot(1, 2, 1)
plt.imshow(tumble_matrix, origin='lower',
           extent=[Pt_vals[0], Pt_vals[-1], Pr_vals[0], Pr_vals[-1]], aspect='auto')
plt.colorbar(label="Tumble Bias (π₁₀ + π₀₀)")
plt.xlabel("P_t")
plt.ylabel("P_r")
plt.title("Tumble Bias")

# Net Drift plot
plt.subplot(1, 2, 2)
plt.imshow(drift_matrix, origin='lower',
           extent=[Pt_vals[0], Pt_vals[-1], Pr_vals[0], Pr_vals[-1]], aspect='auto')
plt.colorbar(label="Net Drift (↑)")
plt.xlabel("P_t")
plt.ylabel("P_r")
plt.title("Up-Gradient Drift")

plt.tight_layout()
plt.show()

# %%
# Fixed agent parameters
Px = 0.25            # spontaneous tumble up
epsilon_t = 0.25     # dwell in tumble
epsilon_d = 0.1      # diffusion within tumble

# Parameter scan ranges for re-engaging run
Pr_vals = np.linspace(0.05, 0.5, 50)
Pt_vals = np.linspace(0.05, 0.5, 50)

# Matrices to store drift in both environments
drift_flat = np.zeros((len(Pr_vals), len(Pt_vals)))
drift_gradient = np.zeros((len(Pr_vals), len(Pt_vals)))
tumble_flat = np.zeros((len(Pr_vals), len(Pt_vals)))
tumble_gradient = np.zeros((len(Pr_vals), len(Pt_vals)))

# State order: (1,1), (1,0), (0,1), (0,0)
for i, Pr in enumerate(Pr_vals):
    for j, Pt in enumerate(Pt_vals):
        # -------- Flat Environment --------
        epsilon_up = 0.5
        epsilon_down = 0.5

        ### testing tumble bias conservation
        # epsilon_d = (1-Pr)/2
        # epsilon_t = (1-Px)/2
        
        P_flat = np.array([
            [epsilon_up,  Px,         1 - epsilon_up, 0         ],
            [Pr,          epsilon_t,  0,              epsilon_d ],
            [1 - epsilon_down, 0,     epsilon_down,      Pt        ],
            [0,            epsilon_d, Px,             epsilon_t ]
        ])
        P_flat /= P_flat.sum(axis=1, keepdims=True)

        eigvals, eigvecs = eig(P_flat.T)
        pi_flat = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
        pi_flat /= pi_flat.sum()
        
        tumble_flat[i, j] = pi_flat[1] + pi_flat[3]
        drift_flat[i, j] = pi_flat[0] + pi_flat[1] - pi_flat[2] - pi_flat[3]

        # -------- Gradient Environment --------
        epsilon_up = 0.9
        epsilon_down = 0.1
        
        ### testing tumble bias conservation
        # epsilon_d = (1-Pr)/2
        # epsilon_t = (1-Px)/2
        
        P_grad = np.array([
            [epsilon_up,  Px,         1 - epsilon_up, 0         ],
            [Pr,          epsilon_t,  0,              epsilon_d ],
            [1 - epsilon_down, 0,     epsilon_down,      Pt        ],
            [0,            epsilon_d, Px,             epsilon_t ]
        ])
        P_grad /= P_grad.sum(axis=1, keepdims=True)

        eigvals, eigvecs = eig(P_grad.T)
        pi_grad = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
        pi_grad /= pi_grad.sum()
        
        tumble_gradient[i, j] = pi_grad[1] + pi_grad[3]
        drift_gradient[i, j] = pi_grad[0] + pi_grad[1] - pi_grad[2] - pi_grad[3]

# -------- Plotting --------
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(tumble_flat, origin='lower',
           extent=[Pt_vals[0], Pt_vals[-1], Pr_vals[0], Pr_vals[-1]], aspect='auto')
plt.colorbar(label="Net Drift (Flat)")
plt.xlabel("Pt")
plt.ylabel("Pr")
plt.title("Tumble Bias in flat", fontsize=20)

plt.subplot(2, 2, 2)
plt.imshow(drift_flat, origin='lower',
           extent=[Pt_vals[0], Pt_vals[-1], Pr_vals[0], Pr_vals[-1]], aspect='auto')
plt.colorbar(label="Net Drift (Flat)")
plt.xlabel("Pt")
plt.ylabel("Pr")
plt.title("Drift in Flat Environment", fontsize=20)

plt.subplot(2, 2, 3)
plt.imshow(tumble_gradient, origin='lower',
           extent=[Pt_vals[0], Pt_vals[-1], Pr_vals[0], Pr_vals[-1]], aspect='auto')
plt.colorbar(label="Net Drift (Gradient)")
plt.xlabel("Pt")
plt.ylabel("Pr")
plt.title("Tumble Bias Linear Gradient", fontsize=20)

plt.subplot(2, 2, 4)
plt.imshow(drift_gradient, origin='lower',
           extent=[Pt_vals[0], Pt_vals[-1], Pr_vals[0], Pr_vals[-1]], aspect='auto')
plt.colorbar(label="Net Drift (Gradient)")
plt.xlabel("Pt")
plt.ylabel("Pr")
plt.title("Drift in Linear Gradient", fontsize=20)

plt.tight_layout()
plt.show()

# %%
# eigvals, eigvecs = np.linalg.eig(P_flat.T)
eigvals, eigvecs = np.linalg.eig(P_grad.T)
i_stationary = np.argmin(np.abs(eigvals - 1))
pi = np.real(eigvecs[:, i_stationary])
pi /= np.sum(pi)

# Probability current: J_ij = pi_i * P_ij - pi_j * P_ji
J = np.zeros_like(P_grad)
for i in range(4):
    for j in range(4):
        J[i, j] = pi[i] * P_grad[i, j] - pi[j] * P_grad[j, i]
print(J)

# %%
P_ = P_flat
# P_ = P_grad
eigvals, eigvecs = np.linalg.eig(P_.T)
i_stationary = np.argmin(np.abs(eigvals - 1))
pi = np.real(eigvecs[:, i_stationary])
pi /= np.sum(pi)
KL_divergence = 0.0
for i in range(4):
    for j in range(4):
        if P_[i, j] > 0 and P_[j, i] > 0:
            KL_divergence += pi[i] * P_[i, j] * np.log(P_[i, j] / P_[j, i])
print(KL_divergence)

# %% NOTES
### compute entropy
### add memory state
### compare to data

# %% test SNR idea
eps_ups = np.linspace(0.1, 0.9,15)
drift = np.zeros(len(eps_ups))
Pr, Pt = 0.3, 0.3

for ii in range(len(eps_ups)):
    epsilon_up = eps_ups[ii]
    epsilon_down = 0.2 #1- eps_ups[ii]
    
    ### testing tumble bias conservation
    # epsilon_d = (1-Pr)/2
    # epsilon_t = (1-Px)/2
    
    P_grad = np.array([
        [epsilon_up,  Px,         1 - epsilon_up, 0         ],
        [Pr,          epsilon_t,  0,              epsilon_d ],
        [1 - epsilon_down, 0,     epsilon_down,      Pt        ],
        [0,            epsilon_d, Px,             epsilon_t ]
    ])
    P_grad /= P_grad.sum(axis=1, keepdims=True)
    
    eigvals, eigvecs = eig(P_grad.T)
    pi_grad = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
    pi_grad /= pi_grad.sum()
    
    drift[ii] = pi_grad[0] + pi_grad[1] - pi_grad[2] - pi_grad[3]

plt.figure()
plt.plot(eps_ups, drift, '-o')

# %% ENTRPOY decomposition
###############################################################################
def compute_markov_entropy(P, pi=None, groups=[[0,1], [2,3]]):
    """
    P: 4x4 Markov transition matrix
    pi: stationary distribution (optional), computed if not given
    groups: list of groups (e.g. [[0,1], [2,3]])
    Returns:
        H_full: entropy over all 16 state transitions
        H_group: entropy over coarse-grained 2x2 transitions
    """

    # Compute stationary distribution if not given
    if pi is None:
        eigvals, eigvecs = np.linalg.eig(P.T)
        stat = eigvecs[:, np.isclose(eigvals, 1)]
        pi = np.real(stat[:, 0])
        pi /= np.sum(pi)

    # Compute full transition entropy
    H_full = 0.0
    for i in range(4):
        for j in range(4):
            if P[i, j] > 0:
                H_full -= pi[i] * P[i, j] * np.log2(P[i, j])

    # Compute grouped transition matrix Q
    num_groups = len(groups)
    Q = np.zeros((num_groups, num_groups))
    for gi, G in enumerate(groups):
        for gj, H in enumerate(groups):
            for i in G:
                for j in H:
                    Q[gi, gj] += pi[i] * P[i, j]

    # Normalize Q (should already sum to 1)
    # Q /= Q.sum()
    Q /= Q.sum(axis=1, keepdims=True)

    # Compute group transition entropy
    H_group = 0.0
    for i in range(num_groups):
        for j in range(num_groups):
            if Q[i, j] > 0:
                H_group -= Q[i, j] * np.log2(Q[i, j])

    return H_full, H_group

# Example transition matrix
P_example = np.array([
    [0.7, 0.2, 0.05, 0.05],
    [0.1, 0.6, 0.2, 0.1],
    [0.1, 0.2, 0.5, 0.2],
    [0.05, 0.05, 0.1, 0.8]
])

H_full, H_group = compute_markov_entropy(P_example)
H_full, H_group

# %% scanning
Hxy =  np.zeros((len(Pr_vals), len(Pt_vals)))
Hxx =  np.zeros((len(Pr_vals), len(Pt_vals)))
Hyy =  np.zeros((len(Pr_vals), len(Pt_vals)))

# State order: (1,1), (1,0), (0,1), (0,0)
for i, Pr in enumerate(Pr_vals):
    for j, Pt in enumerate(Pt_vals):

        # -------- Gradient Environment --------
        epsilon_up = 0.9
        epsilon_down = 0.1
        
        ### testing tumble bias conservation
        # epsilon_d = (1-Pr)/2
        # epsilon_t = (1-Px)/2
        
        P_grad = np.array([
            [epsilon_up,  Px,         1 - epsilon_up, 0         ],
            [Pr,          epsilon_t,  0,              epsilon_d ],
            [1 - epsilon_down, 0,     epsilon_down,      Pt        ],
            [0,            epsilon_d, Px,             epsilon_t ]
        ])
        P_grad /= P_grad.sum(axis=1, keepdims=True)

        eigvals, eigvecs = eig(P_grad.T)
        pi_grad = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
        pi_grad /= pi_grad.sum()
        
        H_full, H_group = compute_markov_entropy(P_grad,  groups=[[0,1], [2,3]])
        _, Hyi = compute_markov_entropy(P_grad,  groups=[[0,2], [1,3]])
        Hxy[i,j], Hxx[i,j], Hyy[i,j] = H_full, H_group, Hyi

# %% plotting
plt.figure(figsize=(20, 7))

plt.subplot(1, 3, 1)
plt.imshow(Hxy, origin='lower',
           extent=[Pt_vals[0], Pt_vals[-1], Pr_vals[0], Pr_vals[-1]], aspect='auto')
plt.colorbar(label="Net Drift (Flat)")
plt.xlabel("Pt")
plt.ylabel("Pr")
plt.title("Tumble Bias in flat", fontsize=20)

plt.subplot(1, 3, 2)
plt.imshow(Hxx/1, origin='lower',
           extent=[Pt_vals[0], Pt_vals[-1], Pr_vals[0], Pr_vals[-1]], aspect='auto')
plt.colorbar(label="Net Drift (Flat)")
plt.xlabel("Pt")
plt.ylabel("Pr")
plt.title("Drift in Flat Environment", fontsize=20)

plt.subplot(1, 3, 3)
plt.imshow(Hyy/1, origin='lower',
           extent=[Pt_vals[0], Pt_vals[-1], Pr_vals[0], Pr_vals[-1]], aspect='auto')
plt.colorbar(label="Net Drift (Gradient)")
plt.xlabel("Pt")
plt.ylabel("Pr")
plt.title("Tumble Bias Linear Gradient", fontsize=20)


