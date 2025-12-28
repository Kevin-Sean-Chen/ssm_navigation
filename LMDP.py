# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 01:48:35 2025

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt

# Define a simple 1D grid world
n_states = 50
goal_state = n_states - 1
states = np.arange(n_states)
actions = [-1, 1]  # left, right

# Transition dynamics: deterministic
def step(state, action):
    return np.clip(state + action, 0, n_states - 1)

# Passive dynamics: do nothing (stay in place)
def passive_transition(state):
    # Choose change from {-1, 0, +1} with equal probability
    delta = np.random.choice([-1, 0, 1])
    next_state = state + delta
    return np.clip(next_state, 0, n_states - 1)


# Reward function: goal state has reward 0, others have -1
Q = -np.ones(n_states)
Q[goal_state] = 0
Q = -np.arange(0, n_states)/n_states
# ### make Q peice-wise linear, with positive and negative slopes
# # Ensure Q is a float array and compute slice lengths robustly
# Q = Q.astype(float)
# l1 = n_states // 3
# l2 = n_states // 3
# l3 = n_states - l1 - l2
# if l1 > 0:
#     Q[:l1] = np.linspace(0.0, 1.0, l1)
# if l2 > 0:
#     Q[l1:l1 + l2] = np.linspace(1.0, -1.0, l2)
# if l3 > 0:
#     Q[l1 + l2:] = np.linspace(-1.0, 0.0, l3)

### test with complex Q form
# Gaussian bumps: tune centers (means), widths (std devs) and amplitudes here
means = [5, 30]                # center locations of the Gaussian bumps (state indices)
sigmas = [2.0, 5.0]            # widths (standard deviations) for each bump
amps = [1.0, 10.0]              # amplitudes for each bump

Q = np.zeros_like(states, dtype=float)
for mu, sigma, amp in zip(means, sigmas, amps):
    Q += amp * np.exp(-0.5 * ((states - mu) / sigma) ** 2)

# Optional: normalize to [0,1] and/or invert to represent costs instead of rewards
Q = Q / Q.max()   # normalize
Q = -Q            # invert so peaks become low-cost valleys (remove if you prefer positive bumps)
Q = Q - Q.min() + 1  # make non-negative

plt.figure()
plt.plot(states, Q, '-o')
plt.xlabel('State')
plt.ylabel('Cost-to-go Q(s)')
plt.title('Cost-to-go Function Q(s)')
plt.grid()
plt.show()
# Compute cost for a given policy at state: Q + beta * KL
def compute_policy_cost(beta):
    policy_costs = []
    policy_entropies = []
    for s in states:
        probs = []
        kl_terms = []
        for a in actions:
            s_next = step(s, a)
            prob = np.exp(-Q[s_next] / beta)
            probs.append(prob)
        probs = np.array(probs)
        probs /= probs.sum()  # Normalize policy
        policy_cost = 0
        entropy = 0
        for i, a in enumerate(actions):
            s_next = step(s, a)
            p = probs[i]
            kl = np.log(p + 1e-12)  # KL to passive (uniform or fixed action)
            policy_cost += p * (Q[s_next] + beta * kl)
            entropy += -p * np.log(p + 1e-12)
        policy_costs.append(policy_cost)
        policy_entropies.append(entropy)
    return np.mean(policy_costs), np.mean(policy_entropies)

# Sweep over beta
betas = np.linspace(0.01, 5, 100)
costs = []
entropies = []

for beta in betas:
    cost, entropy = compute_policy_cost(beta)
    costs.append(cost)
    entropies.append(entropy)

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(betas, costs, label='Average Cost')
plt.xlabel('β (inverse temperature)')
plt.ylabel('Expected Cost')
plt.title('Policy Cost vs β')

plt.subplot(1, 2, 2)
plt.plot(betas, entropies, label='Entropy', color='orange')
plt.xlabel('β (inverse temperature)')
plt.ylabel('Policy Entropy')
plt.title('Entropy vs β')

plt.tight_layout()
plt.show()

# %%
import random
from collections import defaultdict
from sklearn.metrics import mutual_info_score
from matplotlib import cm, colors

# Set random seed for reproducibility
# np.random.seed(42)
# random.seed(42)

# Generate trajectory samples for a given beta
def generate_trajectory(beta, T=1000):
    trajectory = []
    s = 0  # start state
    for t in range(T):
        probs = []
        for a in actions:
            s_next = step(s, a)
            prob = np.exp(-Q[s_next] / beta)
            probs.append(prob)
        probs = np.array(probs)
        probs /= probs.sum()
        a = np.random.choice(actions, p=probs)
        s_next = step(s, a)
        trajectory.append((s, a, s_next))
        s = s_next
    return trajectory

# Estimate TE(s→a) and TE(a→s) via plug-in estimator
def TE_track(trajectory):
    sa_counts = defaultdict(int)
    s_counts = defaultdict(int)
    a_counts = defaultdict(int)
    sas_counts = defaultdict(int)
    as_counts = defaultdict(int)

    for (s, a, s_next) in trajectory:
        sa_counts[(s, a)] += 1
        s_counts[s] += 1
        a_counts[a] += 1
        sas_counts[(s, a, s_next)] += 1
        as_counts[(a, s_next)] += 1

    # Compute TE(s→a)
    te_s_to_a = 0
    for (s, a), count in sa_counts.items():
        p_sa = count / len(trajectory)
        p_s = s_counts[s] / len(trajectory)
        p_a = a_counts[a] / len(trajectory)
        te_s_to_a += p_sa * np.log((p_sa + 1e-12) / (p_s * p_a + 1e-12))

    # Compute TE(a→s')
    te_a_to_s = 0
    for (a, s_next), count in as_counts.items():
        p_as = count / len(trajectory)
        p_a = a_counts[a] / len(trajectory)
        p_snext = sum([v for (s, a_, sn), v in sas_counts.items() if sn == s_next]) / len(trajectory)
        te_a_to_s += p_as * np.log((p_as + 1e-12) / (p_a * p_snext + 1e-12))

    return te_s_to_a, te_a_to_s

def transfer_entropy(X, Y, delay=1):
    n = float(len(X[delay:]))
    binX = len(np.unique(X))
    binY = len(np.unique(Y))
    
    x3 = np.array([X[delay:], Y[:-delay], X[:-delay]])
    x2 = np.array([X[delay:], Y[:-delay]])
    x2_delay = np.array([X[delay:], X[:-delay]])
    
    # Compute histograms
    p3, _ = np.histogramdd(x3.T, bins=[binX, binY, binX])
    p2, _ = np.histogramdd(x2.T, bins=[binX, binY])
    p2delay, _ = np.histogramdd(x2_delay.T, bins=[binX, binX])
    p1, _ = np.histogramdd(np.array(X[delay:]).reshape(-1,1), bins=binX)
    
    # Add small pseudocount to avoid zeros
    epsilon = 1e-8
    p3 = (p3 + epsilon) / np.sum(p3 + epsilon)
    p2 = (p2 + epsilon) / np.sum(p2 + epsilon)
    p2delay = (p2delay + epsilon) / np.sum(p2delay + epsilon)
    p1 = (p1 + epsilon) / np.sum(p1 + epsilon)
    
    # Compute TE
    TE = 0.0
    for i in range(binX):
        for j in range(binY):
            for k in range(binX):
                TE += p3[i,j,k] * np.log2( (p3[i,j,k] * p1[i]) / (p2[i,j] * p2delay[i,k]) )
    
    return TE


def TE_local(trajectory):
    temp = np.array(traj)
    at = temp[:,1][:-1]
    st = (temp[:,-1] - temp[:,0])[1:]
    tsa, tas = transfer_entropy(st, at), transfer_entropy(at, st, delay=1)
    return tsa, tas
    

# %%
# Sweep over beta and compute TE
beta_values = np.linspace(0.01, 10, 50)
te_s_to_a_list = []
te_a_to_s_list = []
free_energy = []

for beta in beta_values:
    ### generate tracks
    traj = generate_trajectory(beta)
    
    ### measure behaivor
    # te_s_a, te_a_s = TE_track(traj)
    te_s_a, te_a_s = TE_local(traj)
    te_s_to_a_list.append(te_s_a)
    te_a_to_s_list.append(te_a_s)
    
    ### measure
    cost, entropy = compute_policy_cost(beta)
    free_energy.append(cost+entropy*1)

# Plotting TE vs beta
plt.figure(figsize=(8, 5))
plt.plot(beta_values, te_s_to_a_list, label='TE(s → a)')
plt.plot(beta_values, te_a_to_s_list, label='TE(a → s)')
# plt.plot(beta_values, free_energy,'--')
plt.xlabel('β (inverse temperature)')
plt.ylabel('Transfer Entropy')
plt.title('Transfer Entropy vs β')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

# First y-axis for TE values
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(beta_values, te_s_to_a_list, label='TE(s → a)')
ax1.plot(beta_values, te_a_to_s_list, label='TE(a → s)')
ax1.set_xlabel('β (inverse temperature)')
ax1.set_ylabel('Transfer Entropy')
ax1.grid(True)

# Second y-axis for free energy
ax2 = ax1.twinx()
ax2.plot(beta_values, free_energy, '--', color='tab:red', label='Free Energy')
ax2.set_ylabel('Free Energy')

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

plt.title('Transfer Entropy and Free Energy vs β')
plt.tight_layout()
plt.show()

### plot some tracks
plt.figure(figsize=(10, 6))
# plot two trajectories for 4 equally-spaced beta values, color-coded by beta
betas_to_plot = np.linspace(beta_values.min(), beta_values.max(), 4)
cmap = cm.viridis
norm = colors.Normalize(vmin=betas_to_plot.min(), vmax=betas_to_plot.max())
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

for b in betas_to_plot:
    c = cmap(norm(b))
    for j in range(2):  # two trajectories per beta
        traj = generate_trajectory(beta=b, T=1000)
        states_traj = [s for (s, a, s_next) in traj]
        plt.plot(states_traj, color=c, alpha=0.6, linewidth=0.8)

plt.xlabel('Time step')
plt.ylabel('State')
plt.title('Sample Trajectories (2 per β, 4 equally spaced β values)')
ax = plt.gca()
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('β')
plt.grid()
plt.show()
