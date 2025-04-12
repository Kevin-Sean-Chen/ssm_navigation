# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 00:11:24 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_search(T=5, noise_type="none"):
    """
    Simulate Bayesian search in 1D with different noise types.

    Args:
        T: number of time steps
        noise_type: 'none', 'spatial', or 'temporal'
    """

    np.random.seed(0)
    x_space = np.linspace(-10, 10, 200)  # 1D space
    true_pos = 3.0  # hidden source location

    belief = np.ones_like(x_space) / len(x_space)  # uniform prior

    # Define likelihood field
    if noise_type == "none":
        likelihood = np.exp(-(x_space - true_pos)**2 / (2*0.9**2))  # Very sharp Gaussian
    elif noise_type == "spatial":
        likelihood = np.exp(-(x_space - true_pos)**2 / (2*5.0**2))  # Broad Gaussian
    elif noise_type == "temporal":
        likelihood = np.exp(-(x_space - true_pos)**2 / (2*0.9**2))  # Sharp Gaussian
    else:
        raise ValueError("noise_type must be 'none', 'spatial', or 'temporal'")

    likelihood /= np.sum(likelihood)  # normalize properly over x

    all_beliefs = [belief.copy()]

    for t in range(T):
        # Pick greedy move: go to maximum a posteriori (MAP)
        idx = np.argmax(belief)
        x_curr = x_space[idx]

        # Simulate observation
        prob_detect = np.interp(x_curr, x_space, likelihood)
        
        if noise_type == "temporal":
            # Temporal noise: detection is noisy even when sitting still
            detection_noise = 20  # not too strong now
            detected = np.random.rand() < (prob_detect * detection_noise)
        else:
            detected = np.random.rand() < prob_detect

        # Bayes update
        if detected:
            obs_likelihood = 1-likelihood
        else:
            obs_likelihood = likelihood

        belief = belief * obs_likelihood
        belief /= np.sum(belief)

        all_beliefs.append(belief.copy())

    return x_space, all_beliefs, true_pos, likelihood

def plot_comparison(x_space, all_beliefs, true_pos, likelihood, title):
    """
    Plot belief evolution with normalized likelihood and true target position.
    """
    fig, axs = plt.subplots(1, 5, figsize=(10, 3), sharex=True, sharey=True)
    axs = axs.ravel()

    likelihood_norm = likelihood / np.max(likelihood)  # normalize for plotting

    for t in range(min(len(all_beliefs), len(axs))):
        belief_norm = all_beliefs[t] / np.max(all_beliefs[t])  # normalize for plotting

        axs[t].plot(x_space, belief_norm, label='Belief p(x) (norm)')
        axs[t].plot(x_space, likelihood_norm, '--', label='Likelihood p(obs|x) (norm)')
        axs[t].axvline(true_pos, color='r', linestyle='--', label='Target')

        axs[t].set_title(f"Time step {t}")

        if t == 0:
            axs[t].legend(fontsize=15)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------
# Simulate all three cases
x_space, beliefs_none, true_pos, likelihood_none = simulate_search(noise_type="none")
x_space, beliefs_spatial, true_pos, likelihood_spatial = simulate_search(noise_type="spatial")
x_space, beliefs_temporal, true_pos, likelihood_temporal = simulate_search(noise_type="temporal")

# ----------------------------------------------------------
# Plot side-by-side comparison

plot_comparison(x_space, beliefs_none, true_pos, likelihood_none, 
                title="Ideal case (No noise): Sharp Likelihood, Reliable Detection")

plot_comparison(x_space, beliefs_spatial, true_pos, likelihood_spatial,
                title="Spatial Noise: Broad Likelihood, Unreliable Spatial Detection")

plot_comparison(x_space, beliefs_temporal, true_pos, likelihood_temporal,
                title="Temporal Noise: Sharp Likelihood, Noisy Detection")
