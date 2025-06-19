# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 00:11:24 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_search(T=15, noise_type="none"):
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

    # # Define likelihood field
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
        
        # Define likelihood field
        if noise_type == "none":
            sensor_likelihood = np.exp(-(x_space - true_pos)**2 / (2*0.9**2))  # Very sharp Gaussian
        elif noise_type == "spatial":
            sensor_likelihood = np.exp(-(x_space - true_pos)**2 / (2*5.0**2))  # Broad Gaussian
        elif noise_type == "temporal":
            sensor_likelihood = np.exp(-(x_space - true_pos)**2 / (2*0.9**2))  # Sharp Gaussian
        else:
            raise ValueError("noise_type must be 'none', 'spatial', or 'temporal'")

        sensor_likelihood /= np.sum(sensor_likelihood)  # normalize properly over x
        

        # Simulate observation
        prob_detect = np.interp(x_curr, x_space, sensor_likelihood)
        
        if noise_type == "temporal":
            # Temporal noise: detection is noisy even when sitting still
            detection_noise = 50  # not too strong now
            detected = np.random.rand() < (prob_detect * detection_noise)
        else:
            detected = np.random.rand() < prob_detect

        # Bayes update
        # if detected:
        #     obs_likelihood = 1-likelihood
        # else:
        #     obs_likelihood = likelihood
            
        # if detected:
        #     obs_likelihood = likelihood
        # else:
        #     obs_likelihood = 1 - likelihood
        
        if detected:
            obs_likelihood = 1 - sensor_likelihood
        else:
            obs_likelihood = sensor_likelihood


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
            axs[t].legend(loc='upper left', bbox_to_anchor=(-2, 1.0), fontsize=15) #(fontsize=15)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.85, 1])#tight_layout()
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

# %%
###############################################################################
# %% simpler demo
from scipy.stats import norm

# Define the 1D domain
x = np.linspace(-10, 10, 500)

# True value (delta function center, hidden from the model)
true_value = 2.0

# Flat prior (uniform distribution over the domain)
prior = np.ones_like(x)
prior /= np.trapz(prior, x)  # Normalize

# Function to compute likelihood for a given observation
def compute_likelihood(x, observation, noise_std):
    return norm.pdf(x, loc=observation, scale=noise_std)

# Bayesian update
def bayesian_update(prior, likelihood, x):
    posterior_unnorm = prior * likelihood
    posterior = posterior_unnorm / np.trapz(posterior_unnorm, x)
    return posterior

# Simulate observations around true value
np.random.seed(42)
noise_std = 1.0
num_observations = 5
observations = np.random.normal(loc=true_value, scale=noise_std, size=num_observations)

# Initialize plot
plt.figure(figsize=(10, 6))
posterior = prior.copy()

for i, obs in enumerate(observations):
    obs = true_value*1
    likelihood = compute_likelihood(x, obs, noise_std)
    posterior = bayesian_update(posterior, likelihood, x)
    
    # Plot each update
    plt.clf()
    plt.plot(x, prior, label='Prior', linestyle='--', alpha=0.5)
    plt.plot(x, likelihood, label=f'Likelihood (obs {i+1})', linestyle=':', alpha=0.7)
    plt.plot(x, posterior, label='Posterior', linewidth=2)
    plt.axvline(true_value, color='k', linestyle='--', label='True Value')
    plt.title(f'Bayesian Update - Step {i+1}')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.pause(0.8)
    
    # Update prior for next round
    prior = posterior

plt.show()
