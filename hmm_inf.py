# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:13:03 2024

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt

# %% generate simple HMM
def generate_hmm_sequence(transition_matrix, means, std_devs, seq_length):
    """
    Generate an HMM sequence given a transition matrix and emission parameters.
    
    Args:
    transition_matrix (np.array): A 2x2 matrix for state transition probabilities.
    means (list): A list of means for the emission distribution of each state.
    std_devs (list): A list of standard deviations for the emission distribution of each state.
    seq_length (int): Length of the sequence to generate.
    
    Returns:
    states (list): The sequence of hidden states.
    emissions (list): The sequence of observed emissions.
    """
    num_states = transition_matrix.shape[0]
    
    # Initialize the sequences
    states = []
    emissions = []
    
    # Initial state (randomly choose the starting state)
    current_state = np.random.choice(num_states)
    states.append(current_state)
    
    # Generate the sequence
    for t in range(1, seq_length):
        # Transition to the next state based on the transition matrix
        current_state = np.random.choice(num_states, p=transition_matrix[current_state])
        states.append(current_state)
        
        # Emit a value based on the state's emission distribution (Gaussian)
        emission = np.random.normal(means[current_state], std_devs[current_state])
        emissions.append(emission)
    
    return states, emissions

# Example usage
transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])  # Transition matrix (2x2)
means = [0, 3]                                          # Means for state 0 and state 1
std_devs = [0.5, 1.0]                                   # Standard deviations for state 0 and state 1
seq_length = 100                                        # Length of the sequence

# Generate the HMM sequence
states, emissions = generate_hmm_sequence(transition_matrix, means, std_devs, seq_length)

# Plot the time series
plt.figure(figsize=(12, 6))

# Plot emissions
plt.subplot(2, 1, 1)
plt.plot(emissions, label="Emissions", color='b')
plt.title('HMM Emissions Time Series')
plt.ylabel('Emissions')
plt.legend()

# Plot states
plt.subplot(2, 1, 2)
plt.step(range(seq_length), states, where='mid', label="States", color='r')
plt.title('HMM States Time Series')
plt.ylabel('States')
plt.xlabel('Time')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# %% inference for HMM
import numpy as np
import matplotlib.pyplot as plt

# Log-sum-exp trick for numerical stability
def log_sum_exp(log_probs):
    max_log_prob = np.max(log_probs)
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

# Forward algorithm using log probabilities
def forward_algorithm_log(transition_matrix, emission_matrix, initial_probs, observations):
    num_states = len(initial_probs)
    seq_length = len(observations)
    log_alpha = np.zeros((seq_length, num_states))
    
    # Initialize log alpha at time t=0
    log_alpha[0, :] = np.log(initial_probs) + np.log(emission_matrix[:, observations[0]])
    
    # Recursion: compute log alpha for t = 1 to seq_length-1
    for t in range(1, seq_length):
        for j in range(num_states):
            log_alpha[t, j] = log_sum_exp(log_alpha[t-1, :] + np.log(transition_matrix[:, j])) + np.log(emission_matrix[j, observations[t]])
    
    return log_alpha

# Backward algorithm using log probabilities
def backward_algorithm_log(transition_matrix, emission_matrix, observations):
    num_states = transition_matrix.shape[0]
    seq_length = len(observations)
    log_beta = np.zeros((seq_length, num_states))
    
    # Initialize log beta at the last time step t = T-1
    log_beta[seq_length - 1, :] = 0  # log(1) = 0 since the probability of the final state is 1
    
    # Recursion: compute log beta for t = T-2 down to 0
    for t in range(seq_length - 2, -1, -1):
        for i in range(num_states):
            log_beta[t, i] = log_sum_exp(np.log(transition_matrix[i, :]) + np.log(emission_matrix[:, observations[t+1]]) + log_beta[t+1, :])
    
    return log_beta

# Compute posterior probabilities using log alpha and log beta
def compute_posterior_log(log_alpha, log_beta):
    log_gamma = log_alpha + log_beta
    log_gamma -= log_sum_exp(log_gamma.T)  # Normalize
    return np.exp(log_gamma)  # Convert back to probabilities

# Test with generating HMM sequence and apply the log-based forward-backward algorithm
def generate_hmm_sequence(transition_matrix, means, std_devs, seq_length):
    num_states = transition_matrix.shape[0]
    
    states = []
    emissions = []
    
    # Start from a random state
    current_state = np.random.choice(num_states)
    states.append(current_state)
    
    # Generate the sequence
    for t in range(seq_length):
        current_state = np.random.choice(num_states, p=transition_matrix[current_state])
        states.append(current_state)
        
        # Emit values based on the current state's Gaussian distribution
        emission = np.random.normal(means[current_state], std_devs[current_state])
        emissions.append(emission)
    
    return states, emissions

# Discretize emissions to match with emission matrix (for simplicity: state 0 or 1)
def discretize_observations(emissions, means):
    return [0 if abs(em - means[0]) < abs(em - means[1]) else 1 for em in emissions]

# Plotting function to compare true hidden states and posterior probabilities
def plot_hmm_inference(true_states, posterior_probs, seq_length):
    time_steps = np.arange(seq_length)
    
    plt.figure(figsize=(12, 6))
    
    # Plot true hidden states
    plt.subplot(2, 1, 1)
    plt.step(time_steps, true_states, where='mid', label="True States", color='r')
    plt.title('True Hidden States')
    plt.ylabel('State')
    
    # Plot posterior probabilities for both states
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, posterior_probs[:, 0], label="P(State=0)", color='b')
    plt.plot(time_steps, posterior_probs[:, 1], label="P(State=1)", color='g')
    plt.title('Posterior Probabilities of States (Log Forward-Backward)')
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example parameters
transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])  # State transition matrix
means = [0, 3]                                          # Means of emissions for states 0 and 1
std_devs = [0.5, 1.0]                                   # Standard deviations of emissions
seq_length = 100                                        # Length of the sequence

# Generate a sequence of states and emissions
true_states, emissions = generate_hmm_sequence(transition_matrix, means, std_devs, seq_length)

# Discretize the emissions to match with emission matrix (for simplicity: state 0 or 1)
observations = discretize_observations(emissions, means)

# Initial HMM parameters
emission_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])    # Example emission matrix
initial_probs = np.array([0.6, 0.4])                   # Initial probabilities for states

# Run the Forward-Backward algorithm using log probabilities
log_alpha = forward_algorithm_log(transition_matrix, emission_matrix, initial_probs, observations)
log_beta = backward_algorithm_log(transition_matrix, emission_matrix, observations)
posterior_probs = compute_posterior_log(log_alpha, log_beta)

# Plot the true hidden states and the posterior probabilities
plot_hmm_inference(true_states[:seq_length], posterior_probs, seq_length)


# %% inference and generation of AR-HMM
import numpy as np
import matplotlib.pyplot as plt

# Generate the AR-HMM sequence
def generate_arhmm_sequence(transition_matrix, means, std_devs, weights, seq_length):
    num_states = transition_matrix.shape[0]
    
    states = []
    emissions = []
    
    # Start from a random state
    current_state = np.random.choice(num_states)
    states.append(current_state)
    
    # Generate the first emission independently (no autoregression at t=0)
    first_emission = np.random.normal(means[current_state], std_devs[current_state])
    emissions.append(first_emission)
    
    # Generate the rest of the sequence
    for t in range(1, seq_length):
        # Transition to the next state
        current_state = np.random.choice(num_states, p=transition_matrix[current_state])
        states.append(current_state)
        
        # Emit values based on autoregressive model: mean + weight * past_emission + noise
        emission = (means[current_state] + 
                    weights[current_state] * emissions[-1] +  # autoregressive component
                    np.random.normal(0, std_devs[current_state]))  # noise
        emissions.append(emission)
    
    return states, emissions

# Discretize observations (this step might change based on AR model behavior)
def discretize_observations(emissions, means):
    return [0 if abs(em - means[0]) < abs(em - means[1]) else 1 for em in emissions]

# Log-sum-exp trick for numerical stability
def log_sum_exp(log_probs):
    max_log_prob = np.max(log_probs)
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

# Forward algorithm using log probabilities for AR-HMM
def forward_algorithm_log_ar(transition_matrix, emission_matrix, initial_probs, observations, emissions, means, weights, std_devs):
    num_states = len(initial_probs)
    seq_length = len(observations)
    log_alpha = np.zeros((seq_length, num_states))
    
    # Initialize log alpha at time t=0
    log_alpha[0, :] = np.log(initial_probs) + np.log(emission_matrix[:, observations[0]])
    
    # Recursion: compute log alpha for t = 1 to seq_length-1
    for t in range(1, seq_length):
        for j in range(num_states):
            # Compute the mean and std dev of the emission based on AR model
            emission_mean = means[j] + weights[j] * emissions[t-1]
            emission_std = std_devs[j]
            emission_prob = (1 / (np.sqrt(2 * np.pi) * emission_std)) * np.exp(
                -0.5 * ((emissions[t] - emission_mean) / emission_std) ** 2)
            
            log_alpha[t, j] = (log_sum_exp(log_alpha[t-1, :] + np.log(transition_matrix[:, j])) +
                               np.log(emission_prob))
    
    return log_alpha

# Backward algorithm using log probabilities for AR-HMM
def backward_algorithm_log_ar(transition_matrix, emission_matrix, observations, emissions, means, weights, std_devs):
    num_states = transition_matrix.shape[0]
    seq_length = len(observations)
    log_beta = np.zeros((seq_length, num_states))
    
    # Initialize log beta at the last time step t = T-1
    log_beta[seq_length - 1, :] = 0  # log(1) = 0 since the probability of the final state is 1
    
    # Recursion: compute log beta for t = T-2 down to 0
    for t in range(seq_length - 2, -1, -1):
        for i in range(num_states):
            log_emission_probs = np.zeros(num_states)
            for j in range(num_states):
                emission_mean = means[j] + weights[j] * emissions[t]
                emission_std = std_devs[j]
                emission_prob = (1 / (np.sqrt(2 * np.pi) * emission_std)) * np.exp(
                    -0.5 * ((emissions[t+1] - emission_mean) / emission_std) ** 2)
                log_emission_probs[j] = np.log(emission_prob)
            
            log_beta[t, i] = log_sum_exp(np.log(transition_matrix[i, :]) + log_emission_probs + log_beta[t+1, :])
    
    return log_beta

# Compute posterior probabilities using log alpha and log beta
def compute_posterior_log(log_alpha, log_beta):
    log_gamma = log_alpha + log_beta
    log_gamma -= log_sum_exp(log_gamma.T)  # Normalize
    return np.exp(log_gamma)  # Convert back to probabilities

# Plotting function to compare true hidden states and posterior probabilities
def plot_hmm_inference(true_states, posterior_probs, seq_length, emissions):
    time_steps = np.arange(seq_length)
    
    plt.figure(figsize=(12, 6))
    
    # Plot true hidden states
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, emissions)
    plt.title('data')
    plt.ylabel('Observation')
    
    plt.subplot(3, 1, 2)
    plt.step(time_steps, true_states, where='mid', label="True States", color='r')
    plt.title('True Hidden States')
    plt.ylabel('State')
    
    # Plot posterior probabilities for both states
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, posterior_probs[:, 0], label="P(State=0)", color='b')
    plt.plot(time_steps, posterior_probs[:, 1], label="P(State=1)", color='g')
    plt.title('Posterior Probabilities of States (Log Forward-Backward AR-HMM)')
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example parameters
transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])  # State transition matrix
means = [0, 3]                                          # Means of emissions for states 0 and 1
std_devs = [0.5, 1.0]                                   # Standard deviations of emissions
weights = [0.7, 0.4]                                    # Autoregressive weights for states
seq_length = 100                                        # Length of the sequence

# Generate a sequence of states and emissions using AR-HMM
true_states, emissions = generate_arhmm_sequence(transition_matrix, means, std_devs, weights, seq_length)

# Discretize the emissions to match with emission matrix (for simplicity: state 0 or 1)
observations = discretize_observations(emissions, means)

# Initial HMM parameters
emission_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])    # Example emission matrix
initial_probs = np.array([0.6, 0.4])                   # Initial probabilities for states

# Run the Forward-Backward algorithm using log probabilities for AR-HMM
log_alpha = forward_algorithm_log_ar(transition_matrix, emission_matrix, initial_probs, observations, emissions, means, weights, std_devs)
log_beta = backward_algorithm_log_ar(transition_matrix, emission_matrix, observations, emissions, means, weights, std_devs)
posterior_probs = compute_posterior_log(log_alpha, log_beta)

# Plot the true hidden states and the posterior probabilities
plot_hmm_inference(true_states[:seq_length], posterior_probs, seq_length, emissions)

# %%
# Example parameters for three states
transition_matrix = np.array([[0.7, 0.2, 0.1], 
                              [0.3, 0.5, 0.2], 
                              [0.2, 0.3, 0.5]])  # 3x3 transition matrix for 3 states
means = [0, 3, 5]                               # Means of emissions for states 0, 1, and 2
std_devs = [0.5, 1.0, 0.7]                      # Standard deviations of emissions for each state
weights = [0.7, 0.4, 0.6]                       # Autoregressive weights for states
seq_length = 100                                # Length of the sequence

# Generate a sequence of states and emissions using AR-HMM for three states
true_states, emissions = generate_arhmm_sequence(transition_matrix, means, std_devs, weights, seq_length)

# Discretize the emissions to match with emission matrix (state 0, 1, or 2)
observations = discretize_observations(emissions, means)

# Initial HMM parameters for three states
emission_matrix = np.array([[0.9, 0.05, 0.05], 
                            [0.1, 0.8, 0.1], 
                            [0.05, 0.1, 0.85]])    # 3x3 emission matrix
initial_probs = np.array([0.5, 0.3, 0.2])          # Initial probabilities for the three states

# Run the Forward-Backward algorithm using log probabilities for AR-HMM
log_alpha = forward_algorithm_log_ar(transition_matrix, emission_matrix, initial_probs, observations, emissions, means, weights, std_devs)
log_beta = backward_algorithm_log_ar(transition_matrix, emission_matrix, observations, emissions, means, weights, std_devs)
posterior_probs = compute_posterior_log(log_alpha, log_beta)

# Plot the true hidden states and the posterior probabilities
plot_hmm_inference(true_states[:seq_length], posterior_probs, seq_length, emissions)

