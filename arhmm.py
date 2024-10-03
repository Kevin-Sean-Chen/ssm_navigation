# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:45:18 2024

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# %%
class ARHMM:
    def __init__(self, transition_matrix, means, covariances, weights, initial_probs, p):
        """
        Initialize the AR-HMM model with parameters.
        
        Parameters:
        - transition_matrix: State transition matrix (KxK).
        - means: Mean vectors of size (KxD) where K is the number of states and D is the observation dimension.
        - covariances: Covariance matrices for each state (KxDxD).
        - weights: Autoregressive weights for each state (KxPxD), where P is the autoregressive lag order.
        - initial_probs: Initial state distribution (K).
        - p: The number of autoregressive steps (AR order).
        """
        self.transition_matrix = transition_matrix
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.initial_probs = initial_probs
        self.p = p  # autoregressive order
        self.num_states = len(initial_probs)
    
    def generate_sequence(self, seq_length):
        """
        Generate a sequence of hidden states and emissions from the AR-HMM.
        
        Parameters:
        - seq_length: The length of the sequence to generate.
        
        Returns:
        - states: List of hidden states (length seq_length).
        - emissions: Array of emissions (seq_length x D).
        """
        states = []
        emissions = []
        
        # Start from a random initial state
        current_state = np.random.choice(self.num_states, p=self.initial_probs)
        states.append(current_state)
        
        # Generate the first emission independently (no autoregression at t=0)
        first_emission = np.random.multivariate_normal(self.means[current_state], self.covariances[current_state])
        emissions.append(first_emission)
        
        # Generate the rest of the sequence
        for t in range(1, seq_length):
            # Transition to the next state
            current_state = np.random.choice(self.num_states, p=self.transition_matrix[states[-1]])
            states.append(current_state)
            
            # Compute the autoregressive term
            autoregressive_term = np.zeros(emissions[-1].shape)
            for lag in range(1, min(self.p, t) + 1):
                autoregressive_term += self.weights[current_state][lag - 1] * emissions[t - lag]
            
            # Generate the emission
            emission_mean = self.means[current_state] + autoregressive_term
            emission = np.random.multivariate_normal(emission_mean, self.covariances[current_state])
            emissions.append(emission)
        
        return states, np.array(emissions)
    
    def forward_algorithm(self, emissions):
        """
        Forward algorithm for AR-HMM with log probabilities.
        
        Parameters:
        - emissions: The observed sequence of emissions (NxD).
        
        Returns:
        - log_alpha: The forward log-probabilities (NxK).
        """
        seq_length = len(emissions)
        log_alpha = np.zeros((seq_length, self.num_states))
        
        # Initialize log alpha at time t=0
        for j in range(self.num_states):
            emission_prob = multivariate_normal.pdf(emissions[0], self.means[j], self.covariances[j])
            log_alpha[0, j] = np.log(self.initial_probs[j] * max(emission_prob, 1e-10))  # Prevent log(0)
        
        # Recursion: compute log alpha for t = 1 to seq_length-1
        for t in range(1, seq_length):
            for j in range(self.num_states):
                autoregressive_term = np.zeros(emissions[t].shape)
                for lag in range(1, min(self.p, t) + 1):
                    autoregressive_term += self.weights[j][lag - 1] * emissions[t - lag]
                
                emission_mean = self.means[j] + autoregressive_term
                emission_prob = multivariate_normal.pdf(emissions[t], emission_mean, self.covariances[j])
                
                # Compute log alpha using the log-sum-exp trick
                log_alpha[t, j] = np.log(max(emission_prob, 1e-10)) + log_sum_exp(log_alpha[t-1, :] + np.log(self.transition_matrix[:, j]))
        
        return log_alpha
    
    def backward_algorithm(self, emissions):
        """
        Backward algorithm for AR-HMM with log probabilities.
        
        Parameters:
        - emissions: The observed sequence of emissions (NxD).
        
        Returns:
        - log_beta: The backward log-probabilities (NxK).
        """
        seq_length = len(emissions)
        log_beta = np.zeros((seq_length, self.num_states))
        
        # Recursion: compute log beta for t = T-2 down to 0
        for t in range(seq_length - 2, -1, -1):
            for i in range(self.num_states):
                log_emission_probs = np.zeros(self.num_states)
                for j in range(self.num_states):
                    autoregressive_term = np.zeros(emissions[t].shape)
                    for lag in range(1, min(self.p, t+1) + 1):
                        autoregressive_term += self.weights[j][lag - 1] * emissions[t + 1 - lag]
                    
                    emission_mean = self.means[j] + autoregressive_term
                    emission_prob = multivariate_normal.pdf(emissions[t + 1], emission_mean, self.covariances[j])
                    log_emission_probs[j] = np.log(max(emission_prob, 1e-10))
                
                log_beta[t, i] = log_sum_exp(np.log(self.transition_matrix[i, :]) + log_emission_probs + log_beta[t+1, :])
        
        return log_beta
    
    def compute_posteriors(self, log_alpha, log_beta):
        """
        Compute the posterior probabilities from log alpha and log beta.
        
        Parameters:
        - log_alpha: The forward log-probabilities (NxK).
        - log_beta: The backward log-probabilities (NxK).
        
        Returns:
        - posterior_probs: Posterior probabilities (NxK).
        """
        log_gamma = log_alpha + log_beta
        log_gamma -= log_sum_exp(log_gamma.T)  # Normalize
        return np.exp(log_gamma)  # Convert back to probabilities
    
    def plot_inference(self, true_states, posterior_probs, emissions):
        """
        Plot the true hidden states, emissions, and posterior probabilities.
        
        Parameters:
        - true_states: The true hidden states (length N).
        - posterior_probs: Posterior probabilities (NxK).
        - emissions: The observed emissions (NxD).
        """
        time_steps = np.arange(len(true_states))
        
        plt.figure(figsize=(12, 6))
        
        # Plot emissions (data)
        plt.subplot(3, 1, 1)
        plt.plot(time_steps, emissions)
        plt.title('Data')
        plt.ylabel('Observation')
        
        # Plot true hidden states
        plt.subplot(3, 1, 2)
        plt.step(time_steps, true_states, where='mid', label="True States", color='r')
        plt.title('True Hidden States')
        plt.ylabel('State')
        
        # Plot posterior probabilities for all states
        plt.subplot(3, 1, 3)
        for i in range(posterior_probs.shape[1]):
            plt.plot(time_steps, posterior_probs[:, i], label=f"P(State={i})")
        
        plt.title('Posterior Probabilities of States (AR-HMM)')
        plt.ylabel('Probability')
        plt.xlabel('Time')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Helper function for log-sum-exp trick
def log_sum_exp(log_probs):
    max_log_prob = np.max(log_probs)
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

# Example usage
if __name__ == "__main__":
    # Define AR-HMM parameters
    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])  # State transition matrix
    means = np.array([[0, 0], [3, 3]])                      # Means of emissions for states 0 and 1
    covariances = np.array([np.eye(2) * 0.5, np.eye(2) * 1.0])  # Covariances for states
    weights = np.array([[[0.7], [0.1]], [[0.2], [0.1]]])     # Autoregressive weights
    initial_probs = np.array([0.6, 0.4])                    # Initial probabilities for states
    p = 2                                                   # Order of autoregression
    seq_length = 100                                        # Length of sequence
    
    # Initialize the AR-HMM model
    model = ARHMM(transition_matrix, means, covariances, weights, initial_probs, p)
    
    # Generate sequence
    true_states, emissions = model.generate_sequence(seq_length)
    
    # Run forward and backward algorithms
    log_alpha = model.forward_algorithm(emissions)
    log_beta = model.backward_algorithm(emissions)
    
    # Compute posterior probabilities
    posterior_probs = model.compute_posteriors(log_alpha, log_beta)
    
    # Plot inference
    model.plot_inference(true_states, posterior_probs, emissions)
    
    # Print log-likelihood for the generated data
    complete_log_likelihood = np.sum(log_alpha[-1])  # Log-likelihood for the complete data
    print(f"Complete Data Log-Likelihood: {complete_log_likelihood}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

class ARHMM:
    def __init__(self, transition_matrix, means, covariances, weights, initial_probs, p):
        """
        Initialize the AR-HMM model with parameters.
        
        Parameters:
        - transition_matrix: State transition matrix (KxK).
        - means: Mean vectors of size (KxD) where K is the number of states and D is the observation dimension.
        - covariances: Covariance matrices for each state (KxDxD).
        - weights: Autoregressive weights for each state (KxPxD), where P is the autoregressive lag order.
        - initial_probs: Initial state distribution (K).
        - p: The number of autoregressive steps (AR order).
        """
        self.transition_matrix = transition_matrix
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.initial_probs = initial_probs
        self.p = p  # autoregressive order
        self.num_states = len(initial_probs)

    def generate_sequence(self, seq_length):
        """
        Generate a sequence of hidden states and emissions from the AR-HMM.
        
        Parameters:
        - seq_length: Length of the sequence to generate.
        
        Returns:
        - states: The sequence of hidden states (list of length seq_length).
        - emissions: The sequence of emissions (list of length seq_length).
        """
        states = []
        emissions = []
        
        # Start from a random state
        current_state = np.random.choice(self.num_states)
        states.append(current_state)
        
        # Generate the first p emissions independently
        for t in range(self.p):
            emission = np.random.multivariate_normal(self.means[current_state], self.covariances[current_state])
            emissions.append(emission)
        
        # Generate the rest of the sequence using AR process
        for t in range(self.p, seq_length):
            # Transition to the next state
            current_state = np.random.choice(self.num_states, p=self.transition_matrix[current_state])
            states.append(current_state)
            
            # Autoregressive component
            ar_component = sum(self.weights[current_state, i] * emissions[-i-1] for i in range(self.p))
            emission = (self.means[current_state] + ar_component + 
                        np.random.multivariate_normal(np.zeros_like(self.means[current_state]), 
                                                      self.covariances[current_state]))
            emissions.append(emission)
        
        return states, emissions

    def forward_algorithm_log_ar(self, observations):
        """
        Forward algorithm using log probabilities for the AR-HMM.
        """
        num_states = self.num_states
        seq_length = len(observations)
        log_alpha = np.zeros((seq_length, num_states))
        
        # Initialize log alpha at time t=0
        log_alpha[0, :] = np.log(self.initial_probs)
        
        for t in range(1, seq_length):
            for j in range(num_states):
                emission_prob = multivariate_normal.pdf(
                    observations[t], 
                    mean=self.means[j], 
                    cov=self.covariances[j]
                )
                log_alpha[t, j] = logsumexp(log_alpha[t-1, :] + np.log(self.transition_matrix[:, j])) + np.log(emission_prob)
        
        return log_alpha

    def backward_algorithm_log_ar(self, observations):
        """
        Backward algorithm using log probabilities for the AR-HMM.
        """
        num_states = self.num_states
        seq_length = len(observations)
        log_beta = np.zeros((seq_length, num_states))
        
        # Initialize log beta at the last time step t = T-1
        log_beta[seq_length - 1, :] = 0  # log(1) = 0 since the probability of the final state is 1
        
        for t in range(seq_length - 2, -1, -1):
            for i in range(num_states):
                log_emission_probs = np.zeros(num_states)
                for j in range(num_states):
                    emission_prob = multivariate_normal.pdf(
                        observations[t+1], 
                        mean=self.means[j], 
                        cov=self.covariances[j]
                    )
                    log_emission_probs[j] = np.log(emission_prob)
                log_beta[t, i] = logsumexp(np.log(self.transition_matrix[i, :]) + log_emission_probs + log_beta[t+1, :])
        
        return log_beta

    def compute_posterior_log(self, log_alpha, log_beta):
        """
        Compute the posterior probabilities using the log alpha and log beta values.
        """
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma.T)  # Normalize
        return np.exp(log_gamma)  # Convert back to probabilities

    def plot_inference(self, true_states, posterior_probs, emissions):
        """
        Plot the posterior probabilities and the true hidden states.
        """
        seq_length = len(true_states)
        time_steps = np.arange(seq_length)
        
        plt.figure(figsize=(12, 8))
        
        # Plot observations
        plt.subplot(3, 1, 1)
        plt.plot(time_steps, np.array(emissions))
        plt.title('Emissions (Observations)')
        
        # Plot true hidden states
        plt.subplot(3, 1, 2)
        plt.step(time_steps, true_states, where='mid', label='True States', color='r')
        plt.title('True Hidden States')
        
        # Plot posterior probabilities
        plt.subplot(3, 1, 3)
        for k in range(self.num_states):
            plt.plot(time_steps, posterior_probs[:, k], label=f'P(State={k})')
        plt.title('Posterior Probabilities')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def em_algorithm(self, observations, max_iter=100, tol=1e-4):
        """
        The Expectation-Maximization (EM) algorithm to estimate parameters.
        
        Parameters:
        - observations: Time series observations (NxD).
        - max_iter: Maximum number of iterations.
        - tol: Convergence tolerance.
        """
        log_likelihoods = []
        for iteration in range(max_iter):
            # E-step: Compute log alpha, log beta, and posterior probabilities
            log_alpha = self.forward_algorithm_log_ar(observations)
            log_beta = self.backward_algorithm_log_ar(observations)
            posterior_probs = self.compute_posterior_log(log_alpha, log_beta)
            
            # M-step: Update model parameters
            # Update transition matrix
            transition_counts = np.zeros_like(self.transition_matrix)
            for t in range(len(observations) - 1):
                for i in range(self.num_states):
                    for j in range(self.num_states):
                        transition_counts[i, j] += np.exp(log_alpha[t, i] + np.log(self.transition_matrix[i, j]) +
                                                          log_beta[t+1, j] + 
                                                          multivariate_normal.logpdf(observations[t+1], 
                                                                                     mean=self.means[j], 
                                                                                     cov=self.covariances[j]))
            self.transition_matrix = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
            
            # Update means, covariances, and autoregressive weights
            # (This is a simplified step, and would involve maximizing the likelihood of the emissions)
            
            # Check convergence
            log_likelihood = np.sum(log_alpha[-1])
            log_likelihoods.append(log_likelihood)
            if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                break
            
        return log_likelihoods

# Example usage
if __name__ == '__main__':
    # Initialize parameters
    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    means = [np.array([0]), np.array([3])]
    covariances = [np.eye(1), np.eye(1)]
    weights = [np.array([0.7]), np.array([0.4])]
    initial_probs = np.array([0.6, 0.4])
    p = 1  # Autoregressive order
    
    # Create AR-HMM model
    model = ARHMM(transition_matrix, means, covariances, weights, initial_probs, p)
    
    # Generate a sequence
    seq_length = 100
    true_states, emissions = model.generate_sequence(seq_length)
    
    # Run EM algorithm to infer parameters from the observed emissions
    log_likelihoods = model.em_algorithm(emissions, max_iter=50)

    # Plot the posterior probabilities and true states
    log_alpha = model.forward_algorithm_log_ar(emissions)
    log_beta = model.backward_algorithm_log_ar(emissions)
    posterior_probs = model.compute_posterior_log(log_alpha, log_beta)
    
    model.plot_inference(true_states, posterior_probs, emissions)
    
    # Print the final log-likelihood after EM
    print(f"Final Log-Likelihood: {log_likelihoods[-1]}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Generate the AR-HMM sequence
def generate_arhmm_sequence(transition_matrix, means, std_devs, weights, seq_length, p):
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
        
        # Emit values based on autoregressive model
        autoregressive_term = np.zeros(emissions[-1].shape)
        for lag in range(1, min(p, t) + 1):
            autoregressive_term += weights[current_state][lag - 1] * emissions[t - lag]
        
        emission = (means[current_state] + autoregressive_term + 
                     np.random.normal(0, std_devs[current_state]))  # noise
        emissions.append(emission)
    
    return states, np.array(emissions)

# Forward algorithm using log probabilities for AR-HMM
def forward_algorithm_log_ar(transition_matrix, initial_probs, emissions, means, weights, covariances, p):
    num_states = len(initial_probs)
    seq_length = len(emissions)
    log_alpha = np.zeros((seq_length, num_states))
    
    # Initialize log alpha at time t=0
    for j in range(num_states):
        emission_prob = multivariate_normal.pdf(emissions[0], means[j], covariances[j])
        log_alpha[0, j] = np.log(initial_probs[j] * max(emission_prob, 1e-10))  # Prevent log(0)
    
    # Recursion: compute log alpha for t = 1 to seq_length-1
    for t in range(1, seq_length):
        for j in range(num_states):
            # Compute autoregressive mean for the current state
            autoregressive_term = np.zeros(emissions[t].shape)
            for lag in range(1, min(p, t) + 1):
                autoregressive_term += weights[j][lag - 1] * emissions[t - lag]
            
            emission_mean = means[j] + autoregressive_term
            emission_prob = multivariate_normal.pdf(emissions[t], emission_mean, covariances[j])
            
            # Compute log alpha using the log-sum-exp trick for numerical stability
            log_alpha[t, j] = np.log(max(emission_prob, 1e-10)) + log_sum_exp(log_alpha[t-1, :] + np.log(transition_matrix[:, j]))
    
    return log_alpha

# Backward algorithm using log probabilities for AR-HMM
def backward_algorithm_log_ar(transition_matrix, emissions, means, weights, covariances, p):
    num_states = transition_matrix.shape[0]
    seq_length = len(emissions)
    log_beta = np.zeros((seq_length, num_states))
    
    # Recursion: compute log beta for t = T-2 down to 0
    for t in range(seq_length - 2, -1, -1):
        for i in range(num_states):
            log_emission_probs = np.zeros(num_states)
            for j in range(num_states):
                autoregressive_term = np.zeros(emissions[t].shape)
                for lag in range(1, min(p, t+1) + 1):
                    autoregressive_term += weights[j][lag - 1] * emissions[t + 1 - lag]
                
                emission_mean = means[j] + autoregressive_term
                emission_prob = multivariate_normal.pdf(emissions[t + 1], emission_mean, covariances[j])
                log_emission_probs[j] = np.log(max(emission_prob, 1e-10))
            
            log_beta[t, i] = log_sum_exp(np.log(transition_matrix[i, :]) + log_emission_probs + log_beta[t+1, :])
    
    return log_beta

# Log-sum-exp trick for numerical stability
def log_sum_exp(log_probs):
    max_log_prob = np.max(log_probs)
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

# Compute posterior probabilities using log alpha and log beta
def compute_posterior_log(log_alpha, log_beta):
    log_gamma = log_alpha + log_beta
    log_gamma -= log_sum_exp(log_gamma.T)  # Normalize
    return np.exp(log_gamma)  # Convert back to probabilities

# Complete data log-likelihood calculation
def compute_complete_data_log_likelihood(states, emissions, transition_matrix, initial_probs, means, weights, covariances, p):
    seq_length = len(emissions)
    log_likelihood = 0.0
    
    # Add initial state probability
    log_likelihood += np.log(initial_probs[states[0]])  # log P(z_0)
    
    # Add transition probabilities
    for t in range(1, seq_length):
        log_likelihood += np.log(transition_matrix[states[t-1], states[t]])  # log P(z_t | z_{t-1})
    
    # Add emission probabilities (autoregressive)
    for t in range(p, seq_length):
        current_state = states[t]
        autoregressive_term = np.zeros(emissions.shape[1])
        
        # Compute autoregressive term from previous p emissions
        for lag in range(1, p + 1):
            autoregressive_term += weights[current_state][lag - 1] * emissions[t - lag]
        
        # Compute emission mean and covariance
        emission_mean = means[current_state] + autoregressive_term
        cov_matrix = covariances[current_state] + np.eye(emissions.shape[1]) * 1e-6  # Regularize covariance
        
        # Compute emission probability
        emission_prob = multivariate_normal.pdf(emissions[t], mean=emission_mean, cov=cov_matrix)
        
        # Clamp emission probability to prevent log(0)
        emission_prob = max(emission_prob, 1e-10)
        
        # Add log emission probability to the log-likelihood
        log_likelihood += np.log(emission_prob)
    
    return log_likelihood

# Plotting function to compare true hidden states and posterior probabilities
def plot_hmm_inference(true_states, posterior_probs, seq_length, emissions):
    time_steps = np.arange(seq_length)
    
    plt.figure(figsize=(12, 6))
    
    # Plot emissions (data)
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, emissions)
    plt.title('Data')
    plt.ylabel('Observation')
    
    # Plot true hidden states
    plt.subplot(3, 1, 2)
    plt.step(time_steps, true_states, where='mid', label="True States", color='r')
    plt.title('True Hidden States')
    plt.ylabel('State')
    
    # Plot posterior probabilities for both states
    plt.subplot(3, 1, 3)
    for i in range(posterior_probs.shape[1]):
        plt.plot(time_steps, posterior_probs[:, i], label=f"P(State={i})")
    
    plt.title('Posterior Probabilities of States (AR-HMM)')
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example parameters
transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])  # State transition matrix
means = np.array([[0, 0], [3, 3]])                      # Means of emissions for states 0 and 1
std_devs = np.array([0.5, 1.0])                          # Standard deviations of emissions
weights = np.array([[0.7, 0.5], [0.4, 0.2]])                       # Autoregressive weights for states
covariances = np.array([np.eye(2) * 0.1, np.eye(2) * 0.1])  # Covariance matrices for states
p = 2                                                  # Order of autoregression
seq_length = 100                                       # Length of the sequence

# Generate a sequence of states and emissions using AR-HMM
true_states, emissions = generate_arhmm_sequence(transition_matrix, means, std_devs, weights, seq_length, p)

# Initial HMM parameters
initial_probs = np.array([0.6, 0.4])                   # Initial probabilities for states

# Compute the complete data log-likelihood
log_likelihood = compute_complete_data_log_likelihood(true_states, emissions, transition_matrix, initial_probs, means, weights, covariances, p)
print(f"Complete Data Log-Likelihood: {log_likelihood}")

# Run the Forward-Backward algorithm using log probabilities for AR-HMM
log_alpha = forward_algorithm_log_ar(transition_matrix, initial_probs, emissions, means, weights, covariances, p)
log_beta = backward_algorithm_log_ar(transition_matrix, emissions, means, weights, covariances, p)
posterior_probs = compute_posterior_log(log_alpha, log_beta)

# Plot the true hidden states and the posterior probabilities
plot_hmm_inference(true_states[:seq_length], posterior_probs, seq_length, emissions)
