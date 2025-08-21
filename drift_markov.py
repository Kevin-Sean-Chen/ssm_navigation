# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 01:09:40 2025

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Parameters
# np.random.seed(42)
num_states = 3          # Number of discrete states
num_steps = 7000        # Number of time steps
alpha_amplitude = 0.3   # Maximum amplitude of time-varying drift

# 2. Create baseline transition matrix P (stationary part)
P = np.random.rand(num_states, num_states)
P = np.eye(num_states)*20 + P
P = P / P.sum(axis=1, keepdims=True)  # Normalize rows

# 3. Create flux matrix T (can be antisymmetric or random)
T = np.random.randn(num_states, num_states)
T = T - T.T  # Make it antisymmetric: T_ij = -T_ji

# 4. Normalize T so that perturbations are small
T = T / np.max(np.abs(T)) * 0.9  # Scale to 20% max

# 5. Define time-dependent alpha(t)
times = np.linspace(0, 10*np.pi, num_steps)  # Slowly varying over time
alpha_t = alpha_amplitude * np.sin(times)*1 + 0

# 6. Initialize
state_sequence = []
current_state = np.random.choice(num_states)

# 7. Simulate
for t in range(num_steps):
    # Build current transition matrix
    Pt = P + alpha_t[t] * T
    
    # Make sure rows are still probability distributions
    Pt = np.maximum(Pt, 0)  # Clip negative values
    Pt = Pt / Pt.sum(axis=1, keepdims=True)  # Renormalize rows

    # Sample next state
    next_state = np.random.choice(num_states, p=Pt[current_state])
    state_sequence.append(current_state)
    current_state = next_state

state_sequence = np.array(state_sequence)

# 8. Visualization
plt.figure(figsize=(10,5))
plt.plot(state_sequence, lw=0.5)
plt.title('Simulated Non-Stationary Markov Chain')
plt.xlabel('Time')
plt.ylabel('State')
plt.grid(True)
plt.show()

# Plot alpha(t) over time
plt.figure(figsize=(10,3))
plt.plot(alpha_t)
plt.title('Time-varying alpha(t) (modulation of flux)')
plt.xlabel('Time')
plt.ylabel('alpha(t)')
plt.grid(True)
plt.show()


def discrete_autocorrelation(states, max_lag=100):
    """
    Compute the autocorrelation function for a discrete state time series.
    
    Args:
        states (np.ndarray): Array of discrete states over time.
        max_lag (int): Maximum lag to compute autocorrelation for.
        
    Returns:
        lags (np.ndarray): Array of lag values.
        autocorr (np.ndarray): Autocorrelation at each lag.
    """
    n = len(states)
    lags = np.arange(0, max_lag+1)
    autocorr = np.zeros(max_lag+1)

    # Encode the states numerically (if needed)
    unique_states = np.unique(states)
    state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
    numeric_states = np.array([state_to_idx[s] for s in states])

    # Subtract mean for zero-mean correlation
    numeric_states = numeric_states - np.mean(numeric_states)
    
    for lag in lags:
        if lag == 0:
            autocorr[lag] = 1.0  # Normalize autocorr(0) = 1
        else:
            autocorr[lag] = np.correlate(numeric_states[:-lag], numeric_states[lag:])[0]
            autocorr[lag] /= np.correlate(numeric_states, numeric_states)[0]

    return lags, autocorr

# Compute autocorrelation
lags, acf = discrete_autocorrelation(state_sequence, max_lag=1000)

# Plot
plt.figure(figsize=(8,4))
plt.loglog(lags, acf, marker='o')
plt.title('Autocorrelation of Discrete State Time Series')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()
