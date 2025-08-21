# -*- coding: utf-8 -*-
"""
Reservoir Computing with Time-Varying Input Field
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# Parameters
N = 100                  # Number of spins
beta = 1.0               # Inverse temperature
timesteps = 1000         # Number of update steps
p_connect = 0.3          # Sparsity of connections

# Initialize spins randomly (+1 or -1)
spins = np.random.choice([-1, 1], size=N)

# Generate random sparse symmetric connectivity matrix J_ij
J = np.random.randn(N, N)*(np.random.rand(N, N) < p_connect) / np.sqrt(N*p_connect)  # Correct scaling for spin glass
J = (J + J.T) / 2        # Ensure symmetry
# np.fill_diagonal(J, 0)   # No self-interaction

# Local field function
def local_field(spins, J, h, i):
    return h[i] + np.dot(J[i], spins)

# Define a time-varying input field h(t)
def time_varying_field(t, N):
    return 0.1 * np.sin(2 * np.pi * t / 100) * np.ones(N)

# Target function to learn (e.g., a sine wave)
def target_function(t):
    return np.sin(2 * np.pi * t / 200)

# Collect reservoir states and target outputs
reservoir_states = []
target_outputs = []

# Time evolution
for t in range(timesteps):
    if t == 0:
        random_vector = np.random.uniform(-1, 1, size=N)  # Generate a fixed random vector
    h = time_varying_field(t, N) * random_vector  # Update input field with the fixed random vector
    for _ in range(N):
        i = np.random.randint(0, N)
        H_i = local_field(spins, J, h, i)
        p_up = np.exp(beta * H_i) / (2 * np.cosh(beta * H_i))
        spins[i] = 1 if np.random.rand() < p_up else -1
    reservoir_states.append(spins.copy())
    target_outputs.append(target_function(t))

# Convert to numpy arrays
reservoir_states = np.array(reservoir_states)
target_outputs = np.array(target_outputs)

# Train a linear readout using Ridge regression
ridge = Ridge(alpha=1e-3)
ridge.fit(reservoir_states, target_outputs)

# Predict the target output
predicted_outputs = ridge.predict(reservoir_states)

# Plot the target and predicted outputs
plt.figure(figsize=(8, 4))
plt.plot(target_outputs, label="Target Output")
plt.plot(predicted_outputs, label="Predicted Output", linestyle="--")
plt.title("Reservoir Computing: Target vs Predicted Output")
plt.xlabel("Time step")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% scanning
def reservoir_computing(beta, N=100, timesteps=1000, p_connect=0.3):
    # Initialize spins randomly (+1 or -1)
    spins = np.random.choice([-1, 1], size=N)

    # Generate random sparse symmetric connectivity matrix J_ij
    # J = np.random.randn(N, N) / np.sqrt(N)  # Correct scaling for spin glass
    J = np.random.randn(N, N)*(np.random.rand(N, N) < p_connect) / np.sqrt(N*p_connect)
    J = (J + J.T) / 2        # Ensure symmetry
    # np.fill_diagonal(J, 0)   # No self-interaction

    # Local field function
    def local_field(spins, J, h, i):
        return h[i] + np.dot(J[i], spins)

    # Define a time-varying input field h(t)
    def time_varying_field(t, N):
        return 0.1 * np.sin(2 * np.pi * t / 100) * np.ones(N)

    # Target function to learn (e.g., a sine wave)
    def target_function(t):
        return np.sin(2 * np.pi * t / 200)

    # Collect reservoir states and target outputs
    reservoir_states = []
    target_outputs = []

    # Time evolution
    for t in range(timesteps):
        if t == 0:
            random_vector = np.random.uniform(-1, 1, size=N)  # Generate a fixed random vector
            h = time_varying_field(t, N) * random_vector  # Update input field with the fixed random vector
        for _ in range(N):
            i = np.random.randint(0, N)
            H_i = local_field(spins, J, h, i)
            p_up = np.exp(beta * H_i) / (2 * np.cosh(beta * H_i))
            spins[i] = 1 if np.random.rand() < p_up else -1
        reservoir_states.append(spins.copy())
        target_outputs.append(target_function(t))

    # Convert to numpy arrays
    reservoir_states = np.array(reservoir_states)
    target_outputs = np.array(target_outputs)

    # Train a linear readout using Ridge regression
    ridge = Ridge(alpha=1e-3)
    ridge.fit(reservoir_states, target_outputs)

    # Predict the target output
    predicted_outputs = ridge.predict(reservoir_states)

    # Compute performance (e.g., mean squared error)
    mse = np.mean((predicted_outputs - target_outputs) ** 2)
    return mse

# Scan through beta values and evaluate performance
betas = np.array([0.1,0.5,1.0,1.5,2.0, 4.0])#np.linspace(0.1, 2.0, 20)
performances = [reservoir_computing(beta) for beta in betas]

# Plot performance vs beta
plt.figure(figsize=(8, 4))
plt.plot(betas, performances, marker="o")
plt.title("Performance vs Beta")
plt.xlabel("Beta")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()


# %% NEXT: test with reinforcement learning
# This section is left for future implementation of reinforcement learning techniques.
# The current setup can be extended to include RL algorithms to optimize the reservoir computing process.