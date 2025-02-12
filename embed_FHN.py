# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:38:25 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

# %%
# FitzHugh-Nagumo model parameters
a = 0.7
b = 0.8
epsilon = 0.08
dt = 0.1  # Time step for Euler method
T = 1000  # Total time
num_steps = int(T / dt)  # Number of steps

# Define input current I(t) (time-dependent input)
def input_current(t):
    return 0.1 * (np.sin(0.2 * t) + 1) + np.random.randn()*.1  # Example: oscillating input

# Initialize v and w
v = np.zeros(num_steps)
w = np.zeros(num_steps)

# Initial conditions (v0, w0)
v[0] = -1.0  # Initial membrane potential
w[0] = -1.0  # Initial recovery variable

# Euler method loop
for t in range(1, num_steps):
    # Current time
    time = t * dt
    
    # FitzHugh-Nagumo equations
    dvdt = v[t-1] - (v[t-1]**3) / 3 - w[t-1] + input_current(time)
    dwdt = epsilon * (v[t-1] + a - b * w[t-1])
    
    # Euler update for v and w
    v[t] = v[t-1] + dvdt * dt
    w[t] = w[t-1] + dwdt * dt

# Plot the results
time = np.linspace(0, T, num_steps)
stim_vec = np.array([input_current(t) for t in time])

plt.figure(figsize=(10, 6))

# Plot v(t) - membrane potential
plt.subplot(2, 1, 1)
plt.plot(time, v, label='Membrane Potential v(t)')
plt.xlabel('time')
plt.ylabel('v(t)')
plt.title('FitzHugh-Nagumo Model')
plt.grid(True)

# Plot w(t) - recovery variable
plt.subplot(2, 1, 2)
plt.plot(time, stim_vec, label='stochastic drive', color='orange')
plt.xlabel('Time')
plt.ylabel('w(t)')
plt.title('stochastic input')
plt.grid(True)

plt.tight_layout()
plt.show()

# %% IDEAS
# recover FHN state space
# embed with stimuli
# find modes of bifurcation

