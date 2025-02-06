# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:41:06 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

# %% Lorantz
def lorenz_attractor(state, sigma=10, beta=8/3, rho=28):
    """
    Compute the derivatives for the Lorenz system.
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

def generate_lorenz_trajectory_euler(initial_state, num_steps, dt):
    """
    Generate the Lorenz trajectory using the Euler method.
    """
    trajectory = np.zeros((num_steps, 3))
    trajectory[0] = initial_state

    for i in range(1, num_steps):
        # Euler step: x_{t+1} = x_t + dt * f(x_t)
        trajectory[i] = trajectory[i - 1] + dt * lorenz_attractor(trajectory[i - 1])
    
    return trajectory

# Parameters
initial_state = np.array([1.0, 1.0, 1.0])
num_steps = 10000
dt = 0.01

# Generate the Lorenz trajectory
trajectory = generate_lorenz_trajectory_euler(initial_state, num_steps, dt)

# Plot the Lorenz attractor
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=0.5)
ax.set_title("Lorenz Attractor (Euler Method)")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()

# %% EZN
### preprocess signal
ipt = trajectory[:-1,:].T*.2
target = ipt*1 #trajectory[1:,:].T*.1

### echo-state network
N = 500
alpha = 1.5
tau = 0.05
lt = len(ipt.T)
rt = np.random.randn(N, lt)
xt = rt*1
readout = np.zeros((3, lt))
Win = np.random.randn(N,3)
Wout = np.random.randn(N,3)/N**0.5
Wij = np.random.randn(N,N)*alpha/np.sqrt(N)
P = np.eye(N)*1

for tt in range(lt-1):
    ### neural dynamics
    xt[:, tt+1] = xt[:, tt] + dt*(-xt[:, tt]/tau + Wij @ rt[:,tt] + Win@ ipt[:,tt]*1)
    rt[:, tt+1] = np.tanh(xt[:, tt+1])
    
    ### readout
    readout[:, tt+1] = Wout.T @ rt[:, tt+1]
    
    ### force learning
    r_new = rt[:,tt+1][:,None]
    err = (target[:,tt+1] - readout[:, tt+1])[None,:]
    Pr = P @ r_new
    P = P - (Pr @ Pr.T) / (1 + Pr.T@r_new)
    Wout = Wout + err * np.dot(P, r_new);

# %% training
# XXT_inv = np.linalg.inv(rt @ rt.T + .1*np.eye(N))  # Inverse of X * X^T
# Wout = XXT_inv @ rt @ target.T

# %% prediction
rt = np.random.randn(N, lt)
xt = rt*1
readout2 = np.random.randn(3, lt)

for tt in range(lt-1):
    ### neural dynamics
    xt[:, tt+1] = xt[:, tt] + dt*(-xt[:, tt]/tau + Wij @ rt[:,tt] + Win @ readout2[:,tt]*1)
    rt[:, tt+1] = np.tanh(xt[:, tt+1])
    
    ### readout
    readout2[:, tt+1] = Wout.T @ rt[:, tt+1]

# %% 
#### might have to rescale for robustness
#### next step is to compute the spectrum
#### if it works start adding input
#### if it works apply to data!