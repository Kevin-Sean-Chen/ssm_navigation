# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 00:35:13 2025

@author: kevin
"""

# Glauber dynamics demo for a 10-spin Ising system with random couplings (no fixed seed)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

rng = np.random.default_rng()

def make_couplings(N: int, scale: float = None, symmetric: bool = True):
    """Create a random coupling matrix J for an Ising model."""
    if scale is None:
        scale = 1.0 / np.sqrt(N)
    J = rng.normal(0.0, scale, size=(N, N))
    if symmetric:
        J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return J

def energy(J: np.ndarray, s: np.ndarray, h: float = 0.0):
    return -0.5 * s @ J @ s - h * np.sum(s)

def glauber_dynamics(J: np.ndarray, beta: float, steps: int, h: np.ndarray):
    """Asynchronous Glauber (heat-bath) dynamics."""
    N = J.shape[0]
    s = rng.choice([-1, 1], size=N)
    m_hist = np.empty(steps)
    e_hist = np.empty(steps, dtype=float)
    s_hist = np.empty((steps, N), dtype=int)
    flips = 0

    for t in range(steps):
        i = rng.integers(0, N)
        h_i = J[i] @ s + h[t]
        p_flip = 1.0 / (1.0 + np.exp(2.0 * beta * s[i] * h_i))
        if rng.random() < p_flip:
            s[i] *= -1
            flips += 1
        m_hist[t] = np.mean(s)
        e_hist[t] = energy(J, s, h=h[t])
        s_hist[t] = s

    stats = {
        "final_magnetization": float(m_hist[-1]),
        "final_energy": float(e_hist[-1]),
        "total_flips": int(flips)
    }
    return s, s_hist, m_hist, e_hist, stats

# Parameters
N = 10
beta = 1.1
steps = 5000
h_field = 0.0
h_field = np.sin(np.arange(steps)/200)*0
h_field = np.random.randn(steps)
h_field = np.convolve(np.ones(10), h_field)

J = make_couplings(N, symmetric=False)
s_final, s_hist, m_hist, e_hist, stats = glauber_dynamics(J, beta=beta, steps=steps, h=h_field)

# Plot spin trajectories
plt.figure(figsize=(8, 4))
plt.imshow(s_hist.T, aspect='auto', cmap='bwr', interpolation='nearest')
plt.colorbar(label='Spin state (-1 or +1)')
plt.xlabel('Time step')
plt.ylabel('Spin index')
plt.title('Spin trajectories during Glauber dynamics')
plt.show()

# Plot magnetization and energy
plt.figure()
plt.plot(m_hist)
plt.title("Magnetization during Glauber dynamics (N=10)")
plt.xlabel("Step")
plt.ylabel("Magnetization m = <s_i>")
plt.show()

plt.figure()
plt.plot(e_hist)
plt.title("Energy during Glauber dynamics (N=10)")
plt.xlabel("Step")
plt.ylabel("Energy")
plt.show()

print("Demo complete.")
print(f"N = {N}, beta = {beta}, steps = {steps}, external field h = {h_field}")
print("Final spin configuration:", s_final)
print("Summary stats:", stats)

# %% termpoal feature
def compute_dwell_times(s_hist: np.ndarray):
    """
    Compute dwell times for each spin.
    Dwell time = consecutive number of steps a spin remains in the same state before flipping.
    """
    N = s_hist.shape[1]
    dwell_times = []
    for i in range(N):
        spin_series = s_hist[:, i]
        changes = np.where(np.diff(spin_series) != 0)[0] + 1
        segments = np.split(spin_series, changes)
        dwell = [len(seg) for seg in segments]
        dwell_times.extend(dwell)
    return np.array(dwell_times)

# compute dwell time distribution
dwell_times = compute_dwell_times(s_hist)

# plot histogram
plt.figure(figsize=(6,4))
plt.hist(dwell_times, bins=np.arange(1, 100), density=True, alpha=0.7)
plt.title("Dwell time distribution across spins")
plt.xlabel("Dwell time (steps)")
plt.ylabel("Probability density")
plt.yscale('log'); plt.xscale('log')
plt.show()

# show mean and variance
mean_dwell = np.mean(dwell_times)
var_dwell = np.var(dwell_times)

print(f"Mean dwell time: {mean_dwell:.2f}")
print(f"Variance: {var_dwell:.2f}")
print(f"Total samples: {len(dwell_times)}")

# %% correlati
C = np.cov(s_hist.T)
plt.figure()
plt.imshow(C)

# %% MaxEnt that will fail
###############################################################################
J_inf = -np.linalg.inv(C)
_, s_inf, _, _, _ = glauber_dynamics(J_inf, beta=beta, steps=steps, h=h_field*0)

# %% corr
C_inf = np.cov(s_inf.T)
plt.figure()
plt.plot(C.reshape(-1), C_inf.reshape(-1),'ko')
plt.xlabel(r'true $C_{ij}$', fontsize=20)
plt.ylabel(r'inferred $C_{ij}$', fontsize=20)

# %% time
dwell_inf = compute_dwell_times(s_inf)
plt.figure(figsize=(6,4))
# Compute histograms as density curves
bins = np.arange(1, 100)
p_inf, _ = np.histogram(dwell_inf, bins=bins, density=True)
p_obs, _ = np.histogram(dwell_times, bins=bins, density=True)
centers = 0.5 * (bins[1:] + bins[:-1])
# Plot as lines instead of overlapping histograms
plt.plot(centers, p_inf, '-o', lw=2, label='inferred')
plt.plot(centers, p_obs, '-o', lw=2, label='observed')

plt.title("Dwell time distribution across spins")
plt.xlabel("Dwell time (steps)")
plt.ylabel("Probability density")
plt.yscale('log'); plt.xscale('log')
plt.legend()
plt.show()
