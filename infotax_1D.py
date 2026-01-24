# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 19:51:49 2026

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))

def p_hit_given_x_and_source(x, xs, L=2.0, p0=0.02):
    d = abs(x - xs)
    return np.clip(p0 + (1 - p0) * np.exp(-d / L), 0.0, 1.0)

def bayes_update(belief, x, hit, xs_grid, L=2.0, p0=0.02):
    ph = np.array([p_hit_given_x_and_source(x, xs, L=L, p0=p0) for xs in xs_grid])
    like = ph if hit else (1 - ph)
    post = belief * like
    s = post.sum()
    if s <= 0:
        return np.ones_like(belief) / len(belief)
    return post / s

def expected_entropy_after_action(belief, x_next, xs_grid, L=2.0, p0=0.02):
    ph_xs = np.array([p_hit_given_x_and_source(x_next, xs, L=L, p0=p0) for xs in xs_grid])
    p_hit = np.sum(belief * ph_xs)
    p_no  = 1.0 - p_hit

    post_hit = bayes_update(belief, x_next, True,  xs_grid, L=L, p0=p0)
    post_no  = bayes_update(belief, x_next, False, xs_grid, L=L, p0=p0)

    return p_hit * entropy(post_hit) + p_no * entropy(post_no)

def infotaxis_1d_with_kinematics(
    xmin=-10, xmax=10, true_source=10, start=0,
    L=2.0, p0=0.02, max_steps=250, seed=2,
    v_max=3, a_max=1, move_cost=0.01
):
    """
    Infotaxis in 1D with kinematics and *stochastic* detections.
    Detection is sampled as a Bernoulli random variable using
    P(hit | x, true_source).
    """
    rng = np.random.default_rng(seed)
    xs_grid = np.arange(xmin, xmax + 1)

    x = int(start)
    v = 0  # velocity (cells / step)

    belief = np.ones_like(xs_grid, dtype=float)
    belief /= belief.sum()

    accel_actions = list(range(-a_max, a_max + 1))

    times = [0]
    path = [x]
    hits = []          # True = actual detection event
    hit_probs = []    # for debugging / sanity checks

    for t in range(max_steps):
        if x == true_source:
            break

        # Choose acceleration that minimizes expected posterior entropy
        candidates = []
        for a in accel_actions:
            v_next = int(np.clip(v + a, -v_max, v_max))
            x_next = int(np.clip(x + v_next, xmin, xmax))

            EH = expected_entropy_after_action(belief, x_next, xs_grid, L=L, p0=p0)
            cost = move_cost * abs(v_next)
            score = EH + cost

            candidates.append((score, -abs(v_next), a, v_next, x_next))

        candidates.sort(key=lambda z: (z[0], z[1]))
        score, _, a, v_next, x_next = candidates[0]

        # --- STOCHASTIC DETECTION ---
        p_true = p_hit_given_x_and_source(x_next, true_source, L=L, p0=p0)
        hit = rng.random() < p_true   # Bernoulli draw

        # Bayesian update
        belief = bayes_update(belief, x_next, hit, xs_grid, L=L, p0=p0)

        # State update
        x, v = x_next, v_next

        times.append(t + 1)
        path.append(x)
        hits.append(hit)
        hit_probs.append(p_true)

    return (
        np.array(times),
        np.array(path),
        np.array(hits, dtype=bool),
        np.array(hit_probs)
    )

# ----------------------
# Run simulation
# ----------------------
###
# params
L = 10
p0 = 0.01
seed = np.random.randint(0,1000)
taget_x = 15
xmin, xmax = -15, 15
###
times, path, hits, hit_probs = infotaxis_1d_with_kinematics( xmin=xmin, xmax=xmax,
    start=0, true_source=taget_x, L=L, p0=p0, seed=seed,
    v_max=2, a_max=1, move_cost=.0
)

# Background gradient: detection probability vs x
x_vals = np.linspace(xmin, xmax, 400)
grad = np.array([p_hit_given_x_and_source(x, taget_x, L=L, p0=p0) for x in x_vals])

# ----------------------
# Plot x–t trajectory
# ----------------------
plt.figure(figsize=(7, 4))

# Background gradient
plt.imshow(
    grad[np.newaxis, :],
    extent=[xmin, xmax, times.min(), times.max()],
    aspect='auto',
    origin='lower'
)

# Trajectory
plt.plot(path, times, linewidth=2)

# ACTUAL detection events (red dots)
hit_times = times[1:][hits]
hit_positions = path[1:][hits]
plt.scatter(hit_positions, hit_times, s=40, c='red')

plt.axvline(taget_x, linestyle='--', linewidth=1)

plt.xlabel("Position x")
plt.ylabel("Time t")
plt.title("1D Infotaxis (kinematics-limited) with Stochastic Detections")
plt.tight_layout()
plt.show()
