# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:57:50 2024

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import hankel
import random

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% model setup
lt = 10000
N = 5
Pij_true = np.random.rand(N,N) + np.eye(N)*5   # baseline transition
Pij_true = Pij_true / Pij_true.sum(axis=1, keepdims=True)
w_true = np.random.randn(N)*.1  # weights on the input  ##### explore NxN later???

smooth_lt = 10
stim = np.random.randn(lt)
stim = np.convolve(stim, np.ones(smooth_lt), mode='same')  # smooth external input

# %% functional
def driven_markov_chain(P, weights, ipts, init=None):
    """
    Sample a sequence from a Markov transition matrix, starting from a random initial state.

    Parameters:
    P (np.ndarray): Transition matrix (size N x N).
    n_steps (int): Number of steps to sample.

    Returns:
    np.ndarray: Sequence of sampled states.
    """
    n_steps = len(ipts)
    n_states = P.shape[0]  # Number of states
    states = np.zeros(n_steps, dtype=int)  # To store the sequence of states
    if init is None:
        states[0] = np.random.choice(n_states) # Randomly choose the initial state
    else:
        states[0] = init
    # Sample the next state based on the current state and transition matrix
    for t in range(1, n_steps):
        current_state = states[t - 1]
        Pt = np.exp(weights[current_state]*ipts[t] + np.log(P[current_state]))
        Pt = np.round(Pt, decimals=6)
        Pt = Pt/np.sum(Pt)
        next_state = np.random.choice(n_states, p=Pt[:])
        states[t] = next_state
    return states

states = driven_markov_chain(Pij_true, w_true, stim)

# %% inferr weights and transition from data
