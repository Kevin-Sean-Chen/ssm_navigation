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
from scipy.optimize import minimize
from scipy.special import logsumexp

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% model setup
lt = 1000
N = 5
u_dim = 2
Pij_true = np.random.rand(N,N) + np.eye(N)*1   # baseline transition
Pij_true = Pij_true / Pij_true.sum(axis=1, keepdims=True)
w_true = np.random.randn(N, u_dim)*1  # weights on the input  ##### explore NxN later???
# w_true = np.array([4,1,0,0])[:,None]

smooth_lt = 1
stim = np.random.randn(lt)*2
stim = np.convolve(stim, np.ones(smooth_lt), mode='same')  # smooth external input

stim = np.random.randn(lt,u_dim)
stim[:,0] = np.sin(np.arange(lt)/50)*.2
stim[:,1] = np.cos(np.arange(lt)/40)*.2

# stim[stim>2] = 2
# stim[stim<2] = 0

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
        Pt = np.exp(np.dot(weights,ipts[t]) + np.log(P[current_state]))
        # Pt = np.exp(weights[current_state]*ipts[t] + np.log(P[current_state]))  ### for scalar input
        # Pt = np.round(Pt, decimals=6)
        Pt = Pt/np.sum(Pt)
        next_state = np.random.choice(n_states, p=Pt[:])
        states[t] = next_state
    return states

states = driven_markov_chain(Pij_true, w_true, stim)

# %% inferr weights and transition from data

class log_linear_Markov:
    def __init__(self, N, u_dim):
        self.N = N
        self.u_dim = u_dim
        self.P00 = np.random.rand(N, N)  #Pij_true ### 
        self.P00 /= self.P00.sum(axis=1, keepdims=True)  ### baseline transition
        self.logP00 = np.log(self.P00)
        self.w0 = np.random.randn(N, u_dim)  #w_true  #   ### stimulus weights
        self.w_regu = 0
    
    def transition_prob(self, x, u):
        """
        Compute P(x' | x, u) using log-linear form.
        """
        correction = np.dot(self.ws, u) #self.ws*u  ### use scalar or vector form
        P = np.exp(correction + self.logP0[x])
        return P / P.sum() #(axis=1, keepdims=True)
    
    def transition_log_prob(self, x, u):
        """
        Compute P(x' | x, u) using log-linear form.
        """
        A = np.dot(self.ws, u) + self.logP0[x]
        logZ = logsumexp(A)
        logp = A - logZ
        return logp
    
    def log_likelihood(self, params, x_seq, u_seq):
        self.logP0 = params[:self.N*self.N].reshape(self.N, self.N)
        self.ws = params[self.N*self.N:].reshape(self.N, self.u_dim)
        
        ll = 0
        ### direct calculation
        for tt in range(len(x_seq)-1):
            P = self.transition_prob(x_seq[tt], u_seq[tt])
            tempP = P[x_seq[tt+1]]  #P[ x_seq[tt], x_seq[tt+1] ]
            # ll += tempP
            if tempP>0:
                ll += np.log(tempP)
            # else:
                # ll = -np.inf
        
        ### logsumexp trick
        # for tt in range(len(x_seq)-1):
        #     logP = self.transition_log_prob(x_seq[tt], u_seq[tt])
        #     tempP = logP[x_seq[tt+1]] 
        #     ll += tempP
            
        return -ll + self.w_regu * np.sum(self.ws**2)
    
    def row_sum_constraint(self, params):
        self.logP = params[:self.N*self.N].reshape(self.N, self.N)
        # cnt = np.exp(self.logP).sum(1)
        cnt = logsumexp(self.logP, axis=1)
        return cnt #- 1  # Each row sum should be 1

    def fit(self, x_seq, u_seq):
        """
        Optimize parameters using MLE.
        """
        params = np.concatenate([self.logP00.flatten(), self.w0.flatten()])
        constraints = {'type': 'eq', 'fun': self.row_sum_constraint}
        # result = minimize(self.log_likelihood, params, args=(x_seq, u_seq), method='L-BFGS-B')
        result = minimize(lambda pars: self.log_likelihood(pars, x_seq, u_seq), params, constraints=constraints)
        print(result)
        learned_param = result.x
        self.P0 = np.exp(learned_param[:self.N*self.N].reshape(self.N, self.N))
        self.ws = learned_param[self.N*self.N:].reshape(self.N, self.u_dim)

# Example Usage
# N = 5  # States
model = log_linear_Markov(N, u_dim)
model.w_regu = 0
model.fit(states, stim)

# %%
plt.figure()
plt.plot(Pij_true.flatten(), model.P0.flatten(),'o'); plt.xlabel('true P0'); plt.ylabel('inferred P0')
plt.figure()
plt.plot(w_true.flatten(), model.ws.flatten(),'o'); plt.xlabel('true w'); plt.ylabel('inferred w')
plt.show()

# %% simulate
def compute_markov_transition_matrix(states):
    unique_states = np.unique(states)  # Identify unique states
    num_states = len(unique_states)  # Number of states
    state_map = {state: i for i, state in enumerate(unique_states)}
    transition_counts = np.zeros((num_states, num_states))
    for t in range(len(states) - 1):
        i = state_map[states[t]]  # Current state index
        j = state_map[states[t+1]]  # Next state index
        transition_counts[i, j] += 1
    # Normalize rows to get transition probabilities
    P = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    P = np.nan_to_num(P)  # Handle divisions by zero
    return P, state_map

### simulations
states_sim = driven_markov_chain(model.P0, model.ws, stim)
Peff_true,_ = compute_markov_transition_matrix(states)
Peff_sim,_ = compute_markov_transition_matrix(states_sim)
plt.figure()
plt.subplot(121); plt.imshow(Peff_true); plt.title('emperical')
plt.subplot(122); plt.imshow(Peff_sim); plt.title('simulated')
plt.show()
