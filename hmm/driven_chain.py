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
lt = 7000
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
stim[:,0] = np.sin(np.arange(lt)/50)*2
stim[:,1] = np.cos(np.arange(lt)/40)*2

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
plt.plot(Pij_true.flatten(), model.P0.flatten(),'.'); plt.xlabel('true P0'); plt.ylabel('inferred P0')
plt.figure()
plt.plot(w_true.flatten(), model.ws.flatten(),'.'); plt.xlabel('true w'); plt.ylabel('inferred w')

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

# %%
###############################################################################
# %% measure entropy at different input angles
lt = 10000
N = 20
u_dim = 1
Pij_true = np.random.rand(N,N) + np.eye(N)*1   # baseline transition
# rand_vec = np.random.rand(N,3)
# Pij_true = Pij_true*.001 + rand_vec @ rand_vec.T ### building in low-rank structure
# Pij_true = Pij_true + Pij_true.T
Pij_true = Pij_true / Pij_true.sum(axis=1, keepdims=True)
# uu,ss,vv = np.linalg.svd(Pij_true)
ee, uu = np.linalg.eig(Pij_true.T)
uu = np.real(uu)
w_true = np.random.randn(N, u_dim)*5  # weights on the input  ##### explore NxN later???
w_true = uu[:,1][:,None]*5

stim = np.random.randn(lt,u_dim)
stim[:,0] = np.sin(np.arange(lt)/10)*2

states = driven_markov_chain(Pij_true, w_true, stim)
plt.figure()
plt.plot(states[:1000])

# %% entropy stuff
def orthogonal_vector(v):
    n = len(v)
    u = np.random.randn(n)  # Random vector
    u -= (np.dot(u, v) / np.dot(v, v)) * v  # Make it orthogonal
    
    # Normalize (optional)
    if np.linalg.norm(u) > 1e-10:  # Avoid division by zero
        u /= np.linalg.norm(u)
    
    return u

def compute_transition_matrix(time_series):
    states = np.unique(time_series)
    n_states = len(states)
    
    # Initialize a transition matrix
    transition_matrix = np.zeros((n_states, n_states))
    
    # Map states to indices
    state_to_index = {state: idx for idx, state in enumerate(states)}
    
    # Count transitions
    for i in range(len(time_series) - 1):
        current_state = time_series[i]
        next_state = time_series[i + 1]
        
        current_idx = state_to_index[current_state]
        next_idx = state_to_index[next_state]
        
        transition_matrix[current_idx, next_idx] += 1
    
    # Normalize to get probabilities (row sums to 1)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix /= row_sums
    
    return transition_matrix, states

def compute_entropy_of_transition_matrix(time_series):
    # Compute the transition matrix
    transition_matrix, states = compute_transition_matrix(time_series)
    
    # Compute the state probabilities P(x)
    state_counts = np.unique(time_series, return_counts=True)[1]
    state_probabilities = state_counts / len(time_series)
    
    # Compute entropy of P(x'|x)
    entropy = 0
    for i, state in enumerate(states):
        # Get the conditional distribution P(x'|x)
        p_x_prime_given_x = transition_matrix[i, :]
        
        # Compute the entropy of P(x'|x)
        entropy += state_probabilities[i] * (-np.sum(p_x_prime_given_x * np.log(p_x_prime_given_x + 1e-10)))  # adding epsilon to avoid log(0)
    
    return entropy

# %% tests
info_vec = np.zeros(N)
for ii in range(N):
    print(ii)
    if ii == 0:
        w_true = uu[:,ii][:,None]*10
        # w_true = np.random.randn(N,1)*1
        # w_true = w_true/np.linalg.norm(w_true)*10
    else:
        w_true = uu[:,ii][:,None]*10
        # w_true = orthogonal_vector(uu[:,ii])[:,None]*10
    states = driven_markov_chain(Pij_true, w_true, stim)
    info_vec[ii] = compute_entropy_of_transition_matrix(states)
    
# %%
plt.figure()
plt.plot(info_vec,'-o')
plt.xlabel('input along the sorted eigenvec'); plt.ylabel('entropy')
