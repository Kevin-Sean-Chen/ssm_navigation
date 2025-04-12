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
plt.plot(Pij_true.flatten(), model.P0.flatten(),'o'); plt.xlabel('true P0'); plt.ylabel('inferred P0')
plt.figure()
plt.plot(w_true.flatten(), model.ws.flatten(),'o'); plt.xlabel('true w'); plt.ylabel('inferred w')

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

# %% torch test
###############################################################################
# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import logsumexp

class LogLinearMarkovTorch(nn.Module):
    def __init__(self, N, u_dim, w_regu=0.0):
        super().__init__()
        self.N = N
        self.u_dim = u_dim
        self.w_regu = w_regu

        # Initialize log P0 and weights
        P00 = np.random.rand(N, N)
        P00 /= P00.sum(axis=1, keepdims=True)
        logP00 = np.log(P00)
        w0 = np.random.randn(N, u_dim)

        # Parameters to optimize (log transition probs and stimulus weights)
        self.logP0 = nn.Parameter(torch.tensor(logP00, dtype=torch.float32))
        self.ws = nn.Parameter(torch.tensor(w0, dtype=torch.float32))

    def transition_log_prob(self, x, u):
        """
        Compute log-probabilities P(x' | x, u) for given state x and input u.
        """
        correction = torch.matmul(self.ws, u)  # (N,)
        logP = correction + self.logP0[x]       # (N,)
        logZ = torch.logsumexp(logP, dim=0)
        log_probs = logP - logZ                 # normalized log-probs
        return log_probs

    def negative_log_likelihood_slow(self, x_seq, u_seq):
        """
        Compute negative log-likelihood for a given sequence of (x, u).
        """
        nll = 0.0
        for t in range(len(x_seq) - 1):
            log_probs = self.transition_log_prob(x_seq[t], u_seq[t])
            nll -= log_probs[x_seq[t+1]]

        # Regularization
        nll += self.w_regu * torch.sum(self.ws**2)
        return nll
    
    def negative_log_likelihood(self, x_seq, u_seq):
        """
        Fully vectorized negative log-likelihood.
        """
    
        x_seq = torch.tensor(x_seq, dtype=torch.long)
        u_seq = torch.tensor(u_seq, dtype=torch.float32)
    
        x_curr = x_seq[:-1]   # current states
        x_next = x_seq[1:]    # next states
        u_curr = u_seq[:-1]   # control inputs at current time
    
        # Compute correction terms: (batch_size, N)
        corrections = torch.matmul(u_curr, self.ws.T)  # (T-1, N)
    
        # Select baseline logP0 based on current states x_curr
        baseline = self.logP0[x_curr]  # (T-1, N)
    
        logits = corrections + baseline  # (T-1, N)
    
        # Normalize
        logZ = torch.logsumexp(logits, dim=1, keepdim=True)  # (T-1, 1)
        log_probs = logits - logZ                            # (T-1, N)
    
        # Pick the probability assigned to actual next state
        selected_log_probs = log_probs.gather(1, x_next.unsqueeze(1)).squeeze()
    
        # Negative log-likelihood
        nll = -selected_log_probs.sum()
    
        # Regularization
        nll += self.w_regu * torch.sum(self.ws**2)
    
        return nll


    def row_normalization(self):
        """
        Project logP0 so that exp(logP0) rows sum to 1.
        (log-sum-exp normalization)
        """
        with torch.no_grad():
            logZ = torch.logsumexp(self.logP0, dim=1, keepdim=True)
            self.logP0.data -= logZ

    def fit(self, x_seq, u_seq, n_epochs=500, lr=1e-2, verbose=True):
        """
        Fit the model using Adam optimizer in PyTorch.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Convert data to torch
        x_seq = torch.tensor(x_seq, dtype=torch.long)
        u_seq = torch.tensor(u_seq, dtype=torch.float32)

        losses = []
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood(x_seq, u_seq)
            loss.backward()
            optimizer.step()

            # Renormalize logP0
            self.row_normalization()

            losses.append(loss.item())

            if verbose and (epoch % 50 == 0 or epoch == n_epochs-1):
                print(f"Epoch {epoch}: loss = {loss.item():.6f}")

        return losses

    def transition_prob(self, x, u):
        """
        Return transition probabilities P(x' | x, u) in numpy array.
        """
        with torch.no_grad():
            correction = torch.matmul(self.ws, torch.tensor(u, dtype=torch.float32))
            logP = correction + self.logP0[x]
            logZ = torch.logsumexp(logP, dim=0)
            log_probs = logP - logZ
            probs = torch.exp(log_probs)
        return probs.numpy()
# %%
class LogLinearMarkovWithBaseline(nn.Module):
    def __init__(self, N, u_dim, w_regu=0.0):
        super().__init__()
        self.N = N
        self.u_dim = u_dim
        self.w_regu = w_regu

        # Initialize baseline logP0: shape (N, N)
        P0 = np.random.rand(N, N)
        P0 /= P0.sum(axis=1, keepdims=True)
        logP0_init = np.log(P0)
        self.logP0 = nn.Parameter(torch.tensor(logP0_init, dtype=torch.float32))

        # Initialize W: shape (N, N-1, u_dim)
        self.W = nn.Parameter(torch.randn(N, N-1, u_dim) * 0.01)

    def transition_log_probs(self, x_curr, u_curr):
        """
        Compute log-probabilities for all transitions given batch of (current states and stimulus inputs).
        """
        T = x_curr.shape[0]
        logits = torch.zeros((T, self.N), device=u_curr.device)

        for curr_state in range(self.N):
            mask = (x_curr == curr_state)
            if torch.any(mask):
                u_selected = u_curr[mask]  # (n_selected, u_dim)

                # Stimulus correction for non-self-transitions
                w_selected = self.W[curr_state]  # (N-1, u_dim)
                stimulus_logits = torch.matmul(u_selected, w_selected.T)  # (n_selected, N-1)

                # Full logits: start from baseline logP0
                baseline_logits = self.logP0[curr_state].unsqueeze(0).expand(u_selected.shape[0], -1)  # (n_selected, N)

                # Insert stimulus logits into the right places
                idx = torch.arange(self.N)
                idx_no_self = idx[idx != curr_state]  # non-self transitions

                full_logits = baseline_logits.clone()
                full_logits[:, idx_no_self] += stimulus_logits

                logits[mask] = full_logits

        # Normalize
        logZ = torch.logsumexp(logits, dim=1, keepdim=True)
        log_probs = logits - logZ

        return log_probs

    def negative_log_likelihood(self, x_seq, u_seq, lag):
        """
        Fully vectorized negative log-likelihood.
        """
        x_seq = torch.tensor(x_seq, dtype=torch.long)
        u_seq = torch.tensor(u_seq, dtype=torch.float32)
        
        x_seq, u_seq = x_seq[lag:], u_seq[:-lag]
        x_curr = x_seq[:-1]
        x_next = x_seq[1:]
        u_curr = u_seq[:-1]

        log_probs = self.transition_log_probs(x_curr, u_curr)

        selected_log_probs = log_probs.gather(1, x_next.unsqueeze(1)).squeeze()

        nll = -selected_log_probs.sum()

        # Regularization term
        nll += self.w_regu * torch.sum(self.W**2)

        return nll

    def row_normalization(self):
        """
        Normalize logP0 row-wise using log-sum-exp trick.
        """
        with torch.no_grad():
            logZ = torch.logsumexp(self.logP0, dim=1, keepdim=True)
            self.logP0.data -= logZ

    def fit(self, x_seq, u_seq, lag=1, n_epochs=500, lr=1e-2, verbose=True):
        """
        Fit the model using Adam optimizer in PyTorch.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)

        x_seq = torch.tensor(x_seq, dtype=torch.long)
        u_seq = torch.tensor(u_seq, dtype=torch.float32)

        losses = []
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood(x_seq, u_seq, lag)
            loss.backward()
            optimizer.step()

            # Normalize baseline logP0 after each step
            self.row_normalization()

            losses.append(loss.item())

            if verbose and (epoch % 50 == 0 or epoch == n_epochs-1):
                print(f"Epoch {epoch}: loss = {loss.item():.6f}")

        return losses

    def predict_transition_probs(self, x, u):
        """
        Predict transition probabilities P(x'|x,u) for a single (x,u).
        """
        x = int(x)
        u = torch.tensor(u, dtype=torch.float32)

        logits = self.logP0[x].clone()

        idx = torch.arange(self.N)
        idx_no_self = idx[idx != x]

        logits[idx_no_self] += torch.matmul(self.W[x], u)

        logZ = torch.logsumexp(logits, dim=0)
        log_probs = logits - logZ
        probs = torch.exp(log_probs)

        return probs.detach().cpu().numpy()


# %% test fitting
# Create model
# model = LogLinearMarkovTorch(N=5, u_dim=1, w_regu=0.01)
model = LogLinearMarkovWithBaseline(N=5, u_dim=1, w_regu=0.0)

# Fit the model
losses = model.fit(reduced_behavior, reduced_behavior[:,None], lag=5, n_epochs=1000, lr=1e-2)

# %% build matrix
plt.figure()
plt.imshow(torch.exp(model.logP0).detach())

ww = model.W.detach().numpy()[:,:,0]
Ws = np.zeros((5,5))
for i in range(5):
    idx = np.arange(5)
    idx_no_self = idx[idx != i]  # indices excluding self-transition
    Ws[i, idx_no_self] = ww[i]
plt.figure()
plt.imshow(Ws)

# %% validation
