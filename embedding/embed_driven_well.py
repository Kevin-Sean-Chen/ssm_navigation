# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:32:42 2024

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import hankel

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% Brownian particle in a driven double-well potential
### simulate driven particle
### then use embedding method to find modes
### then test to infer the driven signal...
### if this works move on to navigation

# %% Simulations
###############################################################################
# %% setup
dt = 0.1              # Time step
T = 100000            # Total time
kBT = 0.001           # temperature
gamma = 1.5             # drag coefficient
lt = int(T / dt)
time = np.linspace(0, T, lt)
D = np.sqrt(2 * gamma * kBT / dt)  # effective diffusion coefficient

# Double-well potential and its gradient
def potential(x, b=0):
    return 0.25 * x**4 - 0.5 * x**2 - b*x

def grad_potential(x, b=0):
    return x**3 - x - b

# Initialize
x = np.zeros(lt)
bt = x*0

# generate a stochastic input
b = .3             # input strength
stim_dur = 30     # input duration
n_stims = 1000     # number of input
vector = np.arange(lt-stim_dur)
stim_pos = np.random.choice(vector, size=n_stims, replace=False)  # input position
for ii in range(n_stims):
    bt[stim_pos[ii]: stim_pos[ii]+stim_dur] = b

# %% dynamics
# Langevin simulation
for t in range(1, lt):
    # noise = np.sqrt(2 * D) * np.random.normal()
    noise = np.random.normal(0, D)
    x[t] = x[t-1] - grad_potential(x[t-1], b=b) * dt/gamma + noise# * np.sqrt(dt)  # Langevin eqn

# Plot results
plt.figure(figsize=(10, 5))
# Trajectory plot
plt.subplot(1, 2, 1)
plt.plot(time[:3000], x[:3000], color='blue')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Particle Trajectory')

# Potential energy landscape
x_vals = np.linspace(-2, 2, 500)
U_vals = potential(x_vals, b=b)

plt.subplot(1, 2, 2)
plt.plot(x_vals, U_vals, color='black', label='Potential $U(x)$')
plt.hist(x, bins=50, density=True, alpha=0.5, color='blue', label='Particle Distribution')
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Potential and Distribution')
plt.legend()

plt.tight_layout()
plt.show()

# %% build embeded Markov model
###############################################################################
# %% params
# should scan through but fix for now
K_star = 7
N_star = 1000
tau_star = 3

# %% building states
def build_X(data, K):
    K = int(K)
    T = len(data)
    X = np.zeros((T-K, K))
    for tt in range(len(data)-K):
        vv = data[tt:tt+K]
        X[tt,:] = vv
    return X

X = build_X(x, K_star)[::tau_star, :]  # build feature vectors with window K and step tau

# %% clustering and assigning states
def discretize(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    return cluster_labels

def kmeans_knn_partition(tseries,n_seeds,batchsize=None,return_centers=False):
    if batchsize==None:
        batchsize = n_seeds*5
    kmeans = MiniBatchKMeans(batch_size=batchsize,n_clusters=n_seeds).fit(tseries)
    labels=kmeans.labels_
    if return_centers:
        return labels,kmeans.cluster_centers_
    return labels

test_label = kmeans_knn_partition(X, N_star)

# %% compute transition and measure entropy
def compute_transition_matrix(time_series, n_states):
    """
    modified from previous function to handle track id and not compute those transitions
    """
    # Initialize the transition matrix (n x n)
    transition_matrix = np.zeros((n_states, n_states))
    # Get the current and next state only for valid transitions
    current_states = time_series[:-1]
    next_states = time_series[1:]
    # Use np.add.at to efficiently accumulate transitions
    np.add.at(transition_matrix, (current_states, next_states), 1)
    
    # Normalize the counts by dividing each row by its sum to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
    return transition_matrix 

def get_steady_state(transition_matrix):
    # Find the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    # Find the index of the eigenvalue that is approximately 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    # The corresponding eigenvector (steady-state)
    steady_state = np.real(eigenvectors[:, idx])
    # Normalize the eigenvector so that its elements sum to 1
    steady_state = steady_state / np.sum(steady_state)
    return steady_state

def trans_entropy(M):
    pi = get_steady_state(M)
    h = 0
    for ii in range(M.shape[0]):
        for jj in range(M.shape[0]):
            h += pi[ii]*M[ii,jj]*np.log(M[ii,jj] + 1e-10)
    return -h

def get_reversible_transition_matrix(P):
    probs = get_steady_state(P)
    P_hat = diags(1/probs)*P.transpose()*diags(probs)
    R=(P+P_hat)/2
    return R

def sorted_spectrum(R,k=5,which='LR'):
    eigvals,eigvecs = eigs(R,k=k,which=which)
    sorted_indices = np.argsort(eigvals.real)[::-1]
    return eigvals[sorted_indices],eigvecs[:,sorted_indices]

Pij = compute_transition_matrix(test_label, N_star)
h_est = trans_entropy(Pij)
print('entropy is: ', h_est)

# %% analyze the Markov model
uu,vv = np.linalg.eig(Pij)
idx = uu.argsort()[::-1]  # Get indices to sort eigenvalues
sorted_eigenvalues = np.real(uu[idx])
plt.figure()
plt.plot((-1)/np.log(sorted_eigenvalues[1:1000]),'-o')
plt.ylabel('relaxation time (s)')
plt.xlabel('eigenvalue index')
plt.yscale('log')

# %% checking the modes
imode = 1
window_show = np.arange(0,3000) 
R = get_reversible_transition_matrix(Pij)
eigvals,eigvecs = sorted_spectrum(R,k=7)  # choose the top k modes
phi2 = eigvecs[test_label,imode].real
color_abs = np.max(np.abs(phi2))
x_back = X[:, 0]
plt.figure()
plt.plot(window_show, x_back[window_show],'k--',alpha=0.2)
plt.scatter(window_show, x_back[window_show],c=phi2[window_show],cmap='coolwarm',s=1.5,vmin=-color_abs,vmax=color_abs)
plt.title(f'mode#{imode}') ; plt.xlabel('Time'); plt.ylabel('Position')

# %% analyze stim...
###############################################################################
# %% condition on stim
Xb = build_X(bt, K_star)[::tau_star, :]
b_back = Xb[:, 0]

# %% try linear regression here!!!
# does it map to generalize Langevin!?
lags = 15
X_phi = hankel(phi2[:lags], phi2[lags-1:]).T
X_x = hankel(x_back[:lags], x_back[lags-1:]).T
y = b_back[:-lags+1]*1

def lin_reg(X,y):
    xx_inv = np.linalg.inv(X.T @ X)
    beta = xx_inv @ X.T @ y
    return beta
beta_phi = lin_reg(X_phi, y)
beta_x = lin_reg(X_x, y)

# %%
plt.figure()
plt.plot(-beta_phi/np.linalg.norm(beta_phi), label=r'$R_{\phi}$')
plt.plot(beta_x/np.linalg.norm(beta_x), label=r'$R_x$')
plt.xlabel('time lag'); plt.ylabel('weights')
plt.legend()
