# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:43:01 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from scipy.linalg import eig, schur, qr
from sklearn.cluster import KMeans
from scipy.optimize import minimize

# %%
def generate_transition_matrix(data, sigma=1.0):
    """
    Generate a transition matrix using a Gaussian kernel.
    :param data: Input data points (N x D, where N = samples, D = dimensions).
    :param sigma: Kernel bandwidth.
    :return: Transition matrix (N x N).
    """
    n = data.shape[0]
    dist_matrix = np.linalg.norm(data[:, None] - data[None, :], axis=2)
    affinity_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    
    # Row-normalize to create a stochastic transition matrix
    row_sums = affinity_matrix.sum(axis=1)
    transition_matrix = affinity_matrix / row_sums[:, None]
    return transition_matrix

def robust_perron_cluster_analysis(P, num_clusters):
    """
    Perform generalized robust Perron cluster analysis on a transition matrix.
    :param P: Transition matrix (N x N).
    :param num_clusters: Number of clusters to identify.
    :return: Cluster labels for each data point.
    """
    # Compute eigenvalues and eigenvectors of the transition matrix
    eigenvalues, eigenvectors = eig(P)
    
    T, eigenvectors = schur(P, output="real")  # T: quasi-triangular matrix, Z: orthogonal/unitary matrix
    # The eigenvalues are on the diagonal of T, extract them
    eigenvalues = np.diag(T)
    
    # Sort eigenvalues and eigenvectors by descending eigenvalue magnitude
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select the top `num_clusters` eigenvectors (after the stationary eigenvector)
    selected_vectors = np.real(eigenvectors[:, 1:num_clusters])
    
    # Normalize the selected eigenvectors (important for clustering)
    selected_vectors = selected_vectors / np.linalg.norm(selected_vectors, axis=1, keepdims=True)
    
    # Perform k-means clustering on the selected eigenvectors
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(selected_vectors)
    labels = kmeans.labels_
    return labels, eigenvalues, eigenvectors

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data
    # np.random.seed(42)
    num_points = 100
    cluster_1 = np.random.normal(loc=[0, 0], scale=0.2, size=(num_points, 2))
    cluster_2 = np.random.normal(loc=[3, 3], scale=0.2, size=(num_points, 2))
    cluster_3 = np.random.normal(loc=[0, 3], scale=0.2, size=(num_points, 2))
    cluster_4 = np.random.normal(loc=[1.5, 1.5], scale=0.2, size=(num_points, 2))
    cluster_5 = np.random.normal(loc=[5, 5], scale=0.2, size=(num_points, 2))
    cluster_6 = np.random.normal(loc=[7, 7], scale=0.2, size=(num_points, 2))
    data = np.vstack([cluster_1, cluster_2, cluster_3])
    data = np.vstack([cluster_1, cluster_2, cluster_3, cluster_4, cluster_6, cluster_5])
    
    # Construct the transition matrix
    P = generate_transition_matrix(data, sigma=1.0)
    # P = P[np.random.permutation(P.shape[0])]
    
    # Perform GRPCA
    num_clusters = 6
    labels, eigenvalues, eigenvectors = robust_perron_cluster_analysis(P, num_clusters)
    
    # Plot results
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis")
    plt.title("Clustering using GRPCA")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(label="Cluster")
    plt.show()
    
    # Display eigenvalues (optional)
    print("Top eigenvalues:", eigenvalues[:num_clusters])
    
# %% Generalized PCCA implementation
### given P compute the basis with Schur decomposition
T, eigenvectors = schur(P, output="real")  # T: quasi-triangular matrix, Z: orthogonal/unitary matrix
eigenvalues = np.diag(T)
eigenvalues, eigenvectors = eig(P) ### for eigen-decomposition
idx = np.argsort(-np.abs(eigenvalues))
eigenvalues = np.real(eigenvalues[idx])
eigenvectors = np.real(eigenvectors[:, idx])  # sorted Schur

size = 4
epsilon=1e-7
X = (eigenvectors[:, :size])

def QR2A(params, size):
    Q_flat = params[:size * size].reshape(size, size)
    R_flat = params[size * size:].reshape(size, size)
    # Ensure Q is orthogonal using QR decomposition
    Q, _ = qr(Q_flat)
    # Ensure R is upper triangular with non-zero diagonal
    R = np.triu(R_flat) + np.eye(size) * epsilon    
    # Reconstruct A
    A = Q @ R
    return A

# Objective function: Optimize A to satisfy the constraints
def chi_objective(params, X):
    # A = A_flat.reshape(X.shape[1], -1)  # Reshape A from flat vector
    A = QR2A(params, X.shape[1])
    XA = X @ A
    # Compute a penalty for deviations from the constraints
    penalty_negativity = np.sum(np.abs(XA[XA < 0]))  # Penalize negative values
    penalty_row_sum = np.sum((np.sum(XA, axis=1) - 1) ** 2)  # Penalize row sum deviations
    return penalty_negativity + penalty_row_sum

# Constraints to ensure positivity and row-stochasticity
def constraint_positivity(params, X):
    # A = A_flat.reshape(X.shape[1], -1)
    A = QR2A(params, X.shape[1])
    XA = X @ A
    return XA.flatten()  # Ensure all elements of XA are non-negative

def constraint_row_sum(params, X):
    # A = A_flat.reshape(X.shape[1], -1)
    A = QR2A(params, X.shape[1])
    XA = X @ A
    return np.sum(XA, axis=0) - 1  # Ensure row sums are 1

def optimize_chi(params, X):
    # Initial guess for A
    m, nc = X.shape
    A_init = params*1

    # Perform optimization
    constraints = [
        {'type': 'ineq', 'fun': constraint_positivity, 'args': (X,)},  # XA >= 0
        {'type': 'eq', 'fun': constraint_row_sum, 'args': (X,)}        # Row sums = 1
    ]
    
    result = minimize(
        chi_objective,
        A_init.flatten(),
        args=(X,),
        constraints=constraints,
        method='SLSQP',
        options={'disp': True}
    )
    A_optimized = result.x.reshape(nc, -1)
    XA = X @ A_optimized
    print(result)
    return XA
    
def trace_S(params, X, size):
    nc = size*1
    N = X.shape[0]
    
    ### direct normalization route
    A = QR2A(params, nc)
    chi = X @ A
    chi = chi + np.abs(np.min(chi))
    chi = chi/chi.sum(1)[:,None]
    
    ### constraint optimization route
    # chi = optimize_chi(params, X)
    
    w = np.ones(N)/N
    Dc = np.diag( 1 / (chi.T @ w) )
    D = np.diag(w)*1
    # D = np.linalg.pinv(X).T @ np.linalg.pinv(X)
    # S = Dc @ chi.T @ D**2 @ chi
    S_prime = chi.T @ D**1 @ chi
    row_sums = np.sum(S_prime, axis=1)
    D_tilde = np.diag(1 / row_sums)  # Normalize rows
    S = D_tilde @ S_prime
    # obj = np.trace(S)
    obj = np.linalg.det(S)
    return -obj

### optimization
# Initialize random Q and R parameters
Q_init = np.random.randn(size, size)
R_init = np.random.randn(size, size)
initial_params = np.concatenate([Q_init.flatten(), R_init.flatten()])

# Use SciPy's minimize function
result = minimize(
    fun=trace_S,
    x0=initial_params,
    args=(X, size),
    method="L-BFGS-B"
)

# %% fuzzy memebership
def assin_membership_from_chi(chi):
    N, nc = chi.shape
    membership = np.zeros(N)
    for nn in range(N):
        prob_vector = chi[nn,:]
        membership[nn] = np.random.choice(len(prob_vector), p=prob_vector)+0
        # membership[nn] = np.argmax(prob_vector)+1
    return membership

def compute_reduced_transition_matrix(P, membership):
    clusters = np.unique(membership)
    C = len(clusters)
    P_reduced = np.zeros((C, C))
    
    for i, cluster_i in enumerate(clusters):
        for j, cluster_j in enumerate(clusters):
            # Create boolean masks for the clusters
            mask_i = membership == cluster_i
            mask_j = membership == cluster_j
            P_reduced[i, j] = np.sum(P[np.ix_(mask_i, mask_j)])
        
        # Normalize row i to ensure stochasticity
        P_reduced[i, :] /= np.sum(P_reduced[i, :])
    
    return P_reduced

# %% analysis
a = result.x
A = QR2A(a, size)
chi = X @ A
chi = chi + np.abs(np.min(chi)) # chi[chi<0]=0
chi = chi/chi.sum(1)[:,None]
N = X.shape[0]
D = np.diag(np.ones(N)/N)*1#**2
# D = np.linalg.pinv(X).T @ np.linalg.pinv(X)

Pc = compute_reduced_transition_matrix(P, assin_membership_from_chi(chi))
# Pc = np.linalg.pinv(chi.T @ D @ chi) @ chi.T @ D @ P @ chi
plt.figure()
plt.imshow(np.real(Pc))
plt.title('coarse-grained')
print(Pc)

plt.figure()
plt.imshow(np.real(chi),aspect='auto', interpolation='none')
plt.title('membership')

### notes
# need to turn fuzzy chi to state assignment
# A might need more constraints
# how robust is it
# apply to data!

# %% study effect of coarse graining
###############################################################################
# %% compute EP
def get_steady_state(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    steady_state = np.real(eigenvectors[:, idx])
    steady_state = steady_state / np.sum(steady_state)
    return steady_state

def trans_entropy(M):
    pi = get_steady_state(M)
    h = 0
    for ii in range(M.shape[0]):
        for jj in range(M.shape[0]):
            h += pi[ii]*M[ii,jj]*np.log(M[ii,jj] + 1e-10)
    return -h

# %%
ep_full = trans_entropy(P)
print(ep_full)
ep_cg = trans_entropy(Pc)
print(ep_cg)

S_prime = chi.T @ D**1 @ chi
row_sums = np.sum(S_prime, axis=1)
D_tilde = np.diag(1 / row_sums)  # Normalize rows
S = D_tilde @ S_prime
print(np.trace(S)/1)

# %% scanning!!
reps = 20
nstates = np.arange(2,10)
crispness = np.zeros((len(nstates), reps))
for rr in range(reps):
    print(rr)
    for ii in range(len(nstates)):
        sizei = nstates[ii]
        Xi = (eigenvectors[:, :sizei])
        ### optimization
        Q_init = np.random.randn(sizei, sizei)
        R_init = np.random.randn(sizei, sizei)
        initial_params = np.concatenate([Q_init.flatten(), R_init.flatten()])
        result = minimize(fun=trace_S,
                        x0=initial_params,
                        args=(Xi, sizei),
                        method="L-BFGS-B")
        ### measurement
        a = result.x
        Ai = QR2A(a, sizei)
        chii = Xi @ Ai
        chii = chii + np.abs(np.min(chii)) # chi[chi<0]=0
        chii = chii/chii.sum(1)[:,None]
        # D = np.diag(np.ones(N)/N)*1
        # S_prime = chii.T @ D**1 @ chii
        # row_sums = np.sum(S_prime, axis=1)
        # D_tilde = np.diag(1 / row_sums)  # Normalize rows
        # Si = D_tilde @ S_prime
        # crispness[ii,rr] = np.trace(Si)
        
        ## use Pc directly
        # Pi = compute_reduced_transition_matrix(P, assin_membership_from_chi(chii))
        # Pi = np.linalg.pinv(chii.T @ D @ chii) @ 
        Pi = chii.T @ D @ P @ chii
        row_sums = np.sum(Pi, axis=1)
        D_tilde = np.diag(1 / row_sums)  # Normalize rows
        Pi = D_tilde @ Pi
        crispness[ii,rr] = np.trace(Pi)

# %%
plt.figure()
plt.plot(nstates, crispness/nstates[:,None])
plt.errorbar(nstates, np.mean(crispness,1)/nstates, np.std(crispness,1), fmt='k-o', ecolor='black', elinewidth=2,)
plt.xlabel('n states'); plt.ylabel('crispness')
        