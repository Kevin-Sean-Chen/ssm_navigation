# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:53:33 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.linalg import hankel
import scipy.sparse
from scipy.linalg import svd
from sklearn.decomposition import FastICA
import pickle

# %% load data
with open("example_data.pkl", "rb") as file:
    vec_vxy, vec_xy, vec_signal, rec_signal, data4fit, rec_tracks = pickle.load(file)

# %% process data
Xtrain = vec_vxy[::3][:10000] ### concatenate fiest... but it is important to do track-based embedding later!!
Xtest = vec_vxy[::3][25000:30000]

# %% S-map from scratch
### delay embedding
def delay_embed(Xt, lags):
    T,d = Xt.shape
    X = []
    for dd in range(d):
        X.append((hankel(Xt[:lags,dd], Xt[lags-1:,dd]).T))
    X = np.concatenate(X, axis=1)
    return X
### build local kernel
def Wxx(Xe, theta):
    T_, Kd = Xe.shape  ### embedded coordinate
    dist_matrix = np.zeros((T_, T_))
    for i in range(T_):
        for j in range(T_):
            dist_matrix[i, j] = np.linalg.norm(Xe[i,:] - Xe[j,:])

    # Compute the shortest distance along each row
    s = np.min(dist_matrix + np.eye(T_) * np.inf, axis=1)  # Ignore diagonal by adding inf

    # Build the kernel matrix
    W = np.zeros((T_, T_))
    for i in range(T_):
        for j in range(T_):
            W[i, j] = np.exp(-theta * dist_matrix[i, j] / s[i])
    return W


def Wxx_sparse(Xe, theta, k):
    """
    Xe: delay embedded T' x Kd matrix
    theta: nonlinearity control
    k: k nearest neighbors
    Returns: scipy.sparse.csr_matrix: Sparse T x T kernel matrix.
    """
    T, d = Xe.shape

    # Compute pairwise Euclidean distances
    time_series_squared = np.sum(Xe**2, axis=1, keepdims=True)  # Shape: (T, 1)
    dist_matrix_squared = (
        time_series_squared 
        + time_series_squared.T 
        - 2 * np.dot(Xe, Xe.T)
    )  # Shape: (T, T)
    dist_matrix = np.sqrt(np.maximum(dist_matrix_squared, 0))

    # Ensure the diagonal is ignored when finding neighbors
    np.fill_diagonal(dist_matrix, np.inf)

    # Retain only the shortest k neighbors per row
    sparse_matrix = scipy.sparse.lil_matrix((T, T))

    # Populate the sparse matrix with k-nearest neighbors
    for i in range(T):
        # Get indices of the k smallest distances in row i
        neighbor_indices = np.argsort(dist_matrix[i])[:k]
        # Compute weights for these neighbors
        weights = np.exp(-theta * dist_matrix[i, neighbor_indices] / dist_matrix[i, neighbor_indices[0]])
        # Assign weights to the sparse matrix
        sparse_matrix[i, neighbor_indices] = weights

    # Convert to CSR format for efficient computation and storage
    sparse_matrix = sparse_matrix.maximum(sparse_matrix.T)  # Symmetrize the matrix
    sparse_matrix = sparse_matrix.tocsr()
    
    return sparse_matrix

### consturction of embedded state and kernel
lags = 100  ### embedding window
theta = 5  ### nonlinearity
tau = 105  ### prediction time step
k = lags+1  ### k nearest neighbors
Xe = delay_embed(Xtrain[:-tau,:], lags)

# %% # 'learning'
# yp = Xtrain[tau:,0]
# Ye = (hankel(yp[:lags], yp[lags-1:]).T)
Xp = Xe[:-tau-lags,:]
W = Wxx_sparse(Xp, theta, k)
Yf = Xe[tau+lags:,[0,lags]]
weights = np.linalg.pinv(Xp.T @ W @ Xp) @ Xp.T @ W @ Yf

# %% making predictions #######################################################
### scan for K, m, and tau (tau should have some type of scaling)
### can test with truncated Xe to make predictions -> low-D representation (ICA and other pre-processing...)
Y_pred = Xp @ weights
err = np.mean(np.linalg.norm(Yf - Y_pred, axis=1))
corr_matrix = np.corrcoef(Yf.T, Y_pred.T)  # Transpose to get pairs of rows
err = corr_matrix[0, 2] 
print(err)

# %% scanning tests
Ks = np.arange(5,100,10) #np.arange(5+1,110,10)
ms = np.arange(2,15)
err_ks = np.zeros(len(Ks))
err_ms = np.zeros(len(ms))
for kk in range(len(Ks)):
# for kk in range(len(ms)):
    print(kk)
    lags = Ks[kk]
    # dim = ms[kk]
    
    ### training
    Xe = delay_embed(Xtrain[:-tau,:], lags)  # embedd training
    Xp = Xe[:-tau-0,:]
    Yf = Xe[tau+0:,[0,lags]]  # make future predictions, just for vx and vy!
    W = Wxx_sparse(Xp, theta, lags+0)  # kernel trick
    weights = np.linalg.pinv(Xp.T @ W @ Xp) @ Xp.T @ W @ Yf  # regresion
    
    ### testing
    Xe = delay_embed(Xtrain[:-tau,:], lags)  # embedd testing
    Xp = Xe[:-tau-0,:]
    Yf = Xe[tau+0:,[0,lags]]  # make future predictions!
    Y_pred = Xp @ weights  # prediction
    # err_ks[kk] = np.mean(np.linalg.norm(Yf - Y_pred, axis=1)) # record error
    corr_matrix = np.corrcoef(Yf.T, Y_pred.T)  # Transpose to get pairs of rows
    err_ks[kk] = corr_matrix[0, 2]  # measure R2
    # err_ms[kk] = np.mean(np.linalg.norm(Ye[:,0] - Y_pred[:,0]))
    print(err_ks[kk])

# %%
plt.figure()
plt.plot(Ks*1/30, (err_ks), '-o')
# plt.plot(ms, (err_ms), '-o')
plt.xlabel('time delay (s)'); plt.ylabel('R^2')
# plt.xlabel('m modes'); plt.ylabel('error')

# %% track-based
###############################################################################
# %% functions for IRWLS !!
def IRLS_Smap(data4fit, lags, dim=None):
    test_errors = np.zeros(len(data4fit))
    P = np.eye(lags*2)  # the embedding dimension, used for tracking inverse Hessian
    for ii in range(len(data4fit)):
        # print(ii)
        ### load data
        Xi = data4fit[ii][::3]
        ### build embedding material
        Xei = delay_embed(Xi[:-tau,:], lags)  # embedd training
        ### dim reduction
        if dim is not None:
            uu,ss,vv = svd(Xei)
            Xei = uu[:,:dim] @ np.diag(ss[:dim]) @ vv[:dim, :]
        Xpi = Xei[:-tau,:]
        Yfi = Xei[tau:,[0,lags]]  # make future predictions, just for vx and vy!
        Wi = Wxx_sparse(Xpi, theta, lags)  # kernel trick
        ### init with weighted OLS
        if ii == 0:
            weights = np.linalg.pinv(Xpi.T @ Wi @ Xpi) @ Xpi.T @ Wi @ Yfi
        ### iterate for IRLS
        else:
            err = Yfi - Xpi @ weights
            P = np.linalg.pinv(Xpi.T @ Wi @ Xpi)
            weights = weights + P @ Xpi.T @ Wi @ err
            # P = P - (P @ Xpi.T @ Wi @ Xpi @ P) / (1 + Xpi.T @ Wi @ Xpi @ P)   ### Woodbury lemma
            # print(weights[0,0])
        ### testing
        Xe = delay_embed(Xi[:-tau,:], lags)  # embedd testing
        Xpi = Xe[:-tau-lags*1,:]
        Yfi = Xe[tau+lags*1:,[0,lags]]  # make future predictions!
        Y_pred = Xpi @ weights  # prediction
        corr_matrix = np.corrcoef(Yfi.T, Y_pred.T)  # Transpose to get pairs of rows
        test_errors[ii] = corr_matrix[0, 2]
        # print(test_errors[ii])
    return weights, test_errors
    
# %% scanning lags via IRLS
Ks = np.arange(5,100,10) #np.arange(5+1,110,10)
ms = np.arange(2,12)
err_ks = np.zeros(len(Ks))
err_ms = np.zeros(len(ms))
err_ks_irls = np.zeros((len(Ks), len(data4fit)))
for kk in range(len(Ks)):
    print(kk)
    # ww, ei = IRLS_Smap(data4fit, Ks[kk])
    ww, ei = IRLS_Smap(data4fit, 35, dim=ms[kk])
    err_ks_irls[kk,:] = ei
    
# %%
plt.figure()
# plt.plot(Ks/30, (err_ks_irls),'k-o')
plt.plot(ms/30, np.mean(err_ks_irls,1),'k-o')
plt.errorbar(ms/30, np.mean(err_ks_irls,1), np.std(err_ks_irls,1)/1)
plt.xlabel('time delay (s)'); plt.ylabel('R^2')
plt.xlabel('m modes'); plt.ylabel('R^2')