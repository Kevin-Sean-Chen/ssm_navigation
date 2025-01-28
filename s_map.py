# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:16:49 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.linalg import hankel
import scipy.sparse

# %% functional
### Logistic
def logistic_map(r, x, n):
    """Generate a logistic map time series."""
    time_series = []
    for _ in range(n):
        x = r * x * (1 - x)
        time_series.append(x)
    return np.array(time_series)

### Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Euler method parameters
dt = 0.01  # Time step
steps = 10000  # Number of steps

# Arrays to store results
x, y, z = 1.0, 1.0, 1.0
x_data = np.zeros(steps)
y_data = np.zeros(steps)
z_data = np.zeros(steps)
time = np.linspace(0, steps * dt, steps)

# Euler method loop
for i in range(steps):
    # dynamics
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    # Update variables using Euler method
    x += dx * dt
    y += dy * dt
    z += dz * dt
    # Store results
    x_data[i] = x
    y_data[i] = y
    z_data[i] = z

# Plot the Lorenz attractor in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_data, y_data, z_data, lw=0.5, color="blue")
ax.set_title("Lorenz Attractor (Euler Method)", fontsize=14)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Optional: Plot time series of x, y, z
plt.figure(figsize=(10, 6))
plt.plot(time, x_data, label="x(t)")
plt.plot(time, y_data, label="y(t)")
plt.plot(time, z_data, label="z(t)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.title("Time Series from Lorenz Attractor")
plt.show()

# %% to-do
# use Hankel for delay embedding
# test different kerenels and regulaization
# write function to judge window size, nonlinearity, and reconstruction... SVD approach?? #################################
# batch test with tracks...

# %% S-map from scratch
### choose data
Xt = np.vstack((x_data, y_data)).T

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
theta = 1  ### nonlinearity
tau = 10  ### prediction time step
k = lags+1  ### k nearest neighbors
Xe = delay_embed(Xt[:-tau,:], lags)
W = Wxx_sparse(Xe, theta, k)

# %% # 'learning'
yp = y_data[tau:]
Ye = (hankel(yp[:lags], yp[lags-1:]).T)
weights = np.linalg.pinv(Xe.T @ W @ Xe) @ Xe.T @ W @ Ye

# %% making predictions #######################################################
### scan for K, m, and tau (tau should have some type of scaling)
### can test with truncated Xe to make predictions -> low-D representation (ICA and other pre-processing...)
Y_pred = Xe @ weights
err = np.mean(np.linalg.norm(Ye - Y_pred, axis=1))
print(err)

# %% scanning tests
Ks = np.arange(tau+1,110,10)
err_ks = np.zeros(len(Ks))
for kk in range(len(Ks)):
    print(kk)
    lags = Ks[kk]
    Xe = delay_embed(Xt[:-tau,:], lags)  # embedd
    W = Wxx_sparse(Xe, theta, lags+1)  # kernel trick
    Ye = (hankel(yp[:lags], yp[lags-1:]).T)  # target
    weights = np.linalg.pinv(Xe.T @ W @ Xe) @ Xe.T @ W @ Ye  # regresion
    Y_pred = Xe @ weights  # prediction
    err_ks[kk] = np.mean(np.linalg.norm(Ye - Y_pred, axis=1)) # record error

# %%
plt.figure()
plt.plot(Ks, err_ks, '-o')
plt.xlabel('time delay'); plt.ylabel('error')

# %% gpt...

def time_delay_embedding(series, embedding_dim, tau):
    """Create a time-delay embedding of a time series."""
    n_samples = len(series) - (embedding_dim - 1) * tau
    embedded = np.array([series[i:n_samples + i:tau] for i in range(embedding_dim)]).T
    return embedded

def s_map(x_data, y_data, theta, l2_penalty=0.01):
    """
    Perform S-Map on the given data.
    
    Parameters:
        x_data: array-like, shape (n_samples, embedding_dim)
            The embedded time series data (reconstructed attractor).
        y_data: array-like, shape (n_samples,)
            The target time series (e.g., future values).
        theta: float
            The tuning parameter for exponential weighting (locality).
        l2_penalty: float, optional
            Ridge regularization penalty for stability (default is 0.01).
    
    Returns:
        predictions: array-like, shape (n_samples,)
            Predicted values for the target time series.
    """
    n_samples = x_data.shape[0]
    predictions = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Compute distances to the target point
        distances = np.linalg.norm(x_data - x_data[i], axis=1)
        
        # Compute weights based on the distances
        weights = np.exp(-theta * distances)
        
        # Fit a Ridge regression model with weighted data
        model = Ridge(alpha=l2_penalty, fit_intercept=True)
        model.fit(x_data, y_data, sample_weight=weights)
        
        # Predict the value for the current target point
        predictions[i] = model.predict(x_data[i].reshape(1, -1))
    
    return predictions

# Step 1: Generate a Logistic Map Time Series
r = 3.8  # Parameter for chaotic dynamics
x0 = 0.4  # Initial value
n_points = 1000  # Length of the time series
logistic_series = logistic_map(r, x0, n_points)

# Step 2: Create Time-Delay Embedding
embedding_dim = 3
tau = 1
embedded_data = time_delay_embedding(logistic_series, embedding_dim, tau)
target = logistic_series[(embedding_dim - 1) * tau:]  # Future values for prediction

# Step 3: Apply S-Map
theta = .0  # Locality parameter
predicted = s_map(embedded_data, target, theta)

# Step 4: Evaluate Predictions
rmse = np.sqrt(np.mean((predicted - target)**2))
print(f"RMSE: {rmse:.4f}")

# Step 5: Plot Results
plt.figure(figsize=(10, 5))
plt.plot(target, label="True Signal", alpha=0.8, lw=1.5)
plt.plot(predicted, label="S-Map Prediction", alpha=0.8, lw=1.5)
plt.legend()
plt.xlabel("Time Index")
plt.ylabel("Value")
plt.title("S-Map Prediction on Logistic Map Time Series")
plt.show()
