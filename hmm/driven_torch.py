# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 19:01:46 2025

@author: ksc75
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# ----------------------------------------------------------
# Define Model Class
class LogLinearMarkovWithBaseline(nn.Module):
    def __init__(self, N, u_dim, w_regu=0.0):
        super().__init__()
        self.N = N
        self.u_dim = u_dim
        self.w_regu = w_regu

        P0 = np.random.rand(N, N)
        P0 /= P0.sum(axis=1, keepdims=True)
        logP0_init = np.log(P0)
        self.logP0 = nn.Parameter(torch.tensor(logP0_init, dtype=torch.float32))

        self.W = nn.Parameter(torch.randn(N, N-1, u_dim) * 0.01)

    def transition_log_probs(self, x_curr, u_curr):
        T = x_curr.shape[0]
        logits = torch.zeros((T, self.N), device=u_curr.device)

        for curr_state in range(self.N):
            mask = (x_curr == curr_state)
            if torch.any(mask):
                u_selected = u_curr[mask]
                w_selected = self.W[curr_state]
                stimulus_logits = torch.matmul(u_selected, w_selected.T)

                baseline_logits = self.logP0[curr_state].unsqueeze(0).expand(u_selected.shape[0], -1)

                idx = torch.arange(self.N)
                idx_no_self = idx[idx != curr_state]

                full_logits = baseline_logits.clone()
                full_logits[:, idx_no_self] += stimulus_logits

                logits[mask] = full_logits

        logZ = torch.logsumexp(logits, dim=1, keepdim=True)
        log_probs = logits - logZ

        return log_probs

    # def negative_log_likelihood(self, x_seq, u_seq, lag):
    #     x_seq = torch.tensor(x_seq, dtype=torch.long)
    #     u_seq = torch.tensor(u_seq, dtype=torch.float32)
        
    #     if lag>0:
    #         x_seq, u_seq = x_seq[lag:], u_seq[:-lag]
    #     x_curr = x_seq[:-1]
    #     x_next = x_seq[1:]
    #     u_curr = u_seq[:-1]

    #     log_probs = self.transition_log_probs(x_curr, u_curr)
    #     selected_log_probs = log_probs.gather(1, x_next.unsqueeze(1)).squeeze()
    #     nll = -selected_log_probs.sum()
    #     nll += self.w_regu * torch.sum(self.W**2)
    #     return nll
    
    def negative_log_likelihood(self, x_seq, u_seq, lag, mask_seq=None):
        """
        Now optionally take a mask sequence (0/1) to include/exclude transitions.
        """
        x_seq = torch.tensor(x_seq, dtype=torch.long)
        u_seq = torch.tensor(u_seq, dtype=torch.float32)
        
        if lag>0:
            x_seq, u_seq, mask_seq = x_seq[lag:], u_seq[:-lag], mask_seq[lag:]
        x_curr = x_seq[:-1]
        x_next = x_seq[1:]
        u_curr = u_seq[:-1]

        log_probs = self.transition_log_probs(x_curr, u_curr)
        selected_log_probs = log_probs.gather(1, x_next.unsqueeze(1)).squeeze()

        if mask_seq is not None:
            mask_seq = torch.tensor(mask_seq, dtype=torch.float32)
            mask_curr = mask_seq[:-1]  # Mask for transitions
            selected_log_probs = selected_log_probs * mask_curr  # Only include masked-in terms
            nll = -selected_log_probs.sum() #/ (mask_curr.sum() + 1e-8)  # Normalize by number of valid points
        else:
            nll = -selected_log_probs.sum() #mean()

        nll += self.w_regu * torch.sum(self.W**2)
        return nll

    def row_normalization(self):
        with torch.no_grad():
            logZ = torch.logsumexp(self.logP0, dim=1, keepdim=True)
            self.logP0.data -= logZ

    def fit(self, x_seq, u_seq, lag=0, n_epochs=500, lr=1e-2, verbose=True, mask_seq=None):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        x_seq = torch.tensor(x_seq, dtype=torch.long)
        u_seq = torch.tensor(u_seq, dtype=torch.float32)

        losses = []
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood(x_seq, u_seq, lag, mask_seq=mask_seq)
            loss.backward()
            optimizer.step()

            self.row_normalization()

            losses.append(loss.item())

            if verbose and (epoch % 50 == 0 or epoch == n_epochs-1):
                print(f"Epoch {epoch}: loss = {loss.item():.6f}")

        return losses

    def predict_transition_probs(self, x, u):
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

# ----------------------------------------------------------
# Reconstruct W
def reconstruct_full_W(model):
    N, N_minus_1, u_dim = model.W.shape
    full_W = np.full((N, N, u_dim), np.nan)

    W_np = model.W.detach().cpu().numpy()

    for i in range(N):
        idx = np.arange(N)
        idx_no_self = idx[idx != i]
        full_W[i, idx_no_self, :] = W_np[i]

    return full_W

# ----------------------------------------------------------
# Simulate Markov chain
def simulate_markov_chain(N, u_dim, T, true_logP0, true_W):
    x_seq = np.zeros(T, dtype=int)
    u_seq = np.random.randn(T, u_dim)*10

    for t in range(T-1):
        x_curr = x_seq[t]
        u_curr = u_seq[t]

        logits = true_logP0[x_curr].copy()
        idx_no_self = np.arange(N)
        idx_no_self = idx_no_self[idx_no_self != x_curr]

        logits[idx_no_self] += true_W[x_curr] @ u_curr
        probs = np.exp(logits - logsumexp(logits))
        x_next = np.random.choice(N, p=probs)

        x_seq[t+1] = x_next

    return x_seq, u_seq

# %% ----------------------------------------------------------
# Main

# Set parameters
N = 3
u_dim = 1
T = 50000

# Ground-truth model
P0 = np.random.rand(N, N)
P0 /= P0.sum(axis=1, keepdims=True)
logP0 = np.log(P0)
true_W = np.random.randn(N, N-1, u_dim) * 0.5

# Simulate dataset
x_seq, u_seq = simulate_markov_chain(N, u_dim, T, logP0, true_W)
mask_seq = (np.random.rand(len(x_seq)) < 0.9999).astype(int)

# Fit model
model = LogLinearMarkovWithBaseline(N=N, u_dim=u_dim, w_regu=0.2)
losses = model.fit(x_seq, u_seq, mask_seq=mask_seq, n_epochs=1000, lr=1e-2)

# %%
# Reconstruct learned W
learned_W = reconstruct_full_W(model)

# Reconstruct true W (for comparison)
true_W_full = np.full((N, N, u_dim), np.nan)
for i in range(N):
    idx_no_self = np.arange(N)
    idx_no_self = idx_no_self[idx_no_self != i]
    true_W_full[i, idx_no_self, :] = true_W[i]

# Compare
diff = np.nanmean(np.abs(learned_W - true_W_full))
print(f"\nMean absolute difference between true W and learned W: {diff:.4f}")

# %% ----------------------------------------------------------
# Visualization

fig, axs = plt.subplots(2, 2, figsize=(12, 6))
d=0
vmin = min(np.nanmin(true_W_full[:,:,d]), np.nanmin(learned_W[:,:,d]))
vmax = max(np.nanmax(true_W_full[:,:,d]), np.nanmax(learned_W[:,:,d]))
axs[0,d].imshow(true_W_full[:,:,d], vmin=vmin, vmax=vmax, cmap='bwr')
axs[0,d].set_title(f"True W")
axs[1,d].imshow(learned_W[:,:,d], vmin=vmin, vmax=vmax, cmap='bwr')
axs[1,d].set_title(f"Learned W")

d=1
inf_P, true_P = np.exp(model.logP0.detach().numpy()), P0
vmin = min(np.nanmin(inf_P), np.nanmin(true_P))
vmax = max(np.nanmax(inf_P), np.nanmax(true_P))
axs[0,d].imshow(true_P, vmin=vmin, vmax=vmax, cmap='bwr')
axs[0,d].set_title(f"True P")
axs[1,d].imshow(inf_P, vmin=vmin, vmax=vmax, cmap='bwr')
axs[1,d].set_title(f"Learned P")

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(15, 15))

for i in range(N):
    for j in range(N):
        # Subplot index needs to be (i * 5 + j + 1)
        plt.subplot(N, N, i * N + j + 1)
        plt.plot(learned_W[i, j, :])
        plt.title(f'({i},{j})', fontsize=8)
        plt.xticks([])  # remove x ticks
        plt.yticks([])  # remove y ticks

plt.tight_layout()
plt.show()

# %% load reduced model and stim valuse from the dceomposed/reduced Markov script!
# u_dim = 150
# stim_matrix = np.zeros((u_dim, len(stim_bin)))
# stim_matrix[0,:] = stim_bin
# for ii in range(1,u_dim):
#     stim_matrix[ii,:] = np.concatenate((stim_bin[ii:] , np.zeros(ii)))
    
# # %% test fitting
# # Create model
# # model = LogLinearMarkovTorch(N=5, u_dim=1, w_regu=0.01)
# model = LogLinearMarkovWithBaseline(N=5, u_dim=u_dim, w_regu=0.01)
# mask_tracks = idsi[:-1] == idsi[1:]

# # Fit the model
# losses = model.fit(reduced_behavior, stim_matrix.T, mask_seq=mask_tracks, lag=0, n_epochs=1000, lr=1e-2) #### NEED mask for tracks here !! ######
