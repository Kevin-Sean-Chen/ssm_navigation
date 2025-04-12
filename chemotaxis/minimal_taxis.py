# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:02:20 2025

@author: ksc75
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

import pickle
import gzip
import glob
import os

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% binarized simple chemotaxis
### model linear gradient climbing with binarized action and states
### analyze the dynamical regiemes
### scan and measure entropy and information transfer

# %% minimal discrete model for R&T navigation
# Parameters
n_steps = 100
alpha = 1.0  # how fast x increases
beta = 2.0   # how big of a reset when flipping
prob_up_run = 0.5  # probability of flip
prob_down_turn = 0.9
prob_up_turn = 1-prob_up_run  # probability of continue
prob_down_run = 1-prob_down_turn

# Initialize
x = np.zeros(n_steps)
s = np.zeros(n_steps, dtype=int)  # 0 = down, 1 = up
a = np.zeros(n_steps, dtype=int)  # 0 = continue, 1 = flip

# Initial conditions
x[0] = 0.0
s[0] = 1  # starting in "up"

# Simple policy: stochastic action selection based on state
def policy(state):
    if state == 1:  # "up"
        return np.random.choice([0, 1], p=[prob_up_run, prob_up_turn])
    else:  # "down"
        return np.random.choice([0, 1], p=[prob_down_run, prob_down_turn]) #p=[prob_continue, prob_flip])

# Simulate
for t in range(n_steps - 1):
    a[t] = policy(s[t])
    
    if a[t] == 0:  # continue
        x[t+1] = x[t] + alpha
        s[t+1] = s[t]
    elif a[t] == 1:  # flip
        x[t+1] = x[t] - beta
        s[t+1] = 1 - s[t]  # flip up/down

# Final action
a[-1] = policy(s[-1])

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axs[0].plot(x, label='Environment variable x(t)')
axs[0].set_ylabel('x')
axs[0].legend()

axs[1].step(np.arange(n_steps), s, where='post', label='State (up=1, down=0)')
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Internal State')
axs[1].legend()

plt.suptitle('Agent-Environment Coupled Dynamics')
plt.tight_layout()
plt.show()

# %% entropic analysis
def transfer_entropy(X,Y,delay=1,gaussian_sigma=None):
	'''
	TE implementation: asymmetric statistic measuring the reduction in uncertainty
	for a future value of X given the history of X and Y. Or the amount
	of information from Y to X. Calculated through the Kullback-Leibler divergence 
	with conditional probabilities

	author: Sebastiano Bontorin
	mail: sbontorin@fbk.eu

	args:
		- X (1D array):
			time series of scalars (1D array)
		- Y (1D array):
			time series of scalars (1D array)
	kwargs:
		- delay (int): 
			step in tuple (x_n, y_{n - delay}, x_(n - delay))
		- gaussian_sigma (int):
			sigma to be used
			default set at None: no gaussian filtering applied
	returns:
		- TE (float):
			transfer entropy between X and Y given the history of X
	'''

	if len(X)!=len(Y):
		raise ValueError('time series entries need to have same length')

	n = float(len(X[delay:]))

	# number of bins for X and Y using Freeman-Diaconis rule
	# histograms built with numpy.histogramdd
	binX = len(np.unique(X))
	binY = len(np.unique(Y))
    
	# Definition of arrays of shape (D,N) to be transposed in histogramdd()
	x3 = np.array([X[delay:],Y[:-delay],X[:-delay]])
	x2 = np.array([X[delay:],Y[:-delay]])
	x2_delay = np.array([X[delay:],X[:-delay]])

	p3,bin_p3 = np.histogramdd(
		sample = x3.T,
		bins = [binX,binY,binX])

	p2,bin_p2 = np.histogramdd(
		sample = x2.T,
		bins=[binX,binY])

	p2delay,bin_p2delay = np.histogramdd(
		sample = x2_delay.T,
		bins=[binX,binX])

	p1,bin_p1 = np.histogramdd(
		sample = np.array(X[delay:]),
		bins=binX)

	# Hists normalized to obtain densities
	p1 = p1/n
	p2 = p2/n
	p2delay = p2delay/n
	p3 = p3/n

	# Ranges of values in time series
	Xrange = bin_p3[0][:-1]
	Yrange = bin_p3[1][:-1]
	X2range = bin_p3[2][:-1]

	# Calculating elements in TE summation
	elements = []
	for i in range(len(Xrange)):
		px = p1[i]
		for j in range(len(Yrange)):
			pxy = p2[i][j]

			for k in range(len(X2range)):
				pxx2 = p2delay[i][k]
				pxyx2 = p3[i][j][k]

				arg1 = float(pxy*pxx2)
				arg2 = float(pxyx2*px)

				# Corrections avoding log(0)
				if arg1 == 0.0: arg1 = float(1e-8)
				if arg2 == 0.0: arg2 = float(1e-8)

				term = pxyx2*np.log2(arg2) - pxyx2*np.log2(arg1) 
				elements.append(term)

	# Transfer Entropy
	TE = np.sum(elements)
	return TE

def transfer_entropy_both(X, Y, delay=1):
    '''
    Compute transfer entropy both ways: TE(Y->X) and TE(X->Y),
    also entropy H(x'|x), H(y'|y), H(x',y'|x,y).

    Args:
        X, Y: 1D arrays
        delay: time lag

    Returns:
        TE_YX: Transfer entropy Y -> X
        TE_XY: Transfer entropy X -> Y
        Hx: H(x'|x)
        Hy: H(y'|y)
        Hxy: H(x',y'|x,y)
    '''

    if len(X) != len(Y):
        raise ValueError('X and Y must have the same length')

    # n = float(len(X[delay:]))

    binX = len(np.unique(X))
    binY = len(np.unique(Y))

    # ## --- For TE(Y -> X)
    # x3 = np.array([X[delay:], Y[:-delay], X[:-delay]])
    # p3, _ = np.histogramdd(sample=x3.T, bins=[binX, binY, binX])

    # x2 = np.array([X[delay:], Y[:-delay]])
    # p2, _ = np.histogramdd(sample=x2.T, bins=[binX, binY])

    # x2delay = np.array([X[delay:], X[:-delay]])
    # p2delay, _ = np.histogramdd(sample=x2delay.T, bins=[binX, binX])

    # p1, _ = np.histogramdd(X[delay:], bins=binX)

    # p1 = p1 / n
    # p2 = p2 / n
    # p2delay = p2delay / n
    # p3 = p3 / n
    
    TE_XY = transfer_entropy(X, Y, delay=delay)
    TE_YX = transfer_entropy(Y, X, delay=delay)
    
    # # Transfer Entropy Y -> X
    # elements = []
    # for i in range(p3.shape[0]):
    #     px = p1[i]
    #     for j in range(p3.shape[1]):
    #         pxy = p2[i, j]
    #         for k in range(p3.shape[2]):
    #             pxx2 = p2delay[i, k]
    #             pxyx2 = p3[i, j, k]

    #             arg1 = float(pxy * pxx2)
    #             arg2 = float(pxyx2 * px)

    #             if arg1 == 0.0: arg1 = 1e-8
    #             if arg2 == 0.0: arg2 = 1e-8

    #             term = pxyx2 * (np.log2(arg2) - np.log2(arg1))
    #             elements.append(term)

    # TE_YX = np.sum(elements)

    # ## --- For TE(X -> Y)
    # y3 = np.array([Y[delay:], X[:-delay], Y[:-delay]])
    # p3_y, _ = np.histogramdd(sample=y3.T, bins=[binY, binX, binY])

    # y2 = np.array([Y[delay:], X[:-delay]])
    # p2_y, _ = np.histogramdd(sample=y2.T, bins=[binY, binX])

    # y2delay = np.array([Y[delay:], Y[:-delay]])
    # p2delay_y, _ = np.histogramdd(sample=y2delay.T, bins=[binY, binY])

    # p1_y, _ = np.histogramdd(Y[delay:], bins=binY)

    # p1_y = p1_y / n
    # p2_y = p2_y / n
    # p2delay_y = p2delay_y / n
    # p3_y = p3_y / n

    # # Transfer Entropy X -> Y
    # elements_y = []
    # for i in range(p3_y.shape[0]):
    #     py = p1_y[i]
    #     for j in range(p3_y.shape[1]):
    #         pyx = p2_y[i, j]
    #         for k in range(p3_y.shape[2]):
    #             pyy2 = p2delay_y[i, k]
    #             pyxy2 = p3_y[i, j, k]

    #             arg1 = float(pyx * pyy2)
    #             arg2 = float(pyxy2 * py)

    #             if arg1 == 0.0: arg1 = 1e-8
    #             if arg2 == 0.0: arg2 = 1e-8

    #             term = pyxy2 * (np.log2(arg2) - np.log2(arg1))
    #             elements_y.append(term)

    # TE_XY = np.sum(elements_y)

    ## --- Entropies H(x'|x), H(y'|y)
    counts_xx = np.histogram2d(X[:-delay], X[delay:], bins=[binX, binX])[0]
    counts_xx /= np.sum(counts_xx)

    Hx = 0.0
    for i in range(binX):
        marginal_x = np.sum(counts_xx[i, :])
        for j in range(binX):
            if counts_xx[i, j] > 0 and marginal_x > 0:
                Hx -= counts_xx[i, j] * np.log2(counts_xx[i, j] / marginal_x)

    counts_yy = np.histogram2d(Y[:-delay], Y[delay:], bins=[binY, binY])[0]
    counts_yy /= np.sum(counts_yy)

    Hy = 0.0
    for i in range(binY):
        marginal_y = np.sum(counts_yy[i, :])
        for j in range(binY):
            if counts_yy[i, j] > 0 and marginal_y > 0:
                Hy -= counts_yy[i, j] * np.log2(counts_yy[i, j] / marginal_y)

    ## --- H(x',y'|x,y)
    stacked = np.vstack([X[:-delay], Y[:-delay], X[delay:], Y[delay:]]).T
    counts_xy_xyprime, _ = np.histogramdd(stacked, bins=[binX, binY, binX, binY])
    counts_xy_xyprime /= np.sum(counts_xy_xyprime)

    Hxy = 0.0
    for i in range(binX):
        for j in range(binY):
            marginal_xy = np.sum(counts_xy_xyprime[i, j, :, :])
            for k in range(binX):
                for l in range(binY):
                    p = counts_xy_xyprime[i, j, k, l]
                    if p > 0 and marginal_xy > 0:
                        Hxy -= p * np.log2(p / marginal_xy)

    return TE_YX, TE_XY, Hx, Hy, Hxy

# %% binarized taxis model
def binary_chemotaxis(pt, n_steps=3000):
    n_steps = 100
    alpha = 1.0  # how fast x increases
    beta = 2.0   # how big of a reset when flipping
    prob_up_run = 0.5  # probability of flip
    prob_down_turn = pt
    prob_up_turn = 1-prob_up_run  # probability of continue
    prob_down_run = 1-prob_down_turn

    # Initialize
    x = np.zeros(n_steps)
    s = np.zeros(n_steps, dtype=int)  # 0 = down, 1 = up
    a = np.zeros(n_steps, dtype=int)  # 0 = continue, 1 = flip

    # Initial conditions
    x[0] = 0.0
    s[0] = 1  # starting in "up"

    # Simple policy: stochastic action selection based on state
    def policy(state):
        if state == 1:  # "up"
            return np.random.choice([0, 1], p=[prob_up_run, prob_up_turn])
        else:  # "down"
            return np.random.choice([0, 1], p=[prob_down_run, prob_down_turn]) #p=[prob_continue, prob_flip])

    # Simulate
    for t in range(n_steps - 1):
        a[t] = policy(s[t])
        
        if a[t] == 0:  # continue
            x[t+1] = x[t] + alpha
            s[t+1] = s[t]
        elif a[t] == 1:  # flip
            x[t+1] = x[t] - beta
            s[t+1] = 1 - s[t]  # flip up/down

    # Final action
    a[-1] = policy(s[-1])
    
    return s,a,x


def nonMarkov_test(pt, n_steps=3000):
    alpha = 1.0  # how fast x increases
    beta = 2.0   # how big of a reset when flipping
    
    # Define transition-dependent probabilities
    # Each (prev_state, current_state) has its own action probabilities
    # prob_run[prev_state][curr_state] = probability to "continue"
    # prob_turn[prev_state][curr_state] = probability to "flip"
    prob_run = np.array([[pt, 0.5],   # when coming from down (0)
                         [1-pt, 0.5]])  # when coming from up (1)
    
    prob_turn = 1 - prob_run
    
    # Initialize
    x = np.zeros(n_steps)
    s = np.zeros(n_steps, dtype=int)  # 0 = down, 1 = up
    a = np.zeros(n_steps, dtype=int)  # 0 = continue, 1 = flip
    
    # Initial conditions
    x[0] = 0.0
    s[0] = 1  # starting in "up"
    prev_s = s[0]  # Previous state initialized as the same
    
    # New policy: depends on (prev_state, current_state)
    def policy(prev_state, current_state):
        return np.random.choice([0, 1], p=[prob_run[prev_state, current_state], prob_turn[prev_state, current_state]])
    
    # Simulate
    for t in range(n_steps - 1):
        a[t] = policy(prev_s, s[t])  # depend on both previous and current state
    
        if a[t] == 0:  # continue
            x[t+1] = x[t] + alpha
            s[t+1] = s[t]
        elif a[t] == 1:  # flip
            x[t+1] = x[t] - beta
            s[t+1] = 1 - s[t]  # flip up/down
    
        prev_s = s[t]  # Update previous state for next step
    
    # Final action
    a[-1] = policy(prev_s, s[-1])
    
    return s,a,x


def run_chemotaxis(N, pt):
    chi = 0
    for nn in range(N):
        _,_,x = binary_chemotaxis(pt)
        # _,_,x = nonMarkov_test(pt)
        if x[0]<x[-1]:
            chi += x[-1]-x[0]
        elif x[0]>=x[-1]:
            chi -= x[0]-x[-1]
    return chi/N

# %% scanning parameter phase for performance
ps = np.arange(0.1,1,.1)
chis = np.zeros(len(ps))
N = 100

for pp in range(len(ps)):
    chis[pp] = run_chemotaxis(N, ps[pp])

# %% plotting
plt.figure()
plt.plot(ps, chis,'-o')
plt.xlabel('P(turn|down)'); plt.ylabel('drift from origin')

# %% scanning parameter phase for performance
ps = np.arange(0.1,1,.1)
N = 100
infos = np.zeros((len(ps), N, 5+1))

for pp in range(len(ps)):
    for nn in range(N):
        s,a,x = binary_chemotaxis(ps[pp])
        # s,a,x = nonMarkov_test(ps[pp])
        TE_YX, TE_XY, Hx, Hy, Hxy = transfer_entropy_both(a, s)  ### X,Y
        
        estimate = Hxy - Hx - Hy + TE_XY + TE_YX
        infos[pp, nn, :] = np.array([ TE_YX, TE_XY, Hx, Hy, Hxy, estimate])
        
# %% plotting
labels = ['TE(s->x)', 'TE(x->s)', 'H(x\'|x)', 'H(s\'|s)', 'H(x\',s\'|x,s)', 'finite-scale']
plt.figure()
# plt.plot(ps, np.mean(infos,1), '-o')
for ii in range(6):
    plt.plot(ps, np.mean(infos[:,:,ii],1),'-o', label=labels[ii])
plt.legend()
plt.xlabel('P(turn|down)')
