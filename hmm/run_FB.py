# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:10:51 2024

@author: ksc75
"""

import numpy as np
from scipy.special import logsumexp

def runFB_GLMHMM(mm, xx, yy, mask=None):
    """
    Run forward-backward algorithm for GLM-HMM with discrete observations using logSumExp trick.
    
    Parameters
    ----------
    mm : dict
        Parameter dictionary with keys:
            - 'A': Transition matrix (k x k)
            - 'loglifun': Function to compute log-likelihoods
    xx : ndarray
        Input features (for the GLM)
    yy : ndarray
        Observations for each time bin
    mask : ndarray, optional
        Mask array for valid time bins
    
    Returns
    -------
    logp : float
        Log marginal probability log P(X|theta)
    gams : ndarray
        Marginal posterior over states at each time
    xisum : ndarray
        Summed marginal over pairs of states
    logcs : ndarray
        Log conditional marginal probabilities
    """
    
    nStates = mm['A'].shape[0]
    nT = len(yy)
    if mask is None:
        mask = np.ones(nT, dtype=bool)
    
    logpi0 = np.log(np.ones(nStates) / nStates)  # uniform prior
    logpy = mm['loglifun'](mm, xx, yy, mask)     # compute log-likelihood for each state and time
    
    logaa = np.zeros((nStates, nT))  # forward log-probabilities
    logbb = np.zeros((nStates, nT))  # backward log-probabilities
    logcs = np.zeros(nT)             # forward marginal likelihoods

    # Forward pass
    if mask[0]:
        logpyz = logpi0 + logpy[0]
        logcs[0] = logsumexp(logpyz)
        logaa[:, 0] = logpyz - logcs[0]
    else:
        logcs[0] = 0
        logaa[:, 0] = logpi0

    for jj in range(1, nT):
        logaaprior = logsumexp(logaa[:, jj-1] + np.log(mm['A']).T, axis=1)
        if mask[jj]:
            logpyz = logaaprior + logpy[jj]
            logcs[jj] = logsumexp(logpyz)
            logaa[:, jj] = logpyz - logcs[jj]
        else:
            logcs[jj] = 0
            logaa[:, jj] = logaaprior

    # Backward pass
    for jj in range(nT-2, -1, -1):
        if mask[jj+1]:
            logbb[:, jj] = logsumexp(np.log(mm['A']) + logbb[:, jj+1] + logpy[jj+1], axis=1) - logcs[jj+1]
        else:
            logbb[:, jj] = logsumexp(np.log(mm['A']) + logbb[:, jj+1], axis=1) - logcs[jj+1]

    # Outputs
    logp = np.sum(logcs)
    loggams = logaa + logbb
    gams = np.exp(loggams - logsumexp(loggams, axis=0))

    # Compute summed xis
    xisum = np.zeros((nStates, nStates))
    logA = np.log(mm['A'])
    for jj in range(nT - 1):
        if mask[jj + 1]:
            xisum += np.exp(logA + (logaa[:, jj] - logcs[jj+1])[:, None] + (logbb[:, jj+1] + logpy[jj+1]))
        else:
            xisum += np.exp(logA + (logaa[:, jj] - logcs[jj+1])[:, None] + logbb[:, jj+1])

    return logp, gams, xisum, logcs
