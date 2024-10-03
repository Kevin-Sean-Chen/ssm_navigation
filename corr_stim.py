# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:45:58 2024

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt


# %% OU function
def OU_intensity(tau, dt, min_val, max_val, duration, mu=0, sigma=1):
    """
    Generate an Ornstein-Uhlenbeck (OU) process with clamped values.
    
    Parameters:
    - tau: timescale of the OU process (controls the rate of mean reversion)
    - dt: time step (update rate)
    - min_val: minimum clamp value
    - max_val: maximum clamp value
    - duration: total duration of the process
    - mu: long-term mean (default is 0)
    - sigma: volatility (default is 1)
    
    Returns:
    - intensities: the generated OU process clamped between min_val and max_val
    """
    vec_length = int(duration / dt)  # seconds to frames
    dWt = np.random.normal(0, np.sqrt(dt), size=vec_length)
    
    # Set initial value
    intensities = np.zeros(vec_length)
    intensities[0] = mu #dWt[0]
    binned_int = intensities*0
    
    # OU process update rule
    for i in range(1, vec_length):
        # Calculate next step using the OU process equation
        intensities[i] = intensities[i-1] + (mu - intensities[i-1]) * (dt / tau) + sigma**2 * dWt[i]
        
        # Clamp the value between min_val and max_val
        intensities[i] = np.clip(intensities[i], min_val, max_val)
        
        if intensities[i]>mu:
            binned_int[i] = 1
            
    return intensities, binned_int

# %% plotting
def compute_autocorrelation(data, max_lag):
    """
    Compute the autocorrelation for each lag from 1 to max_lag.

    Parameters:
    - data: 1D numpy array or list of time series data
    - max_lag: Maximum number of lags to compute

    Returns:
    - lags: Array of lags (from 1 to max_lag)
    - autocorr_values: Autocorrelation values corresponding to each lag
    """
    n = len(data)
    mean = np.nanmean(data)
    autocorr_values = []
    
    # Compute autocorrelation for each lag
    for lag in range(1, max_lag + 1):
        numerator = np.nansum((data[:-lag] - mean) * (data[lag:] - mean))
        denominator = np.nansum((data - mean) ** 2)
        autocorrelation = numerator / denominator
        autocorr_values.append(autocorrelation)
    
    return np.arange(1, max_lag + 1), autocorr_values

intensities, binned_int = OU_intensity(1, 1/60, 0, 1, 30, 0.5, .5)
plt.figure()
plt.plot(intensities)
plt.plot(binned_int)

lags,acf = compute_autocorrelation(binned_int, 1000)
plt.figure()
plt.plot(lags, acf)


