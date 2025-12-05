# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 10:45:53 2025

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# %%
def dwell_from_jumps(times, states, *,
                     start_time=None, start_state=None,
                     end_time=None, state=None):
    """
    Compute dwell durations from jump times and jumped-to states.

    Parameters
    ----------
    times : (K,) array-like
        Strictly increasing jump times t0 < t1 < ... < tK.
    states : (K,) array-like
        State after each jump: s[i] is occupied on [t[i], t[i+1]).
    start_time : float, optional
        If provided with start_state, include leading dwell [start_time, times[0]).
    start_state : int/float, optional
        State occupied before the first jump at times[0].
    end_time : float, optional
        If provided, include trailing dwell [times[-1], end_time) in states[-1].
    state : value or None
        If given, return only dwell durations for this state value.
        If None, return (durations, states_for_each_duration).

    Returns
    -------
    If state is None:
        durations : (M,) float array
        state_seq : (M,) array with the state for each duration
    Else:
        durations_for_state : (m,) float array
    """
    t = np.asarray(times, dtype=float)
    s = np.asarray(states)

    if t.ndim != 1 or s.ndim != 1 or len(t) != len(s):
        raise ValueError("times and states must be 1D and of equal length (jumped-to semantics).")
    if not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing.")

    starts = []
    ends = []
    vals = []

    # Optional leading interval
    if start_time is not None and start_state is not None and start_time < t[0]:
        starts.append(float(start_time))
        ends.append(float(t[0]))
        vals.append(start_state)

    # Intervals between jumps: [t[i], t[i+1]) has state s[i]
    if len(t) >= 2:
        starts.extend(t[:-1])
        ends.extend(t[1:])
        vals.extend(s[:-1])

    # Optional trailing interval
    if end_time is not None and end_time > t[-1]:
        starts.append(float(t[-1]))
        ends.append(float(end_time))
        vals.append(s[-1])

    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    vals = np.asarray(vals)

    durations = ends - starts
    if np.any(durations < 0):
        raise ValueError("Found negative interval; check start/end times.")

    if state is None:
        return durations, vals
    else:
        return durations[vals == state]

# %% two state system
def two_state_OUdrive(kt):
    """
    Input time series of bounded OU drive on the kinetic rate of two states
    return the time series of two states
    """    
    t = 0.0
    state = np.random.randint(0,2)  # start in state 0
    times = [t]
    states = [state]
    t_max = len(kt)
    ii = 0
    while ii < t_max: #t < t_max:
        # unpack rates
        k01, k10 = kt[ii], 0.5 #kt[ii] #kt[ii,1]
        # Pick rate depending on current state
        if state == 0:
            rate = k01
        else:
            rate = k10
        
        # Sample waiting time ~ Exp(rate)
        dt = np.random.exponential(1.0 / rate)
        t += dt
        if ii > t_max:
            break
        
        # Flip state
        state = 1 - state
        
        # Record
        times.append(t)
        states.append(state)
        ii += 1
    return np.array(times), np.array(states)

# %% run time-varying kinetics
# Example usage with OU process
kt = np.ones(100000)+0.1
dt = 0.01
k0 = .5 
tau = 5
for tt in range(len(kt)-1):
    kt[tt+1] = kt[tt] + dt/tau*(-(kt[tt] - k0)) + np.random.randn()*dt**0.5
kt[kt<0] = 0.1

#### periodic drive
# freq = 0.1
# kt = (np.cos(np.arange(len(kt))*freq)+1)*k0

#### constant
# kt = kt*0+k0

times, states = two_state_OUdrive(kt)

# Plot time series
plt.step(times, states, where="post")
plt.xlabel("time")
plt.ylabel("state")
plt.title("Two-state system (0 ↔ 1)")
plt.show()

# %% dwell time
dwell,_ = dwell_from_jumps(times, states)
plt.figure()
plt.hist(dwell,30)

dwell = np.asarray(dwell)
dwell = dwell[dwell > 0]
x = np.sort(dwell)
y = 1.0 - np.arange(1, len(x)+1) / len(x)
plt.figure()
plt.loglog(x, y, marker='o', linestyle='none')
plt.xlabel("dwell time")
plt.ylabel("P(Dwell ≥ x)")
plt.title("Dwell-time CCDF (log–log)")
plt.tight_layout()
