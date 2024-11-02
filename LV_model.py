# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:15:04 2024

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# np.random.seed(42)

# %%
r = 0.7
eps = .22
def fx(x):
    if x<r:
        xx = x/r
    elif x>=r:
        xx = (1-x)/(1-r)
    return xx

def neural_dynamics(xy):
    x,y = xy[0],xy[1]
    xx = (1-eps)*fx(x) + eps*fx(y)
    yy = (1-eps)*fx(y) + eps*fx(x)
    return np.array([xx,yy])

def set_c(xy):
    # c = np.pi/np.abs(np.diff(xy))
    # if np.abs(np.diff(xy))==0:
    #     c = np.pi
    c = np.pi
    return c

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# %%
T = 5000
xys = np.zeros((T,2))
xys[0,:] = np.random.rand(2)
XYs = xys*0
dtheta = np.zeros(T)
theta = dtheta*0

for tt in range(T-1):
    xys[tt+1,:] = neural_dynamics(xys[tt,:])  #### can replace with a reasonable neural dynamcis, for heading!!!
    c = set_c(xys[tt+1,:])
    dtheta[tt+1] = c*(xys[tt+1,0] - xys[tt+1,1])
    theta[tt+1] = wrap_angle(theta[tt] + dtheta[tt+1])
    XYs[tt+1,:] = XYs[tt,:] + np.array([np.cos(theta[tt+1]), np.sin(theta[tt+1])])
    
# %%
plt.figure()
plt.plot(XYs[:,0],XYs[:,1])

# %% respones
### F: distance of perturbed and unperturbed tracks
### G: distance to the perturbed location
# %%
tp = 1000
tau = 200
reps = 100
Ss = 10**np.arange(-5,0,0.5)
Gs = np.zeros((len(Ss), reps))

for rr in range(reps):
    print(rr)
    for ss in range(len(Ss)):
        T = 2000
        xys = np.zeros((T,2))
        xys[0,:] = np.random.rand(2)
        XYs = xys*0
        dtheta = np.zeros(T)
        theta = dtheta*0
        
        xy_at_tp = np.zeros(2)
        for tt in range(T-1):
            xys[tt+1,:] = neural_dynamics(xys[tt,:])
            if tt==tp:
                w = np.random.rand()*2 - np.pi
                xys[tt+1,:] = 0.5 + np.array([Ss[ss]*np.cos(w), Ss[ss]*np.sin(w)])
            c = set_c(xys[tt+1,:])
            dtheta[tt+1] = c*(xys[tt+1,0] - xys[tt+1,1])
            theta[tt+1] = wrap_angle(theta[tt] + dtheta[tt+1])
            XYs[tt+1,:] = XYs[tt,:] + np.array([np.cos(theta[tt+1]), np.sin(theta[tt+1])])
            if tt==tp:
                xy_at_tp = XYs[tt+1,:]
        Gs[ss,rr] = np.sum((XYs[tp+tau,:] - xy_at_tp)**2)**0.5
    
# %%
plt.figure()
plt.plot(Ss, np.mean(Gs,1))
plt.xlabel('perturbation')
plt.ylabel('distance to stim')
plt.xscale('log')
