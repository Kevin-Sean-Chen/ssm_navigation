# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 00:18:03 2025

@author: kevin
"""


import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

# np.random.seed(3)

# %% params
### entry memory
a_en = 0.727
b_en = 0.437
r_re = 2
p_re = 0.505
c_re = 0.733
sig_re = 3.23

### exit memory
a_ex = 0.956
b_ex = 0.482
r_le = 1
p_le = 0.81
c_le = 0.6
sig_le = 3.255

wind_bias = 6.2 *1

# %% numerical test
M_en = np.array([[a_en,  b_en],
              [1-c_re,  c_re]])

M_le = np.array([[a_ex,  b_ex],
              [1-c_le,  c_le]])

# %% environment
right_border = 100
def edge_environment(x, edge=0):
    x_,y_ = x[0],x[1]
    if x_>edge and x_<right_border:
        return 1
    else:
        return 0

T = 500
xs = np.zeros((T,2))
vecs = np.ones((T, 2))  #### fix this for memory!
m_entry = np.array([0,1]) #vecs*1
m_exit = np.array([0,1]) #vecs*1
vs = vecs*1
timer = 0
stay_interval = 10 ###???
state=0
vec_memory = np.array([0,1])
state = np.zeros(T) ################### use internal state!!!
# %% updates
for tt in range(5,T):
    ### check environment
    env_next = edge_environment(xs[tt,:])
    env_prev = edge_environment(xs[tt-1,:])
    ### states ###
    # crossing odor edge
    if env_next==1 and env_prev==0 and state[tt-1]==0:
        stay_interval = np.random.negative_binomial(r_re, 1-p_re, size=1)
        state[tt] = 1
        timer = 0      
    # exit odor
    elif env_next==0 and env_prev==1 and state[tt-1]==1:
        stay_interval = np.random.negative_binomial(r_le, 1-p_le, size=1)
        state[tt] = 0
        timer = 0
    else:
        state[tt] = state[tt-1]
        
    ### check state clock ###
    timer += .2
    if timer>stay_interval+1 and env_next==0 and state[tt]==0:
        state[tt] = 1
    elif timer>stay_interval+1 and env_next==1 and state[tt]==1:
        state[tt] = 0
            # state[tt] = 1 - state[tt]  # flip state
        
    ### memory ###
    # in odor
    if env_next==1 and env_prev==0: #state[tt]==1 and state[tt-1]==0:
        vec_hist = np.mean(vs[tt-5:tt-1,:],0)
        vec_memory = vec_hist / np.linalg.norm(vec_hist)
        m_entry = a_en*m_entry + b_en*vec_memory + np.random.randn(2)*sig_re*0
    elif env_next==0 and env_prev==1: #state[tt]==0 and state[tt-1]==1:
        vec_hist = np.mean(vs[tt-5:tt-1,:],0)
        vec_memory = vec_hist / np.linalg.norm(vec_hist)
        m_exit = a_ex*m_exit + b_ex*vec_memory + np.random.randn(2)*sig_le*0
        
    ### actions ###
    if state[tt]==1:
        vs[tt,:] = c_le*vs[tt-1,:] + (1-c_le)*m_exit + np.random.randn(2)*sig_re**2 + np.array([0,wind_bias])
    elif state[tt]==0:
        vs[tt,:] = c_re*vs[tt-1,:] + (1-c_re)*m_entry + np.random.randn(2)*sig_le**2 + np.array([0,wind_bias])
    
    ### track location
    xs[tt,:] = xs[tt-1,:] + vs[tt,:]
    

# %%
plt.figure()
pos = np.where(state==0)[0]
plt.plot(xs[:,0],xs[:,1],'-o')
plt.plot(xs[pos,0],xs[pos, 1],'ro')     
plt.plot(xs[0,0],xs[0,1],'o')
# plt.figure()
# plt.fill_between(np.linspace(0, max(xs[:,0])), min(xs[:,1]), max(xs[:,1]), color='gray', alpha=0.3)
plt.fill_between(np.linspace(0, right_border), min(xs[:,1]), max(xs[:,1]), color='gray', alpha=0.3)

# %% generate many
# reps = 100
# window = 3
# en_angs = []
# ex_angs = []

# def angle_between_vectors(a, b):
#     """Computes the angle (in radians) between two vectors a and b."""
#     dot_product = np.dot(a, b)
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)

#     # Compute angle in radians
#     cos_theta = dot_product / (norm_a * norm_b)
#     theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical errors

#     return theta  # Returns angle in radians

# for rr in range(reps):
#     print(rr)
#     for tt in range(5,T):
#         ### check environment
#         env_next = edge_environment(xs[tt,:])
#         env_prev = edge_environment(xs[tt-1,:])
#         ### states ###
#         # crossing odor edge
#         if env_next==1 and env_prev==0 and state[tt-1]==0:
#             stay_interval = np.random.negative_binomial(r_re, 1-p_re, size=1)
#             state[tt] = 1
#             timer = 0      
#         # exit odor
#         elif env_next==0 and env_prev==1 and state[tt-1]==1:
#             stay_interval = np.random.negative_binomial(r_le, 1-p_le, size=1)
#             state[tt] = 0
#             timer = 0
#         else:
#             state[tt] = state[tt-1]
            
#         ### check state clock ###
#         timer += .2
#         if timer>stay_interval+1 and env_next==0 and state[tt]==0:
#             state[tt] = 1
#         elif timer>stay_interval+1 and env_next==1 and state[tt]==1:
#             state[tt] = 0
#                 # state[tt] = 1 - state[tt]  # flip state
            
#         ### memory ###
#         # in odor
#         if env_next==1 and env_prev==0: #state[tt]==1 and state[tt-1]==0:
#             vec_hist = np.mean(vs[tt-window:tt-1,:],0)
#             vec_memory = vec_hist / np.linalg.norm(vec_hist)
#             en_angs.append(180 - angle_between_vectors(vec_memory, np.array([0,1])) * 180/np.pi)
#             m_entry = a_en*m_entry + b_en*vec_memory + np.random.randn(2)*sig_re*0
#         elif env_next==0 and env_prev==1: #state[tt]==0 and state[tt-1]==1:
#             vec_hist = np.mean(vs[tt-window:tt-1,:],0)
#             vec_memory = vec_hist / np.linalg.norm(vec_hist)
#             ex_angs.append(angle_between_vectors(vec_memory, np.array([0,1])) * 180/np.pi)
#             m_exit = a_ex*m_exit + b_ex*vec_memory + np.random.randn(2)*sig_le*0
            
#         ### actions ###
#         if state[tt]==1:
#             vs[tt,:] = c_le*vs[tt-1,:] + (1-c_le)*m_exit + np.random.randn(2)*sig_re**2 + np.array([0,wind_bias])
#         elif state[tt]==0:
#             vs[tt,:] = c_re*vs[tt-1,:] + (1-c_re)*m_entry + np.random.randn(2)*sig_le**2 + np.array([0,wind_bias])
        
#         ### track location
#         xs[tt,:] = xs[tt-1,:] + vs[tt,:]


# # %%
# plt.figure()
# plt.bar(['out','inbound'], [np.mean(ex_angs), np.mean(en_angs)], yerr=[np.std(ex_angs), np.std(en_angs)])
# plt.ylabel('angle (degs)', fontsize=20)