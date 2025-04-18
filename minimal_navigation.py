# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:06:26 2024

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from autograd import grad
import autograd.numpy as np
from scipy.linalg import hankel

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %% minimal model for navigation (run-and-tumble); then test the inference
### the idea is to explore feedback and environmental drive
### check if the conditions can be inferred with filters

# %% parameter setup
dt = 1   # characteristic time scale
t_M = 10  # memory
N = 1  # receptor gain
H = 1  # motor gain
F0 = 0.5  # adapted internal state
v0 = 1.  # run speed

target = np.array([100,100])  ### location of source
tau_env = 10   ### fluctuation time scale of the environment
C0 = 100
sigC = 50

lt = 10000   #max time lapse
eps = 3  # criteria to reach sourc

# %% kinematics
def r_F(F):
    return 1/(1+np.exp(-H*F))+.0  # run rate

def tumble(r):
    p = np.random.rand()
    if r>p:
        angle = np.random.randn()*.1  # run
        tumb = 0
    else:
        angle = (np.random.rand()*2-1)*np.pi  # tumble
        tumb = 1
    return angle, tumb

def theta2state(pos, theta):
    """
    take continuous angle dth and put back to discrete states
    """
    dv = v0 + np.random.randn()*0.  # draw speed
    vec = np.array([np.cos(theta)*dv , np.sin(theta)*dv] )
    pos = pos + vec #np.array([dx, dy])
    return pos, vec
    
# %% factorized environment
def temporal_ou_process(lt, tau, A, dt=0.01):
    n_steps = lt*1  # Number of steps
    x = np.ones(n_steps)  # OU process values
    
    # Variance of the noise
    sigma = A * np.sqrt(2 / tau)
    
    for i in range(1, n_steps):
        # Update step for OU process
        x[i] = x[i-1] - ((x[i-1]-1) / tau) * dt + sigma * np.sqrt(dt) * np.random.randn()
    
    return x

# Parameters
tau = 10.0      # Correlation time
A = 0.5        # Amplitude
# Simulate and plot the process
fluctuation_t = temporal_ou_process(lt, tau, A)#*.1 + 1
plt.figure()
plt.plot(fluctuation_t)

def dist2source(x):
    return np.sum( (x-target)**2 )**0.5
    
def env_space(x, tt=-1):
    C = np.exp(-np.sum((x - target)**2)/sigC**2)*C0 + np.random.randn()*0.
    if tt==-1:
        return C
    else:
        return C*fluctuation_t[tt]

def env_space_xy(x,y):
    # C = np.exp(-np.sum((x - target)**2)/sigC**2)*C0
    C = -((x - target[0])**2 / (sigC**2) + (y - target[1])**2 / (sigC**2))
    return C

def env_time(x):
    return 1   ### let's not have environmental dynamics for now

def gaussian(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1):
    return np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))

gradient = grad(env_space, argnum=0)
def grad_env(x, u_x, tt=-1):
    grad_x = gradient(x, tt)
    if np.linalg.norm(grad_x)==0:
        percieved = C0
    else:
        percieved = np.dot(grad_x/np.linalg.norm(grad_x), u_x/(1))  # dot product between motion and local gradient
    # print(np.linalg.norm(grad_x))
    return percieved

# # Example point
# inputs = np.array([1.0, 1.5])
# grad_values = gradient(inputs)

# %% setup timee
xys = []
cs = []
vecs = []
Fs = []
tumbles = []
pos_t = np.random.randn(2)  # random init location
vec_t = np.random.randn(2)  # random init direction
df_dt = np.random.randn()     # init internal state
theta = 0
tt = 0

while tt<lt and dist2source(pos_t)>eps:
    ### compute impulse
    d_phi = grad_env(pos_t, vec_t, tt)  #np.random.randn()*1 #
    ### internal dynamics
    df_dt = df_dt + dt*(-1/t_M*(df_dt - F0) + d_phi)
    ### draw actions
    r_t = r_F(df_dt)*dt
    dth,tumb_t = tumble(r_t)
    ### make movements
    theta = theta + dth
    new_pos, new_vec = theta2state(pos_t, theta)
    ### record
    xys.append(pos_t)
    # cs.append(env_space(pos_t))
    # cs.append(np.log(env_space(pos_t)))
    cs.append(d_phi)
    vecs.append(vec_t)
    Fs.append(df_dt)
    tumbles.append(tumb_t)
    ### update
    pos_t, vec_t = new_pos*1, new_vec*1
    tt += 1

def gen_tracks():
    lt = 10000   #max time lapse
    eps = 1  # criteria to reach source
    xys = []
    cs = []
    vecs = []
    Fs = []
    tumbles = []
    pos_t = np.random.randn(2)  # random init location
    vec_t = np.random.randn(2)  # random init direction
    df_dt = np.random.randn()     # init internal state
    theta = 0
    tt = 0

    while tt<lt and dist2source(pos_t)>eps:
        ### compute impulse
        d_phi = grad_env(pos_t, vec_t)  #np.random.randn()*1 #
        ### internal dynamics
        df_dt = df_dt + dt*(-1/t_M*(df_dt - F0) + d_phi)
        ### draw actions
        r_t = r_F(df_dt)*dt
        dth,tumb_t = tumble(r_t)
        ### make movements
        theta = theta + dth
        new_pos, new_vec = theta2state(pos_t, theta)
        ### record
        xys.append(pos_t)
        ### choose input
        # cs.append(env_space(pos_t))
        # cs.append(np.log(env_space(pos_t)))
        cs.append(d_phi)
        ####
        vecs.append(vec_t)
        Fs.append(df_dt)
        tumbles.append(tumb_t)
        ### update
        pos_t, vec_t = new_pos*1, new_vec*1
        tt += 1
    ### vectorize
    vec_xy = np.array(xys)
    vec_cs = np.array(cs)
    vec_Fs = np.array(Fs)
    vec_tumb = np.array(tumbles)
    return vec_xy, vec_cs, vec_Fs, vec_tumb

# %% vectorize lists
vec_xy = np.array(xys)
vec_cs = np.array(cs)
vec_Fs = np.array(Fs)
vec_tumb = np.array(tumbles)

# %% plotting
min_x, max_x = np.min(vec_xy[:,0]), np.max(vec_xy[:,0])
min_y, max_y = np.min(vec_xy[:,1]), np.max(vec_xy[:,1])

x = np.linspace(min_x, max_x, 150)
y = np.linspace(min_y, max_y, 150)
x, y = np.meshgrid(x, y)
c_env = env_space_xy(x,y)

plt.figure()
plt.imshow(c_env, extent=[x[0,0], x[-1,-1], y[0,0], y[-1,-1]], origin="lower")
plt.plot(vec_xy[:,0], vec_xy[:,1])
plt.plot(vec_xy[0,0], vec_xy[0,1],'go')
plt.plot(vec_xy[-1,0], vec_xy[-1,1],'r*')

# %% now test inference!
###############################################################################
# %% setup design matrix 
lags = 20
X_s = hankel(vec_cs[:lags], vec_cs[lags-1:]).T  ### time by lag
vec_causal = vec_tumb[:-1]
vec_causal = np.insert(vec_causal, 0, 0)
X_a = hankel(vec_causal[:lags], vec_causal[lags-1:]).T  ### time by lag
X = np.concatenate((X_s,X_a), 1)
X = np.concatenate((X, np.ones((X_s.shape[0],1))*1 ), 1)
y = vec_tumb[:-lags+1]*1  # actions

# %% likelihood function and optimization
def neg_ll(theta,   y, X):
    p = X @ theta
    eps = 1e-10  # Small value to prevent log(0)
    p = np.clip(p, eps, 1 - eps)
    # Compute the log-likelihood
    log_likelihood = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    nll = -log_likelihood
    return nll

def neg_ll_parametric(theta,   y, X):
    a_s, tau_s, a_h, tau_h, b = theta
    xl = (X.shape[1]-1)//2  # vector length, removing the intersect
    tvec = np.arange(0, xl)
    K_s = a_s*np.exp(-tvec*(tau_s))
    K_h = a_h*np.exp(-tvec*(tau_h))
    beta = np.concatenate([K_s, K_h, [b]])
    p = X @ beta
    eps = 1e-10  # Small value to prevent log(0)
    p = np.clip(p, eps, 1 - eps)
    # Compute the log-likelihood
    log_likelihood = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    nll = -log_likelihood
    return nll

def grad_neg_log_likelihood(theta, y, X):
    # a_s, tau_s, a_h, tau_h, b = theta
    # xl = (X.shape[1]-1)//2  # vector length, removing the intersect
    # tvec = np.arange(0, xl)
    # K_s = a_s*np.exp(-tvec*(tau_s))
    # K_h = a_h*np.exp(-tvec*(tau_h))
    # beta = np.concatenate([K_s, K_h, [b]])
    linear_pred = X @ theta
    p = 1 / (1 + np.exp(-linear_pred))  # Sigmoid function
    gradient = X.T @ (p - y)  # Gradient
    return gradient

# %%
# %% test gradient
beta_init = np.zeros(lags*2 + 1)
# Minimize the negative log-likelihood with gradient
result = sp.optimize.minimize(
    neg_ll, 
    beta_init, 
    args=(y, X), 
    method='BFGS', 
    jac=grad_neg_log_likelihood
)

# %% IDEAS.... Gradient or basis function...
###############################################################################
###############################################################################

# %% optimization (non-parametric)
n_params = lags*2 + 1  ### two kernels and one offset
theta0 = np.ones(n_params)*0
res = sp.optimize.minimize( neg_ll, theta0, args=(y, X), options={'disp':True,'gtol':1e-9})
                            # , method='Nelder-Mead',options={'disp':True,'maxiter':3000})#
                            # method="BFGS") #, tol=1e-3, options={'disp':True,'gtol':1e-2})#
                            
# %% plot sensory and history kernels
betas = res.x
K_s, K_a, base = betas[:lags], betas[lags:lags*2], betas[-1]
plt.figure(figsize=(12, 4))
plt.subplot(121); plt.plot(K_s)
plt.subplot(122); plt.plot(K_a)

# %% optimization (non-parametric)
n_params = 2*2 + 1  ### two kernels and one offset
theta0 = np.ones(n_params)*.0
bounds = ([-100,100],[0.1,100],[-100,100],[0.1,100], [-100, 100])
res = sp.optimize.minimize( neg_ll_parametric, theta0, args=(y, X), options={'disp':True,'gtol':1e-9})#,
                           # bounds = bounds, method='SLSQP')

# %% plot for parametric inference
betas = res.x
a_s, tau_s, a_h, tau_h, b = betas
tvec = np.arange(0,lags)
K_s = a_s*np.exp(-tvec*(tau_s))
K_a = a_h*np.exp(-tvec*(tau_h))
plt.figure(figsize=(12, 4))
plt.subplot(121); plt.plot(K_s)
plt.subplot(122); plt.plot(K_a)

# %% group analysis
###############################################################################
# %% make group dictionary
reps = 50
tracks_dic = {}
for rr in range(reps):
    vec_xy, vec_cs, vec_Fs, vec_tumb = gen_tracks()
    if len(vec_xy)<lt:  ### if reached goal...
        tracks_dic[rr] = {'xy':vec_xy, 'cs':vec_cs, 'Fs':vec_Fs, 'tumb': vec_tumb}
    print(rr)
# %% group nll
def group_nll(theta, tracks, lamb, nll_function):
    gnll = 0
    for ii in range(len(tracks)):
        tracki = tracks[ii]
        _, vec_cs, _, vec_tumb = tracki['xy'], tracki['cs'], tracki['Fs'], tracki['tumb']
        # lags = 10
        X_s = hankel(vec_cs[:lags], vec_cs[lags-1:]).T  ### time by lag
        vec_causal = vec_tumb[:-1]
        vec_causal = np.insert(vec_causal, 0, 0)*1
        X_a = hankel(vec_causal[:lags], vec_causal[lags-1:]).T  ### time by lag
        X = np.concatenate((X_s,X_a), 1)
        X = np.concatenate((X, np.ones((X_s.shape[0],1))*1  ), 1)
        y = vec_tumb[:-lags+1]*1  # actions
        
        gnll += nll_function(theta, y, X)
    return gnll + lamb*np.sum(np.diff(theta)**2)

# %% optimize for group
n_params = lags*2 + 1  ### two kernels and one offset
n_params = 2*2 + 1  ### if it is for non-parametric
lamb = 0
theta0 = np.ones(n_params)*5
# theta0 = np.array([-1, 10, 1, 10, 0])
bounds = ([-100,100],[0.1,100],[-100,100],[0.1,100], [-100, 100])
### non-parametric
# res = sp.optimize.minimize( group_nll, theta0, args=(tracks_dic, lamb, neg_ll), options={'disp':True,'gtol':1e-9})
### parametric 
res = sp.optimize.minimize( group_nll, theta0, args=(tracks_dic, lamb, neg_ll_parametric), options={'disp':True,'gtol':1e-9})#,
                            # bounds = bounds, method='SLSQP')

# %% plot for parametric inference
betas = res.x
a_s, tau_s, a_h, tau_h, b = betas
tvec = np.arange(0,lags)
K_s = a_s*np.exp(-tvec/(tau_s))
K_a = a_h*np.exp(-tvec/(tau_h))
plt.figure(figsize=(12, 4))
plt.subplot(121); plt.plot(K_s)
plt.subplot(122); plt.plot(K_a)

# %% plot sensory and history kernels
betas = res.x
K_s, K_a, base = betas[:lags], betas[lags:lags*2], betas[-1]
plt.figure(figsize=(12, 4))
plt.subplot(121); plt.plot(K_s); plt.title('K_s'); plt.xlabel('time lag')
plt.subplot(122); plt.plot(K_a); plt.title('K_a'); plt.xlabel('time lag')

# %% test inference matching
###############################################################################
# %% scanning

Tms = np.array([1,2,4,8])  # scanning memory
reps_sims = 5  # repeated inference
reps_tracks = 50  # repeated tracks
n_params = 2*2 + 1  ### if it is for non-parametric
ks_tau = np.zeros((reps_sims, len(Tms), n_params))  # inferred time scale
bounds = ([-100,100],[0,100],[-100,100],[0,100], [-100, 100])

for ss in range(reps_sims):
    print('repeats:', ss)
    for mm in range(len(Tms)):
        print('memory', Tms[mm])
        t_M = Tms[mm]
        ### group tracks for inference
        tracks_dic = {}
        for rr in range(reps):
            vec_xy, vec_cs, vec_Fs, vec_tumb = gen_tracks()
            if len(vec_xy)<lt:  ### if reached goal...
                tracks_dic[rr] = {'xy':vec_xy, 'cs':vec_cs, 'Fs':vec_Fs, 'tumb': vec_tumb}
            # print(rr)
        
        ### doing group inference
        lamb = 0
        theta0 = np.ones(n_params)*5
        res = sp.optimize.minimize( group_nll, theta0, args=(tracks_dic, lamb, neg_ll_parametric))#, 
                                   # options={'disp':True,'gtol':1e-9}, bounds = bounds, method='SLSQP')
        ks_tau[ss,mm, :] = res.x
            

# %% plotting inferred parameters versus true time scales
plt.figure()
plt.plot(Tms, -(ks_tau[:,:,1].T),'.')
plt.xlabel(r'true $\tau_M$')
plt.ylabel(r'inferred $\hat{\tau}$')
plt.ylim([0,10])
