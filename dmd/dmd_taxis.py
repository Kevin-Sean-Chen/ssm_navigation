# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:38:47 2025

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

# %% minimal model for navigation (run-and-tumble); then test mode discovery methods
### try it with Markov model
### try it with DMD with control
### later try with RNN for dynamic modes

# %% parameter setup
dt = 1   # characteristic time scale
t_M = 2  # memory
N = 1  # receptor gain
H = 1  # motor gain
F0 = 0.  # adapted internal state
v0 = 1.  # run speed
vn = .1

target = np.array([100,100])  ### location of source
tau_env = 10   ### fluctuation time scale of the environment
C0 = 100
sigC = 50

lt = 10000   #max time lapse
eps = 5  # criteria to reach sourc

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
    dv = v0 + np.random.randn()*vn  # draw speed
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
    C = np.exp(-np.sum((x - target)**2)/sigC**2)*C0 + np.random.randn()*0.1
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
    d_phi = grad_env(pos_t, vec_t, tt)
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
    cs.append(env_space(pos_t))
    # cs.append(np.log(env_space(pos_t)))
    # cs.append(d_phi)
    # cs.append(env_space(new_pos) - env_space(pos_t))
    vecs.append(vec_t)
    Fs.append(df_dt)
    tumbles.append(tumb_t)
    ### update
    pos_t, vec_t = new_pos*1, new_vec*1
    tt += 1

def gen_tracks():
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
        d_phi = grad_env(pos_t, vec_t)
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
        # cs.append(d_phi)
        cs.append(env_space(new_pos) - env_space(pos_t))
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
    vec_vxy = np.array(vecs)
    vec_tumb = np.array(tumbles)
    return vec_xy, vec_cs, vec_Fs, vec_tumb, vec_vxy

# %% measure track statistics
reps = 30
tracks_dic = {}
behavior = []
stimuli = []
for rr in range(reps):
    vec_xy, vec_cs, vec_Fs, vec_tumb, vec_vxy = gen_tracks()
    if len(vec_xy)<lt:  ### if reached goal...
        tracks_dic[rr] = {'xy':vec_xy, 'cs':vec_cs, 'Fs':vec_Fs, 'tumb': vec_tumb, 'vxy': vec_vxy}
        behavior.append(vec_vxy)
        stimuli.append(vec_cs)
    print(rr)
behavior = np.concatenate((behavior),0)
stimuli = np.concatenate((stimuli))

# %% build delay matrix
### delay embedding
def delay_embed(Xt, lags):
    T,d = Xt.shape
    X = []
    for dd in range(d):
        X.append((hankel(Xt[:lags,dd], Xt[lags-1:,dd]).T))
    X = np.concatenate(X, axis=1)
    return X

# %% scan delays
def err_given_delay(behavior, stimuli, lags):
    tau = 1
    Xb = delay_embed(behavior[:,[0,1]], lags)
    Xo = delay_embed(stimuli[:,None], lags)[:-tau-lags:] ## odor
    Xbp = Xb[:-tau-lags]
    Ybf = Xb[tau+lags:,:]
    Omega = np.concatenate((Xbp, Xo),1)
    uu,ss,vv = np.linalg.svd(Omega.T, full_matrices=False)
    ux,sx,vx = np.linalg.svd(Ybf.T, full_matrices=False)
    u1, u2 = uu[:lags*2,:], uu[lags*2:,:]
    A_matrix = ux.T.conj() @ Ybf.T @ vv.T @ np.diag(np.reciprocal(ss)) @ u1.T.conj() @ ux
    B_matrix = ux.T.conj() @ Ybf.T @ vv.T @ np.diag(np.reciprocal(ss)) @ u2.T.conj()
    
    X_pred = ux @ (A_matrix @ (ux.T @ Xbp.T)*1 + B_matrix @ (Xo.T)*1)
    err = np.linalg.norm(X_pred[lags,tau:] - Ybf[:-tau,lags])
    return err
    
scan_d = np.arange(5,200,20)
errs = np.zeros(len(scan_d))
for ii in range(len(scan_d)):
    print(ii)
    errs[ii] = err_given_delay(behavior, stimuli, scan_d[ii])

plt.figure()
plt.plot(scan_d, errs,'-o')
    
# %% # DMDc
lags = 50
tau = 1
Xb = delay_embed(behavior[:,[0,1]], lags)
Xo = delay_embed(stimuli[:,None], lags)[:-tau-lags:] ## odor
Xbp = Xb[:-tau-lags]
Ybf = Xb[tau+lags:,:]

# %%
Omega = np.concatenate((Xbp, Xo),1)
uu,ss,vv = np.linalg.svd(Omega.T, full_matrices=False)
ux,sx,vx = np.linalg.svd(Ybf.T, full_matrices=False)
u1, u2 = uu[:lags*2,:], uu[lags*2:,:]

# %% DMDc solution
A_matrix = ux.T.conj() @ Ybf.T @ vv.T @ np.diag(np.reciprocal(ss)) @ u1.T.conj() @ ux
B_matrix = ux.T.conj() @ Ybf.T @ vv.T @ np.diag(np.reciprocal(ss)) @ u2.T.conj()

# %% analyze matrix
A_real = ux @ A_matrix
B_real = ux @ B_matrix

plt.figure()
plt.subplot(121); plt.plot(B_real[:lags,:]);#plt.plot(uu[:lags,:5])
plt.subplot(122); plt.plot(B_real[-lags:,:]);#plt.plot(uu[-lags:,:5])

plt.figure()
plt.subplot(121); plt.plot(A_real[:lags, :]);#plt.plot(uu[:lags,:5])
plt.subplot(122); plt.plot(A_real[-lags:, :]);#plt.plot(uu[-lags:,:5])

# %% making predictions
X_pred = ux @ (A_matrix @ (ux.T @ Xbp.T) + B_matrix @ (Xo.T))

T_wind = 500
plt.figure()
plt.plot(X_pred[lags,tau:T_wind+tau], label='data')
plt.plot(Ybf[:T_wind,lags],'--', label='prediction')
plt.legend(); plt.xlabel('time steps'); plt.ylabel('v_x')

# %% show modes
top_m = 15
uu,ss,vv = np.linalg.svd(Xb, full_matrices=False)
uuo,sso,vvo = np.linalg.svd(Xo, full_matrices=False)
mode = vv[:top_m,:]
modeo = vvo[:top_m,:]
plt.figure()
plt.imshow(mode, aspect='auto')
plt.figure()
plt.imshow(modeo, aspect='auto')

# %%
plt.figure(figsize=(14,6))
for ii in range(5):
    # modei = (mode_tracks[ii,4,:].squeeze())
    modei = mode[ii,:]
    # modei = Gamm[:,ii]
    vxyi = -modei.reshape(2,lags)
    vxyi -= vxyi[:,0][:,None]*1
    plt.subplot(121)
    plt.plot(vxyi[0,:], vxyi[1,:], 'o')  ### try density!!
    plt.xlabel('vx')
    plt.ylabel('vy')
    plt.subplot(122)
    plt.plot(modeo[ii,:])
    plt.xlabel('lags')
    plt.ylabel('weight')
plt.title('top modes')  

# %% plot tracks
plt.figure()
x_range = np.linspace(-10, 140, 150)  # 100 points along the x-axis
y_range = np.linspace(-10, 140, 150)  # 100 points along the y-axis

# Create meshgrid for the grid points
X, Y = np.meshgrid(x_range, y_range)

# Compute Z values for each grid point
Z = env_space_xy(X, Y)
plt.imshow(Z, origin='lower')
for ii in range(reps):
    temp = tracks_dic[ii]['xy']
    plt.plot(temp[:,0], temp[:,1])
    
# %% projection analysis for controlled modes
ui,si,vi = np.linalg.svd(A_real, full_matrices=False)
# idx = ui.argsort()[::-1]   
# ui = ui[idx]
# vi = vi[:,idx]
projs = np.zeros(len(ui))
ub,sb,vb = np.linalg.svd(B_real, full_matrices=False)

plt.figure()
for ii in range(len(ui)):
    plt.plot(ii, ui[:,ii] @ ub[:,0],'o')
plt.xlabel('sorted'); plt.ylabel('projection (input on modes)')

# %% plot in state space
plt.figure()
for ii in range(4):
    vec_action = ui[:,ii].reshape(2,-1)
    plt.plot(vec_action[0,:]-vec_action[0,-1], vec_action[1,:]-vec_action[1,-1],'-o')
plt.title('action mode')

plt.figure()
for ii in range(4):
    vec_drive = ub[:,ii].reshape(2,-1)
    plt.plot(vec_drive[0,:], vec_drive[1,:],'-o')
plt.title('driving modes')

# %%
###############################################################################
# %% can we chemotaxis with DMDc??
#### try this!!
def DMD_taxis(A, B):
    A, B = A_real*1, B_real*1
    lags = B.shape[1]
    eps = 5  # criteria to reach source
    xys = []
    cs = []
    vecs = []
    vec_behavior_t = np.random.randn(lags*2)
    vec_stimuli_t = np.random.randn(lags)
    pos_t = np.random.randn(2)*0  # random init location
    tt = 0
    
    while tt<lt and dist2source(pos_t)>eps:
        ### use DMDc for next step
        # vec_behavior_next = (A @ vec_behavior_t + B @ vec_stimuli_t)*1
        # vec_behavior_next = (ui @ vec_behavior_t + ub @ vec_stimuli_t)*1
        vec_behavior_next = ux @ (A @ (ux.T @ vec_behavior_t) + B @ (vec_stimuli_t))
        
        ### update next value
        vxy = vec_behavior_t.reshape(2,-1)
        vxy = np.roll(vxy, shift=1, axis=1)  # Shift elements left
        vxy_next = vec_behavior_next.reshape(2,-1)[:,0]*1
        vxy[:, -0] = vxy_next
        vec_behavior_t = vxy.reshape(-1)
        
        ### update location
        new_pos = pos_t + vxy_next*1
        
        ### stim signal
        new_stim = env_space(new_pos) - env_space(pos_t)
        vec_stimuli_t = np.roll(vec_stimuli_t, shift=1)  # Shift elements left
        vec_stimuli_t[-0] = new_stim
    
        ### record
        xys.append(pos_t)
        ### choose input
        # cs.append(env_space(pos_t))
        # cs.append(np.log(env_space(pos_t)))
        # cs.append(d_phi)
        cs.append(env_space(new_pos) - env_space(pos_t))
        ####
        vecs.append(vxy_next)
        ### update
        pos_t = new_pos*1
        tt += 1
    ### vectorize
    vec_xy = np.array(xys)
    vec_cs = np.array(cs)
    vec_vxy = np.array(vecs)
    return vec_xy, vec_cs, vec_vxy

plt.figure()
# plt.imshow(Z, origin='lower')
# plt.plot(vec_xy[:,0], vec_xy[:,1])
reps = 10
for rr in range(reps):
    vec_xy, vec_cs, vec_vxy = DMD_taxis(A_matrix, B_matrix)
    plt.plot(vec_xy[:,0], vec_xy[:,1])
    print(rr)