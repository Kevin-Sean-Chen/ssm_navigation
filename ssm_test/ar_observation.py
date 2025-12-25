# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 22:43:03 2024

@author: kevin
"""

import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.special import gammaln, digamma, logsumexp
from autograd.scipy.special import logsumexp

from ssm.util import random_rotation, ensure_args_are_lists, \
    logistic, logit, one_hot
from ssm.regression import fit_linear_regression, generalized_newton_studentst_dof, fit_scalar_glm
from ssm.preprocessing import interpolate_data
from ssm.cstats import robust_ar_statistics
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
import ssm.stats as stats

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %%
class Observations(object):
    # K = number of discrete states
    # D = number of observed dimensions
    # M = exogenous input dimensions (the inputs modulate the probability of discrete state transitions via a multiclass logistic regression)

    def __init__(self, K, D, M):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, init_method="random"):
        Ts = [data.shape[0] for data in datas]

        # Get initial discrete states
        if init_method.lower() == 'kmeans':
            # KMeans clustering
            from sklearn.cluster import KMeans
            km = KMeans(self.K)
            km.fit(np.vstack(datas))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])

        elif init_method.lower() =='random':
            # Random assignment
            zs = [npr.choice(self.K, size=T) for T in Ts]

        else:
            raise Exception('Not an accepted initialization type: {}'.format(init_method))

        # Make a one-hot encoding of z and treat it as HMM expectations
        Ezs = [one_hot(z, self.K) for z in zs]
        expectations = [(Ez, None, None) for Ez in Ezs]

        # Set the variances all at once to use the setter
        self.m_step(expectations, datas, inputs, masks, tags)

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag):
        raise NotImplementedError

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        raise NotImplementedError

    def m_step(self, expectations, datas, inputs, masks, tags,
               optimizer="bfgs", **kwargs):
        """
        If M-step cannot be done in closed form for the observations, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, _, _) \
                in zip(datas, inputs, masks, tags, expectations):
                lls = self.log_likelihoods(data, input, mask, tag)
                elbo += np.sum(expected_states * lls)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(_objective, self.params, **kwargs)

    def smooth(self, expectations, data, input, tag):
        raise NotImplementedError

    def neg_hessian_expected_log_dynamics_prob(self, Ez, data, input, mask, tag=None):
        raise NotImplementedError
        
# %%
class GLM_PoissonObservations(Observations):

    def __init__(self, K, D, M):
        super(GLM_PoissonObservations, self).__init__(K, D, M)
        self.Wk = npr.randn(K, D, M)
        self.K, self.D, self.M = K, D, M   #K-state, D-dim output, M-dim input

    @property
    def params(self):
        return self.Wk

    @params.setter
    def params(self, value):
        self.Wk = value

    def permute(self, perm):
        self.Wk = self.Wk[perm]
        
#    def calculate_logPoisson(self, input):
#        # np.sum(Y * np.log(f(X@w)) - f(X@w)*dt - sp.special.gammaln(Y+1) + Y*np.log(dt))
#        Wk_trans = np.transpose(self.Wk, (1,0,2))

    def log_likelihoods(self, data, input, mask, tag):
        lambdas = np.exp((self.Wk @ input.T) * 1.0)#.astype(float)  #exponential Poisson nonlinearity
        assert self.D == 1, "InputDrivenObservations written for D = 1!"
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        ############################################ fix log-ll here ##################################
        return stats.poisson_logpdf(data[None,:,:], np.transpose(lambdas,(0,2,1)), mask=mask[None,:,:]).T
#        print(lambdas.shape)
#        print(data.shape)
#        return stats.poisson_logpdf(data[:,None,:], lambdas, mask=mask[:,None,:])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        assert self.D == 1, "InputDrivenObservations written for D = 1!"
        if input.ndim == 1 and input.shape == (self.M,): # if input is vector of size self.M (one time point), expand dims to be (1, M)
            input = np.expand_dims(input, axis=0)
        lambdas = np.exp(self.Wk @ input.T)
        return npr.poisson(lambdas[z])   #y = Poisson(exp(W @ x))

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, optimizer="bfgs", **kwargs)
##        x = np.concatenate(datas)
##        weights = np.concatenate([Ez for Ez, _, _ in expectations])
##        for k in range(self.K):
##            self.log_lambdas[k] = np.log(np.average(x, axis=0, weights=weights[:,k]) + 1e-16)
#        
#        X = np.concatenate(inputs)
#        y = np.concatenate(datas)
#        expt = np.concatenate([Ez for Ez, _, _ in expectations]) #expectation=gamma,xi,ll
##        print(X.shape)
##        print(y.shape)
##        print(expt.shape)
#        for k in range(self.K):
#            weigthed_y = y[:,0]*(expt[:,k]) /sum(expt[:,k])  ## weighted here
##            print(weigthed_y.shape)
#            what, bhat = fit_scalar_glm(X, weigthed_y, model="poisson", mean_function="exp")
#            self.Wk[k] = what


    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
#        return expectations.dot(np.exp(self.log_lambdas))
        raise NotImplementedError

# %%
class AR_Observations(Observations):
    def __init__(self, K, D, M, L):
        """
        K = number of discrete states
        D = number of observed dimensions
        M = exogenous input dimensions
        L = delay
        
        E[x_t | x_{t-1}, z_t=k, u_t]
        = \sum_{l=1}^{L} A_k^{(l)} x_{t-l} + b_k + V_k u_t.
        
        """
        super(AR_Observations, self).__init__(K, D, M, L)
        self.As = npr.randn(K, D, L)  # per state, map delay L to D dim signal
        # self.Bs = npr.randn(K, D, M, L)  # per state, map input to output, with delay L
        self.bs = npr.randn(K, D)  # per state, D dim baseline
        self.log_kappas = np.log(-1*npr.uniform(low=-1, high=0, size=(K, D)))

    @property
    def params(self):
        return self.mus, self.log_kappas  #make mus a vector here for GLM-like emission

    @params.setter
    def params(self, value):
        self.mus, self.log_kappas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self.log_kappas = self.log_kappas[perm]

    def log_likelihoods(self, data, input, mask, tag):
        mus, kappas = self.mus, np.exp(self.log_kappas)
        ###
        driven_angle = self.mus @ input.T
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
#        return stats.vonmises_logpdf(data[:, None, :], driven_angle, kappas, mask=mask[:, None, :])
        kappa_t = np.repeat(kappas[:,:,None], input.T.shape[-1], axis=2)  #time-independent
        return stats.vonmises_logpdf(data[None,:,:], np.transpose(driven_angle,(0,2,1)), \
                                     np.transpose(kappa_t,(0,2,1)), mask=mask[None,:,:]).T


    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, kappas = self.D, self.mus, np.exp(self.log_kappas)
        assert self.D == 1, "InputDrivenObservations written for D = 1!"
        if input.ndim == 1 and input.shape == (self.M,): # if input is vector of size self.M (one time point), expand dims to be (1, M)
            input = np.expand_dims(input, axis=0)
        ###
        driven_angle = mus @ input.T
        kappas_t = np.repeat(kappas[:,:,None], input.T.shape[-1], axis=2)  #time-independent
        return npr.vonmises(driven_angle[z], kappas_t[z] )#, D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, optimizer="bfgs", **kwargs)

    def smooth(self, expectations, data, input, tag):
        mus = self.mus
        return expectations.dot(mus)


# %% testing
###############################################################################
# %%
#import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ssm
from ssm.util import one_hot, find_permutation

# %%
num_states = 2        # number of discrete states           K
obs_dim = 1           # number of observed dimensions       D
num_categories = 2    # number of categories for output     C  #should not be used here for continous output
input_dim = 3         # input dimensions                    M

# Make a GLM-HMM
true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="categorical", 
                   observation_kwargs=dict(C=num_categories), transitions="inputdriven")  #fix to be both driven!

# %%  replace from here for now
true_glmhmm.observations = GLM_PoissonObservations(num_states, obs_dim, input_dim)
# true_glmhmm.observations = InputVonMisesObservations(num_states, obs_dim, input_dim)
print(true_glmhmm.transitions.Ws.shape)
# print(true_glmhmm.observations.mus.shape)

# %%
#input sequence
true_glmhmm.transitions.Ws *= 3
num_sess = 10 # number of example sessions
num_trials_per_sess = 1000 # number of trials in a session
inpts = np.ones((num_sess, num_trials_per_sess, input_dim)) # initialize inpts array
stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]

### discrete input
inpts[:,:,0] = np.random.choice(stim_vals, (num_sess, num_trials_per_sess)) # generate random sequence of stimuli
inpts[:,:,1] = np.random.choice(stim_vals, (num_sess, num_trials_per_sess)) 
### random input
#inpts = np.random.randn(num_sess, num_trials_per_sess, input_dim)
### noisy sine input
inpts = np.sin(2*np.pi*np.arange(num_trials_per_sess)/50)[:,None]*.5+.5*npr.randn(num_trials_per_sess,input_dim)
inpts = np.repeat(inpts[None,:,:], num_sess, axis=0)

inpts = list(inpts) #convert inpts to correct format

# %%
# Generate a sequence of latents and choices for each session
true_latents, true_choices = [], []
for sess in range(num_sess):
    true_z, true_y = true_glmhmm.sample(num_trials_per_sess, input=inpts[sess])  #changed hmm.py line206!
    true_latents.append(true_z)
    true_choices.append(true_y)
    
# %%
# Calculate true loglikelihood
true_ll = true_glmhmm.log_probability(true_choices, inputs=inpts) 
print("true ll = " + str(true_ll))

# %% fix this~~~ for inference
############################################################################### "input_driven_obs"
new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="categorical", 
                   observation_kwargs=dict(C=num_categories), transitions="inputdriven")
#new_glmhmm.observations = GLM_PoissonObservations(num_states, obs_dim, input_dim) ##obs:"input_driven"
new_glmhmm.observations = InputVonMisesObservations(num_states, obs_dim, input_dim)

N_iters = 100 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
fit_ll = new_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=N_iters)#, tolerance=10**-4)

# %%
new_glmhmm.permute(find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_choices[0], input=inpts[0])))

# %%
true_obs_ws = true_glmhmm.observations.mus #Wk
inferred_obs_ws = new_glmhmm.observations.mus

cols = ['r', 'g', 'b']
plt.figure()
for ii in range(num_states):
    plt.plot(true_obs_ws[ii][0],linewidth=5, label='ture', color=cols[ii])
    plt.plot(inferred_obs_ws[ii][0],'--',linewidth=5,label='inferred',color=cols[ii])
plt.legend(fontsize=40)
plt.title('Emission weights', fontsize=40)

# %%
true_trans_ws = true_glmhmm.transitions.Ws
inferred_trans_ws = new_glmhmm.transitions.Ws

plt.figure()
for ii in range(num_states):
    plt.plot(true_trans_ws[ii],linewidth=5, label='ture', color=cols[ii])
    plt.plot(inferred_trans_ws[ii],'--',linewidth=5,label='inferred', color=cols[ii])
#plt.legend(fontsize=40)
plt.title('Transition weights', fontsize=40)