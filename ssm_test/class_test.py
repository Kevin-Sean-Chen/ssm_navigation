# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:35:18 2024

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
from autograd.scipy.special import logsumexp
import scipy as sp
from scipy.special import gammaln

import ssm
import ssm.stats as stats
from ssm.transitions import StickyTransitions, Transitions
from ssm.observations import Observations, GaussianObservations
import autograd.numpy as np

# %% writing classes for driven state transition and mixture emission
###############################################################################
# %% input driven across states
class InputDrivenTransitions_k(StickyTransitions):
    """
    Hidden Markov Model whose transition probabilities are
    determined by a generalized linear model applied to the
    exogenous input.
    """
    def __init__(self, K, D, M, alpha=1, kappa=0, l2_penalty=0.0):
        super(InputDrivenTransitions_k, self).__init__(K, D, M=M, alpha=alpha, kappa=kappa)

        # Parameters linking input to state distribution
        self.Ws = npr.randn(K, K, M)

        # Regularization of Ws
        self.l2_penalty = l2_penalty

    @property
    def params(self):
        return self.log_Ps, self.Ws

    @params.setter
    def params(self, value):
        self.log_Ps, self.Ws = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.Ws = self.Ws[perm]

    def log_prior(self):
        lp = super(InputDrivenTransitions_k, self).log_prior()
        lp = lp + np.sum(-0.5 * self.l2_penalty * self.Ws**2)
        return lp

    def log_transition_matrices(self, data, input, mask, tag):
        T = data.shape[0]
        assert input.shape[0] == T
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))

# %% mixture emsision

class MixtureObservations(Observations):
    """
    test with emissions that are two independent statistical models
    ex: Gamma for speed and VonMesis for angle
    """
    def __init__(self, K, D, M=0, theta_init=None):
        super(MixtureObservations, self).__init__(K, D, M)
        ### custom parameters
        _D = 1 ## for one-D case now
        num_params_per_state = 4  # mu, sig, alpha, beta
        if theta_init is not None:
            self.theta_k = theta_init
        else:
            self.theta_k = npr.rand(K, num_params_per_state)
        ### for Gaussian angle change
        # self.mus = npr.randn(K, num_params_per_state)
        # self._sqrt_Sigmas = npr.randn(K, _D, _D)
        ### for Gamma speed
        # self.alphas = npr.rand(K, _D)
        # self.betas = npr.rand(K, _D)
        self._D = _D

    @property
    def params(self):
        return self.theta_k

    @params.setter
    def params(self, value):
        self.theta_k = value

    def permute(self, perm):
        self.theta_k = self.theta_k[perm]

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    def log_likelihoods(self, data, input, mask, tag):
        data_ang = data[:,0][:,None]
        data_spd = data[:,1][:,None]
        THETA = self.theta_k
        mus, sigs, alphas, betas = THETA[:,0], THETA[:,1], THETA[:,2], THETA[:,3]
        alphas, betas = alphas**2, betas**2  ### posititvity
        if mask is not None and np.any(~mask) and not isinstance(mus, np.ndarray):
            raise Exception("Current implementation of multivariate_normal_logpdf for masked data"
                            "does not work with autograd because it writes to an array. "
                            "Use DiagonalGaussian instead if you need to support missing data.")

        # stats.multivariate_normal_logpdf supports broadcasting, but we get
        # significant performance benefit if we call it with (TxD), (D,), and (D,D)
        # arrays as inputs
        log_p_tk = []
        for kk in range(self.K):
            mu,sig,alpha,beta = mus[kk], sigs[kk], alphas[kk], betas[kk]
            logp_gauss_k = self.logP_gauss(data_ang, mu, sig)
            logp_gamma_k = self.logP_gamma(data_spd, alpha, beta)
            # print(logp_gamma_k.shape)
            ll = logp_gauss_k + logp_gamma_k
            log_p_tk.append(ll)
        log_p_tk = np.array(log_p_tk).T
        # logp_gauss = np.column_stack([stats.multivariate_normal_logpdf(data_ang, mu, Sigma)
        #                        for mu, Sigma in zip(mus, Sigmas)])  ### T by K states
        # logp_gamma = np.column_stack([sp.stats.gamma.logpdf(data_spd, alpha, loc=0, scale=1/beta)
        #                         for alpha, beta in zip(alphas, betas)])
        # logp_gamma = np.column_stack([stats.multivariate_normal_logpdf(data_spd, mu, Sigma)
                               # for mu, Sigma in zip(alphas, Sigmas)])  ### T by K states
        return log_p_tk
    
    
#     def log_likelihoods_old(self, data, input, mask, tag):
#         """
#         new version to avoid auto-grad issues
#         """
#         ### load data
#         dth = data.squeeze()
#         dc, dcp = input[:,:int(input.shape[1]/2)], input[:,int(input.shape[1]/2):]
#         log_p_tk = []#np.zeros((data.shape[0], self.K))  # TxK log-likelihood
#         for kk in range(self.K):
#             THETA = self.theta_k[kk]  #parameter in a state
#             ### mixture model
#             k_, A_, B_, Amp, tau = THETA[0], THETA[1], THETA[2:4], THETA[4], THETA[5]
#             P = self.sigmoid2(A_,B_,dc)
#             C_ = -Amp * np.exp(-np.arange(len(B_)) / tau)
# #            C_ = np.array([Amp,tau])  #use direct weights for now
#             xx = (dth-np.dot(dcp,C_))*(np.pi/180)
#             VM = np.exp(k_**2*np.cos(xx))/(2*np.pi*i0(k_**2))
#             marginalP = (1-P)*VM + (1/(2*np.pi))*P
#             ll = np.log(np.abs(marginalP)+1e-10)
#             log_p_tk.append(ll)
#         log_p_tk = np.array(log_p_tk).T
#         return log_p_tk
    
    def logP_gauss(self, data, mu, sig):
        log_pdf_values = -0.5 * np.sum(((data - mu) / sig) ** 2 + np.log(2 * np.pi * sig ** 2 +1e-10), axis=1)
        return log_pdf_values
    
    def logP_gamma(self, data, alpha, beta):
        ### gamma distribution for speed
        # beta = 1/beta
        log_pdf_values = np.sum(
        (alpha - 1) * np.log(data+1e-10) - beta * data + alpha * np.log(beta+1e-10) - self.stirling_approx_gammaln(alpha),axis=1
        )
        ### log-normal
        # log_pdf_values = -np.log(data +1e-10) - np.log(beta * np.sqrt(2 * np.pi)) - ((np.log(data +1e-10) - alpha) ** 2) / (2 * beta ** 2)
        # log_pdf_values = log_pdf_values.squeeze()
    
        return log_pdf_values
    
    def stirling_approx_gammaln(self, alpha):
        return alpha * np.log(alpha+1e-10) - alpha + 0.5 * np.log(2 * np.pi / alpha+1e-10)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        THETA = self.theta_k
        mus, sigs, alphas, betas = THETA[:,0], THETA[:,1], THETA[:,2], THETA[:,3]
        alphas, betas = alphas**2, betas**2
        D = self._D
        # sqrt_Sigmas = self._sqrt_Sigmas if with_noise else np.zeros((self.K, self._D, self._D))
        samp_ang = mus[z] + np.dot(sigs[z], npr.randn(D))
        samp_speed = [np.random.gamma(alphas[z], scale=1/betas[z])]
        return np.array([samp_ang, samp_speed]).squeeze()

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, optimizer="bfgs", **kwargs)
        
        # K, D = self.K, self.D
        # J = np.zeros((K, D))
        # h = np.zeros((K, D))
        # for (Ez, _, _), y in zip(expectations, datas):
        #     J += np.sum(Ez[:, :, None], axis=0)
        #     h += np.sum(Ez[:, :, None] * y[:, None, :], axis=0)
        # self.mus = h / J

        # # Update the variance
        # sqerr = np.zeros((K, D, D))
        # weight = np.zeros((K,))
        # for (Ez, _, _), y in zip(expectations, datas):
        #     resid = y[:, None, :] - self.mus
        #     sqerr += np.sum(Ez[:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)
        #     weight += np.sum(Ez, axis=0)
        # self._sqrt_Sigmas = np.linalg.cholesky(sqerr / weight[:, None, None] + 1e-8 * np.eye(self.D))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)

# %% main tests
# Make a GLM-HMM
num_states = 5
obs_dim = 2  # for each speed and angle
test_mixhmm = ssm.HMM(num_states, obs_dim)  #fix to be both driven!

# %%  replace from here for now
test_mixhmm.observations = MixtureObservations(num_states, obs_dim)

test_mixhmm.observations.theta_k = np.array([[10, 10, 5, 5],
                                             [-10, 1, 1,10]])

# %% sample
true_z, true_y = test_mixhmm.sample(10000)

# %%
inf_mixhmm = ssm.HMM(num_states, obs_dim)  #fix to be both driven!
inf_mixhmm.observations = MixtureObservations(num_states, obs_dim)
hmm_lls = inf_mixhmm.fit(true_y,  method="em", num_iters=50, init_method="kmeans")
plt.figure()
plt.plot(hmm_lls, label="EM")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")

# %% now fit to flies!
### load data4fit
data_train = data4fit*1
fly_mixhmm = ssm.HMM(num_states, obs_dim)  #fix to be both driven!
fly_mixhmm.observations = MixtureObservations(num_states, obs_dim)
hmm_lls = fly_mixhmm.fit(data_train,  method="em", num_iters=50, init_method="kmeans")
plt.figure()
plt.plot(hmm_lls, label="EM")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")

# %%
ltr = len(data_train)
post_z = []
for ll in range(ltr):
    most_likely_states = fly_mixhmm.most_likely_states(data_train[ll])
    post_z.append(most_likely_states[:,None])

# %%
cmap = plt.get_cmap('tab10')
vec_states = np.concatenate(post_z)
vec_ego = np.concatenate(data_train)
plt.figure()
for ii in range(num_states):
    pos = np.where(vec_states==ii)[0]
    plt.plot(vec_ego[pos,0], vec_ego[pos,1], '.', color=cmap(ii),alpha=.1)
plt.xlabel("angular change (deg/s)")
plt.ylabel("speed (mm/s)")

# %% plot track
pick_id = 81  # 0,7
most_likely_states = fly_mixhmm.most_likely_states(data_train[pick_id])
track_i = rec_tracks[pick_id]

most_likely_states = most_likely_states[:] #:6
track_i = track_i[:] #:6

# Create a colormap for the two states
unique_states = np.unique(most_likely_states)
cmap = plt.get_cmap('tab10')

plt.figure(figsize=(8, 6))
for ii in range(num_states): #(len(unique_states)):
    state_mask = np.where(most_likely_states==ii)[0]
    plt.plot(track_i[state_mask,0], track_i[state_mask,1], 'o', color=cmap(ii), alpha=0.5)

plt.title("state-code trajectories")
plt.xlabel("X")
plt.ylabel("Y")

# %% basis function test
###############################################################################
# %%
# lamb = 1
# k = 5
# betas = np.array([1,2,3,4,5])
# betas = np.random.randn(k)
# time = np.arange(0,10,.2)

# Kt = lamb*np.exp(-lamb*time)*np.sum(np.array([betas[ii]*(lamb*time)**ii for ii in range(k)]),0)

# plt.figure()
# plt.plot(time, Kt)

# # %%
# for ii in range(k):
#     temp = lamb*np.exp(-lamb*time)*betas[ii]*(lamb*time)**ii
#     plt.plot(time, temp)