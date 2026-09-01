# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 14:15:55 2026

@author: ksc75
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# %% mixture of logistics to predict crossing events
def _log_bern_prob(p, y):
    """
    log p(y | p) for Bernoulli with probability p.
    p: (N,) in (0,1)
    y: (N,) in {0,1}
    """
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


class MixtureLogisticRegressionEM:
    """
    Mixture of K logistic regressions:
        p(y=1|x) = sum_k pi_k * sigmoid(w_k^T x + b_k)

    Fit with EM using sklearn LogisticRegression in M-step via sample_weight.

    Notes:
    - If use_scaler=True, models are Pipelines(StandardScaler -> LogisticRegression)
      so sample_weight must be passed as logisticregression__sample_weight.
    - Use L2 first for stability; L1 can cause component collapse.
    """

    def __init__(
        self,
        n_components=2,
        penalty="l2",
        C=1.0,
        solver=None,
        max_iter_lr=2000,
        max_iter_em=200,
        tol=1e-5,
        n_restarts=5,
        random_state=0,
        verbose=False,
        use_scaler=True,
    ):
        self.K = int(n_components)
        self.penalty = penalty
        self.C = float(C)
        self.solver = solver
        self.max_iter_lr = int(max_iter_lr)
        self.max_iter_em = int(max_iter_em)
        self.tol = float(tol)
        self.n_restarts = int(n_restarts)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.use_scaler = bool(use_scaler)

        self.pis_ = None
        self.models_ = None
        self.best_loglik_ = None

    def _make_lr(self):
        # Choose a sane default solver if not specified
        if self.solver is not None:
            solver = self.solver
        else:
            solver = "liblinear" if self.penalty == "l1" else "lbfgs"

        lr = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=solver,
            max_iter=self.max_iter_lr,
            fit_intercept=True
        )
        return make_pipeline(StandardScaler(), lr) if self.use_scaler else lr

    def _component_prob(self, model, X):
        # returns p_k(y=1|X) shape (N,)
        return model.predict_proba(X)[:, 1]

    def _log_likelihood(self, X, y, pis, models):
        """
        total log-likelihood:
            sum_i log( sum_k pi_k * p_k(y_i|x_i) )
        """
        N = X.shape[0]
        log_terms = np.zeros((N, self.K), dtype=float)

        for k in range(self.K):
            pk = self._component_prob(models[k], X)
            log_py = _log_bern_prob(pk, y)
            log_terms[:, k] = np.log(np.clip(pis[k], 1e-12, 1.0)) + log_py

        # log-sum-exp over components for stability
        m = np.max(log_terms, axis=1, keepdims=True)
        ll_i = m[:, 0] + np.log(np.sum(np.exp(log_terms - m), axis=1))
        return float(np.sum(ll_i))

    def _e_step(self, X, y, pis, models):
        """
        responsibilities:
            r_ik ∝ pi_k * p_k(y_i|x_i)
        """
        N = X.shape[0]
        log_r = np.zeros((N, self.K), dtype=float)

        for k in range(self.K):
            pk = self._component_prob(models[k], X)
            log_r[:, k] = np.log(np.clip(pis[k], 1e-12, 1.0)) + _log_bern_prob(pk, y)

        # normalize in log space
        m = np.max(log_r, axis=1, keepdims=True)
        r = np.exp(log_r - m)
        r /= np.sum(r, axis=1, keepdims=True)
        return r  # (N, K)

    def _m_step(self, X, y, r):
        """
        Update:
            pi_k = (1/N) sum_i r_ik
            beta_k = weighted logistic regression with weights r_ik
        """
        Nk = r.sum(axis=0)
        pis = Nk / np.sum(Nk)

        models = []
        for k in range(self.K):
            mk = self._make_lr()

            # IMPORTANT FIX:
            # If mk is a Pipeline, route sample_weight to the LR step.
            if self.use_scaler:
                mk.fit(X, y, logisticregression__sample_weight=r[:, k])
            else:
                mk.fit(X, y, sample_weight=r[:, k])

            models.append(mk)

        return pis, models

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("y must be binary {0,1}")

        rng = np.random.default_rng(self.random_state)

        best_ll = -np.inf
        best_params = None

        for rr in range(self.n_restarts):
            # random soft assignments for initialization
            # r = rng.random((X.shape[0], self.K))
            r = np.random.random((X.shape[0], self.K))
            r /= r.sum(axis=1, keepdims=True)

            # one M-step to initialize models
            pis, models = self._m_step(X, y, r)

            prev_ll = -np.inf
            for it in range(self.max_iter_em):
                r = self._e_step(X, y, pis, models)
                pis, models = self._m_step(X, y, r)
                ll = self._log_likelihood(X, y, pis, models)

                if self.verbose and (it % 10 == 0 or it == self.max_iter_em - 1):
                    print(f"[restart {rr+1}/{self.n_restarts}] iter {it:03d} ll={ll:.3f}")

                if np.abs(ll - prev_ll) < self.tol:
                    break
                prev_ll = ll

            if ll > best_ll:
                best_ll = ll
                best_params = (pis, models)

        self.pis_, self.models_ = best_params
        self.best_loglik_ = best_ll
        return self

    def predict_proba(self, X):
        """
        Mixture probability of y=1:
            p(y=1|x) = sum_k pi_k p_k(y=1|x)
        """
        if self.models_ is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X)

        p1 = np.zeros(X.shape[0], dtype=float)
        for k in range(self.K):
            p1 += self.pis_[k] * self._component_prob(self.models_[k], X)

        p1 = np.clip(p1, 1e-12, 1 - 1e-12)
        return np.column_stack([1 - p1, p1])

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

    def component_predict_proba(self, X):
        """
        Return per-component probabilities p_k(y=1|x), shape (N, K).
        """
        if self.models_ is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X)
        P = np.zeros((X.shape[0], self.K), dtype=float)
        for k in range(self.K):
            P[:, k] = self._component_prob(self.models_[k], X)
        return P
    
    def posterior_z(self, X, y):
        """
        Posterior responsibilities r_ik = P(z=k | x, y).
        Returns r with shape (N, K).
        """
        if self.models_ is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        return self._e_step(X, y, self.pis_, self.models_)

    
    
def extract_em_weights(em):
    """
    Returns:
        W: (K, d) weights
        b: (K,) intercepts
        pis: (K,) mixture weights
    Works for em.use_scaler True/False.
    """
    K = em.K
    pis = np.asarray(em.pis_)
    W = []
    b = []

    for k in range(K):
        mk = em.models_[k]

        # If pipeline: StandardScaler -> LogisticRegression
        if hasattr(mk, "named_steps"):
            lr = mk.named_steps["logisticregression"]
        else:
            lr = mk

        W.append(lr.coef_[0].copy())
        b.append(float(lr.intercept_[0]))

    return np.asarray(W), np.asarray(b), pis


def plot_em_component_weights(em, feature_names, sort_by_pi=True):
    """
    Plot weights for each mixture component as separate panels.
    """
    W, b, pis = extract_em_weights(em)

    K, d = W.shape
    if len(feature_names) != d:
        raise ValueError(f"feature_names length {len(feature_names)} != #features {d}")

    order = np.argsort(-pis) if sort_by_pi else np.arange(K)

    fig, axes = plt.subplots(1, K, figsize=(4.5*K, 4), sharey=True)
    if K == 1:
        axes = [axes]

    x = np.arange(d)
    for ax, kk in zip(axes, order):
        ax.plot(x, W[kk], "o")
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.set_title(f"Component {kk}\npi={pis[kk]:.2f}, b={b[kk]:.2f}")
        ax.set_ylabel("Weight (z-scored features)" if em.use_scaler else "Weight")

    plt.tight_layout()
    plt.show()  

def plot_em_weight_summary(em, feature_names):
    W, b, pis = extract_em_weights(em)
    w_avg = (pis[:, None] * W).sum(axis=0)

    x = np.arange(len(feature_names))
    plt.figure(figsize=(7,4))
    plt.plot(x, w_avg, "o")
    plt.axhline(0, color="gray", ls="--", lw=1)
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.ylabel("Weighted-average weight")
    plt.title("Mixture-of-logistics: aggregated weights (Σ pi_k β_k)")
    plt.tight_layout()
    plt.show()
    
def get_em_component_weights(em):
    """
    Returns sorted weights by mixture weight.
    Output:
        W: (K, d) ordered by descending pi
        pis: (K,)
    """
    pis = np.asarray(em.pis_)
    K = len(pis)

    W = []
    for k in range(K):
        mk = em.models_[k]
        if hasattr(mk, "named_steps"):
            lr = mk.named_steps["logisticregression"]
        else:
            lr = mk
        W.append(lr.coef_[0].copy())

    W = np.asarray(W)

    # sort components by pi (largest = state 0)
    order = np.argsort(-pis)
    return W[order], pis[order]

# -----------------------
# Minimal demo usage
# -----------------------
if __name__ == "__main__":
    # Example: X_all, y_all already exist in your workspace
    # X_all: (N, d), y_all: (N,)

    em = MixtureLogisticRegressionEM(
        n_components=2,
        penalty="l2",      # start with l2 for stability
        C=1.0,
        n_restarts=5,
        max_iter_em=100,
        tol=1e-5,
        random_state=0,
        verbose=True,
        use_scaler=True
    )

    em.fit(X_all, y_all)
    print("pis:", em.pis_)
    print("best train log-likelihood:", em.best_loglik_)

    p = em.predict_proba(X_all)[:, 1]
    print("p[:10] =", p[:10])
    
    ### cross valudation
    from sklearn.model_selection import GroupShuffleSplit

    K_states = 2
    gss = GroupShuffleSplit(n_splits=30, test_size=0.3, random_state=0)
    
    all_component_weights = []   # list of (K_states, d)
    
    for train_idx, test_idx in gss.split(X_all, y_all, groups):
    
        X_train, y_train = X_all[train_idx], y_all[train_idx]
    
        em = MixtureLogisticRegressionEM(
            n_components=K_states,
            penalty="l2",
            C=1.0,
            n_restarts=3,
            max_iter_em=80,
            random_state=1,
            verbose=False
        )
    
        em.fit(X_train, y_train)
    
        W, pis = get_em_component_weights(em)
        all_component_weights.append(W)
    
    all_component_weights = np.array(all_component_weights)
    # shape: (n_splits, K_states, n_features)
    
    
    ### posterior
    r_all  = em.posterior_z(X_all,  y_all)    # (N_test,  K)  (for analysis only)
    z_hat_all = r_all.argmax(axis=1)         # hard assignment
    
    # %%
    
    def plot_em_cv_weights(all_component_weights, feature_names):
    
        n_splits, K, d = all_component_weights.shape
        x = np.arange(d)
    
        fig, axes = plt.subplots(1, K, figsize=(5*K,4), sharey=True)
    
        if K == 1:
            axes = [axes]
    
        for k in range(K):
            mean_w = all_component_weights[:,k,:].mean(axis=0)
            std_w  = all_component_weights[:,k,:].std(axis=0)
    
            ax = axes[k]
            ax.errorbar(x, mean_w, yerr=std_w, fmt='o', capsize=3)
            ax.axhline(0, color='gray', linestyle='--')
    
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_title(f"Latent state {k}")
            ax.set_ylabel("Weight (z-scored features)")
    
        plt.suptitle("Mixture-of-Logistics Feature Weights (mean ± std across CV splits)")
        plt.tight_layout()
        plt.show()

    plot_em_cv_weights(all_component_weights, feature_names)
    plot_em_weight_summary(em, feature_names)
    plot_em_component_weights(em, feature_names)

    
# %% visualize
    
    offset=0
    plt.figure()
    for ii in range(len(z_hat_all)//1):
        if z_hat_all[ii] == 1:
            segi = hist_test[ii]
            plt.plot(segi[-150:,0]-segi[-1,0]*offset, segi[-150:,1]-segi[-1,1]*offset,'k,', alpha=.2)
            # plt.plot(segi[-1,0]-segi[-1,0]*offset, segi[-1,1]-segi[-1,1]*offset,'r.')
            # plt.plot(segi[0,0]-segi[-1,0]*offset, segi[0,1]-segi[-1,1]*offset,'g.')
        elif z_hat_all[ii] == 0:
            plt.plot(segi[:,0]-segi[-1,0]*offset, segi[:,1]-segi[-1,1]*offset,'r,', alpha=.2)
