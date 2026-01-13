"""
estimator_gibbs.py — Collapsed/Indicator DPMM Gibbs Samplers (SciPy & PyTorch)
==============================================================================
Refactored from `Gibbs_sampler_indicator_theta_first.m` and `Gibbs_sampler_indicator.m`.

Provides:
- `DPMMGibbsSciPy`: Cholesky-accelerated (`cho_factor`/`cho_solve`), log-domain
  (`scipy.special.logsumexp`) MCMC Gibbs sampler with vectorized state updates.
- `DPMMGibbsPyTorch`: GPU/MPS/CPU PyTorch MCMC Gibbs sampler using
  `torch.distributions.MultivariateNormal(loc, scale_tril=...)` and
  `Categorical(logits=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from data_generation import assign_outliers_and_reindex, sym_solve
from estimator_dbscan import dbscan_kdtree, estimate_epsilon_knee

try:
    import torch
    from torch.distributions import Categorical, MultivariateNormal
    TORCH_AVAILABLE = True
    _no_grad = torch.no_grad
except ImportError:
    TORCH_AVAILABLE = False

    def _no_grad():
        def decorator(fn):
            return fn
        return decorator


@dataclass
class GibbsSamples:
    """Container for MCMC posterior samples."""
    x: Union[np.ndarray, "torch.Tensor"]      # Shape: (K_iter, N, dim)
    theta: Union[np.ndarray, "torch.Tensor"]  # Shape: (K_iter, N, dim)
    last_labels: np.ndarray                   # Final cluster assignments: (N,)


class DPMMGibbsSciPy:
    """
    SciPy-powered Multidimensional DPMM Indicator Gibbs Sampler.
    Supports both `theta_first=True` (`Gibbs_sampler_indicator_theta_first.m`)
    and `theta_first=False` (`Gibbs_sampler_indicator.m`).
    """

    def __init__(
        self,
        var_v: np.ndarray,
        var_u: np.ndarray,
        var_theta: np.ndarray,
        mean_theta: np.ndarray,
        alpha_dp: float = 1.0,
        n_iters: int = 800,
        burn_in: int = 199,
        theta_first: bool = True,
    ):
        self.var_v = np.asarray(var_v, dtype=float)
        self.var_u = np.asarray(var_u, dtype=float)
        self.var_theta = np.asarray(var_theta, dtype=float)
        self.mean_theta = np.asarray(mean_theta, dtype=float).ravel()
        self.dim = self.mean_theta.shape[0]
        self.alpha_dp = float(alpha_dp)
        self.n_iters = int(n_iters)
        self.burn_in = int(burn_in)
        self.theta_first = bool(theta_first)

        # Precompute Cholesky linear operators
        self.sum_uv = self.var_u + self.var_v
        self.sum_u_theta = self.var_u + self.var_theta

        self.w_v = sym_solve(self.sum_uv, self.var_v).T
        self.w_u = sym_solve(self.sum_uv, self.var_u).T
        self.cov_x_post = self.w_v @ self.var_u
        self.chol_x_post = cholesky(self.cov_x_post, lower=True)

        self.w_theta_1 = sym_solve(self.sum_u_theta, self.var_u).T
        self.w_x_1 = sym_solve(self.sum_u_theta, self.var_theta).T
        self.cov_theta_1 = self.w_theta_1 @ self.var_theta
        self.chol_theta_1 = cholesky(self.cov_theta_1, lower=True)

        self.chol_u = cholesky(self.var_u, lower=True)
        self.chol_u_factor = cho_factor(self.var_u, lower=True)
        self.log_det_u = 2.0 * float(np.sum(np.log(np.diag(self.chol_u))))

    def _init_dbscan(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = y.shape[0]
        if n_samples == 1:
            idx = np.array([0], dtype=int)
        else:
            min_pts = self.dim + 1
            if int(np.ceil(np.log(n_samples))) <= 2:
                eps = 1.0
            else:
                max_pts = max(2, int(np.ceil(self.alpha_dp * np.log(n_samples))))
                eps = estimate_epsilon_knee(y, 2, max_pts)
            idx = dbscan_kdtree(y, eps, min_pts)

        c, m = assign_outliers_and_reindex(idx)
        num_clusters = len(m)
        theta_0 = np.zeros((num_clusters, self.dim), dtype=float)

        for i in range(num_clusters):
            mask = (c == i)
            mean_y_i = np.mean(y[mask], axis=0)
            cov_total = self.sum_uv + m[i] * self.var_theta
            w_prior = sym_solve(cov_total, self.sum_uv).T
            w_obs = sym_solve(cov_total, m[i] * self.var_theta).T
            theta_0[i] = self.mean_theta @ w_prior + mean_y_i @ w_obs

        return theta_0, c

    def sample(
        self, y: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> GibbsSamples:
        if rng is None:
            rng = np.random.default_rng()

        y = np.asarray(y, dtype=float)
        n_samples = y.shape[0]
        K = self.n_iters

        samples_x = np.zeros((K, n_samples, self.dim), dtype=float)
        samples_theta = np.zeros((K, n_samples, self.dim), dtype=float)

        previous_theta_star, c = self._init_dbscan(y)
        log_alpha = np.log(self.alpha_dp)

        x_k = previous_theta_star[c] + rng.standard_normal((n_samples, self.dim)) @ self.chol_u.T

        for k in range(K):
            if not self.theta_first:
                # Sample states x first (`Gibbs_sampler_indicator.m`)
                mean_x_all = previous_theta_star[c] @ self.w_v + y @ self.w_u
                x_k = mean_x_all + rng.standard_normal((n_samples, self.dim)) @ self.chol_x_post.T
                samples_x[k] = x_k
            elif k > 0:
                x_k = samples_x[k - 1]

            c_max = previous_theta_star.shape[0]
            log_prior_marginal = multivariate_normal.logpdf(
                x_k, mean=self.mean_theta, cov=self.sum_u_theta
            )
            if np.ndim(log_prior_marginal) == 0:
                log_prior_marginal = np.array([float(log_prior_marginal)])

            # Log-domain CRP indicator sampling
            for n in range(n_samples):
                c_counts = np.bincount(c, minlength=c_max)
                if c[n] < c_max:
                    c_counts[c[n]] -= 1

                log_weights = np.full(c_max + 1, -np.inf, dtype=float)
                active_t = np.where(c_counts > 0)[0]
                if len(active_t) > 0:
                    diffs = previous_theta_star[active_t] - x_k[n]
                    sol = cho_solve(self.chol_u_factor, diffs.T, check_finite=False)
                    maha = np.sum(diffs.T * sol, axis=0)
                    log_pdfs = -0.5 * (self.dim * np.log(2.0 * np.pi) + self.log_det_u + maha)
                    log_weights[active_t] = np.log(c_counts[active_t]) + log_pdfs

                log_weights[c_max] = log_alpha + log_prior_marginal[n]
                probs = np.exp(log_weights - logsumexp(log_weights))

                c_n = int(rng.choice(c_max + 1, p=probs))
                if c_n == c_max:
                    mean_1 = self.mean_theta @ self.w_theta_1 + x_k[n] @ self.w_x_1
                    theta_k = mean_1 + rng.standard_normal(self.dim) @ self.chol_theta_1.T
                    previous_theta_star = np.vstack([previous_theta_star, theta_k])
                    c[n] = c_max
                    c_max += 1
                else:
                    c[n] = c_n

            # Re-index active clusters and resample cluster centers theta^*_c
            unique_classes, c = np.unique(c, return_inverse=True)
            num_active = len(unique_classes)
            new_theta_star = np.zeros((num_active, self.dim), dtype=float)

            for idx_c in range(num_active):
                mask = (c == idx_c)
                x_k_class = x_k[mask]
                num_c = x_k_class.shape[0]
                cov_c = self.var_u + num_c * self.var_theta
                w_prior_c = sym_solve(cov_c, self.var_u).T
                w_obs_c = sym_solve(cov_c, num_c * self.var_theta).T
                mean_2 = self.mean_theta @ w_prior_c + np.mean(x_k_class, axis=0) @ w_obs_c
                var_2 = w_prior_c @ self.var_theta
                chol_2 = cholesky(var_2, lower=True)
                new_theta_star[idx_c] = mean_2 + rng.standard_normal(self.dim) @ chol_2.T

            samples_theta[k] = new_theta_star[c]
            previous_theta_star = new_theta_star

            if self.theta_first:
                # Sample states x after theta (`Gibbs_sampler_indicator_theta_first.m`)
                mean_x_all = previous_theta_star[c] @ self.w_v + y @ self.w_u
                samples_x[k] = (
                    mean_x_all + rng.standard_normal((n_samples, self.dim)) @ self.chol_x_post.T
                )

        return GibbsSamples(x=samples_x, theta=samples_theta, last_labels=c.copy())

    def fit_predict(
        self, y: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs MCMC Gibbs sampling and returns `(mean_x_post, last_cluster_labels)`."""
        samples = self.sample(y, rng=rng)
        mean_x = np.mean(samples.x[self.burn_in :], axis=0)
        return mean_x, samples.last_labels


class DPMMGibbsPyTorch:
    """
    PyTorch-accelerated DPMM Indicator Gibbs Sampler using `torch.distributions`.
    """

    def __init__(
        self,
        var_v: np.ndarray,
        var_u: np.ndarray,
        var_theta: np.ndarray,
        mean_theta: np.ndarray,
        alpha_dp: float = 1.0,
        n_iters: int = 800,
        burn_in: int = 199,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch (`torch`) is required for `DPMMGibbsPyTorch`.")

        self.device = torch.device(device)
        self.dtype = torch.float64
        self.alpha_dp = float(alpha_dp)
        self.n_iters = int(n_iters)
        self.burn_in = int(burn_in)

        self.var_v = torch.as_tensor(var_v, device=self.device, dtype=self.dtype)
        self.var_u = torch.as_tensor(var_u, device=self.device, dtype=self.dtype)
        self.var_theta = torch.as_tensor(var_theta, device=self.device, dtype=self.dtype)
        self.mean_theta = torch.as_tensor(mean_theta, device=self.device, dtype=self.dtype).view(-1)
        self.dim = int(self.mean_theta.numel())

        self.sum_uv = self.var_u + self.var_v
        self.sum_u_theta = self.var_u + self.var_theta

        self.w_v = torch.linalg.solve(self.sum_uv, self.var_v).T
        self.w_u = torch.linalg.solve(self.sum_uv, self.var_u).T
        self.tril_x_post = torch.linalg.cholesky(self.w_v @ self.var_u)

        self.w_theta_1 = torch.linalg.solve(self.sum_u_theta, self.var_u).T
        self.w_x_1 = torch.linalg.solve(self.sum_u_theta, self.var_theta).T
        self.tril_theta_1 = torch.linalg.cholesky(self.w_theta_1 @ self.var_theta)

        self.tril_u = torch.linalg.cholesky(self.var_u)
        self.tril_u_theta = torch.linalg.cholesky(self.sum_u_theta)
        self.prior_marginal_dist = MultivariateNormal(
            loc=self.mean_theta, scale_tril=self.tril_u_theta
        )

    @_no_grad()
    def fit_predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scipy_helper = DPMMGibbsSciPy(
            self.var_v.cpu().numpy(),
            self.var_u.cpu().numpy(),
            self.var_theta.cpu().numpy(),
            self.mean_theta.cpu().numpy(),
            alpha_dp=self.alpha_dp,
            n_iters=self.n_iters,
            burn_in=self.burn_in,
        )
        theta_0_np, c_np = scipy_helper._init_dbscan(np.asarray(y, dtype=float))

        y_t = torch.as_tensor(y, device=self.device, dtype=self.dtype)
        n_samples = y_t.shape[0]
        K = self.n_iters

        samples_x = torch.zeros((K, n_samples, self.dim), device=self.device, dtype=self.dtype)
        previous_theta_star = torch.as_tensor(theta_0_np, device=self.device, dtype=self.dtype)
        c = torch.as_tensor(c_np, device=self.device, dtype=torch.long)
        log_alpha = float(np.log(self.alpha_dp))

        x_k = (
            previous_theta_star[c]
            + torch.randn((n_samples, self.dim), device=self.device, dtype=self.dtype)
            @ self.tril_u.T
        )

        for k in range(K):
            if k > 0:
                x_k = samples_x[k - 1]

            c_max = previous_theta_star.shape[0]
            log_prior_all = self.prior_marginal_dist.log_prob(x_k)

            for n in range(n_samples):
                c_counts = torch.bincount(c, minlength=c_max).to(self.dtype)
                if c[n] < c_max:
                    c_counts[c[n]] -= 1.0

                logits = torch.full((c_max + 1,), -float("inf"), device=self.device, dtype=self.dtype)
                active_mask = c_counts > 0
                if torch.any(active_mask):
                    active_thetas = previous_theta_star[active_mask]
                    cluster_dist = MultivariateNormal(loc=active_thetas, scale_tril=self.tril_u)
                    logits[:c_max][active_mask] = (
                        torch.log(c_counts[active_mask]) + cluster_dist.log_prob(x_k[n])
                    )

                logits[c_max] = log_alpha + log_prior_all[n]
                c_n = int(Categorical(logits=logits).sample().item())

                if c_n == c_max:
                    mean_1 = self.mean_theta @ self.w_theta_1 + x_k[n] @ self.w_x_1
                    z = torch.randn(self.dim, device=self.device, dtype=self.dtype)
                    theta_k = mean_1 + z @ self.tril_theta_1.T
                    previous_theta_star = torch.cat([previous_theta_star, theta_k.unsqueeze(0)], dim=0)
                    c[n] = c_max
                    c_max += 1
                else:
                    c[n] = c_n

            unique_classes, c = torch.unique(c, sorted=True, return_inverse=True)
            num_active = unique_classes.numel()
            new_theta_star = torch.zeros((num_active, self.dim), device=self.device, dtype=self.dtype)

            for idx_c in range(num_active):
                mask = (c == idx_c)
                x_k_class = x_k[mask]
                num_c = float(x_k_class.shape[0])
                cov_c = self.var_u + num_c * self.var_theta
                w_prior_c = torch.linalg.solve(cov_c, self.var_u).T
                w_obs_c = torch.linalg.solve(cov_c, num_c * self.var_theta).T
                mean_2 = self.mean_theta @ w_prior_c + x_k_class.mean(dim=0) @ w_obs_c
                var_2 = w_prior_c @ self.var_theta
                tril_2 = torch.linalg.cholesky(var_2)
                z = torch.randn(self.dim, device=self.device, dtype=self.dtype)
                new_theta_star[idx_c] = mean_2 + z @ tril_2.T

            previous_theta_star = new_theta_star
            mean_x_all = previous_theta_star[c] @ self.w_v + y_t @ self.w_u
            z_x = torch.randn((n_samples, self.dim), device=self.device, dtype=self.dtype)
            samples_x[k] = mean_x_all + z_x @ self.tril_x_post.T

        mean_x_np = samples_x[self.burn_in :].mean(dim=0).detach().cpu().numpy()
        return mean_x_np, c.detach().cpu().numpy()