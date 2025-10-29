"""
data_generation.py — Synthetic CRP-DPMM Data Generator & Shared LMMSE Utilities
===============================================================================
Provides:
- `DPMMData`: Dataclass container for observations `y`, states `x`, cluster
  centers `theta`, and 0-indexed cluster assignments `classes`.
- `generate_dpmm_data`: Vectorized NumPy/SciPy Chinese Restaurant Process (CRP)
  and hierarchical Gaussian generator (`generate_simple.m`).
- `sym_solve`: Cholesky-based SPD linear solver (`scipy.linalg.cho_factor` / `cho_solve`).
- `assign_outliers_and_reindex`: Maps DBSCAN noise labels (`< 0`) to singleton
  clusters and re-indexes contiguously `0 .. C-1`.
- `empirical_bayes_lmmse`: Shared Case 3b Empirical Bayes / LMMSE state update
  given any hard cluster assignment vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.stats import multivariate_normal


@dataclass
class DPMMData:
    """Container for synthetic DPMM data."""
    y: np.ndarray        # Observations: shape (N, dim)
    x: np.ndarray        # True latent states: shape (N, dim)
    theta: np.ndarray    # True cluster centers: shape (K_clusters, dim)
    classes: np.ndarray  # True 0-indexed cluster labels: shape (N,)


def sym_solve(cov_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """
    Computes `inv(cov_a) @ mat_b` using Cholesky factorization (`cho_factor`
    and `cho_solve`) instead of explicit matrix inversion.
    """
    c, lower = cho_factor(cov_a, lower=True, check_finite=True)
    return cho_solve((c, lower), mat_b, check_finite=False)


def assign_outliers_and_reindex(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts noise/outlier labels (`idx < 0`) into distinct singleton clusters
    and re-indexes all clusters contiguously `0, 1, ..., C-1`.

    Returns
    -------
    c : np.ndarray
        Contiguous 0-indexed cluster assignments of shape (N,).
    m : np.ndarray
        Cluster counts of shape (C,).
    """
    idx_clean = np.asarray(idx, dtype=int).copy()
    next_id = int(idx_clean.max()) + 1 if np.any(idx_clean >= 0) else 0
    for i in range(len(idx_clean)):
        if idx_clean[i] < 0:
            idx_clean[i] = next_id
            next_id += 1
    _, c, m = np.unique(idx_clean, return_inverse=True, return_counts=True)
    return c, m


def empirical_bayes_lmmse(
    y: np.ndarray,
    cluster_labels: np.ndarray,
    var_v: np.ndarray,
    var_u: np.ndarray,
    var_theta: np.ndarray,
    mean_theta: np.ndarray,
) -> np.ndarray:
    """
    Case 3b Empirical Bayes / LMMSE state estimator given any hard cluster
    assignment vector `cluster_labels` (from DBSCAN, K-Means, Variational DPMM,
    or Agglomerative clustering).
    """
    y = np.asarray(y, dtype=float)
    mean_theta = np.asarray(mean_theta, dtype=float).ravel()

    c, m = assign_outliers_and_reindex(cluster_labels)
    num_clusters = len(m)

    sum_uv = var_u + var_v
    w_u = sym_solve(sum_uv, var_u).T
    w_v = sym_solve(sum_uv, var_v).T
    est_x = np.zeros_like(y, dtype=float)

    for ci in range(num_clusters):
        mask = (c == ci)
        m_ci = m[ci]
        mean_y_ci = np.mean(y[mask], axis=0)
        cov_total = sum_uv + m_ci * var_theta
        w_theta_total = sym_solve(cov_total, var_v).T
        w_mean_y = sym_solve(cov_total, w_v @ (m_ci * var_theta)).T
        est_x[mask] = y[mask] @ w_u + mean_theta @ w_theta_total + mean_y_ci @ w_mean_y

    return est_x


def sample_crp_tables(
    max_n: int,
    alpha_dp: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int]:
    """
    Samples cluster assignments for `max_n` targets from a Chinese Restaurant
    Process (CRP) with concentration parameter `alpha_dp`.
    """
    if rng is None:
        rng = np.random.default_rng()

    c = np.zeros(max_n, dtype=int)
    counts = [1]
    u = rng.random(max_n)

    for n in range(1, max_n):
        probs = np.cumsum(counts) / (alpha_dp + n)
        idx = int(np.searchsorted(probs, u[n]))
        if idx < len(counts):
            c[n] = idx
            counts[idx] += 1
        else:
            c[n] = len(counts)
            counts.append(1)

    return c, len(counts)


def generate_dpmm_data(
    max_n: int,
    alpha_dp: float,
    mean_theta: np.ndarray,
    var_v: np.ndarray,
    var_u: np.ndarray,
    var_theta: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> DPMMData:
    """
    Generates synthetic multidimensional DPMM observations `y`, latent states `x`,
    cluster parameters `theta`, and CRP table assignments `classes` (refactored
    from `generate_simple.m`).
    """
    if rng is None:
        rng = np.random.default_rng()

    mean_theta = np.asarray(mean_theta, dtype=float).ravel()
    dim = mean_theta.shape[0]
    var_v = np.asarray(var_v, dtype=float)
    var_u = np.asarray(var_u, dtype=float)
    var_theta = np.asarray(var_theta, dtype=float)

    c, num_classes = sample_crp_tables(max_n, alpha_dp, rng=rng)

    classes_theta = multivariate_normal.rvs(
        mean=mean_theta, cov=var_theta, size=num_classes, random_state=rng
    )
    if num_classes == 1:
        classes_theta = classes_theta.reshape(1, dim)

    chol_u = cholesky(var_u, lower=True)
    chol_v = cholesky(var_v, lower=True)

    x = classes_theta[c] + rng.standard_normal((max_n, dim)) @ chol_u.T
    y = x + rng.standard_normal((max_n, dim)) @ chol_v.T

    return DPMMData(y=y, x=x, theta=classes_theta, classes=c)