"""
bounds.py — Analytical MMSE Bounds & Performance Metrics
========================================================
Provides:
- `immse`: Mean Squared Error matching MATLAB's `immse(est_x, true_x)`.
- `mmse_known_theta_bound`: Case 2 Oracle Known-Theta MMSE bound (`MMSE_known_theta`).
- `mmse_unknown_theta_bound`: Case 1 Unclustered Unknown-Theta MMSE bound (`MMSE_unknown_theta`).
- `partition_mmse_bound`: Case 4 Partition-Averaged MMSE lower bound (`average_over_partitions_1D.m`)
  using `scipy.special.gammaln` for overflow-free evaluation at arbitrary N.
- `clustering_gain_db`: Clustering Gain in decibels `10 * log10(MMSE_unknown / MSE)`.
"""

from __future__ import annotations

from typing import Dict, Union
import numpy as np
from scipy.special import gammaln

from data_generation import sym_solve


def immse(est_x: np.ndarray, true_x: np.ndarray) -> float:
    """Mean Squared Error per coordinate (matches MATLAB's `immse`)."""
    return float(np.mean((np.asarray(est_x, dtype=float) - np.asarray(true_x, dtype=float)) ** 2))


def mmse_known_theta_bound(var_v: np.ndarray, var_u: np.ndarray) -> float:
    """
    Case 2 Analytical Bound (`MMSE_known_theta`):
    `(1 / dim) * trace(var_v @ inv(var_u + var_v) @ var_u)`
    """
    var_v = np.asarray(var_v, dtype=float)
    var_u = np.asarray(var_u, dtype=float)
    dim = var_v.shape[0]
    w_v = sym_solve(var_u + var_v, var_v).T
    return float((1.0 / dim) * np.trace(w_v @ var_u))


def mmse_unknown_theta_bound(
    var_v: np.ndarray,
    var_u: np.ndarray,
    var_theta: np.ndarray,
) -> float:
    """
    Case 1 Analytical Bound (`MMSE_unknown_theta`, unclustered baseline):
    `(1 / dim) * trace(var_v @ inv(var_theta + var_u + var_v) @ (var_u + var_theta))`
    """
    var_v = np.asarray(var_v, dtype=float)
    var_u = np.asarray(var_u, dtype=float)
    var_theta = np.asarray(var_theta, dtype=float)
    dim = var_v.shape[0]
    w_v_total = sym_solve(var_theta + var_u + var_v, var_v).T
    return float((1.0 / dim) * np.trace(w_v_total @ (var_u + var_theta)))


def partition_mmse_bound(
    var_theta: Union[float, np.ndarray],
    var_u: Union[float, np.ndarray],
    var_v: Union[float, np.ndarray],
    n_objects: int,
    alpha_dp: float,
) -> float:
    """
    Case 4 Partition-Averaged Analytical MMSE Lower Bound (refactored from
    `average_over_partitions_1D.m`).

    Uses log-gamma (`scipy.special.gammaln`) to compute the Beta-Binomial /
    Pochhammer marginal cluster-size distribution without factorial overflow
    even for `n_objects > 170`.
    """
    vt1 = float(np.asarray(var_theta).flat[0])
    vu1 = float(np.asarray(var_u).flat[0])
    vv1 = float(np.asarray(var_v).flat[0])

    k_vals = np.arange(1, n_objects + 1, dtype=float)
    log_num = (
        gammaln(alpha_dp + n_objects - k_vals)
        - gammaln(alpha_dp)
        + gammaln(n_objects)
        - gammaln(n_objects - k_vals + 1.0)
    )
    log_den = gammaln(alpha_dp + n_objects) - gammaln(alpha_dp)
    partition_p = np.exp(log_num - log_den)

    mse_k = (vv1 * vu1) / (vu1 + vv1) + ((vt1 * vv1) / (vv1 + vu1)) * (
        vv1 / (vv1 + vu1 + k_vals * vt1)
    )
    return float(np.sum(mse_k * partition_p))


def clustering_gain_db(
    mse: Union[float, np.ndarray],
    mmse_unknown: float,
) -> Union[float, np.ndarray]:
    """
    Computes Clustering Gain in dB relative to the unclustered bound:
    `CG = 10 * log10(MMSE_unknown_theta / MSE)`.
    """
    return 10.0 * np.log10(mmse_unknown / np.asarray(mse, dtype=float))


def compute_all_bounds(
    var_theta: np.ndarray,
    var_u: np.ndarray,
    var_v: np.ndarray,
    n_objects: int,
    alpha_dp: float,
) -> Dict[str, float]:
    """Convenience function returning all theoretical bounds for a given setup."""
    m_known = mmse_known_theta_bound(var_v, var_u)
    m_unknown = mmse_unknown_theta_bound(var_v, var_u, var_theta)
    m_part = partition_mmse_bound(var_theta, var_u, var_v, n_objects, alpha_dp)
    return {
        "mmse_known_theta": m_known,
        "mmse_unknown_theta": m_unknown,
        "mmse_partition_bound": m_part,
        "cg_partition_bound_db": float(clustering_gain_db(m_part, m_unknown)),
    }