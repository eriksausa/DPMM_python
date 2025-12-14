"""
estimator_variational_dpmm.py — Variational Inference DPMM & Hierarchical Estimators
====================================================================================
Provides `scikit-learn` library baselines for non-parametric/hierarchical clustering
paired with the Case 3b Empirical Bayes LMMSE state estimator:
- `VariationalDPMMEstimator`: Uses `sklearn.mixture.BayesianGaussianMixture`
  with `weight_concentration_prior_type='dirichlet_process'` (Mean-Field
  Variational Inference counterpart to your MCMC Gibbs sampler).
- `AgglomerativeEstimator`: Uses `sklearn.cluster.AgglomerativeClustering`
  (Ward linkage with noise-calibrated distance threshold).
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import multivariate_normal

from data_generation import assign_outliers_and_reindex, empirical_bayes_lmmse, sym_solve

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.mixture import BayesianGaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class VariationalDPMMEstimator:
    """
    Mean-Field Variational Inference Dirichlet Process Mixture Model
    (`sklearn.mixture.BayesianGaussianMixture`) + Case 3b LMMSE state estimator.
    """

    def __init__(
        self,
        var_v: np.ndarray,
        var_u: np.ndarray,
        var_theta: np.ndarray,
        mean_theta: np.ndarray,
        alpha_dp: float = 1.0,
        max_components: int = 10,
    ):
        self.var_v = np.asarray(var_v, dtype=float)
        self.var_u = np.asarray(var_u, dtype=float)
        self.var_theta = np.asarray(var_theta, dtype=float)
        self.mean_theta = np.asarray(mean_theta, dtype=float).ravel()
        self.alpha_dp = float(alpha_dp)
        self.max_components = int(max_components)

    def _map_crp_fallback(self, y: np.ndarray) -> np.ndarray:
        """Sequential MAP Chinese Restaurant clusterer used when `sklearn` is absent."""
        n_samples = y.shape[0]
        sum_uv = self.var_u + self.var_v
        labels = np.zeros(n_samples, dtype=int)
        clusters: List[List[int]] = [[0]]

        for n in range(1, n_samples):
            log_probs = np.zeros(len(clusters) + 1, dtype=float)
            for c_idx, members in enumerate(clusters):
                m_c = len(members)
                mean_y_c = np.mean(y[members], axis=0)
                cov_post_theta = sym_solve(sum_uv + m_c * self.var_theta, sum_uv).T @ self.var_theta
                mean_post_theta = (
                    self.mean_theta @ sym_solve(sum_uv + m_c * self.var_theta, sum_uv).T
                    + mean_y_c @ sym_solve(sum_uv + m_c * self.var_theta, m_c * self.var_theta).T
                )
                log_probs[c_idx] = np.log(m_c) + multivariate_normal.logpdf(
                    y[n], mean=mean_post_theta, cov=sum_uv + cov_post_theta
                )
            log_probs[-1] = np.log(self.alpha_dp) + multivariate_normal.logpdf(
                y[n], mean=self.mean_theta, cov=sum_uv + self.var_theta
            )
            best_c = int(np.argmax(log_probs))
            if best_c == len(clusters):
                labels[n] = len(clusters)
                clusters.append([n])
            else:
                labels[n] = best_c
                clusters[best_c].append(n)

        return labels

    def fit_predict(
        self, y: np.ndarray, random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits Variational DPMM to `y` and returns `(x_hat, cluster_labels)`.
        """
        y = np.asarray(y, dtype=float)
        n_samples = y.shape[0]

        if SKLEARN_AVAILABLE and n_samples >= 2:
            bgmm = BayesianGaussianMixture(
                n_components=min(n_samples, self.max_components),
                covariance_type="spherical",
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=self.alpha_dp,
                mean_prior=self.mean_theta,
                max_iter=200,
                n_init=3,
                random_state=random_state,
            )
            raw_labels = bgmm.fit_predict(y)
        else:
            raw_labels = self._map_crp_fallback(y)

        clean_labels, _ = assign_outliers_and_reindex(raw_labels)
        x_hat = empirical_bayes_lmmse(
            y, clean_labels, self.var_v, self.var_u, self.var_theta, self.mean_theta
        )
        return x_hat, clean_labels


class AgglomerativeEstimator:
    """
    Parameter-free Hierarchical Ward Clustering (`sklearn.cluster.AgglomerativeClustering`
    or `scipy.cluster.hierarchy.linkage`) + Case 3b LMMSE state estimator.
    """

    def __init__(
        self,
        var_v: np.ndarray,
        var_u: np.ndarray,
        var_theta: np.ndarray,
        mean_theta: np.ndarray,
        threshold_scale: float = 1.5,
    ):
        self.var_v = np.asarray(var_v, dtype=float)
        self.var_u = np.asarray(var_u, dtype=float)
        self.var_theta = np.asarray(var_theta, dtype=float)
        self.mean_theta = np.asarray(mean_theta, dtype=float).ravel()
        self.dist_thresh = float(threshold_scale) * float(np.sqrt(np.trace(self.var_u + self.var_v)))

    def fit_predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        n_samples = y.shape[0]

        if n_samples == 1:
            raw_labels = np.array([0], dtype=int)
        elif SKLEARN_AVAILABLE:
            agg = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.dist_thresh,
                linkage="ward",
            )
            raw_labels = agg.fit_predict(y)
        else:
            z_mat = linkage(y, method="ward")
            raw_labels = fcluster(z_mat, t=self.dist_thresh, criterion="distance") - 1

        clean_labels, _ = assign_outliers_and_reindex(raw_labels)
        x_hat = empirical_bayes_lmmse(
            y, clean_labels, self.var_v, self.var_u, self.var_theta, self.mean_theta
        )
        return x_hat, clean_labels