"""
estimator_kmeans.py — Case 3b K-Means++ Empirical Bayes LMMSE Estimator
=======================================================================
Refactored from `sim_3b_kmeans.m`.
Uses `sklearn.cluster.KMeans` (`n_init=10`) when `scikit-learn` is installed,
with an automatic fallback to `scipy.cluster.vq.kmeans2(..., minit='++')`.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from scipy.cluster.vq import kmeans2

from data_generation import assign_outliers_and_reindex, empirical_bayes_lmmse

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def kmeans_scipy(
    y: np.ndarray,
    n_clusters: int,
    replicates: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Multi-replicate K-Means++ built on `scipy.cluster.vq.kmeans2`.
    """
    if rng is None:
        rng = np.random.default_rng()

    y_2d = np.asarray(y, dtype=float)
    n_samples = y_2d.shape[0]
    k = max(1, min(int(n_clusters), n_samples))
    if k == 1:
        return np.zeros(n_samples, dtype=int)
    if k == n_samples:
        return np.arange(n_samples, dtype=int)

    best_inertia = np.inf
    best_labels = np.zeros(n_samples, dtype=int)

    for _ in range(max(1, replicates)):
        seed_val = int(rng.integers(0, 2**31 - 1))
        centroids, labels = kmeans2(
            y_2d, k, iter=50, minit="++", missing="warn", check_finite=False, rng=seed_val
        )
        _, labels_clean = np.unique(labels, return_inverse=True)
        diffs = y_2d - centroids[labels]
        inertia = float(np.sum(diffs ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels_clean.copy()

    return best_labels


class KMeansEstimator:
    """
    Case 3b K-Means++ Plug-in LMMSE State Estimator (`sim_3b_kmeans.m`).
    """

    def __init__(
        self,
        var_v: np.ndarray,
        var_u: np.ndarray,
        var_theta: np.ndarray,
        mean_theta: np.ndarray,
        n_replicates: int = 3,
    ):
        self.var_v = np.asarray(var_v, dtype=float)
        self.var_u = np.asarray(var_u, dtype=float)
        self.var_theta = np.asarray(var_theta, dtype=float)
        self.mean_theta = np.asarray(mean_theta, dtype=float).ravel()
        self.n_replicates = int(n_replicates)

    def fit_predict(
        self,
        y: np.ndarray,
        true_classes: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clusters observations `y` via K-Means++ (using either explicit `n_clusters`
        or oracle cluster count `len(np.unique(true_classes))`) and returns
        `(x_hat, cluster_labels)`.
        """
        y = np.asarray(y, dtype=float)
        n_samples = y.shape[0]

        if n_clusters is None:
            if true_classes is not None:
                n_clusters = len(np.unique(true_classes))
            else:
                n_clusters = max(1, int(np.sqrt(n_samples)))
        k = max(1, min(int(n_clusters), n_samples))

        if SKLEARN_AVAILABLE:
            km = KMeans(n_clusters=k, n_init=self.n_replicates, random_state=random_state)
            raw_labels = km.fit_predict(y)
        else:
            rng = np.random.default_rng(random_state)
            raw_labels = kmeans_scipy(y, k, replicates=self.n_replicates, rng=rng)

        clean_labels, _ = assign_outliers_and_reindex(raw_labels)
        x_hat = empirical_bayes_lmmse(
            y, clean_labels, self.var_v, self.var_u, self.var_theta, self.mean_theta
        )
        return x_hat, clean_labels