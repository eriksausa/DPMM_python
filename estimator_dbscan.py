"""
estimator_dbscan.py — Case 3b DBSCAN Empirical Bayes LMMSE Estimator
====================================================================
Refactored from `sim_3b.m`.
Uses `sklearn.cluster.DBSCAN` + `sklearn.neighbors.NearestNeighbors` when
`scikit-learn` is installed, with an automatic pure-`SciPy` fallback built on
`scipy.spatial.KDTree` (`query` and `query_ball_point`).
"""

from __future__ import annotations

from collections import deque
from typing import Tuple
import numpy as np
from scipy.spatial import KDTree

from data_generation import assign_outliers_and_reindex, empirical_bayes_lmmse

try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def estimate_epsilon_knee(y: np.ndarray, min_pts: int, max_pts: int) -> float:
    """
    Replicates MATLAB's `clusterDBSCAN.estimateEpsilon(y, min_pts, max_pts)`
    using sorted k-NN distance curves across k in `[min_pts, max_pts]` and
    maximum perpendicular distance (knee detection).
    """
    y_2d = np.atleast_2d(np.asarray(y, dtype=float))
    if y_2d.shape[0] == 1 and y_2d.shape[1] > 1 and np.asarray(y).ndim == 1:
        y_2d = y_2d.T

    n_samples = y_2d.shape[0]
    if n_samples <= 1:
        return 1.0

    k_min = max(1, min(int(min_pts), n_samples - 1))
    k_max = max(k_min, min(int(max_pts), n_samples - 1))

    if SKLEARN_AVAILABLE:
        nbrs = NearestNeighbors(n_neighbors=k_max + 1, algorithm="auto").fit(y_2d)
        distances, _ = nbrs.kneighbors(y_2d)
        distances = distances[:, k_min : k_max + 1]
    else:
        tree = KDTree(y_2d)
        distances, _ = tree.query(y_2d, k=list(range(k_min + 1, k_max + 2)))
        if distances.ndim == 1:
            distances = distances.reshape(-1, 1)

    knees = []
    for col_idx in range(distances.shape[1]):
        k_distances = np.sort(distances[:, col_idx])
        if len(k_distances) <= 2:
            knees.append(float(np.mean(k_distances)))
            continue

        coords = np.vstack((np.arange(len(k_distances), dtype=float), k_distances)).T
        first_pt = coords[0]
        line_vec = coords[-1] - first_pt
        line_norm = np.linalg.norm(line_vec)
        if line_norm < 1e-6:
            knees.append(float(k_distances[-1]))
            continue

        line_unit = line_vec / line_norm
        vec_from_first = coords - first_pt
        proj = np.outer(np.dot(vec_from_first, line_unit), line_unit)
        perp_dist = np.linalg.norm(vec_from_first - proj, axis=1)
        knees.append(float(k_distances[int(np.argmax(perp_dist))]))

    eps = float(np.mean(knees)) if knees else 1.0
    return eps if eps > 1e-4 else 1.0


def dbscan_kdtree(y: np.ndarray, epsilon: float, min_pts: int) -> np.ndarray:
    """
    SciPy `KDTree.query_ball_point` implementation of DBSCAN.
    Returns 0-indexed cluster IDs and `-1` for noise/outliers.
    """
    y_2d = np.asarray(y, dtype=float)
    if y_2d.ndim == 1:
        y_2d = y_2d.reshape(-1, 1)

    n_samples = y_2d.shape[0]
    tree = KDTree(y_2d)
    neighborhoods = tree.query_ball_point(y_2d, r=float(epsilon))

    labels = np.full(n_samples, -1, dtype=int)
    visited = np.zeros(n_samples, dtype=bool)
    cluster_id = 0

    for i in range(n_samples):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = neighborhoods[i]
        if len(neighbors) < min_pts:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = deque(nb for nb in neighbors if nb != i)
            in_queue = set(queue)
            while queue:
                curr = queue.popleft()
                if not visited[curr]:
                    visited[curr] = True
                    curr_neighbors = neighborhoods[curr]
                    if len(curr_neighbors) >= min_pts:
                        for nb in curr_neighbors:
                            if not visited[nb] and nb not in in_queue:
                                queue.append(nb)
                                in_queue.add(nb)
                if labels[curr] == -1:
                    labels[curr] = cluster_id
            cluster_id += 1

    return labels


class DBSCANEstimator:
    """
    Case 3b DBSCAN Plug-in LMMSE State Estimator (`sim_3b.m`).
    """

    def __init__(
        self,
        var_v: np.ndarray,
        var_u: np.ndarray,
        var_theta: np.ndarray,
        mean_theta: np.ndarray,
        alpha_dp: float = 1.0,
    ):
        self.var_v = np.asarray(var_v, dtype=float)
        self.var_u = np.asarray(var_u, dtype=float)
        self.var_theta = np.asarray(var_theta, dtype=float)
        self.mean_theta = np.asarray(mean_theta, dtype=float).ravel()
        self.dim = self.mean_theta.shape[0]
        self.alpha_dp = float(alpha_dp)

    def fit_predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clusters observations `y` via DBSCAN and returns `(x_hat, cluster_labels)`.
        """
        y = np.asarray(y, dtype=float)
        n_samples = y.shape[0]

        if n_samples == 1:
            raw_labels = np.array([0], dtype=int)
        else:
            min_pts = 2 * self.dim
            if int(np.ceil(np.log(n_samples))) <= 2:
                eps = 1.0
            else:
                max_pts = max(min_pts, int(np.ceil(self.alpha_dp * np.log(n_samples))))
                eps = estimate_epsilon_knee(y, min_pts, max_pts)

            if SKLEARN_AVAILABLE:
                raw_labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(y)
            else:
                raw_labels = dbscan_kdtree(y, eps, min_pts)

        clean_labels, _ = assign_outliers_and_reindex(raw_labels)
        x_hat = empirical_bayes_lmmse(
            y, clean_labels, self.var_v, self.var_u, self.var_theta, self.mean_theta
        )
        return x_hat, clean_labels