from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bounds import (
    clustering_gain_db,
    compute_all_bounds,
    immse,
    mmse_known_theta_bound,
    mmse_unknown_theta_bound,
    partition_mmse_bound,
)
from data_generation import (
    assign_outliers_and_reindex,
    empirical_bayes_lmmse,
    generate_dpmm_data,
    sample_crp_tables,
    sym_solve,
)
from estimator_dbscan import DBSCANEstimator, estimate_epsilon_knee
from estimator_gibbs import DPMMGibbsSciPy
from estimator_kmeans import KMeansEstimator
from estimator_variational_dpmm import AgglomerativeEstimator, VariationalDPMMEstimator


def test_sym_solve_matches_explicit_inverse():
    cov = np.array([[2.5, 0.4], [0.4, 1.8]], dtype=float)
    rhs = np.array([[1.0, -0.5], [0.2, 3.0]], dtype=float)
    np.testing.assert_allclose(sym_solve(cov, rhs), np.linalg.inv(cov) @ rhs, rtol=1e-10)


def test_crp_and_data_generation_shapes():
    rng = np.random.default_rng(42)
    c, k = sample_crp_tables(25, alpha_dp=0.8, rng=rng)
    assert c.shape == (25,)
    assert k == len(np.unique(c))

    data = generate_dpmm_data(
        max_n=20,
        alpha_dp=0.5,
        mean_theta=np.zeros(2),
        var_v=np.eye(2),
        var_u=np.eye(2),
        var_theta=5.0 * np.eye(2),
        rng=rng,
    )
    assert data.y.shape == (20, 2)
    assert data.x.shape == (20, 2)
    assert data.classes.shape == (20,)


def test_mmse_bounds_ordering():
    var_v = np.eye(2)
    var_u = np.eye(2)
    var_theta = 5.0 * np.eye(2)
    b = compute_all_bounds(var_theta, var_u, var_v, n_objects=20, alpha_dp=0.5)
    assert b["mmse_known_theta"] < b["mmse_partition_bound"] < b["mmse_unknown_theta"]
    assert b["cg_partition_bound_db"] > 0.0


def test_dbscan_all_outlier_reindexing_and_tight_clusters():
    c, m = assign_outliers_and_reindex(np.array([-1, -1, -1]))
    assert np.array_equal(c, np.array([0, 1, 2]))
    assert np.array_equal(m, np.array([1, 1, 1]))
    tight_y = np.array([[0.0, 0.0], [1e-6, 0.0], [2e-6, 1e-6], [1.0, 1.0]])
    eps = estimate_epsilon_knee(tight_y, 1, 2)
    assert eps > 1e-8


def test_classical_estimators_shapes():
    rng = np.random.default_rng(7)
    v = np.eye(2)
    vt = 5.0 * np.eye(2)
    m = np.zeros(2)
    data = generate_dpmm_data(12, 0.5, m, v, v, vt, rng=rng)

    for est in [
        DBSCANEstimator(v, v, vt, m, alpha_dp=0.5),
        KMeansEstimator(v, v, vt, m),
        VariationalDPMMEstimator(v, v, vt, m, alpha_dp=0.5),
        AgglomerativeEstimator(v, v, vt, m),
    ]:
        x_hat, lbl = est.fit_predict(data.y)
        assert x_hat.shape == (12, 2)
        assert lbl.shape == (12,)