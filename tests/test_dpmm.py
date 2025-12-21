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