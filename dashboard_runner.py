"""
dashboard_runner.py — Unified CLI Benchmark Runner & Interactive Streamlit Dashboard
====================================================================================
Python / Machine Learning / Deep Learning Extension of the Master's Thesis:
"Advanced Bayesian Estimation in Hierarchical Gaussian Models:
 Dirichlet Process Mixtures and Clustering Gain"
Author: Dipl.-Ing. Erik Šauša (Institute of Telecommunications, TU Wien, March 2024)
Supervision: Ao.Univ.Prof. Dipl.-Ing. Dr.techn. Franz Hlawatsch, Dipl.-Ing. Thomas John Bucco

Orchestrates all modular DPMM files:
- `data_generation.py`            (Eq. 3.64-3.67, 4.1-4.8, 5.155 Scenario 4 LMMSE)
- `bounds.py`                     (Eq. 4.61 Scenario 1, Eq. 4.88 Scenario 2, Eq. 5.162 Scenario 4, Eq. 6.3 CG)
- `estimator_gibbs.py`            (Section 5.2.3 Algorithm 3 MacEachern Gibbs Sampler: SciPy & PyTorch)
- `estimator_dbscan.py`           (Section 6.3.2 Modified Scenario 4 with DBSCAN: MSE^(DB)_MC)
- `estimator_kmeans.py`           (Section 6.3.2 Modified Scenario 4 with K-Means++: MSE^(K)_MC)
- `estimator_variational_dFpmm.py` (Section 1.2 / Ch. 7 CAVI / Variational DPMM [15] & Ward Agglomerative)
- `estimator_neural.py`           (PyTorch Set-Attention Transformer & Pointwise Residual MLP)

How to Run from VS Code Terminal:
1. CLI Benchmark (Thesis Fig. 12b & 13b — MSE & CG vs. alpha, N=20):
   `python dashboard_runner.py --mode alpha --max-objects 20 --iterations 30 --save-plot fig12b_13b.png`
2. CLI Benchmark (Thesis Fig. 12a & 13a — MSE & CG vs. N, alpha=0.5):
   `python dashboard_runner.py --mode n-sweep --alpha 0.5 --max-objects 30 --iterations 30 --save-plot fig12a_13a.png`
3. Interactive Streamlit Cockpit:
   `streamlit run dashboard_runner.py`
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Sequence
import numpy as np
import matplotlib.pyplot as plt

# Ensure local modular imports resolve cleanly regardless of working directory in VS Code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bounds import (
    clustering_gain_db,
    immse,
    mmse_known_theta_bound,
    mmse_unknown_theta_bound,
    partition_mmse_bound,
)
from data_generation import DPMMData, generate_dpmm_data
from estimator_dbscan import DBSCANEstimator
from estimator_gibbs import DPMMGibbsSciPy
from estimator_kmeans import KMeansEstimator
from estimator_neural import NeuralDenoiserSuite
from estimator_variational_dpmm import AgglomerativeEstimator, VariationalDPMMEstimator


THESIS_METADATA = {
    "title": (
        "Advanced Bayesian Estimation in Hierarchical Gaussian Models: "
        "Dirichlet Process Mixtures and Clustering Gain"
    ),
    "author": "Dipl.-Ing. Erik Šauša",
    "institution": "Institute of Telecommunications, TU Wien (March 2024)",
    "supervisors": (
        "Ao.Univ.Prof. Dipl.-Ing. Dr.techn. Franz Hlawatsch & "
        "Dipl.-Ing. Thomas John Bucco"
    ),
    "abstract": (
        "This Master's thesis explores the use of a Dirichlet process (DP) prior to enhance "
        "Bayesian estimation of the parameters of multiple objects. Specifically, it focuses on "
        "a hierarchical Gaussian model where each object is linked to one parameter of interest, "
        "one noisy measurement, and one hyperparameter, and the hyperparameters are shared among "
        "objects within the same cluster. The model permits the derivation of closed-form "
        "performance bounds, enabling a quantification of performance improvements relative to "
        "the theoretically achievable performance. Our primary objective is to estimate the "
        "parameter of interest for each object based on its associated noisy measurement while "
        "leveraging the cluster structure of the hyperparameters. Because a closed-form "
        "calculation of the posterior distribution is not possible, we employ a Markov chain "
        "Monte Carlo sampling method to approximate the minimum mean square error (MMSE) "
        "estimator. This methodology yields an estimator that exploits the inherent cluster "
        "structure and, as we show through simulations, consistently achieves a mean squared "
        "error (MSE) that is lower than the MSE of the MMSE estimator for a scenario without a "
        "cluster structure. Additionally, we derive a closed-form MMSE estimator assuming known "
        "object-cluster associations and demonstrate its performance through simulations. Our "
        "approach of combining estimation and clustering demonstrates superior performance "
        "compared to the widely used method of first clustering and then performing estimation "
        "within each cluster; however, this performance advantage comes at the cost of a higher "
        "computational complexity."
    ),
}

METHOD_METADATA = [
    ("gibbs_dpmm", "Proposed DPMM Gibbs (Alg. 3)", "#1b9e77", "-o", 2.4),
    ("sklearn_vi_dpmm", "Variational DPMM (CAVI / Sklearn BGMM)", "#d95f02", "--s", 1.8),
    ("sklearn_kmeans", "K-Means++ (Oracle K)", "#7570b3", "-.^", 1.8),
    ("pytorch_set_attention", "PyTorch Set-Attention DNN", "#e7298a", "-v", 1.8),
    ("sklearn_agglomerative", "Agglomerative (Ward)", "#66a61e", ":P", 1.6),
    ("sklearn_dbscan", "DBSCAN (k-NN Knee)", "#e6ab02", "--d", 1.6),
    ("pytorch_pointwise_mlp", "PyTorch Pointwise MLP (Scenario 1 Limit)", "#a6761d", ":x", 1.6),
]


def evaluate_single_realization(
    max_objects: int = 20,
    dim: int = 2,
    alpha_dp: float = 1.5,
    sigma_theta_sq: float = 5.0,
    gibbs_iters: int = 400,
    burn_in: int = 150,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Replicates Figure 10 (p. 102 of Thesis: N=20, alpha=1) and evaluates all
    estimators on a single realization.
    """
    rng = np.random.default_rng(seed)
    var_v = np.eye(dim, dtype=float)
    var_u = np.eye(dim, dtype=float)
    var_theta = np.eye(dim, dtype=float) * float(sigma_theta_sq)
    mean_theta = np.zeros(dim, dtype=float)

    data: DPMMData = generate_dpmm_data(
        max_objects, alpha_dp, mean_theta, var_v, var_u, var_theta, rng=rng
    )

    gibbs_est = DPMMGibbsSciPy(
        var_v, var_u, var_theta, mean_theta, alpha_dp=alpha_dp, n_iters=gibbs_iters, burn_in=burn_in
    )
    vi_est = VariationalDPMMEstimator(var_v, var_u, var_theta, mean_theta, alpha_dp=alpha_dp)
    km_est = KMeansEstimator(var_v, var_u, var_theta, mean_theta)
    db_est = DBSCANEstimator(var_v, var_u, var_theta, mean_theta, alpha_dp=alpha_dp)
    agg_est = AgglomerativeEstimator(var_v, var_u, var_theta, mean_theta)

    neural_suite = NeuralDenoiserSuite(
        var_v, var_u, var_theta, mean_theta, alpha_dp=alpha_dp, max_n=max_objects
    ).fit(train_steps=180, rng=rng)

    x_gibbs, lbl_gibbs = gibbs_est.fit_predict(data.y, rng=rng)
    x_vi, lbl_vi = vi_est.fit_predict(data.y, random_state=seed)
    x_km, lbl_km = km_est.fit_predict(data.y, true_classes=data.classes, random_state=seed)
    x_db, lbl_db = db_est.fit_predict(data.y)
    x_agg, lbl_agg = agg_est.fit_predict(data.y)
    x_attn = neural_suite.predict_attention(data.y)
    x_mlp = neural_suite.predict_mlp(data.y)

    mmse_unk = mmse_unknown_theta_bound(var_v, var_u, var_theta)
    estimates = {
        "gibbs_dpmm": (x_gibbs, lbl_gibbs),
        "sklearn_vi_dpmm": (x_vi, lbl_vi),
        "sklearn_kmeans": (x_km, lbl_km),
        "pytorch_set_attention": (x_attn, data.classes),
        "sklearn_agglomerative": (x_agg, lbl_agg),
        "sklearn_dbscan": (x_db, lbl_db),
        "pytorch_pointwise_mlp": (x_mlp, np.arange(max_objects)),
    }

    metrics = {}
    for key, (x_hat, lbl) in estimates.items():
        m_val = immse(x_hat, data.x)
        metrics[key] = {
            "mse": m_val,
            "cg_db": float(clustering_gain_db(m_val, mmse_unk)),
            "num_clusters": int(len(np.unique(lbl))),
            "x_hat": x_hat,
            "labels": lbl,
        }

    return {
        "data": data,
        "metrics": metrics,
        "mmse_known": mmse_known_theta_bound(var_v, var_u),
        "mmse_unknown": mmse_unk,
        "mmse_partition_bound": partition_mmse_bound(var_theta, var_u, var_v, max_objects, alpha_dp),
    }


def run_modular_benchmark(
    max_objects: int = 20,
    dim: int = 2,
    iterations: int = 25,
    alpha_list: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0),
    sigma_theta_sq: float = 5.0,
    gibbs_iters: int = 300,
    burn_in: int = 100,
    save_plot_path: str = "",
) -> Dict[str, np.ndarray]:
    """
    Replicates Thesis Figures 12b & 13b (MSE and Clustering Gain vs. concentration
    parameter `alpha` at fixed `N = max_objects`).
    """
    alpha_arr = np.asarray(alpha_list, dtype=float)
    num_alphas = len(alpha_arr)

    var_v = np.eye(dim, dtype=float)
    var_u = np.eye(dim, dtype=float)
    var_theta = np.eye(dim, dtype=float) * float(sigma_theta_sq)
    mean_theta = np.zeros(dim, dtype=float)

    mmse_known = mmse_known_theta_bound(var_v, var_u)
    mmse_unknown = mmse_unknown_theta_bound(var_v, var_u, var_theta)

    method_keys = [m[0] for m in METHOD_METADATA]
    mse_records = {k: np.zeros((num_alphas, iterations), dtype=float) for k in method_keys}
    mse_bound = np.zeros(num_alphas, dtype=float)

    rng = np.random.default_rng(2026)

    for i, alpha_dp in enumerate(alpha_arr):
        gibbs_est = DPMMGibbsSciPy(
            var_v, var_u, var_theta, mean_theta, alpha_dp=float(alpha_dp), n_iters=gibbs_iters, burn_in=burn_in
        )
        vi_est = VariationalDPMMEstimator(var_v, var_u, var_theta, mean_theta, alpha_dp=float(alpha_dp))
        km_est = KMeansEstimator(var_v, var_u, var_theta, mean_theta)
        db_est = DBSCANEstimator(var_v, var_u, var_theta, mean_theta, alpha_dp=float(alpha_dp))
        agg_est = AgglomerativeEstimator(var_v, var_u, var_theta, mean_theta)

        neural_suite = NeuralDenoiserSuite(
            var_v, var_u, var_theta, mean_theta, alpha_dp=float(alpha_dp), max_n=max_objects
        ).fit(train_steps=180, rng=rng)

        for ii in range(iterations):
            data = generate_dpmm_data(
                max_objects, float(alpha_dp), mean_theta, var_v, var_u, var_theta, rng=rng
            )
            seed_ii = int(rng.integers(0, 2**31 - 1))

            x_gibbs, _ = gibbs_est.fit_predict(data.y, rng=rng)
            x_vi, _ = vi_est.fit_predict(data.y, random_state=seed_ii)
            x_km, _ = km_est.fit_predict(data.y, true_classes=data.classes, random_state=seed_ii)
            x_db, _ = db_est.fit_predict(data.y)
            x_agg, _ = agg_est.fit_predict(data.y)
            x_attn = neural_suite.predict_attention(data.y)
            x_mlp = neural_suite.predict_mlp(data.y)

            mse_records["gibbs_dpmm"][i, ii] = immse(x_gibbs, data.x)
            mse_records["sklearn_vi_dpmm"][i, ii] = immse(x_vi, data.x)
            mse_records["sklearn_kmeans"][i, ii] = immse(x_km, data.x)
            mse_records["sklearn_dbscan"][i, ii] = immse(x_db, data.x)
            mse_records["sklearn_agglomerative"][i, ii] = immse(x_agg, data.x)
            mse_records["pytorch_set_attention"][i, ii] = immse(x_attn, data.x)
            mse_records["pytorch_pointwise_mlp"][i, ii] = immse(x_mlp, data.x)

        mse_bound[i] = partition_mmse_bound(var_theta, var_u, var_v, max_objects, float(alpha_dp))

    summary: Dict[str, np.ndarray] = {
        "x_axis": alpha_arr,
        "x_label": r"DP Concentration Parameter ($\alpha$)",
        "alpha": alpha_arr,
        "mse_bound": mse_bound,
        "cg_bound": clustering_gain_db(mse_bound, mmse_unknown),
        "mmse_known_theta": np.array(mmse_known),
        "mmse_unknown_theta": np.array(mmse_unknown),
    }
    for key in method_keys:
        mean_mse = np.mean(mse_records[key], axis=1)
        summary[f"mse_{key}"] = mean_mse
        summary[f"cg_{key}"] = clustering_gain_db(mean_mse, mmse_unknown)

    if save_plot_path:
        fig = plot_benchmark_curves(summary)
        fig.savefig(save_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    return summary


def run_n_sweep_benchmark(
    n_list: Sequence[int] = (2, 5, 10, 15, 20, 30),
    alpha_dp: float = 0.5,
    dim: int = 2,
    iterations: int = 25,
    sigma_theta_sq: float = 5.0,
    gibbs_iters: int = 300,
    burn_in: int = 100,
    save_plot_path: str = "",
) -> Dict[str, np.ndarray]:
    """
    Replicates Thesis Figures 11, 12a & 13a (MSE and Clustering Gain vs. number
    of objects `N` at fixed `alpha = 0.5`).
    """
    n_arr = np.asarray(n_list, dtype=int)
    num_n = len(n_arr)

    var_v = np.eye(dim, dtype=float)
    var_u = np.eye(dim, dtype=float)
    var_theta = np.eye(dim, dtype=float) * float(sigma_theta_sq)
    mean_theta = np.zeros(dim, dtype=float)

    mmse_known = mmse_known_theta_bound(var_v, var_u)
    mmse_unknown = mmse_unknown_theta_bound(var_v, var_u, var_theta)

    method_keys = [m[0] for m in METHOD_METADATA]
    mse_records = {k: np.zeros((num_n, iterations), dtype=float) for k in method_keys}
    mse_bound = np.zeros(num_n, dtype=float)

    rng = np.random.default_rng(2026)

    gibbs_est = DPMMGibbsSciPy(
        var_v, var_u, var_theta, mean_theta, alpha_dp=alpha_dp, n_iters=gibbs_iters, burn_in=burn_in
    )
    vi_est = VariationalDPMMEstimator(var_v, var_u, var_theta, mean_theta, alpha_dp=alpha_dp)
    km_est = KMeansEstimator(var_v, var_u, var_theta, mean_theta)
    db_est = DBSCANEstimator(var_v, var_u, var_theta, mean_theta, alpha_dp=alpha_dp)
    agg_est = AgglomerativeEstimator(var_v, var_u, var_theta, mean_theta)

    for i, n_obj in enumerate(n_arr):
        neural_suite = NeuralDenoiserSuite(
            var_v, var_u, var_theta, mean_theta, alpha_dp=alpha_dp, max_n=int(n_obj)
        ).fit(train_steps=150, rng=rng)

        for ii in range(iterations):
            data = generate_dpmm_data(
                int(n_obj), alpha_dp, mean_theta, var_v, var_u, var_theta, rng=rng
            )
            seed_ii = int(rng.integers(0, 2**31 - 1))

            x_gibbs, _ = gibbs_est.fit_predict(data.y, rng=rng)
            x_vi, _ = vi_est.fit_predict(data.y, random_state=seed_ii)
            x_km, _ = km_est.fit_predict(data.y, true_classes=data.classes, random_state=seed_ii)
            x_db, _ = db_est.fit_predict(data.y)
            x_agg, _ = agg_est.fit_predict(data.y)
            x_attn = neural_suite.predict_attention(data.y)
            x_mlp = neural_suite.predict_mlp(data.y)

            mse_records["gibbs_dpmm"][i, ii] = immse(x_gibbs, data.x)
            mse_records["sklearn_vi_dpmm"][i, ii] = immse(x_vi, data.x)
            mse_records["sklearn_kmeans"][i, ii] = immse(x_km, data.x)
            mse_records["sklearn_dbscan"][i, ii] = immse(x_db, data.x)
            mse_records["sklearn_agglomerative"][i, ii] = immse(x_agg, data.x)
            mse_records["pytorch_set_attention"][i, ii] = immse(x_attn, data.x)
            mse_records["pytorch_pointwise_mlp"][i, ii] = immse(x_mlp, data.x)

        mse_bound[i] = partition_mmse_bound(var_theta, var_u, var_v, int(n_obj), alpha_dp)

    summary: Dict[str, np.ndarray] = {
        "x_axis": n_arr,
        "x_label": "Number of Objects (N)",
        "objects": n_arr,
        "mse_bound": mse_bound,
        "cg_bound": clustering_gain_db(mse_bound, mmse_unknown),
        "mmse_known_theta": np.array(mmse_known),
        "mmse_unknown_theta": np.array(mmse_unknown),
    }
    for key in method_keys:
        mean_mse = np.mean(mse_records[key], axis=1)
        summary[f"mse_{key}"] = mean_mse
        summary[f"cg_{key}"] = clustering_gain_db(mean_mse, mmse_unknown)

    if save_plot_path:
        fig = plot_benchmark_curves(summary)
        fig.savefig(save_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    return summary


def plot_benchmark_curves(summary: Dict[str, np.ndarray]) -> plt.Figure:
    """Creates a two-panel Matplotlib figure for MSE and Clustering Gain (dB)."""
    x_arr = summary.get("x_axis", summary.get("alpha"))
    x_label = str(summary.get("x_label", r"DP Concentration Parameter ($\alpha$)"))
    mmse_unk = float(summary["mmse_unknown_theta"])
    mmse_kn = float(summary["mmse_known_theta"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    ax.axhline(mmse_unk, color="#333333", linestyle="--", linewidth=1.3, label=r"Scenario 1: $\mathrm{MSE}_{\mathrm{min}}^{(1)}$ (Eq. 4.61)")
    ax.axhline(mmse_kn, color="#333333", linestyle=":", linewidth=1.3, label=r"Scenario 2: $\mathrm{MSE}_{\mathrm{min}}^{(2)}$ (Eq. 4.88)")
    ax.plot(x_arr, summary["mse_bound"], color="#111111", linestyle="-", linewidth=2.0, label=r"Scenario 4: $\mathrm{MSE}_{\mathrm{min}}^{(4)}$ (Eq. 5.162)")
    for key, label, color, fmt, lw in METHOD_METADATA:
        ax.plot(x_arr, summary[f"mse_{key}"], fmt, color=color, linewidth=lw, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("State Estimation MSE Comparison")
    ax.legend(fontsize=8.0, loc="best")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2)
    ax2.plot(x_arr, summary["cg_bound"], color="#111111", linestyle="-", linewidth=2.0, label=r"Scenario 4 Bound: $\mathrm{CG}^{(4)}$ (Eq. 6.4)")
    for key, label, color, fmt, lw in METHOD_METADATA:
        ax2.plot(x_arr, summary[f"cg_{key}"], fmt, color=color, linewidth=lw, label=label)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Clustering Gain (dB)")
    ax2.set_title("Clustering Gain (Eq. 6.3, dB)")
    ax2.legend(fontsize=8.0, loc="best")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_single_realization_2d(single_res: Dict[str, object]) -> plt.Figure:
    """Plots a 2D spatial comparison of true states vs. estimated states (Thesis Fig. 10)."""
    data: DPMMData = single_res["data"]
    metrics = single_res["metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
    panels = [
        ("gibbs_dpmm", "Proposed DPMM Gibbs (Alg. 3)"),
        ("sklearn_vi_dpmm", "CAVI / Sklearn Variational DPMM"),
        ("sklearn_kmeans", " K-Means++"),
    ]

    for ax, (key, title) in zip(axes, panels):
        x_hat = metrics[key]["x_hat"]
        mse_val = metrics[key]["mse"]
        cg_val = metrics[key]["cg_db"]

        ax.scatter(data.y[:, 0], data.y[:, 1], c="#94a3b8", marker="x", s=45, label="Noisy Obs $y_n$")
        ax.scatter(data.x[:, 0], data.x[:, 1], c="#2563eb", marker="o", s=55, alpha=0.8, label="True State $x_n$")
        ax.scatter(x_hat[:, 0], x_hat[:, 1], c="#16a34a", marker="^", s=60, label=r"Estimated $\hat{x}_n$")
        ax.scatter(
            data.theta[:, 0], data.theta[:, 1], c="#eab308", edgecolors="k", marker="*", s=160, label=r"Cluster $\theta^*_l$"
        )
        for n in range(data.x.shape[0]):
            ax.plot([data.x[n, 0], x_hat[n, 0]], [data.x[n, 1], x_hat[n, 1]], "k-", alpha=0.25, linewidth=0.9)

        ax.set_title(f"{title}\nMSE: {mse_val:.3f} ({cg_val:+.2f} dB)")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    return fig


def run_streamlit_dashboard() -> None:
    """Interactive Streamlit UI (`streamlit run dashboard_runner.py`)."""
    import streamlit as st

    st.set_page_config(page_title="DPMM Bayesian Estimation Cockpit", layout="wide")
    st.title("Nonparametric Bayesian DPMM State Estimation Cockpit")
    st.markdown(
        f"**Python, Machine Learning & Deep Learning Extension of the TU Wien Master's Thesis**  \n"
        f"*{THESIS_METADATA['title']}* — **{THESIS_METADATA['author']}** "
        f"({THESIS_METADATA['institution']}; Supervision: {THESIS_METADATA['supervisors']})"
    )

    with st.expander("Master's Thesis Abstract & Python/ML Extension Overview", expanded=True):
        col_abs, col_ext = st.columns([1.25, 1.0])
        with col_abs:
            st.markdown("#### Master's Thesis Abstract (TU Wien, March 2024)")
            st.markdown(f"> {THESIS_METADATA['abstract']}")
        with col_ext:
            st.markdown("#### What This Python / ML Extension Adds")
            st.markdown(
                "- **Vectorized SciPy & PyTorch MCMC Engine (`estimator_gibbs.py`)**: "
                "Refactors Scenario 3 MacEachern Indicator Gibbs Sampling (Algorithm 3, Eq. 5.77–5.110) "
                "using Cholesky SPD solvers (`cho_factor`/`cho_solve`), log-domain CRP weights (`logsumexp`), "
                "and GPU-ready `torch.distributions`.\n"
                "- **Overflow-Free Analytical Bounds (`bounds.py`)**: "
                "Evaluates Scenario 1 ($\\mathrm{MSE}_{\\mathrm{min}}^{(1)}$, Eq. 4.61), "
                "Scenario 2 ($\\mathrm{MSE}_{\\mathrm{min}}^{(2)}$, Eq. 4.88), and Scenario 4 "
                "($\\mathrm{MSE}_{\\mathrm{min}}^{(4)}$, Eq. 5.162) in log-gamma space (`gammaln`) for arbitrary $N$.\n"
                "- **Variational Inference & Classical Clustering (`estimator_variational_dpmm.py`, `estimator_dbscan.py`, `estimator_kmeans.py`)**: "
                "Benchmarks Coordinate Ascent Variational Inference (`BayesianGaussianMixture`, proposed as future work in Ch. 7) "
                "alongside Ward Agglomerative, DBSCAN ($\\mathrm{MSE}_{\\mathrm{MC}}^{(\\mathrm{DB})}$), and K-Means++ ($\\mathrm{MSE}_{\\mathrm{MC}}^{(\\mathrm{K})}$).\n"
                "- **Deep Learning Set-Attention Denoisers (`estimator_neural.py`)**: "
                "Trains a permutation-equivariant Multi-Head `SetAttentionDenoiser` (`nn.TransformerEncoderLayer`) "
                "against a pointwise `ResidualMLPDenoiser` to show why cross-object context is required for positive Clustering Gain (Eq. 6.3)."
            )

    with st.sidebar:
        st.header("Simulation Setup (Table 2)")
        max_objects = st.slider("Number of Objects (N)", min_value=4, max_value=50, value=20, step=2)
        alpha_dp = st.slider("DP Concentration (alpha)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        sigma_theta_sq = st.slider("Cluster Prior Variance (sigma_theta^2)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
        gibbs_iters = st.slider("Gibbs MCMC Samples (Q)", min_value=100, max_value=1000, value=400, step=50)
        seed = st.number_input("Random Seed", min_value=1, max_value=99999, value=42)
        mc_trials = st.slider("Monte Carlo Runs (J)", min_value=5, max_value=100, value=20, step=5)

    tab1, tab2, tab3 = st.tabs([
        "Fig. 10: Single-Realization Spatial Inspector",
        "Fig. 12b & 13b: MSE & CG vs. Alpha",
        "Fig. 12a & 13a: MSE & CG vs. Number of Objects (N)",
    ])

    with tab1:
        res_single = evaluate_single_realization(
            max_objects=max_objects,
            dim=2,
            alpha_dp=alpha_dp,
            sigma_theta_sq=sigma_theta_sq,
            gibbs_iters=gibbs_iters,
            burn_in=max(20, gibbs_iters // 3),
            seed=int(seed),
        )
        gibbs_m = res_single["metrics"]["gibbs_dpmm"]
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Scenario 1 Unclustered Bound", f"{res_single['mmse_unknown']:.4f}")
        kpi2.metric("Scenario 4 Partition Bound", f"{res_single['mmse_partition_bound']:.4f}")
        kpi3.metric("Scenario 3 Gibbs MSE", f"{gibbs_m['mse']:.4f}")
        kpi4.metric("Gibbs Clustering Gain (Eq. 6.3)", f"{gibbs_m['cg_db']:+.2f} dB")

        st.pyplot(plot_single_realization_2d(res_single))

        rows = []
        for key, label, _, _, _ in METHOD_METADATA:
            m = res_single["metrics"][key]
            rows.append({
                "Estimator": label,
                "MSE": round(m["mse"], 4),
                "Clustering Gain (dB)": round(m["cg_db"], 2),
                "Inferred Clusters S(N)": m["num_clusters"],
            })
        st.table(rows)

    with tab2:
        if st.button("Run Alpha Sweep (Figures 12b & 13b)"):
            with st.spinner("Running Monte Carlo simulations across alpha..."):
                summary_a = run_modular_benchmark(
                    max_objects=max_objects,
                    dim=2,
                    iterations=mc_trials,
                    sigma_theta_sq=sigma_theta_sq,
                    gibbs_iters=gibbs_iters,
                    burn_in=max(20, gibbs_iters // 3),
                )
            st.pyplot(plot_benchmark_curves(summary_a))

    with tab3:
        if st.button("Run N-Objects Sweep (Figures 12a & 13a)"):
            with st.spinner("Running Monte Carlo simulations across N..."):
                summary_n = run_n_sweep_benchmark(
                    n_list=(2, 5, 10, 15, 20, 30),
                    alpha_dp=alpha_dp,
                    dim=2,
                    iterations=mc_trials,
                    sigma_theta_sq=sigma_theta_sq,
                    gibbs_iters=gibbs_iters,
                    burn_in=max(20, gibbs_iters // 3),
                )
            st.pyplot(plot_benchmark_curves(summary_n))


def main() -> None:
    parser = argparse.ArgumentParser(description="Modular DPMM Thesis Benchmark Runner")
    parser.add_argument("--mode", choices=["alpha", "n-sweep"], default="alpha", help="Sweep 'alpha' (Fig 12b) or 'n-sweep' (Fig 12a)")
    parser.add_argument("--max-objects", type=int, default=10, help="Number of objects N (for alpha sweep)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fixed alpha (for n-sweep)")
    parser.add_argument("--dim", type=int, default=2, help="State dimension D")
    parser.add_argument("--iterations", type=int, default=15, help="Monte Carlo runs J")
    parser.add_argument("--gibbs-iters", type=int, default=250, help="Gibbs MCMC samples Q")
    parser.add_argument("--burn-in", type=int, default=80, help="Gibbs burn-in samples")
    parser.add_argument("--save-plot", type=str, default="", help="Optional path to save PNG plot")
    args, _ = parser.parse_known_args()

    if args.mode == "n-sweep":
        print(f"Running N-Sweep (Thesis Fig. 12a & 13a: alpha={args.alpha}, dim={args.dim}, J={args.iterations})...")
        summary = run_n_sweep_benchmark(
            n_list=(2, 5, 10, 15, 20),
            alpha_dp=args.alpha,
            dim=args.dim,
            iterations=args.iterations,
            gibbs_iters=args.gibbs_iters,
            burn_in=args.burn_in,
            save_plot_path=args.save_plot,
        )
        x_key, col_title = "objects", "    N"
    else:
        print(f"Running Alpha-Sweep (Thesis Fig. 12b & 13b: N={args.max_objects}, dim={args.dim}, J={args.iterations})...")
        summary = run_modular_benchmark(
            max_objects=args.max_objects,
            dim=args.dim,
            iterations=args.iterations,
            gibbs_iters=args.gibbs_iters,
            burn_in=args.burn_in,
            save_plot_path=args.save_plot,
        )
        x_key, col_title = "alpha", "Alpha"

    header = (
        f"{col_title:>5} | {'Scen.4':>6} | {'Scen.3 Gibbs':>12} | {'CAVI/BGMM':>10} | "
        f"{'K-Means++':>10} | {'Torch-Attn':>10} | {'Agglom.':>8} | {'DBSCAN':>8} | {'Torch-MLP':>9}"
    )
    print(header)
    print("-" * len(header))
    for i, val in enumerate(summary[x_key]):
        print(
            f"{val:5.1f} | {summary['mse_bound'][i]:6.4f} | "
            f"{summary['mse_gibbs_dpmm'][i]:12.4f} | "
            f"{summary['mse_sklearn_vi_dpmm'][i]:10.4f} | "
            f"{summary['mse_sklearn_kmeans'][i]:10.4f} | "
            f"{summary['mse_pytorch_set_attention'][i]:10.4f} | "
            f"{summary['mse_sklearn_agglomerative'][i]:8.4f} | "
            f"{summary['mse_sklearn_dbscan'][i]:8.4f} | "
            f"{summary['mse_pytorch_pointwise_mlp'][i]:9.4f}"
        )


if __name__ == "__main__":
    if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT"):
        run_streamlit_dashboard()
    else:
        main()