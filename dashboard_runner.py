"""
dashboard_runner.py — Unified CLI Benchmark Runner & Interactive Streamlit Dashboard
====================================================================================
Orchestrates all modular DPMM files:
- `data_generation.py`          (Synthetic CRP-DPMM data & Case 3b LMMSE core)
- `bounds.py`                   (Analytical MMSE bounds & Clustering Gain dB)
- `estimator_gibbs.py`          (Proposed MCMC Indicator Gibbs Sampler: SciPy & PyTorch)
- `estimator_dbscan.py`         (DBSCAN Case 3b LMMSE estimator)
- `estimator_kmeans.py`         (Oracle-K K-Means++ Case 3b LMMSE estimator)
- `estimator_variational_dpmm.py` (Scikit-Learn Variational DPMM & Agglomerative)
- `estimator_neural.py`         (PyTorch Set-Attention & Pointwise Residual MLP)

Usage:
1. Command-Line Benchmark Runner:
   `python3 dashboard_runner.py --iterations 30 --max-objects 10`
2. Interactive Streamlit Cockpit:
   `streamlit run dashboard_runner.py`
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import sys
from typing import Dict, Sequence
import numpy as np
import matplotlib.pyplot as plt

# Ensure local modular imports resolve cleanly
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


METHOD_METADATA = [
    ("gibbs_dpmm", "Proposed DPMM Gibbs Sampler (MCMC)", "#1b9e77", "-o", 2.4),
    ("sklearn_vi_dpmm", "Sklearn Variational DPMM (BGMM)", "#d95f02", "--s", 1.8),
    ("sklearn_kmeans", "Sklearn K-Means++ (Oracle K)", "#7570b3", "-.^", 1.8),
    ("pytorch_set_attention", "PyTorch Set-Attention DNN", "#e7298a", "-v", 1.8),
    ("sklearn_agglomerative", "Sklearn Agglomerative (Ward)", "#66a61e", ":P", 1.6),
    ("sklearn_dbscan", "Sklearn DBSCAN (k-NN Knee)", "#e6ab02", "--d", 1.6),
    ("pytorch_pointwise_mlp", "PyTorch Pointwise MLP", "#a6761d", ":x", 1.6),
]


def evaluate_single_realization(
    max_objects: int = 12,
    dim: int = 2,
    alpha_dp: float = 0.8,
    sigma_theta_sq: float = 5.0,
    gibbs_iters: int = 350,
    burn_in: int = 120,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Generates one 2D/ND DPMM realization and runs every modular estimator on it
    for visual inspection in the dashboard or CLI.
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
    max_objects: int = 10,
    dim: int = 2,
    iterations: int = 25,
    alpha_list: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0),
    sigma_theta_sq: float = 5.0,
    gibbs_iters: int = 300,
    burn_in: int = 100,
    save_plot_path: str = "",
) -> Dict[str, np.ndarray]:
    """
    Runs a Monte Carlo sweep across `alpha_list` invoking every modular estimator.
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

    rng = np.random.default_rng(1000)

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
        fig.savefig(save_plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    return summary


def plot_benchmark_curves(summary: Dict[str, np.ndarray]) -> plt.Figure:
    """Creates a two-panel Matplotlib figure for MSE and Clustering Gain (dB)."""
    alpha_arr = summary["alpha"]
    mmse_unk = float(summary["mmse_unknown_theta"])
    mmse_kn = float(summary["mmse_known_theta"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    ax.axhline(mmse_unk, color="#333333", linestyle="--", linewidth=1.3, label=r"$\mathrm{MSE}_{\mathrm{min}}^{(1)}$ (Unclustered)")
    ax.axhline(mmse_kn, color="#333333", linestyle=":", linewidth=1.3, label=r"$\mathrm{MSE}_{\mathrm{min}}^{(2)}$ (Known $\theta$)")
    ax.plot(alpha_arr, summary["mse_bound"], color="#111111", linestyle="-", linewidth=2.0, label=r"$\mathrm{MSE}_{\mathrm{min}}^{(4)}$ (Partition Bound)")
    for key, label, color, fmt, lw in METHOD_METADATA:
        ax.plot(alpha_arr, summary[f"mse_{key}"], fmt, color=color, linewidth=lw, label=label)
    ax.set_xlabel(r"DP Concentration Parameter ($\alpha$)")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("State Estimation MSE vs. Concentration Parameter")
    ax.legend(fontsize=8.2, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2)
    ax2.plot(alpha_arr, summary["cg_bound"], color="#111111", linestyle="-", linewidth=2.0, label="Analytical Partition Bound")
    for key, label, color, fmt, lw in METHOD_METADATA:
        ax2.plot(alpha_arr, summary[f"cg_{key}"], fmt, color=color, linewidth=lw, label=label)
    ax2.set_xlabel(r"DP Concentration Parameter ($\alpha$)")
    ax2.set_ylabel("Clustering Gain (dB)")
    ax2.set_title("Clustering Gain over Unclustered LMMSE (dB)")
    ax2.legend(fontsize=8.2, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_single_realization_2d(single_res: Dict[str, object]) -> plt.Figure:
    """Plots a 2D spatial comparison of true states vs. estimated states."""
    data: DPMMData = single_res["data"]
    metrics = single_res["metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
    panels = [
        ("gibbs_dpmm", "Proposed DPMM Gibbs Sampler"),
        ("sklearn_vi_dpmm", "Sklearn Variational DPMM"),
        ("sklearn_kmeans", "Sklearn K-Means++ (Oracle K)"),
    ]

    for ax, (key, title) in zip(axes, panels):
        x_hat = metrics[key]["x_hat"]
        mse_val = metrics[key]["mse"]
        cg_val = metrics[key]["cg_db"]

        ax.scatter(data.y[:, 0], data.y[:, 1], c="#94a3b8", marker="x", s=45, label="Noisy Obs $y_n$")
        ax.scatter(data.x[:, 0], data.x[:, 1], c="#2563eb", marker="o", s=55, alpha=0.8, label="True State $x_n$")
        ax.scatter(x_hat[:, 0], x_hat[:, 1], c="#16a34a", marker="^", s=60, label=r"Estimated $\hat{x}_n$")
        ax.scatter(
            data.theta[:, 0], data.theta[:, 1], c="#eab308", edgecolors="k", marker="*", s=160, label=r"Cluster $\theta^*_c$"
        )
        for n in range(data.x.shape[0]):
            ax.plot([data.x[n, 0], x_hat[n, 0]], [data.x[n, 1], x_hat[n, 1]], "k-", alpha=0.25, linewidth=0.9)

        ax.set_title(f"{title}\nMSE: {mse_val:.3f} ({cg_val:+.2f} dB)")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="best")
# 
    fig.tight_layout()
    return fig


def run_streamlit_dashboard() -> None:
    """Interactive Streamlit UI (`streamlit run dashboard_runner.py`)."""
    import streamlit as st

    st.set_page_config(page_title="DPMM State Estimation Cockpit", layout="wide")
    st.title("Nonparametric Bayesian DPMM State Estimation Cockpit")
    st.caption(
        "Modular Comparison: MCMC Indicator Gibbs vs. Scikit-Learn (BGMM, KMeans, DBSCAN, Agglomerative) & PyTorch DNNs"
    )

    with st.sidebar:
        st.header("Simulation Parameters")
        max_objects = st.slider("Number of Targets (N)", min_value=4, max_value=40, value=12, step=2)
        alpha_dp = st.slider("DP Concentration (alpha)", min_value=0.1, max_value=5.0, value=0.8, step=0.1)
        sigma_theta_sq = st.slider("Cluster Prior Variance (sigma_theta^2)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
        gibbs_iters = st.slider("Gibbs MCMC Iterations", min_value=100, max_value=1000, value=350, step=50)
        seed = st.number_input("Random Seed", min_value=1, max_value=99999, value=42)
        mc_trials = st.slider("Monte Carlo Sweep Trials", min_value=5, max_value=100, value=20, step=5)

    tab1, tab2 = st.tabs(["Single-Realization Spatial Inspector", "Monte Carlo Alpha Benchmark"])

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
        st.pyplot(plot_single_realization_2d(res_single))

        rows = []
        for key, label, _, _, _ in METHOD_METADATA:
            m = res_single["metrics"][key]
            rows.append({
                "Estimator": label,
                "MSE": round(m["mse"], 4),
                "Clustering Gain (dB)": round(m["cg_db"], 2),
                "Inferred Clusters": m["num_clusters"],
            })
        st.table(rows)

    with tab2:
        if st.button("Run Monte Carlo Benchmark Across Alpha"):
            with st.spinner("Running Monte Carlo simulations across all 7 modular estimators..."):
                summary = run_modular_benchmark(
                    max_objects=max_objects,
                    dim=2,
                    iterations=mc_trials,
                    sigma_theta_sq=sigma_theta_sq,
                    gibbs_iters=gibbs_iters,
                    burn_in=max(20, gibbs_iters // 3),
                )
            st.pyplot(plot_benchmark_curves(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description="Modular DPMM Benchmark Runner")
    parser.add_argument("--max-objects", type=int, default=10, help="Number of targets N")
    parser.add_argument("--dim", type=int, default=2, help="State dimension")
    parser.add_argument("--iterations", type=int, default=15, help="Monte Carlo trials per alpha")
    parser.add_argument("--gibbs-iters", type=int, default=250, help="Gibbs MCMC iterations")
    parser.add_argument("--burn-in", type=int, default=80, help="Gibbs burn-in iterations")
    parser.add_argument("--save-plot", type=str, default="", help="Optional path to save PNG plot")
    args, _ = parser.parse_known_args()

    print(
        f"Running Modular DPMM Suite (N={args.max_objects}, dim={args.dim}, "
        f"MC trials={args.iterations}, Gibbs iters={args.gibbs_iters})..."
    )
    summary = run_modular_benchmark(
        max_objects=args.max_objects,
        dim=args.dim,
        iterations=args.iterations,
        gibbs_iters=args.gibbs_iters,
        burn_in=args.burn_in,
        save_plot_path=args.save_plot,
    )

    header = (
        f"{'Alpha':>5} | {'Bound':>6} | {'Gibbs(Proposed)':>12} | {'Sklearn-BGMM':>12} | "
        f"{'Sklearn-KM':>10} | {'Torch-Attn':>10} | {'Sklearn-Agg':>11} | {'Sklearn-DB':>10} | {'Torch-MLP':>9}"
    )
    print(header)
    print("-" * len(header))
    for i, a in enumerate(summary["alpha"]):
        print(
            f"{a:5.1f} | {summary['mse_bound'][i]:6.4f} | "
            f"{summary['mse_gibbs_dpmm'][i]:12.4f} | "
            f"{summary['mse_sklearn_vi_dpmm'][i]:12.4f} | "
            f"{summary['mse_sklearn_kmeans'][i]:10.4f} | "
            f"{summary['mse_pytorch_set_attention'][i]:10.4f} | "
            f"{summary['mse_sklearn_agglomerative'][i]:11.4f} | "
            f"{summary['mse_sklearn_dbscan'][i]:10.4f} | "
            f"{summary['mse_pytorch_pointwise_mlp'][i]:9.4f}"
        )


if __name__ == "__main__":
    # Detect if launched via `streamlit run dashboard_runner.py`
    if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT"):
        run_streamlit_dashboard()
    else:
        main()