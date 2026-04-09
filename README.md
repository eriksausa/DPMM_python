# Nonparametric Bayesian Estimation in Hierarchical Gaussian Models (DPMM)

Modular Python (`SciPy`, `PyTorch`, `scikit-learn`, `Streamlit`) implementation and benchmark suite accompanying the TU Wien Master's Thesis:

> **Advanced Bayesian Estimation in Hierarchical Gaussian Models: Dirichlet Process Mixtures and Clustering Gain**
> *Dipl.-Ing. Erik Ĺ auĹˇa*, Institute of Telecommunications, TU Wien (Advisor: Prof. Franz Hlawatsch).

## Modular Architecture

| Module | Mathematical Scope & Estimators |
| :--- | :--- |
| `data_generation.py` | Chinese Restaurant Process (CRP) generator (`sample_crp_tables`), hierarchical Gaussian model (Eq. 3.64â€“3.67), Cholesky SPD linear solver (`sym_solve`), and Scenario 4 LMMSE state update (`empirical_bayes_lmmse`, Eq. 5.155). |
| `bounds.py` | Scenario 1 unclustered MMSE bound (`mmse_unknown_theta_bound`, Eq. 4.61), Scenario 2 known-$\boldsymbol{\theta}$ MMSE bound (`mmse_known_theta_bound`, Eq. 4.88), Scenario 4 partition-averaged lower bound (`partition_mmse_bound`, Eq. 5.162 via `gammaln`), and Clustering Gain in dB (Eq. 6.3). |
| `estimator_gibbs.py` | Scenario 3 MacEachern Collapsed/Indicator Gibbs Sampler (Algorithm 3, Eq. 5.77â€“5.110) implemented in both `SciPy` (`cho_factor`/`cho_solve` + `logsumexp`) and `PyTorch` (`torch.distributions.MultivariateNormal` & `Categorical`). |
| `estimator_dbscan.py` | Scenario 4 Modified Plug-in LMMSE estimator with k-NN knee $\varepsilon$-detection and `sklearn.cluster.DBSCAN` / `scipy.spatial.KDTree` fallback ($\mathrm{MSE}_{\mathrm{MC}}^{(\mathrm{DB})}$, Section 6.3.2). |
| `estimator_kmeans.py` | Scenario 4 Modified Plug-in LMMSE estimator with multi-replicate `K-Means++` ($\mathrm{MSE}_{\mathrm{MC}}^{(\mathrm{K})}$, Section 6.3.2). |
| `estimator_variational_dpmm.py` | Mean-Field Coordinate Ascent Variational Inference (`BayesianGaussianMixture` with Dirichlet Process prior) and parameter-free Ward `AgglomerativeClustering` LMMSE estimators. |
| `estimator_neural.py` | Permutation-equivariant Multi-Head `SetAttentionDenoiser` (`nn.TransformerEncoderLayer`) and pointwise `ResidualMLPDenoiser` (`nn.Module`). |
| `dashboard_runner.py` | Unified CLI Monte Carlo benchmark runner (reproducing Thesis Figures 10, 11, 12a/b, 13a/b) and 3-tab interactive `Streamlit` cockpit. |
## Quickstart & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit Cockpit
```bash
streamlit run dashboard_runner.py
```

### 3. Run Command-Line Monte Carlo Benchmarks
* **Sweep Concentration Parameter $\alpha$ (Thesis Figures 12b & 13b)**:
  ```bash
  python dashboard_runner.py --mode alpha --max-objects 20 --iterations 30 --save-plot fig12b_13b.png
  ```
* **Sweep Number of Objects $N$ (Thesis Figures 11, 12a & 13a)**:
  ```bash
  python dashboard_runner.py --mode n-sweep --alpha 0.5 --max-objects 30 --iterations 30 --save-plot fig12a_13a.png
  ```

### 4. Run Unit Test Suite
```bash
pytest -v
```