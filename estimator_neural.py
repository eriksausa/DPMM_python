"""
estimator_neural.py — PyTorch Deep Learning Denoisers (Set-Attention & Residual MLP)
====================================================================================
Provides two neural network estimators trained on synthetic CRP-DPMM sets:
- `SetAttentionDenoiser`: Permutation-equivariant Multi-Head Self-Attention
  (`torch.nn.TransformerEncoderLayer`) that attends across all N targets
  `{y_1, ..., y_N}` to learn contextual shrinkage.
- `ResidualMLPDenoiser`: Pointwise Residual MLP (`y_n -> x_hat_n`), illustrating
  why pointwise estimators cannot beat the unclustered bound `MMSE_unknown_theta`.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from scipy.special import logsumexp

from data_generation import generate_dpmm_data, sym_solve

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class ResidualMLPDenoiser(nn.Module):
        """Pointwise Residual MLP (`y_n -> x_hat_n`)."""

        def __init__(self, dim: int = 2, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            self.skip = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.skip.weight)

        def forward(self, y: torch.Tensor) -> torch.Tensor:
            return 0.85 * self.skip(y) + 0.15 * self.net(y)


    class SetAttentionDenoiser(nn.Module):
        """
        Permutation-Equivariant Multi-Head Self-Attention Set Network
        (`nn.TransformerEncoderLayer` without positional encoding).
        """

        def __init__(self, dim: int = 2, d_model: int = 64, n_heads: int = 4, num_layers: int = 2):
            super().__init__()
            self.input_proj = nn.Linear(dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_head = nn.Linear(d_model, dim)
            self.skip = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.skip.weight)

        def forward(self, y: torch.Tensor) -> torch.Tensor:
            single_set = (y.dim() == 2)
            if single_set:
                y = y.unsqueeze(0)
            h = self.input_proj(y)
            h = self.encoder(h)
            out = 0.70 * self.skip(y) + 0.30 * self.output_head(h)
            return out.squeeze(0) if single_set else out


class NeuralDenoiserSuite:
    """
    Trains and evaluates both the Pointwise Residual MLP and the Permutation-Equivariant
    Set-Attention Denoiser on synthetic CRP-DPMM realizations.
    """

    def __init__(
        self,
        var_v: np.ndarray,
        var_u: np.ndarray,
        var_theta: np.ndarray,
        mean_theta: np.ndarray,
        alpha_dp: float = 1.0,
        max_n: int = 10,
        device: str = "cpu",
    ):
        self.var_v = np.asarray(var_v, dtype=float)
        self.var_u = np.asarray(var_u, dtype=float)
        self.var_theta = np.asarray(var_theta, dtype=float)
        self.mean_theta = np.asarray(mean_theta, dtype=float).ravel()
        self.dim = self.mean_theta.shape[0]
        self.alpha_dp = float(alpha_dp)
        self.max_n = int(max_n)
        self.device = device

        self._mlp_model = None
        self._attn_model = None
        self._w_y_fallback = None
        self._w_prior_fallback = None
        self._w_self = 0.50
        self._w_ctx = 0.42
        self._log_tau = np.log(2.0 * float(np.trace(self.var_u + self.var_v)))

    def fit(
        self,
        train_steps: int = 150,
        batch_size: int = 32,
        lr: float = 3e-3,
        rng: Optional[np.random.Generator] = None,
    ) -> "NeuralDenoiserSuite":
        if rng is None:
            rng = np.random.default_rng(42)

        if TORCH_AVAILABLE:
            dev = torch.device(self.device)
            self._mlp_model = ResidualMLPDenoiser(dim=self.dim, hidden_dim=64).to(dev)
            self._attn_model = SetAttentionDenoiser(
                dim=self.dim, d_model=64, n_heads=4, num_layers=2
            ).to(dev)

            opt_mlp = optim.AdamW(self._mlp_model.parameters(), lr=lr, weight_decay=1e-4)
            opt_attn = optim.AdamW(self._attn_model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.MSELoss()

            self._mlp_model.train()
            self._attn_model.train()

            for _ in range(train_steps):
                y_batch = np.zeros((batch_size, self.max_n, self.dim), dtype=np.float32)
                x_batch = np.zeros((batch_size, self.max_n, self.dim), dtype=np.float32)
                for b in range(batch_size):
                    d = generate_dpmm_data(
                        self.max_n,
                        self.alpha_dp,
                        self.mean_theta,
                        self.var_v,
                        self.var_u,
                        self.var_theta,
                        rng=rng,
                    )
                    y_batch[b] = d.y
                    x_batch[b] = d.x

                y_t = torch.from_numpy(y_batch).to(dev)
                x_t = torch.from_numpy(x_batch).to(dev)

                opt_mlp.zero_grad()
                loss_mlp = criterion(self._mlp_model(y_t), x_t)
                loss_mlp.backward()
                opt_mlp.step()

                opt_attn.zero_grad()
                loss_attn = criterion(self._attn_model(y_t), x_t)
                loss_attn.backward()
                opt_attn.step()

            self._mlp_model.eval()
            self._attn_model.eval()
        else:
            # Lightweight gradient-trained kernel attention + pointwise shrinkage fallback
            cov_total = self.var_v + self.var_u + self.var_theta
            self._w_y_fallback = sym_solve(cov_total, self.var_u + self.var_theta).T
            self._w_prior_fallback = sym_solve(cov_total, self.var_v).T

            step_lr = 0.02
            for _ in range(120):
                d = generate_dpmm_data(
                    self.max_n,
                    self.alpha_dp,
                    self.mean_theta,
                    self.var_v,
                    self.var_u,
                    self.var_theta,
                    rng=rng,
                )
                pred, attn_ctx = self._fallback_attn_forward(d.y)
                err = pred - d.x
                grad_self = 2.0 * float(np.mean(err * d.y))
                grad_ctx = 2.0 * float(np.mean(err * attn_ctx))
                self._w_self -= step_lr * np.clip(grad_self, -1.0, 1.0)
                self._w_ctx -= step_lr * np.clip(grad_ctx, -1.0, 1.0)

        return self

    def _fallback_attn_forward(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sq_dists = np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        logits = -sq_dists / np.exp(self._log_tau)
        weights = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        attn_ctx = weights @ y
        pred = self._w_self * y + self._w_ctx * attn_ctx
        return pred, attn_ctx

    def predict_mlp(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if TORCH_AVAILABLE and self._mlp_model is not None:
            with torch.no_grad():
                y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
                return self._mlp_model(y_t).cpu().numpy()
        return y @ self._w_y_fallback + self.mean_theta @ self._w_prior_fallback

    def predict_attention(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if TORCH_AVAILABLE and self._attn_model is not None:
            with torch.no_grad():
                y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
                return self._attn_model(y_t).cpu().numpy()
        pred, _ = self._fallback_attn_forward(y)
        return pred