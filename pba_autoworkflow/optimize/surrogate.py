# SPDX-License-Identifier: GPL-3.0-or-later
"""Gaussian-process surrogate for mixed continuous/categorical design spaces.

Implemented directly on numpy/scipy rather than pulled from a modelling framework,
for three reasons that matter on a self-driving lab: the hyperparameter fit must
stay well behaved with 10-40 observations, the noise level must be *learned* (the
platform's run-to-run scatter is the dominant uncertainty early in a campaign and
fixing it wrongly makes the acquisition either paranoid or reckless), and the
whole thing must be serializable into the provenance record without an external
dependency's version pinning.

Kernel: Matern-5/2 with automatic relevance determination over the continuous
dimensions, multiplied by an exchangeable (Hamming-style) kernel over each
categorical block.  The exchangeable form shares information across metals --
which is what makes a five-metal campaign tractable in a few dozen runs -- while
still letting each metal keep its own offset through the learned correlation.

Hyperparameters are fitted by maximizing the exact log marginal likelihood with
multi-start L-BFGS-B on log-transformed parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def matern52(dists: np.ndarray) -> np.ndarray:
    s5 = np.sqrt(5.0) * dists
    return (1.0 + s5 + 5.0 / 3.0 * dists ** 2) * np.exp(-s5)


@dataclass
class GPFit:
    log_marginal_likelihood: float
    lengthscales: np.ndarray
    signal_var: float
    noise_var: float
    cat_correlation: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    converged: bool = True


class MixedGP:
    """Zero-mean-on-standardized-residual GP with a mixed kernel.

    Parameters
    ----------
    n_continuous
        Number of leading columns treated as continuous (already in the unit cube).
    categorical_blocks
        ``[(name, [column indices]), ...]`` one-hot blocks following the
        continuous columns.
    """

    def __init__(self, n_continuous: int,
                 categorical_blocks: list[tuple[str, list[int]]] | None = None,
                 jitter: float = 1e-8) -> None:
        self.n_continuous = int(n_continuous)
        self.categorical_blocks = categorical_blocks or []
        self.jitter = float(jitter)
        self._fitted = False

        self.lengthscales = np.ones(self.n_continuous)
        self.signal_var = 1.0
        self.noise_var = 1e-2
        self.cat_rho = {name: 0.5 for name, _ in self.categorical_blocks}
        self.fit_info: GPFit | None = None

    # ------------------------------------------------------------------ #
    # Kernel
    # ------------------------------------------------------------------ #

    def _kernel(self, X1: np.ndarray, X2: np.ndarray, lengthscales: np.ndarray,
                signal_var: float, cat_rho: dict[str, float]) -> np.ndarray:
        nc = self.n_continuous
        if nc > 0:
            d = cdist(X1[:, :nc] / lengthscales, X2[:, :nc] / lengthscales)
            K = matern52(d)
        else:
            K = np.ones((X1.shape[0], X2.shape[0]))
        for name, cols in self.categorical_blocks:
            same = X1[:, cols] @ X2[:, cols].T  # 1 if same level, else 0
            rho = cat_rho[name]
            K = K * (rho + (1.0 - rho) * same)
        return signal_var * K

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def _unpack(self, theta: np.ndarray) -> tuple[np.ndarray, float, float, dict[str, float]]:
        nc = self.n_continuous
        ls = np.exp(theta[:nc]) if nc else np.ones(0)
        sv = float(np.exp(theta[nc]))
        nv = float(np.exp(theta[nc + 1]))
        rho = {}
        for i, (name, _) in enumerate(self.categorical_blocks):
            # logit-transformed so rho stays in (0, 1)
            rho[name] = float(1.0 / (1.0 + np.exp(-theta[nc + 2 + i])))
        return ls, sv, nv, rho

    def _nll(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        ls, sv, nv, rho = self._unpack(theta)
        n = X.shape[0]
        K = self._kernel(X, X, ls, sv, rho)
        K[np.diag_indices_from(K)] += nv + self.jitter
        try:
            c, lower = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = cho_solve((c, lower), y)
        logdet = 2.0 * float(np.sum(np.log(np.diag(c))))
        nll = 0.5 * float(y @ alpha) + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)
        # Weak log-normal priors keep lengthscales and noise from running away
        # in the small-n regime that dominates the first half of a campaign.
        nll += 0.5 * float(np.sum((np.log(ls) - np.log(0.35)) ** 2) / 1.5 ** 2)
        nll += 0.5 * ((np.log(nv) - np.log(0.02)) ** 2) / 1.5 ** 2
        return nll if np.isfinite(nll) else 1e12

    def fit(self, X: np.ndarray, y: np.ndarray, n_restarts: int = 6,
            rng: np.random.Generator | None = None) -> GPFit:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.size:
            raise ValueError(f"X has {X.shape[0]} rows, y has {y.size}")
        if y.size < 2:
            raise ValueError("need at least 2 observations")

        rng = rng or np.random.default_rng(0)
        self.X_train = X
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) or 1.0
        y_s = (y - self.y_mean) / self.y_std

        nc = self.n_continuous
        n_theta = nc + 2 + len(self.categorical_blocks)
        bounds = (
            [(np.log(0.03), np.log(8.0))] * nc
            + [(np.log(1e-3), np.log(50.0)), (np.log(1e-6), np.log(2.0))]
            + [(-4.0, 4.0)] * len(self.categorical_blocks)
        )
        starts = [np.concatenate([
            np.full(nc, np.log(0.4)), [np.log(1.0), np.log(0.02)],
            np.zeros(len(self.categorical_blocks)),
        ])]
        for _ in range(max(0, n_restarts - 1)):
            starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))

        best = (np.inf, starts[0], False)
        for x0 in starts:
            try:
                res = minimize(self._nll, x0, args=(X, y_s), method="L-BFGS-B",
                               bounds=bounds, options={"maxiter": 300})
            except Exception:
                continue
            if res.fun < best[0]:
                best = (float(res.fun), res.x, bool(res.success))

        nll, theta, ok = best
        self.lengthscales, self.signal_var, self.noise_var, self.cat_rho = self._unpack(theta)
        K = self._kernel(X, X, self.lengthscales, self.signal_var, self.cat_rho)
        K[np.diag_indices_from(K)] += self.noise_var + self.jitter
        self._chol = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._chol, y_s)
        self._y_s = y_s
        self._fitted = True
        _ = n_theta
        self.fit_info = GPFit(
            log_marginal_likelihood=-nll,
            lengthscales=self.lengthscales.copy(),
            signal_var=self.signal_var,
            noise_var=self.noise_var,
            cat_correlation=dict(self.cat_rho),
            n_train=X.shape[0],
            converged=ok,
        )
        return self.fit_info

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray, return_std: bool = True,
                include_noise: bool = False):
        if not self._fitted:
            raise RuntimeError("MixedGP.fit must be called first")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Ks = self._kernel(X, self.X_train, self.lengthscales, self.signal_var,
                          self.cat_rho)
        mu_s = Ks @ self._alpha
        mu = mu_s * self.y_std + self.y_mean
        if not return_std:
            return mu
        v = cho_solve(self._chol, Ks.T)
        prior = self.signal_var + (self.noise_var if include_noise else 0.0)
        var_s = np.clip(prior - np.einsum("ij,ji->i", Ks, v), 1e-12, None)
        return mu, np.sqrt(var_s) * self.y_std

    def sample_posterior(self, X: np.ndarray, n_samples: int,
                         rng: np.random.Generator) -> np.ndarray:
        """Joint posterior samples, shape ``(n_samples, len(X))``."""
        if not self._fitted:
            raise RuntimeError("MixedGP.fit must be called first")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Ks = self._kernel(X, self.X_train, self.lengthscales, self.signal_var,
                          self.cat_rho)
        mu_s = Ks @ self._alpha
        Kss = self._kernel(X, X, self.lengthscales, self.signal_var, self.cat_rho)
        v = cho_solve(self._chol, Ks.T)
        cov = Kss - Ks @ v
        cov[np.diag_indices_from(cov)] += 1e-9
        # Eigen-decomposition is more forgiving than Cholesky on the near-singular
        # covariances produced by densely sampled candidate grids.
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 0.0, None)
        L = V * np.sqrt(w)
        z = rng.standard_normal((X.shape[0], n_samples))
        samples_s = mu_s[:, None] + L @ z
        return (samples_s * self.y_std + self.y_mean).T

    def loo_diagnostics(self) -> dict[str, float]:
        """Leave-one-out predictive checks from the fitted factorization.

        Cheap closed-form LOO for a GP: the residuals and variances follow from
        the inverse covariance, so this costs one triangular solve rather than n
        refits.  ``loo_r2`` below about 0.3 means the surrogate is not yet
        informative and the planner should stay in exploration.
        """
        if not self._fitted:
            raise RuntimeError("MixedGP.fit must be called first")
        n = self.X_train.shape[0]
        K_inv = cho_solve(self._chol, np.eye(n))
        diag = np.diag(K_inv)
        # Sundararajan-Keerthi identities: LOO residual = alpha_i / [K^-1]_ii and
        # LOO predictive variance = 1 / [K^-1]_ii.
        resid_s = self._alpha / diag
        var_s = 1.0 / diag
        resid = resid_s * self.y_std
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((self._y_s - self._y_s.mean()) ** 2)) * self.y_std ** 2
        z = resid_s / np.sqrt(var_s)
        return {
            "loo_rmse": float(np.sqrt(ss_res / n)),
            "loo_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "loo_z_std": float(np.std(z)),
            "noise_fraction": float(self.noise_var / (self.noise_var + self.signal_var)),
        }
