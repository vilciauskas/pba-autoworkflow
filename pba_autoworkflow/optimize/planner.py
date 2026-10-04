# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment planners: what to run next.

Three strategies, all behind :class:`Planner`, so the campaign loop can switch
without knowing the difference:

:class:`SobolPlanner`
    Space-filling seed design.  Uses a scrambled Sobol sequence, which gives
    markedly lower discrepancy than Latin hypercube at the batch sizes a physical
    platform actually runs (8-24), and stratifies the categorical choice so every
    metal appears in the seed batch.

:class:`RandomPlanner`
    Uniform random, kept as the honest baseline any claim of optimizer benefit
    must be measured against.

:class:`BayesPlanner`
    Constrained multi-objective Bayesian optimization.  One GP per objective plus
    one per constraint.  Batches are selected by *q*-noisy expected hypervolume
    improvement estimated by Monte-Carlo over joint posterior samples, with the
    feasibility probability from the constraint GP multiplying the improvement.
    Within a batch, points are chosen greedily under the "kriging believer"
    convention -- each accepted point is inserted as a pseudo-observation at its
    posterior mean -- which is what prevents a batch of eight nearly identical
    recipes.

The reference-point and hypervolume machinery is deliberately explicit rather than
delegated, because the objectives here are bounded on [0, 1] and the natural
reference point (the origin, i.e. no sodium in a fully defective framework) is
known a priori.  That removes the usual instability of adaptive reference points
in early campaigns.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm, qmc

from ..schema import DesignSpace, Experiment, SynthesisParameters
from ..analysis.objectives import OBJECTIVE_NAMES
from .surrogate import MixedGP


# --------------------------------------------------------------------------- #
# Pareto / hypervolume utilities
# --------------------------------------------------------------------------- #


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (all objectives maximized)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    n = Y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
        if np.any(dominated):
            keep[i] = False
    return keep


def hypervolume(Y: np.ndarray, ref: np.ndarray) -> float:
    """Dominated hypervolume above ``ref`` (maximization).

    For two objectives the exact value comes from sorting the front, which is what
    this campaign uses.  Higher dimensions fall back to a quasi-Monte-Carlo
    estimate, adequate for ranking candidate batches.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    ref = np.asarray(ref, dtype=float)
    Y = Y[np.all(Y > ref, axis=1)]
    if Y.size == 0:
        return 0.0
    front = Y[pareto_mask(Y)]
    if front.shape[1] == 2:
        order = np.argsort(-front[:, 0])
        front = front[order]
        hv, prev_y = 0.0, ref[1]
        for x, y in front:
            if y > prev_y:
                hv += (x - ref[0]) * (y - prev_y)
                prev_y = y
        return float(hv)
    # d > 2: QMC estimate inside the bounding box.
    hi = front.max(axis=0)
    sampler = qmc.Sobol(d=front.shape[1], scramble=True, seed=0)
    pts = ref + sampler.random(4096) * (hi - ref)
    inside = np.any(np.all(pts[:, None, :] <= front[None, :, :], axis=2), axis=1)
    return float(np.prod(hi - ref) * inside.mean())


# --------------------------------------------------------------------------- #
# Planner interface
# --------------------------------------------------------------------------- #


@dataclass
class Suggestion:
    parameters: SynthesisParameters
    origin: str
    diagnostics: dict[str, float] = field(default_factory=dict)


class Planner(abc.ABC):
    name = "planner"

    def __init__(self, space: DesignSpace, seed: int = 0) -> None:
        self.space = space
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

    @abc.abstractmethod
    def suggest(self, batch_size: int, history: list[Experiment]) -> list[Suggestion]:
        ...

    def _random_unit(self, n: int) -> np.ndarray:
        X = self.rng.random((n, self.space.dim))
        for _, cols in self.space.categorical_blocks():
            X[:, cols] = 0.0
            pick = self.rng.integers(0, len(cols), size=n)
            X[np.arange(n), np.array(cols)[pick]] = 1.0
        return X


class RandomPlanner(Planner):
    name = "random"

    def suggest(self, batch_size: int, history: list[Experiment]) -> list[Suggestion]:
        if batch_size <= 0:
            return []
        X = self._random_unit(batch_size)
        return [Suggestion(self.space.decode(x), "random") for x in X]


class SobolPlanner(Planner):
    """Scrambled-Sobol seed design with stratified categorical assignment."""

    name = "sobol"

    def __init__(self, space: DesignSpace, seed: int = 0) -> None:
        super().__init__(space, seed)
        self._sampler = qmc.Sobol(d=len(space.continuous), scramble=True, seed=seed)

    def suggest(self, batch_size: int, history: list[Experiment]) -> list[Suggestion]:
        if batch_size <= 0:
            # Reachable whenever replicates consume the whole batch
            # (n_rep == n in Campaign._plan), and it used to crash the campaign
            # mid-iteration: `random_base2` rejects n=0, and the categorical
            # round-robin builds `np.array([])`, whose float64 dtype is not a
            # legal index.  An empty request is a well-formed request.
            return []
        n_cont = len(self.space.continuous)
        # Sobol' balance properties only hold for blocks of 2**m points, and a
        # physical batch is rarely a power of two.  Draw the next full block and
        # keep the prefix: the retained points are still a balanced subsequence,
        # and the discarded remainder is consumed so successive batches do not
        # re-cover the same stratum.
        m = max(1, int(math.ceil(math.log2(max(batch_size, 1)))))
        U = self._sampler.random_base2(m)[:batch_size]
        X = np.zeros((batch_size, self.space.dim))
        X[:, :n_cont] = U
        for _, cols in self.space.categorical_blocks():
            # Round-robin so every level is represented before any repeats.
            assign = np.array([cols[i % len(cols)] for i in range(batch_size)])
            self.rng.shuffle(assign)
            X[np.arange(batch_size), assign] = 1.0
        _ = n_cont
        return [Suggestion(self.space.decode(x), "sobol") for x in X]


# --------------------------------------------------------------------------- #
# Bayesian planner
# --------------------------------------------------------------------------- #


@dataclass
class PlannerDiagnostics:
    n_train: int
    models: dict[str, dict[str, float]] = field(default_factory=dict)
    current_hypervolume: float = 0.0
    reference_point: tuple[float, ...] = ()
    batch_acquisition: list[float] = field(default_factory=list)
    exploration_fallback: bool = False
    note: str = ""


class BayesPlanner(Planner):
    """Constrained q-NEHVI over a mixed design space.

    Parameters
    ----------
    n_candidates
        Size of the candidate pool per batch slot.  A pool plus greedy selection
        is used instead of gradient-based inner optimization because the space is
        mixed and the objective GPs are cheap to evaluate in bulk.
    n_mc_samples
        Posterior samples used to estimate expected hypervolume improvement.
    min_train
        Below this many usable observations the planner refuses to trust the
        surrogate and returns space-filling points instead.  Guarding this
        explicitly avoids the classic failure where a 4-point GP collapses a
        campaign onto its first lucky recipe.
    """

    name = "qnehvi"

    def __init__(self, space: DesignSpace, seed: int = 0, n_candidates: int = 2048,
                 n_mc_samples: int = 96, min_train: int = 8,
                 objective_names: tuple[str, ...] = OBJECTIVE_NAMES,
                 constraint_names: tuple[str, ...] = ("isolated_yield",),
                 local_fraction: float = 0.4) -> None:
        super().__init__(space, seed)
        self.n_candidates = int(n_candidates)
        self.n_mc_samples = int(n_mc_samples)
        self.min_train = int(min_train)
        self.objective_names = tuple(objective_names)
        self.constraint_names = tuple(constraint_names)
        self.local_fraction = float(local_fraction)
        self._fallback = SobolPlanner(space, seed + 991)
        self.last_diagnostics: PlannerDiagnostics | None = None

    # -- training data ------------------------------------------------- #

    def _training_data(self, history: list[Experiment]):
        X, Yo, Yc = [], [], []
        for exp in history:
            if exp.objectives is None:
                continue
            if exp.status.value in ("failed", "quarantined"):
                continue
            try:
                X.append(self.space.encode(exp.parameters))
            except ValueError:
                continue
            Yo.append([exp.objectives.values.get(n, 0.0) for n in self.objective_names])
            Yc.append([exp.objectives.constraints.get(n, 0.0)
                       for n in self.constraint_names])
        if not X:
            return None, None, None
        return np.array(X), np.array(Yo), np.array(Yc)

    def _candidate_pool(self, X_train: np.ndarray, Y_obj: np.ndarray) -> np.ndarray:
        """Global Sobol coverage plus local perturbations of the current front."""
        n_global = int(self.n_candidates * (1.0 - self.local_fraction))
        n_local = self.n_candidates - n_global
        nc = len(self.space.continuous)

        sampler = qmc.Sobol(d=nc, scramble=True, seed=int(self.rng.integers(1 << 30)))
        m = int(np.ceil(np.log2(max(n_global, 2))))
        U = sampler.random_base2(m)[:n_global]
        Xg = np.zeros((n_global, self.space.dim))
        Xg[:, :nc] = U
        for _, cols in self.space.categorical_blocks():
            pick = self.rng.integers(0, len(cols), size=n_global)
            Xg[np.arange(n_global), np.array(cols)[pick]] = 1.0

        if n_local > 0 and X_train.shape[0] >= 2:
            front = X_train[pareto_mask(Y_obj)]
            base = front[self.rng.integers(0, front.shape[0], size=n_local)]
            Xl = base.copy()
            Xl[:, :nc] = np.clip(
                Xl[:, :nc] + self.rng.normal(0.0, 0.09, size=(n_local, nc)), 0.0, 1.0
            )
            # Occasionally hop the categorical level to keep metals comparable.
            for _, cols in self.space.categorical_blocks():
                hop = self.rng.random(n_local) < 0.25
                if np.any(hop):
                    Xl[np.ix_(hop, cols)] = 0.0
                    pick = self.rng.integers(0, len(cols), size=int(hop.sum()))
                    Xl[np.where(hop)[0], np.array(cols)[pick]] = 1.0
            return np.vstack([Xg, Xl])
        return Xg

    # -- acquisition ---------------------------------------------------- #

    def suggest(self, batch_size: int, history: list[Experiment]) -> list[Suggestion]:
        if batch_size <= 0:
            return []
        X_train, Y_obj, Y_con = self._training_data(history)
        n_train = 0 if X_train is None else X_train.shape[0]

        if X_train is None or n_train < self.min_train:
            self.last_diagnostics = PlannerDiagnostics(
                n_train=n_train, exploration_fallback=True,
                note=f"{n_train} usable observations < min_train={self.min_train}",
            )
            return [Suggestion(s.parameters, "sobol-warmup")
                    for s in self._fallback.suggest(batch_size, history)]

        nc = len(self.space.continuous)
        blocks = self.space.categorical_blocks()
        obj_models: list[MixedGP] = []
        diag: dict[str, dict[str, float]] = {}
        for j, name in enumerate(self.objective_names):
            gp = MixedGP(nc, blocks)
            gp.fit(X_train, Y_obj[:, j], rng=self.rng)
            obj_models.append(gp)
            diag[name] = gp.loo_diagnostics()

        con_models: list[MixedGP] = []
        for j, name in enumerate(self.constraint_names):
            gp = MixedGP(nc, blocks)
            gp.fit(X_train, Y_con[:, j], rng=self.rng)
            con_models.append(gp)
            diag[f"constraint::{name}"] = gp.loo_diagnostics()

        ref = np.zeros(len(self.objective_names))
        feasible = np.all(Y_con >= 0.0, axis=1)
        Y_ref_set = Y_obj[feasible] if np.any(feasible) else Y_obj
        hv_now = hypervolume(Y_ref_set, ref)

        pool = self._candidate_pool(X_train, Y_obj)
        # Posterior samples for every candidate, per objective.
        obj_samples = np.stack(
            [gp.sample_posterior(pool, self.n_mc_samples, self.rng)
             for gp in obj_models], axis=-1
        )  # (n_mc, n_pool, n_obj)
        # Feasibility probability from the constraint GPs.
        p_feas = np.ones(pool.shape[0])
        for gp in con_models:
            mu, sd = gp.predict(pool, return_std=True, include_noise=True)
            p_feas *= norm.cdf(mu / np.clip(sd, 1e-9, None))

        chosen: list[int] = []
        acq_trace: list[float] = []
        current_front = [Y_ref_set]

        for _ in range(batch_size):
            base = np.vstack(current_front)
            hv_base = hypervolume(base, ref)
            # Expected hypervolume improvement, MC over the joint posterior.
            gains = np.empty(pool.shape[0])
            # Vectorized over candidates is not possible for exact HV, so restrict
            # the expensive evaluation to the most promising candidates by an
            # upper-confidence pre-screen.
            screen_mu = np.mean(obj_samples, axis=0)
            screen_sd = np.std(obj_samples, axis=0)
            ucb = np.sum(screen_mu + 1.5 * screen_sd, axis=1) * p_feas
            top = np.argsort(-ucb)[: min(220, pool.shape[0])]
            gains[:] = -np.inf
            for i in top:
                if i in chosen:
                    continue
                sub = obj_samples[:, i, :]
                g = 0.0
                for s in range(0, self.n_mc_samples, 2):  # stride: halves MC cost
                    g += max(0.0, hypervolume(np.vstack([base, sub[s][None, :]]), ref)
                             - hv_base)
                gains[i] = g / max(1, len(range(0, self.n_mc_samples, 2)))
            acq = gains * p_feas
            best = int(np.argmax(acq))
            if not np.isfinite(acq[best]) or acq[best] <= 0.0:
                # Front cannot be improved by any candidate under the current
                # posterior: switch this slot to pure uncertainty sampling.
                unc = np.sum(np.std(obj_samples, axis=0), axis=1) * p_feas
                unc[chosen] = -np.inf
                best = int(np.argmax(unc))
            chosen.append(best)
            acq_trace.append(float(acq[best]))
            # Kriging believer: pretend we observed the posterior mean here.
            current_front.append(np.mean(obj_samples[:, best, :], axis=0)[None, :])

        self.last_diagnostics = PlannerDiagnostics(
            n_train=n_train, models=diag, current_hypervolume=hv_now,
            reference_point=tuple(float(v) for v in ref),
            batch_acquisition=acq_trace,
        )
        out: list[Suggestion] = []
        for i, idx in enumerate(chosen):
            out.append(Suggestion(
                self.space.decode(pool[idx]), "qnehvi",
                diagnostics={
                    "acquisition": acq_trace[i],
                    "p_feasible": float(p_feas[idx]),
                    **{f"pred_{n}": float(np.mean(obj_samples[:, idx, j]))
                       for j, n in enumerate(self.objective_names)},
                },
            ))
        return out


def make_planner(kind: str, space: DesignSpace, seed: int = 0, **kwargs) -> Planner:
    kinds = {"random": RandomPlanner, "sobol": SobolPlanner, "bayes": BayesPlanner,
             "qnehvi": BayesPlanner}
    if kind not in kinds:
        raise ValueError(f"unknown planner {kind!r}; choose from {sorted(kinds)}")
    return kinds[kind](space, seed=seed, **kwargs)
