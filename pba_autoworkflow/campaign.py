# SPDX-License-Identifier: GPL-3.0-or-later
"""The closed loop.

:class:`Campaign` is the top-level object a user drives.  One iteration is:

    plan -> register -> execute (scheduled, concurrent) -> analyse -> record -> stop?

Everything that makes this survivable on real hardware lives here rather than in
the notebook that calls it:

* **Resumption.**  Construct a campaign with an existing ``campaign_id`` and the
  history is reloaded from the provenance store; the planner is refitted on it and
  the loop continues at the next batch index.  A crash costs at most the batch in
  flight.
* **Stopping rules.**  A campaign ends on iteration budget, on convergence of the
  hypervolume indicator, or on a platform-health trip.  The health trip is the
  important one: if the failure rate in a batch exceeds a threshold the loop halts
  and asks for a human rather than burning reagent on a broken deck.
* **Quarantine, not deletion.**  Samples that fail QC stay in the record with
  their flags and are excluded from surrogate training.  Silently dropping them
  makes an optimizer look better than it is and hides systematic instrument
  problems.
* **Replication.**  A configurable fraction of each batch re-runs the current best
  recipe.  Without it there is no measurement of platform reproducibility, and the
  optimizer will happily chase noise.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .analysis.objectives import (
    DEFAULT_TARGET_PHASE,
    OBJECTIVE_CATALOGUE,
    OBJECTIVE_NAMES,
    echem_plan,
    scalarize,
    validate_objectives,
)
from .devices.base import Platform
from .optimize.planner import Planner, hypervolume, make_planner, pareto_mask
from .clock import tick
from .provenance import ProvenanceStore
from .scheduler import BatchScheduler, StationPool, default_capacities
from .schema import (
    DesignSpace,
    Experiment,
    ExperimentStatus,
    default_design_space,
)
from .workflow import ExperimentWorkflow, WorkflowConfig


@dataclass
class CampaignConfig:
    """Everything that defines a campaign's behaviour, and nothing else."""

    campaign_id: str = field(default_factory=lambda: f"pba-{uuid.uuid4().hex[:8]}")
    #: objectives to maximise (``analysis.objectives.OBJECTIVE_CATALOGUE``).
    #: None: those recorded for an existing campaign, else the XRD pair
    #: (target_phase_fraction, crystallinity).  Electrochemical objectives add
    #: the cycling and competitive-insertion stage to every run.
    objectives: tuple[str, ...] | None = None
    planner: str = "bayes"
    seed_planner: str = "sobol"
    n_seed: int = 12
    batch_size: int = 6
    max_iterations: int = 8
    max_experiments: int = 200
    replicate_fraction: float = 0.15
    max_in_flight: int = 4
    reactor_capacity: int = 4
    per_experiment_timeout_s: float | None = None
    #: framework phase to make (``pba_fm3m``, ``pba_p21n``, ``znhcf_r3c``).
    #: None: the phase recorded for an existing campaign, else ``pba_fm3m``.
    target_phase: str | None = None

    # Stopping and safety
    hv_convergence_tol: float = 5e-4
    hv_convergence_patience: int = 3
    max_batch_failure_rate: float = 0.5
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class IterationRecord:
    iteration: int
    n_suggested: int
    counts: dict[str, int]
    wall_time_s: float
    hypervolume: float
    hv_delta: float
    best_scalar: float
    bottleneck: str | None
    planner_note: str = ""
    planner_diagnostics: dict = field(default_factory=dict)
    stop_reason: str | None = None
    #: Force-field-versus-measurement comparison, when a ThermoAdvisor is
    #: attached.  Advisory and recorded only -- it never alters what was
    #: proposed, because the hull is indexed by an outcome the planner cannot
    #: set.  See ``pba_autoworkflow.thermo.advisor``.
    thermo_advice: dict = field(default_factory=dict)


class Campaign:
    """Closed-loop optimization campaign over a physical or simulated platform."""

    def __init__(self, platform: Platform, store: ProvenanceStore,
                 config: CampaignConfig | None = None,
                 space: DesignSpace | None = None,
                 workflow_config: WorkflowConfig | None = None,
                 planner: Planner | None = None,
                 progress: Callable[[str], None] | None = None,
                 thermo_advisor=None) -> None:
        self.platform = platform
        self.store = store
        #: Optional ``pba_autoworkflow.thermo.advisor.ThermoAdvisor``.  Compares a
        #: force-field hull against what the platform actually made, once per
        #: iteration.  Deliberately not passed to the planner: see that
        #: module's docstring for why a hull cannot score a proposal.
        self.thermo_advisor = thermo_advisor
        self.config = config or CampaignConfig()
        self.space = space or default_design_space()
        self.workflow_config = workflow_config or WorkflowConfig()
        self._resolve_target_phase(store)
        self._resolve_objectives(store)
        self._progress = progress or (lambda msg: None)

        self.pool = StationPool(default_capacities(self.config.reactor_capacity))
        self.scheduler = BatchScheduler(
            self.pool, max_in_flight=self.config.max_in_flight,
            per_experiment_timeout_s=self.config.per_experiment_timeout_s,
        )
        self.workflow = ExperimentWorkflow(platform, store, self.workflow_config,
                                           station_pool=self.pool)

        extra = ({"objective_names": self.objective_names}
                 if self.config.planner in ("bayes", "qnehvi") else {})
        self.planner = planner or make_planner(
            self.config.planner, self.space, seed=self.config.seed, **extra
        )
        self.seed_planner = make_planner(
            self.config.seed_planner, self.space, seed=self.config.seed + 17
        )

        resumed = store.campaign_exists(self.config.campaign_id)

        if not resumed:
            store.create_campaign(
                self.config.campaign_id, self.config.as_dict(),
                self.space.model_dump_json(),
            )
        self.history: list[Experiment] = store.load_campaign(self.config.campaign_id)
        stale = sorted({k for e in self.history if e.objectives is not None
                        for k in e.objectives.values} - set(self.objective_names))
        if stale:
            raise ValueError(
                f"campaign {self.config.campaign_id!r} was recorded with objectives "
                f"{stale}, but this campaign optimises {list(self.objective_names)}; "
                "its results cannot be combined -- start a new campaign id")
        # Iteration records are logged as ``iteration_complete`` events; reload
        # them so a resumed campaign (or a report built from the store) keeps its
        # hypervolume trajectory, its iteration numbering, and the convergence
        # test's memory.  These used to start empty on every load, which emptied
        # the report's progress panel and iterations.csv after any resume.
        self.iterations: list[IterationRecord] = self._restore_iterations(store)
        self.resumed = resumed and bool(self.history)
        self._hv_history: list[float] = [r.hypervolume for r in self.iterations]
        if self.resumed:
            self._progress(
                f"resumed {self.config.campaign_id} with {len(self.history)} experiments"
            )

    def _resolve_target_phase(self, store: ProvenanceStore) -> None:
        from .analysis.phases import CANDIDATES, FRAMEWORK_PHASES

        stored = store.campaign_config(self.config.campaign_id) or {}
        recorded = stored.get("target_phase")
        if recorded and self.config.target_phase and self.config.target_phase != recorded:
            raise ValueError(
                f"campaign {self.config.campaign_id!r} was scored for target phase "
                f"{recorded!r}, not {self.config.target_phase!r}; pass the same target "
                "phase (or none) or start a new campaign id")
        target = self.config.target_phase or recorded or DEFAULT_TARGET_PHASE
        if target not in FRAMEWORK_PHASES:
            raise ValueError(f"target phase {target!r} is not one of {FRAMEWORK_PHASES}")
        metals = next((c.choices for c in self.space.categorical if c.name == "metal"), ())
        if not any(k.split("/")[0] == target for m in metals for k in CANDIDATES.get(m, ())):
            raise ValueError(f"no metal in the design space {tuple(metals)} can form {target!r}")
        self.config.target_phase = target
        self.workflow_config.target_phase = target

    def _resolve_objectives(self, store: ProvenanceStore) -> None:
        stored = store.campaign_config(self.config.campaign_id) or {}
        recorded = tuple(stored["objectives"]) if stored.get("objectives") else None
        asked = tuple(self.config.objectives) if self.config.objectives else None
        if recorded and asked and asked != recorded:
            raise ValueError(
                f"campaign {self.config.campaign_id!r} optimises {list(recorded)}, not "
                f"{list(asked)}; pass the same objectives (or none) or start a new campaign id")
        names = validate_objectives(asked or recorded or OBJECTIVE_NAMES)
        self.config.objectives = names
        self.objective_names = names
        self.workflow_config.objectives = names
        single, mixed = echem_plan(names)
        if single or mixed:
            if self.platform.electrochem is None:
                needs = [n for n in names if OBJECTIVE_CATALOGUE[n] == "echem"]
                raise ValueError(f"objectives {needs} need a potentiostat; the platform has none")
            # Keep any electrolytes the caller configured, add what the objectives need.
            wc = self.workflow_config
            wc.echem_single_ions = tuple(dict.fromkeys(tuple(wc.echem_single_ions) + single))
            wc.echem_mixed_electrolytes = tuple(wc.echem_mixed_electrolytes) + tuple(
                m for m in mixed if m not in tuple(wc.echem_mixed_electrolytes))

    # ------------------------------------------------------------------ #
    # Views over the history
    # ------------------------------------------------------------------ #

    @property
    def usable(self) -> list[Experiment]:
        return [e for e in self.history
                if e.status is ExperimentStatus.COMPLETE and e.objectives is not None]

    def objective_matrix(self, feasible_only: bool = True) -> np.ndarray:
        rows = [
            [e.objectives.values[n] for n in self.objective_names]
            for e in self.usable
            if (e.objectives.feasible or not feasible_only)
        ]
        return np.array(rows) if rows else np.empty((0, len(self.objective_names)))

    def current_hypervolume(self) -> float:
        Y = self.objective_matrix()
        if Y.size == 0:
            return 0.0
        return hypervolume(Y, np.zeros(len(self.objective_names)))

    def pareto_front(self) -> list[Experiment]:
        feasible = [e for e in self.usable if e.objectives.feasible]
        if not feasible:
            return []
        Y = np.array([[e.objectives.values[n] for n in self.objective_names]
                      for e in feasible])
        return [e for e, keep in zip(feasible, pareto_mask(Y)) if keep]

    def best_experiment(self) -> Experiment | None:
        cands = [e for e in self.usable if e.objectives.feasible] or self.usable
        if not cands:
            return None
        return max(cands, key=lambda e: scalarize(e.objectives))

    # ------------------------------------------------------------------ #
    # One iteration
    # ------------------------------------------------------------------ #

    def _plan(self, iteration: int) -> list[Experiment]:
        cfg = self.config
        n_usable = len(self.usable)
        if n_usable == 0 and iteration == 0:
            n = cfg.n_seed
            suggestions = self.seed_planner.suggest(n, self.history)
            note = f"seed design, n={n}"
        else:
            n = min(cfg.batch_size, cfg.max_experiments - len(self.history))
            n_rep = int(round(cfg.replicate_fraction * n))
            best = self.best_experiment()
            suggestions = self.planner.suggest(max(0, n - n_rep), self.history)
            note = getattr(self.planner, "last_diagnostics", None)
            note = getattr(note, "note", "") if note else ""
            if best is not None and n_rep > 0:
                from .optimize.planner import Suggestion

                suggestions += [
                    Suggestion(best.parameters, "replicate",
                               {"replicate_of_batch": float(best.batch_index)})
                    for _ in range(n_rep)
                ]

        batch_index = self.store.next_batch_index(cfg.campaign_id)
        experiments: list[Experiment] = []
        for k, s in enumerate(suggestions):
            exp = Experiment(
                experiment_id=f"{cfg.campaign_id}-b{batch_index:02d}-e{k:02d}",
                campaign_id=cfg.campaign_id,
                batch_index=batch_index,
                parameters=s.parameters,
                origin=s.origin,
                metadata={"planner_diagnostics": s.diagnostics},
            )
            self.store.register_experiment(exp)
            experiments.append(exp)
        self.store.log_event("batch_planned", campaign_id=cfg.campaign_id,
                             payload={"iteration": iteration, "n": len(experiments),
                                      "note": note,
                                      "origins": [e.origin for e in experiments]})
        return experiments

    def _check_stop(self, record: IterationRecord) -> str | None:
        cfg = self.config
        if len(self.history) >= cfg.max_experiments:
            return f"experiment budget reached ({cfg.max_experiments})"
        n_bad = record.counts.get("failed", 0) + record.counts.get("quarantined", 0)
        if record.n_suggested and n_bad / record.n_suggested > cfg.max_batch_failure_rate:
            return (f"platform health: {n_bad}/{record.n_suggested} of the batch "
                    "failed or was quarantined; halting for inspection")
        if len(self._hv_history) > cfg.hv_convergence_patience:
            recent = self._hv_history[-(cfg.hv_convergence_patience + 1):]
            deltas = np.diff(recent)
            if np.all(np.abs(deltas) < cfg.hv_convergence_tol):
                return (f"hypervolume converged: |dHV| < {cfg.hv_convergence_tol} "
                        f"for {cfg.hv_convergence_patience} consecutive batches")
        return None

    async def run_iteration(self, iteration: int) -> IterationRecord:
        t0 = tick()
        experiments = self._plan(iteration)
        if not experiments:
            rec = IterationRecord(iteration, 0, {}, 0.0, self.current_hypervolume(),
                                  0.0, float("nan"), None,
                                  stop_reason="no experiments left in budget")
            self.iterations.append(rec)
            return rec

        self._progress(
            f"iteration {iteration}: running {len(experiments)} experiments "
            f"({', '.join(sorted({e.origin for e in experiments}))})"
        )
        hv_before = self.current_hypervolume()

        def on_complete(exp: Experiment) -> None:
            tag = exp.status.value
            self._progress(f"  {exp.experiment_id} {tag}"
                           + (f" — {exp.error}" if exp.error else ""))

        outcome = await self.scheduler.run_batch(
            experiments, self.workflow.run, on_complete=on_complete
        )
        self.history.extend(outcome.experiments)

        hv_after = self.current_hypervolume()
        self._hv_history.append(hv_after)
        best = self.best_experiment()
        pdiag = getattr(self.planner, "last_diagnostics", None)
        rec = IterationRecord(
            iteration=iteration,
            n_suggested=len(experiments),
            counts=outcome.counts(),
            wall_time_s=tick() - t0,
            hypervolume=hv_after,
            hv_delta=hv_after - hv_before,
            best_scalar=scalarize(best.objectives) if best and best.objectives else float("nan"),
            bottleneck=outcome.bottleneck,
            planner_note=getattr(pdiag, "note", "") if pdiag else "",
            planner_diagnostics=asdict(pdiag) if pdiag else {},
        )
        if self.thermo_advisor is not None:
            try:
                rec.thermo_advice = self.thermo_advisor.review(self.usable).as_payload()
            except Exception as err:  # noqa: BLE001
                # An advisory check must never take down a campaign that has
                # already spent reagents and instrument time on this batch.
                rec.thermo_advice = {"error": f"{type(err).__name__}: {err}"}

        rec.stop_reason = self._check_stop(rec)
        self.iterations.append(rec)
        self.store.log_event("iteration_complete", campaign_id=self.config.campaign_id,
                             payload={k: v for k, v in asdict(rec).items()
                                      if k != "planner_diagnostics"})
        return rec

    def _restore_iterations(self, store: ProvenanceStore) -> list[IterationRecord]:
        known = set(IterationRecord.__dataclass_fields__)
        out: list[IterationRecord] = []
        for ev in store.event_log(self.config.campaign_id, kinds=["iteration_complete"]):
            payload = json.loads(ev["payload_json"] or "{}")
            rec = IterationRecord(**{k: v for k, v in payload.items() if k in known})
            out.append(rec)
        # Keep the event log's chronological order -- it is the ground truth --
        # and renumber.  Stores written before iterations were restored restarted
        # the count at 0 on every resume, so sorting by the stored number would
        # interleave batches and make the hypervolume trajectory non-monotone.
        for i, rec in enumerate(out):
            rec.iteration = i
        return out

    async def run(self, max_iterations: int | None = None) -> list[IterationRecord]:
        """Run the closed loop until a stopping rule fires."""
        n_iter = max_iterations if max_iterations is not None else self.config.max_iterations
        start = len(self.iterations)
        for i in range(start, start + n_iter):
            rec = await self.run_iteration(i)
            self._progress(
                f"iteration {i}: HV={rec.hypervolume:.4f} (Δ{rec.hv_delta:+.4f}) "
                f"best={rec.best_scalar:.4f} counts={rec.counts} "
                f"bottleneck={rec.bottleneck} {rec.wall_time_s:.1f}s"
            )
            if rec.stop_reason:
                self._progress(f"stopping: {rec.stop_reason}")
                self.store.log_event("campaign_stopped",
                                     campaign_id=self.config.campaign_id,
                                     payload={"reason": rec.stop_reason,
                                              "iteration": i})
                break
        return self.iterations

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        best = self.best_experiment()
        front = self.pareto_front()
        counts: dict[str, int] = {}
        for e in self.history:
            counts[e.status.value] = counts.get(e.status.value, 0) + 1
        return {
            "campaign_id": self.config.campaign_id,
            "resumed": self.resumed,
            "n_experiments": len(self.history),
            "status_counts": counts,
            "n_iterations": len(self.iterations),
            "hypervolume": self.current_hypervolume(),
            "hv_trajectory": list(self._hv_history),
            "pareto_size": len(front),
            "best": None if best is None else {
                "experiment_id": best.experiment_id,
                "recipe": best.parameters.model_dump(),
                "objectives": best.objectives.values if best.objectives else {},
                "scalar": scalarize(best.objectives) if best.objectives else None,
                "formula": (best.descriptors.composition.formula
                            if best.descriptors.composition else None),
                "capacity_mAh_g": best.descriptors.capacity_mAh_g,
                "domain_size_nm": (best.descriptors.xrd.domain_size_nm
                                   if best.descriptors.xrd else None),
                "isolated_yield": best.descriptors.isolated_yield,
            },
            "station_report": self.pool.report(),
            "bottleneck": self.pool.bottleneck(),
            "stop_reason": next((r.stop_reason for r in reversed(self.iterations)
                                 if r.stop_reason), None),
        }

    def dataframe(self):
        return self.store.to_dataframe(self.config.campaign_id)

    def write_summary(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.summary(), indent=2, default=str))
        return p

    def replicate_statistics(self) -> dict[str, float]:
        """Within-recipe scatter measured from the replicate runs.

        Reported as the relative standard deviation of each objective over groups
        of experiments that share a recipe.  This is the number that tells you
        whether an apparent improvement is real.
        """
        groups: dict[str, list[Experiment]] = {}
        for e in self.usable:
            # Group on recipes rounded to 4 decimals, so that replicates entered by
            # hand or re-read from an instrument log still group with the original
            # even when the last digits of a float differ.
            dump = e.parameters.model_dump()
            key = json.dumps({k: round(v, 4) if isinstance(v, float) else v
                              for k, v in dump.items()}, sort_keys=True)
            groups.setdefault(key, []).append(e)
        reps = [g for g in groups.values() if len(g) > 1]
        if not reps:
            return {}
        out: dict[str, float] = {}
        for name in self.objective_names:
            rsds = []
            for g in reps:
                vals = np.array([e.objectives.values[name] for e in g])
                if vals.mean() > 1e-9:
                    rsds.append(vals.std(ddof=1) / vals.mean())
            if rsds:
                out[f"rsd_{name}"] = float(np.mean(rsds))
        out["n_replicate_groups"] = float(len(reps))
        out["n_replicate_runs"] = float(sum(len(g) for g in reps))
        return out


async def run_campaign(platform: Platform, store: ProvenanceStore,
                       config: CampaignConfig | None = None,
                       space: DesignSpace | None = None,
                       progress: Callable[[str], None] | None = None) -> Campaign:
    """Convenience entry point: build a campaign, run it, return it."""
    campaign = Campaign(platform, store, config, space, progress=progress)
    await campaign.run()
    return campaign


def run_campaign_sync(*args, **kwargs) -> Campaign:
    return asyncio.run(run_campaign(*args, **kwargs))


__all__ = [
    "Campaign", "CampaignConfig", "IterationRecord",
    "run_campaign", "run_campaign_sync",
]


def _unused(x: Sequence[object]) -> None:  # pragma: no cover
    del x
