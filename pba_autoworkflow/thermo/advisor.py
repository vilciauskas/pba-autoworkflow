# SPDX-License-Identifier: GPL-3.0-or-later
"""Where a force-field hull may and may not touch a running campaign.

The obvious wiring -- score each proposed recipe with the hull and add a term to
the planner's objective -- **cannot be built**, and the reason is structural
rather than a matter of tuning.

``HullPrior`` is indexed by ``(metal, vacancy_fraction)``.  Of those two,
``metal`` is a synthesis parameter, but ``vacancy_fraction`` is an *outcome*: it
is derived from a measured Fe/M ratio and a CHN carbon assay, and it appears in
``CompositionDescriptors``, not in ``SynthesisParameters``.  The planner proposes
concentrations, temperature, pH, aging time and stir rate.  Nothing it can set
tells you the vacancy fraction that will result.  Scoring a proposal with the
hull would require a recipe -> composition model, and that model is the
surrogate, which is fitted to the data the campaign is collecting.  Feeding the
hull in through it would make the prior a function of the measurements it is
supposed to be independent of.

So the hull enters in the one place it is valid: **after** measurement, as a
falsifiable claim about the samples that came back.  That direction is sound
because both quantities are then measured, and it is the direction that can tell
you the force field is wrong -- which, for the current models, is the outcome to
plan for.

Two checks run per iteration, both advisory and both logged:

``stability``
    ``HullPrior.disagreement`` on accumulated ``(metal, y, objective)``.  A
    negative correlation means the hull ranked compositions backwards relative
    to what the platform actually made.

``lattice``
    The sharper test.  Along a fixed-metal composition series the models make
    *different structural predictions*: mace-mp-small contracts the Mn cell by
    0.350 A at y = 0.25, while uma-s-1p1-omat moves it by 0.060 A over the whole
    range.  XRD measures the cell edge on every sample anyway, so the campaign
    adjudicates between them at no extra experimental cost.  Absolute edges are
    not comparable (UMA's Mn is 0.55 A small), so this compares *changes*
    relative to the least-vacancy sample in the series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .prior import HullPrior

#: Below this many measured samples in a metal series, neither check reports a
#: verdict.  Three points is the minimum that can show a trend rather than a
#: line, and two of them are the endpoints.
MIN_SERIES = 4


@dataclass
class ThermoAdvice:
    """One iteration's worth of force-field-versus-measurement comparison."""

    #: ``"agrees"`` | ``"disagrees"`` | ``"uninformative"`` | ``"insufficient-data"``
    stability_verdict: str = "insufficient-data"
    stability_correlation: float = float("nan")
    n_stability: int = 0

    lattice_verdict: str = "insufficient-data"
    #: measured minus predicted change in cell edge, per metal, in angstrom
    lattice_residual_A: dict[str, float] = field(default_factory=dict)
    n_lattice: int = 0

    notes: list[str] = field(default_factory=list)

    def as_payload(self) -> dict:
        return {"stability_verdict": self.stability_verdict,
                "stability_correlation": self.stability_correlation,
                "n_stability": self.n_stability,
                "lattice_verdict": self.lattice_verdict,
                "lattice_residual_A": dict(self.lattice_residual_A),
                "n_lattice": self.n_lattice,
                "notes": list(self.notes)}


class ThermoAdvisor:
    """Compares a hull against measurements. Never influences what is proposed.

    ``predicted_lattice`` maps ``metal -> {vacancy_fraction: relaxed_a_A}`` from
    whatever model produced the hull.  Leave it empty to skip the lattice check.
    """

    def __init__(self, prior: HullPrior,
                 predicted_lattice: dict[str, dict[float, float]] | None = None) -> None:
        self.prior = prior
        self.predicted_lattice = predicted_lattice or {}

    # ------------------------------------------------------------------ #

    def review(self, experiments) -> ThermoAdvice:
        """Score the force field against completed, QC-passing experiments."""
        advice = ThermoAdvice()
        rows = _measured_rows(experiments)
        if not rows:
            advice.notes.append("no completed sample carries both a composition "
                                "and an objective")
            return advice

        self._review_stability(rows, advice)
        self._review_lattice(rows, advice)
        return advice

    # ------------------------------------------------------------------ #

    def _review_stability(self, rows, advice) -> None:
        metals = {m for m, _, _, _ in rows}
        informative = [m for m in metals if self.prior.is_informative(m)]
        if not informative:
            advice.stability_verdict = "uninformative"
            advice.notes.append(
                "the hull carries no resolvable feature for any measured metal, "
                "so there is nothing to falsify; this is the expected state for "
                "an unfine-tuned foundation model")
            return

        measured = [(m, y, obj) for m, y, obj, _ in rows if m in informative]
        advice.n_stability = len(measured)
        if len(measured) < MIN_SERIES:
            advice.notes.append(
                f"{len(measured)} usable sample(s) on an informative metal; "
                f"need {MIN_SERIES} before reading a correlation")
            return

        verdict = self.prior.disagreement(measured)
        advice.stability_correlation = float(verdict.get("correlation", float("nan")))
        advice.stability_verdict = (
            "disagrees" if "misleading" in str(verdict.get("verdict", "")) else "agrees")
        advice.notes.append(str(verdict.get("verdict", "")))

    # ------------------------------------------------------------------ #

    def _review_lattice(self, rows, advice) -> None:
        if not self.predicted_lattice:
            advice.lattice_verdict = "not-configured"
            return

        n_used = 0
        for metal, pred in self.predicted_lattice.items():
            series = sorted(((y, a) for m, y, _, a in rows
                             if m == metal and not math.isnan(y) and not math.isnan(a)),
                            key=lambda t: t[0])
            if len(series) < MIN_SERIES or len(pred) < 2:
                continue
            # Compare CHANGES, not absolute edges: every model tested is off by
            # 0.2-0.5 A in absolute terms, which would swamp the composition
            # dependence being tested.
            y_ref, a_ref = series[0]
            ys = sorted(pred)
            p_ref = _interp(y_ref, ys, [pred[y] for y in ys])
            resid = []
            for y, a in series[1:]:
                d_meas = a - a_ref
                d_pred = _interp(y, ys, [pred[y] for y in ys]) - p_ref
                resid.append(d_meas - d_pred)
            if resid:
                advice.lattice_residual_A[metal] = sum(resid) / len(resid)
                n_used += len(resid) + 1

        advice.n_lattice = n_used
        if not advice.lattice_residual_A:
            advice.lattice_verdict = "insufficient-data"
            advice.notes.append(
                f"no metal has {MIN_SERIES} composition-resolved samples with "
                "both a vacancy fraction and a cell edge")
            return

        worst = max(abs(v) for v in advice.lattice_residual_A.values())
        # 0.05 A is roughly Rietveld reproducibility on a lab diffractometer for
        # a cubic cell of this size; beyond that the model's composition
        # dependence is wrong by more than the measurement can excuse.
        advice.lattice_verdict = "agrees" if worst <= 0.05 else "disagrees"
        if worst > 0.05:
            advice.notes.append(
                f"predicted composition dependence of the cell edge is wrong by "
                f"{worst:.3f} A (worst metal); the model's structural response to "
                "vacancies does not match the samples")


# ---------------------------------------------------------------------- #


def _measured_rows(experiments):
    """(metal, vacancy_fraction, objective, lattice_a_A) for usable samples."""
    from ..analysis.objectives import scalarize

    out = []
    for e in experiments:
        d = getattr(e, "descriptors", None)
        comp = getattr(d, "composition", None) if d else None
        if comp is None or e.objectives is None:
            continue
        y = getattr(comp, "vacancy_fraction", float("nan"))
        if y is None or math.isnan(y):
            # NaN means the assay could not determine it -- notably Fe without
            # CHN.  Excluded rather than imputed.
            continue
        xrd = getattr(d, "xrd", None)
        a = getattr(xrd, "lattice_a_A", float("nan")) if xrd else float("nan")
        out.append((e.parameters.metal, float(y),
                    float(scalarize(e.objectives)),
                    float(a) if a is not None else float("nan")))
    return out


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


__all__ = ["ThermoAdvisor", "ThermoAdvice", "MIN_SERIES"]
