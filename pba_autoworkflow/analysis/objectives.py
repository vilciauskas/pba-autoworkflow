# SPDX-License-Identifier: GPL-3.0-or-later
"""Objective construction and data-quality gating.

Two responsibilities:

**Quality control.**  An automated platform will happily feed a garbage
measurement to its optimizer, and a single mis-indexed pattern in a small dataset
can steer a campaign for many iterations.  :func:`quality_flags` applies the
checks a human would apply while glancing at the data -- did the pattern index,
is the lattice constant physically possible for this analogue, is the composition
charge-balanced, does the gravimetric yield exceed unity -- and returns the
reasons a sample should be quarantined instead of trusted.

**Objectives.**  The campaign is multi-objective: maximize sodium inventory,
minimize the vacancy fraction, and keep the isolated yield high enough for the
material to be worth making.  :func:`compute_objectives` normalizes everything to
"larger is better" and records the constraint values so the optimizer can treat
yield as a feasibility threshold rather than a third thing to trade off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schema import (
    IONIC_RADIUS_A,
    LATTICE_A0_A,
    Objectives,
    SampleDescriptors,
    SynthesisParameters,
)

#: Plausible cubic lattice-constant window per analogue (Angstrom).  Outside this
#: the indexing has almost certainly latched onto the wrong solution.
#: The window is wide enough to admit the real vacancy- and sodium-driven
#: variation in the cell edge (roughly +/-0.15 A over the accessible composition
#: range) but far narrower than the spacing between spurious indexing solutions.
LATTICE_WINDOW_A: dict[str, tuple[float, float]] = {
    m: (a - 0.35, a + 0.35) for m, a in LATTICE_A0_A.items()
}


@dataclass
class QualityReport:
    passed: bool
    flags: list[str]

    def reason(self) -> str:
        return "; ".join(self.flags) if self.flags else "ok"


def quality_flags(desc: SampleDescriptors, params: SynthesisParameters
                  ) -> QualityReport:
    """Reasons this sample's descriptors should not be trusted."""
    flags: list[str] = []

    if desc.xrd is None:
        flags.append("no diffraction data")
    else:
        x = desc.xrd
        if x.n_peaks_indexed < 2:
            flags.append(f"only {x.n_peaks_indexed} reflection(s) indexed")
        if not math.isfinite(x.lattice_a_A):
            flags.append("lattice constant not determined")
        else:
            lo, hi = LATTICE_WINDOW_A[params.metal]
            if not (lo <= x.lattice_a_A <= hi):
                flags.append(
                    f"lattice a={x.lattice_a_A:.3f} A outside "
                    f"[{lo:.2f}, {hi:.2f}] for {params.metal}"
                )
        if math.isfinite(x.fit_residual) and x.fit_residual > 0.12:
            flags.append(f"indexing residual {x.fit_residual:.3f} deg")
        if math.isfinite(x.domain_size_nm) and not (1.0 <= x.domain_size_nm <= 400.0):
            flags.append(f"implausible domain size {x.domain_size_nm:.1f} nm")

    if desc.composition is None:
        flags.append("no elemental assay")
    else:
        c = desc.composition
        if not math.isfinite(c.vacancy_fraction):
            # Undetermined, not zero.  Letting this through would train the
            # surrogate on a fabricated composition.
            flags.append("vacancy fraction not determinable from the assay taken")
        else:
            # Charge balance: Na+ occupancy cannot exceed 2 per completed framework.
            na_ceiling = 2.0 * (1.0 - c.vacancy_fraction) + 0.15
            if c.na_per_fu > na_ceiling:
                flags.append(
                    f"Na={c.na_per_fu:.2f} exceeds charge-balance ceiling "
                    f"{na_ceiling:.2f}"
                )
            # The same threshold catches an implausibly *negative* vacancy, which is
            # the mirror-image failure: more hexacyanoferrate than sites to hold it.
            if c.fe_per_metal > 1.15:
                flags.append(
                    f"Fe/M={c.fe_per_metal:.2f} above framework stoichiometry")

    if desc.isolated_yield is not None and desc.isolated_yield > 1.05:
        flags.append(f"isolated yield {desc.isolated_yield:.2f} exceeds theory")

    if desc.uvvis is not None and desc.uvvis.conversion < 0.02:
        flags.append("essentially no conversion detected")

    return QualityReport(passed=not flags, flags=flags)


#: Objective names in canonical order.  All are maximized.
OBJECTIVE_NAMES: tuple[str, ...] = ("na_inventory", "framework_integrity")

#: Minimum isolated yield for a run to count as feasible.
YIELD_FLOOR = 0.35


def compute_objectives(desc: SampleDescriptors, params: SynthesisParameters,
                       yield_floor: float = YIELD_FLOOR) -> Objectives:
    """Map descriptors onto maximization objectives plus feasibility constraints."""
    comp = desc.composition
    na = comp.na_per_fu if comp is not None else 0.0
    vacancy = comp.vacancy_fraction if comp is not None else 1.0

    values = {
        # Sodium per formula unit, normalized to the theoretical maximum of 2.
        "na_inventory": float(na / 2.0),
        # Completeness of the hexacyanoferrate sublattice.  Not clipped at 1.0:
        # the measured value carries assay error, and truncating it there would
        # make every near-perfect framework score identically, collapsing the
        # Pareto front onto a single point and leaving the planner nothing to
        # discriminate between good and excellent samples.
        "framework_integrity": float(1.0 - vacancy),
    }
    y = desc.isolated_yield if desc.isolated_yield is not None else 0.0
    constraints = {"isolated_yield": float(y - yield_floor)}
    return Objectives(values=values, constraints=constraints,
                      feasible=constraints["isolated_yield"] >= 0.0)


def scalarize(obj: Objectives, weights: dict[str, float] | None = None,
              infeasible_penalty: float = 0.35) -> float:
    """Single figure of merit for reporting and for single-objective mode.

    Uses an augmented Chebyshev (max-min) scalarization rather than a plain
    weighted sum: the weighted-sum front cannot reach non-convex regions of the
    Pareto set, and PBA composition space is demonstrably non-convex in the
    Na/vacancy trade-off.

    The feasibility handling is lexicographic rather than a soft penalty: any
    constraint violation returns a strictly negative score whose magnitude grows
    with the shortfall, while every feasible point scores at or above zero.  A
    tunable penalty subtracted from the objective term cannot guarantee this --
    with a large enough objective value an infeasible run outranks a feasible one,
    and the campaign then reports a "best" recipe that did not produce enough
    material to characterize.  Ordering by feasibility first removes that failure
    mode, and ranking the infeasible points among themselves by shortfall keeps
    the signal that tells the optimizer which direction restores feasibility.
    """
    w = weights or {n: 1.0 / len(OBJECTIVE_NAMES) for n in OBJECTIVE_NAMES}
    terms = [w[n] * obj.values.get(n, 0.0) for n in OBJECTIVE_NAMES]
    if not obj.feasible:
        shortfall = sum(-min(0.0, v) for v in obj.constraints.values())
        return float(-infeasible_penalty * (1.0 + shortfall))
    # Augmented Chebyshev: the min term drives the front, the sum term breaks ties
    # and keeps weakly dominated points from being scored identically.
    return float(min(terms) + 0.05 * sum(terms))
