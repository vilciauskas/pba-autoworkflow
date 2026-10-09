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

**Objectives.**  The campaign targets the formation of one chosen phase,
judged from powder XRD: maximize the weight fraction of the target phase in the
crystalline product (polymorph and secondary-phase selectivity) and the
crystallinity of the framework (Bragg order rather than amorphous scattering),
and keep the isolated yield high enough for the material to be worth making.  The ICP
composition is still measured -- it feeds the yield and the charge-balance
checks -- but it is not optimized.  :func:`compute_objectives` normalizes everything to
"larger is better" and records the constraint values so the optimizer can treat
yield as a feasibility threshold rather than a third thing to trade off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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
        # An amorphous product is a legitimate (poor) outcome that the phase
        # objectives score directly; quarantining it would hide from the
        # optimiser exactly where phase formation fails.  The structural
        # checks below only make sense for an indexed crystalline pattern.
        if x.phase != "amorphous":
            if x.n_peaks_indexed < 2:
                flags.append(f"only {x.n_peaks_indexed} reflection(s) indexed")
            # The cubic window applies when the cubic framework dominates; the
            # other polymorphs are refined inside a window around their own
            # reference cell, so an implausible cell cannot be returned.
            if x.phase != "pba_fm3m":
                pass
            elif not math.isfinite(x.lattice_a_A):
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
            # Charge balance: A+ (Na + K) occupancy cannot exceed 2 per completed
            # framework.  (Loose: the exact Fe(II) ceiling is 4(1 - y) - 2.)
            na_ceiling = 2.0 * (1.0 - c.vacancy_fraction) + 0.15
            if c.a_per_fu > na_ceiling:
                flags.append(
                    f"Na+K={c.a_per_fu:.2f} exceeds charge-balance ceiling "
                    f"{na_ceiling:.2f}"
                )
            # The same threshold catches an implausibly *negative* vacancy, which is
            # the mirror-image failure: more hexacyanoferrate than sites to hold it.
            if c.fe_per_metal > 1.15:
                flags.append(
                    f"Fe/M={c.fe_per_metal:.2f} above framework stoichiometry")

    if desc.isolated_yield is not None and desc.isolated_yield > 1.05:
        flags.append(f"isolated yield {desc.isolated_yield:.2f} exceeds theory")

    return QualityReport(passed=not flags, flags=flags)


#: Default objectives, in canonical order.  All are maximized.
OBJECTIVE_NAMES: tuple[str, ...] = ("target_phase_fraction", "crystallinity")

#: Every objective a campaign can choose (``CampaignConfig.objectives``).  All
#: are scaled to [0, 1] with 0 the worst value, because the hypervolume uses the
#: origin as its reference point.
#:
#: * ``*_selectivity``: separation factor alpha_A/B from competitive insertion,
#:   mapped as 0.5 + log10(alpha)/8 (alpha = 1 -> 0.5; 1e4 -> 1; 1e-4 -> 0).
#: * ``zn_tolerance``: 1 - k_zn_selectivity (prefers Zn2+ uptake despite K+).
#: * ``zn_retention``: capacity retention in the Zn2+ electrolyte.
#: * ``zn_capacity``: second-cycle capacity in Zn2+, / 150 mAh/g.
#: * ``framework_stability``: 1 - Fe dissolved into the mixed electrolyte.
OBJECTIVE_CATALOGUE: dict[str, str] = {
    "target_phase_fraction": "xrd",
    "crystallinity": "xrd",
    "k_zn_selectivity": "echem",
    "na_zn_selectivity": "echem",
    "k_na_selectivity": "echem",
    "zn_tolerance": "echem",
    "zn_retention": "echem",
    "zn_capacity": "echem",
    "framework_stability": "echem",
}

#: Single-ion electrolytes (1 mol/L each) and mixed electrolytes each
#: electrochemical objective needs.  The minor ion's concentration is chosen so
#: that both ions take a measurable share of the charge for a formal-potential
#: gap of ~0.1-0.25 V: at 0.1 M K+ against 1 M Zn2+, K+ carries > 99 % and the
#: Zn uptake (by coulometric difference) is lost in the noise.
ECHEM_REQUIREMENTS: dict[str, tuple[tuple[str, ...], tuple[dict[str, float], ...]]] = {
    "k_zn_selectivity": (("Zn", "K"), ({"Zn": 1.0, "K": 0.005},)),
    "zn_tolerance": (("Zn", "K"), ({"Zn": 1.0, "K": 0.005},)),
    "na_zn_selectivity": (("Zn", "Na"), ({"Zn": 1.0, "Na": 0.02},)),
    "k_na_selectivity": (("Na", "K"), ({"Na": 1.0, "K": 0.01},)),
    "zn_retention": (("Zn",), ()),
    "zn_capacity": (("Zn",), ()),
    "framework_stability": ((), ({"Zn": 1.0, "K": 0.005},)),
}
_PAIR = {"k_zn_selectivity": "K/Zn", "na_zn_selectivity": "Na/Zn", "k_na_selectivity": "K/Na"}


def echem_plan(objectives) -> tuple[tuple[str, ...], tuple[dict[str, float], ...]]:
    """Union of the electrolytes the chosen objectives need."""
    single: list[str] = []
    mixed: list[dict[str, float]] = []
    for o in objectives:
        s, m = ECHEM_REQUIREMENTS.get(o, ((), ()))
        single += [i for i in s if i not in single]
        mixed += [d for d in m if d not in mixed]
    return tuple(single), tuple(mixed)


def validate_objectives(objectives) -> tuple[str, ...]:
    objectives = tuple(objectives)
    unknown = [o for o in objectives if o not in OBJECTIVE_CATALOGUE]
    if unknown:
        raise ValueError(f"unknown objective(s) {unknown}; choose from {sorted(OBJECTIVE_CATALOGUE)}")
    if len(objectives) < 1 or len(set(objectives)) != len(objectives):
        raise ValueError(f"objectives must be distinct and non-empty: {objectives}")
    return objectives


def _selectivity_score(alpha: float) -> float:
    if not math.isfinite(alpha):
        return 1.0 if alpha == float("inf") else 0.0
    if alpha <= 0:
        return 0.0
    return float(np.clip(0.5 + math.log10(alpha) / 8.0, 0.0, 1.0))


def _finite01(v, default: float = 0.0) -> float:
    return float(np.clip(v, 0.0, 1.0)) if v is not None and math.isfinite(v) else default

#: Target phase when a campaign does not name one.
DEFAULT_TARGET_PHASE = "pba_fm3m"

#: Minimum isolated yield for a run to count as feasible.
YIELD_FLOOR = 0.35


def target_phase_fraction(desc: SampleDescriptors, target_phase: str) -> float:
    """Weight fraction of ``target_phase`` in the crystalline product.

    Discounted by the share of Bragg intensity no library phase explains: an
    unknown crystalline phase is a real impurity, but its weight cannot be
    estimated without a structure, so its intensity share is used instead.
    """
    x = desc.xrd
    if x is None:
        return 0.0
    w = float(x.phase_fractions.get(target_phase, 0.0))
    return float(np.clip(w * (1.0 - x.unidentified_fraction), 0.0, 1.0))


def compute_objectives(desc: SampleDescriptors, params: SynthesisParameters,
                       yield_floor: float = YIELD_FLOOR,
                       target_phase: str = DEFAULT_TARGET_PHASE,
                       objectives: tuple[str, ...] = OBJECTIVE_NAMES) -> Objectives:
    """Map descriptors onto maximization objectives plus feasibility constraints.

    A missing measurement scores 0 (the worst value), as a missing pattern
    always has.
    """
    x = desc.xrd
    ec = desc.echem
    all_values = {
        # Polymorph selectivity: how much of the crystalline product is the
        # phase this campaign wants (Hill-Howard weight fraction).
        "target_phase_fraction": lambda: target_phase_fraction(desc, target_phase),
        # Order: ordered framework scattering against that plus amorphous halo,
        # independent of which framework polymorph formed.
        "crystallinity": lambda: float(x.crystallinity_index) if x is not None else 0.0,
        "zn_retention": lambda: _finite01(ec.retention.get("Zn")) if ec else 0.0,
        "zn_capacity": lambda: _finite01(ec.capacity_mAh_g.get("Zn", float("nan")) / 150.0) if ec else 0.0,
        "framework_stability": lambda: (_finite01(1.0 - ec.dissolved_fe_fraction)
                                        if ec and math.isfinite(ec.dissolved_fe_fraction) else 0.0),
    }
    for name, pair in _PAIR.items():
        all_values[name] = (lambda pair=pair: _selectivity_score(ec.separation_factor.get(pair, float("nan")))
                            if ec else 0.0)
    all_values["zn_tolerance"] = lambda: (1.0 - _selectivity_score(ec.separation_factor["K/Zn"])
                                          if ec and "K/Zn" in ec.separation_factor
                                          and not math.isnan(ec.separation_factor["K/Zn"]) else 0.0)
    values = {n: float(all_values[n]()) for n in objectives}
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
    names = list(obj.values) or list(OBJECTIVE_NAMES)
    w = weights or {n: 1.0 / len(names) for n in names}
    terms = [w[n] * obj.values.get(n, 0.0) for n in w]
    if not obj.feasible:
        shortfall = sum(-min(0.0, v) for v in obj.constraints.values())
        return float(-infeasible_penalty * (1.0 + shortfall))
    # Augmented Chebyshev: the min term drives the front, the sum term breaks ties
    # and keeps weakly dominated points from being scored identically.
    return float(min(terms) + 0.05 * sum(terms))
