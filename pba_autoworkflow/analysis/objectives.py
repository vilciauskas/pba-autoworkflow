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

#: Objectives a campaign can choose (``CampaignConfig.objectives``).  All are
#: scaled to [0, 1] with 0 the worst value, because the hypervolume uses the
#: origin as its reference point.  Electrochemical names are generated from the
#: ion table (:mod:`pba_autoworkflow.ions`), for any ions ``a``, ``b`` there:
#:
#: * ``<a>_<b>_selectivity``: separation factor alpha_a/b from competitive
#:   insertion out of a mixed a/b electrolyte, mapped as 0.5 + log10(alpha)/8
#:   (alpha = 1 -> 0.5; 1e4 -> 1; 1e-4 -> 0).  ``<b>_<a>_selectivity`` is
#:   1 minus it.  e.g. ``k_zn_selectivity``, ``li_na_selectivity``.
#: * ``<a>_retention``: discharge capacity retention in the a electrolyte.
#: * ``<a>_capacity``: second-cycle discharge capacity in a, / 150 mAh/g.
#: * ``framework_stability``: 1 - fraction of the electrode's Fe dissolved
#:   into the mixed electrolyte(s).
#: * ``zn_tolerance``: alias of ``zn_k_selectivity`` (Zn2+ uptake despite K+).
XRD_OBJECTIVES = ("target_phase_fraction", "crystallinity")
ALIASES = {"zn_tolerance": "zn_k_selectivity"}
#: alpha range mapped onto [0, 1]: 10^-4 .. 10^4.  Changing it changes the
#: scores of stored campaigns, which are not recomputed on resume.
SELECTIVITY_SCALE_DECADES = 4.0


def parse_objective(name: str) -> tuple:
    """``("xrd", name)``, ``("selectivity", a, b)``, ``("retention", a)``,
    ``("capacity", a)`` or ``("stability",)``.  Raises on unknown names."""
    from .. import ions as I
    if name in XRD_OBJECTIVES:
        return ("xrd", name)
    if name == "framework_stability":
        return ("stability",)
    canon = ALIASES.get(name, name)
    parts = canon.split("_")
    try:
        if len(parts) == 3 and parts[2] == "selectivity":
            a, b = I.from_key(parts[0]), I.from_key(parts[1])
            if a == b:
                raise ValueError
            return ("selectivity", a, b)
        if len(parts) == 2 and parts[1] in ("retention", "capacity"):
            return (parts[1], I.from_key(parts[0]))
    except ValueError:
        pass
    raise ValueError(f"unknown objective {name!r}; see objective_catalogue()")


def objective_catalogue() -> dict[str, str]:
    """Every objective name with its measurement ("xrd" or "echem")."""
    from .. import ions as I
    out = {n: "xrd" for n in XRD_OBJECTIVES}
    keys = [I.key(x) for x in I.IONS]
    for a in keys:
        for b in keys:
            if a != b:
                out[f"{a}_{b}_selectivity"] = "echem"
        out[f"{a}_retention"] = "echem"
        out[f"{a}_capacity"] = "echem"
    out["framework_stability"] = "echem"
    out.update({k: "echem" for k in ALIASES})
    return out


#: Kept for callers of the earlier fixed table.
OBJECTIVE_CATALOGUE = objective_catalogue()


def echem_requirements(name: str) -> tuple[tuple[str, ...], tuple[dict[str, float], ...]]:
    """(single-ion electrolytes, mixed electrolytes) an objective needs."""
    from .. import ions as I
    kind = parse_objective(name)
    if kind[0] == "selectivity":
        a, b = kind[1], kind[2]
        return tuple(sorted((a, b))), (I.competitive_electrolyte(a, b),)
    if kind[0] in ("retention", "capacity"):
        return (kind[1],), ()
    if kind[0] == "stability":
        return (), ()          # uses whatever mixed electrolytes run; Zn/K if none
    return (), ()


def echem_plan(objectives) -> tuple[tuple[str, ...], tuple[dict[str, float], ...]]:
    """Union of the electrolytes the chosen objectives need."""
    from .. import ions as I
    single: list[str] = []
    mixed: list[dict[str, float]] = []
    for o in objectives:
        s, m = echem_requirements(o)
        single += [i for i in s if i not in single]
        mixed += [d for d in m if d not in mixed]
    if "framework_stability" in objectives and not mixed:
        mixed.append(I.competitive_electrolyte("Zn", "K"))
    return tuple(single), tuple(mixed)


def validate_objectives(objectives) -> tuple[str, ...]:
    objectives = tuple(objectives)
    if len(objectives) < 1 or len(set(objectives)) != len(objectives):
        raise ValueError(f"objectives must be distinct and non-empty: {objectives}")
    for o in objectives:
        try:
            parse_objective(o)
        except ValueError:
            raise ValueError(f"unknown objective(s) [{o!r}]; e.g. {list(XRD_OBJECTIVES)}, "
                             "'k_zn_selectivity', 'li_retention', 'framework_stability' "
                             "(see objective_catalogue())") from None
    return objectives


def is_echem(name: str) -> bool:
    return parse_objective(name)[0] != "xrd"


def _selectivity_score(alpha: float) -> float:
    if not math.isfinite(alpha):
        return 1.0 if alpha == float("inf") else 0.0
    if alpha <= 0:
        return 0.0
    return float(np.clip(0.5 + math.log10(alpha) / (2 * SELECTIVITY_SCALE_DECADES), 0.0, 1.0))


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


def target_phase_fraction_sd(desc: SampleDescriptors, target_phase: str) -> float:
    """Standard uncertainty of :func:`target_phase_fraction` (0 if not reported)."""
    x = desc.xrd
    if x is None:
        return 0.0
    sd = (x.phase_fraction_sd or {}).get(target_phase, 0.0)
    if sd is None or not math.isfinite(sd):
        return 0.0
    return float(sd * (1.0 - x.unidentified_fraction))


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
    from .. import ions as I

    def value(name: str) -> float:
        kind = parse_objective(name)
        if kind[0] == "xrd":
            if name == "target_phase_fraction":
                # Polymorph selectivity: how much of the crystalline product is
                # the phase this campaign wants (Hill-Howard weight fraction).
                return target_phase_fraction(desc, target_phase)
            # Order: ordered framework scattering against that plus the halo.
            return float(x.crystallinity_index) if x is not None else 0.0
        if ec is None:
            return 0.0
        if kind[0] == "selectivity":
            a, b = kind[1], kind[2]
            key = I.pair_name(a, b)
            alpha = ec.separation_factor.get(key, float("nan"))
            if alpha is None or (isinstance(alpha, float) and math.isnan(alpha)):
                return 0.0
            score = _selectivity_score(alpha)
            return score if key == f"{a}/{b}" else 1.0 - score
        if kind[0] == "retention":
            return _finite01(ec.retention.get(kind[1]))
        if kind[0] == "capacity":
            return _finite01(ec.capacity_mAh_g.get(kind[1], float("nan")) / 150.0)
        return (_finite01(1.0 - ec.dissolved_fe_fraction)
                if math.isfinite(ec.dissolved_fe_fraction) else 0.0)

    values = {n: float(value(n)) for n in objectives}
    y = desc.isolated_yield if desc.isolated_yield is not None else 0.0
    constraints = {"isolated_yield": float(y - yield_floor)}
    uncertainty = {}
    if "target_phase_fraction" in objectives and desc.xrd is not None:
        sd = target_phase_fraction_sd(desc, target_phase)
        if sd > 0:
            uncertainty["target_phase_fraction"] = sd
    return Objectives(values=values, constraints=constraints,
                      feasible=constraints["isolated_yield"] >= 0.0, uncertainty=uncertainty)


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
