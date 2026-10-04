# SPDX-License-Identifier: GPL-3.0-or-later
"""The Na/vacancy pseudo-binary convex hull.

What this computes, precisely
-----------------------------
For one N-site metal, the mixing energy of Na_x M[Fe(CN)6]_(1-y) relative to the
two framework endpoints:

    E_mix(x, y) = E(x, y) - (1 - y) E(x_full, 0) - y E(x_empty, 1)

with the endpoints being the fully intact framework and the fully vacant one, each
at its own charge-balanced Na content.  The lower convex hull of E_mix against y
then separates compositions that are stable against phase separation (on the hull)
from those that would demix into a mixture of two hull compositions (above it).
The vertical distance above the hull is the driving force for that demixing, in
eV per formula unit.

What it does *not* compute
--------------------------
This is not a formation-energy hull.  It says nothing about whether the PBA is
stable against decomposition into metal oxides, cyanides or the aqueous ions --
only about which *framework compositions* are stable relative to each other.  That
restriction is deliberate and is what makes the result defensible: the errors of a
foundation force field on this chemistry are large in absolute terms and largely
systematic along a homologous series, so differences within a series survive where
absolute energies do not.

Water: why the reference does *not* rescue this hull
----------------------------------------------------
A vacancy brings six aqua ligands, so the water count varies along the vacancy
axis and one might expect a water chemical potential to be needed.  It is
recorded (:func:`water_reference_eV`, ``mu_water_eV`` on the result) because a
grand potential in water is the physically correct object -- but for *this* hull
it changes nothing, and the reason is worth stating so nobody re-derives it:

    n_H2O = 6y is linear in y, and the mixing energy subtracts a straight line in
    y.  Any term linear in y therefore cancels identically, for any number of
    compositions.  Verified numerically: E_mix is bit-for-bit identical with and
    without the correction.

The reference matters for a *formation-energy* hull, where no such subtraction
happens, and it would matter here if the waters-per-vacancy ratio varied along the
series.  Neither applies, so it is kept for correctness of the recorded quantity
and not relied upon.

Which means the large features this module currently reports on MACE-MP -- of
order -900 meV/f.u. -- are **not** a bookkeeping artefact that a reference fixes.
They are what the force field says, and at roughly fifty times the thermal scale
they are not physically credible for vacancy mixing in a framework this open.
Together with the failed lattice-trend validation (rank correlation -0.70 across
the metal series, :func:`pba_autoworkflow.thermo.cli.cmd_validate`) the conclusion is that
MACE-MP is not quantitative for this chemistry.  The machinery here is correct and
tested; the numbers it currently produces should be treated as untrustworthy until
the force field is replaced or fine-tuned on PBA reference data.

The T = 0 K caveat
------------------
A convex hull is a zero-temperature construction.  Configurational entropy at
synthesis temperature will shrink or close a shallow miscibility gap: the rough
scale is k_B T ln(2) ~ 18 meV per site at 25 C, so a hull depth below ~20 meV/f.u.
should not be read as a real two-phase region.  :func:`hull_report` compares each
depth against that scale rather than leaving the reader to do it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .mlff import EnergyModel, EnergyResult
from .structures import PBAComposition, build_pba, charge_balanced_na

#: Boltzmann constant, eV/K.
K_B_EV = 8.617333262e-5

#: A mixing energy larger than this is not credible for hexacyanoferrate vacancy
#: ordering, whatever the force field reports.  Vacancy-vacancy interactions in a
#: framework that is roughly half empty by volume are weak; published cluster
#: expansions for related systems put the scale at tens of meV per formula unit.
#: Treating anything near an electron-volt as a real miscibility gap would be
#: reading a model failure as a physical result.
IMPLAUSIBLE_FEATURE_EV = 0.20


@dataclass
class HullPoint:
    """One composition on or above the hull."""

    na_per_fu: float
    vacancy_fraction: float
    energy_per_fu_eV: float
    #: ``energy_per_fu_eV`` minus ``n_water * mu_H2O``: the grand potential in
    #: water, which is what the hull is actually built from.  Equal to the raw
    #: energy when no water reference was supplied.
    grand_potential_per_fu_eV: float
    mixing_energy_eV: float
    #: Vertical distance above the lower convex hull, eV per formula unit.  Zero
    #: (within tolerance) means the composition is a hull vertex.
    e_above_hull_eV: float
    on_hull: bool
    lattice_a_A: float
    converged: bool
    warnings: list[str] = field(default_factory=list)
    #: Standard deviation over configurational samples, if measured.  ``None``
    #: means a single decoration was used and the spread is unknown -- not zero.
    config_std_eV: float | None = None
    formula: str = ""


@dataclass
class HullResult:
    """The hull for one metal, plus what it means at finite temperature."""

    metal: str
    points: list[HullPoint]
    model_name: str
    supercell: tuple[int, int, int]
    temperature_C: float
    #: Largest hull *feature*, eV/f.u., signed by its character.  Negative means
    #: an ordering tendency: some intermediate composition lies below the endpoint
    #: tie-line and is a stable intermediate phase.  Positive means a demixing
    #: tendency: the intermediates lie above the tie-line and the system prefers a
    #: two-phase mixture of the endpoints.
    #:
    #: Taking ``min(e_mix)`` alone (an earlier version) reports 0.0 for a purely
    #: demixing system -- every interior point is *above* the hull, so the minimum
    #: is just the endpoint -- which then reads as "no resolvable feature" and
    #: silently switched the campaign prior to uninformative for exactly the
    #: systems whose miscibility gap it was built to find.
    max_hull_depth_eV: float
    #: k_B T ln 2 at ``temperature_C``, the configurational entropy scale.
    entropy_scale_eV: float
    #: Water chemical potential subtracted, eV per molecule.  ``None`` means no
    #: correction was applied, in which case mixing energies along the vacancy axis
    #: contain the energy of forming water from nothing and are not usable.
    mu_water_eV: float | None = None
    #: Problems with the composition path itself, as opposed to any single point.
    #: A non-empty list means the mixing energies are not interpretable -- see
    #: :func:`check_path_linearity`.
    path_warnings: list[str] = field(default_factory=list)

    @property
    def hull_vertices(self) -> list[HullPoint]:
        return [p for p in self.points if p.on_hull]

    @property
    def resolvable(self) -> bool:
        """Is the deepest hull feature larger than the entropy scale?

        If not, the hull is reporting structure that thermal disorder erases at
        synthesis temperature, and it should not be used to exclude compositions.
        """
        if self.path_warnings:
            return False   # the energies are not interpretable at all
        return abs(self.max_hull_depth_eV) > self.entropy_scale_eV

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([{
            "na_per_fu": p.na_per_fu, "vacancy_fraction": p.vacancy_fraction,
            "energy_per_fu_eV": p.energy_per_fu_eV,
            "grand_potential_per_fu_eV": p.grand_potential_per_fu_eV,
            "mixing_energy_eV": p.mixing_energy_eV,
            "e_above_hull_eV": p.e_above_hull_eV, "on_hull": p.on_hull,
            "lattice_a_A": p.lattice_a_A, "converged": p.converged,
            "config_std_eV": p.config_std_eV,
            "n_warnings": len(p.warnings), "formula": p.formula,
        } for p in self.points])


def lower_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the lower convex hull of the 2-D point set, left to right.

    Implemented directly rather than via ``scipy.spatial.ConvexHull`` because that
    returns the *full* hull and separating the lower branch from the upper one is
    the part that is easy to get wrong -- and because it degenerates on fewer than
    three points, which is a legitimate case here (two endpoints and nothing else).
    """
    order = np.argsort(x)
    stack: list[int] = []
    for idx in order:
        while len(stack) >= 2:
            x1, y1 = x[stack[-2]], y[stack[-2]]
            x2, y2 = x[stack[-1]], y[stack[-1]]
            # Drop the middle point if it sits on or above the line through its
            # neighbours (cross product >= 0 means non-convex downward).
            cross = (x2 - x1) * (y[idx] - y1) - (y2 - y1) * (x[idx] - x1)
            if cross <= 0:
                stack.pop()
            else:
                break
        stack.append(int(idx))
    return np.array(stack, dtype=int)


def _hull_distance(x: np.ndarray, y: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Vertical distance of every point above the piecewise-linear lower hull."""
    hx, hy = x[verts], y[verts]
    return y - np.interp(x, hx, hy)


def compute_hull(
    metal: str,
    model: EnergyModel,
    vacancy_values: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0),
    supercell: tuple[int, int, int] | None = None,
    temperature_C: float = 25.0,
    use_sqs: bool = True,
    n_config_samples: int = 0,
    fmax: float = 0.05,
    steps: int = 300,
    seed: int = 0,
    hull_tol_eV: float = 1e-4,
    mu_water_eV: float | None = None,
    auto_water_reference: bool = True,
    progress: bool = False,
) -> HullResult:
    """Relax a composition series and construct the pseudo-binary hull.

    Parameters
    ----------
    vacancy_values:
        Vacancy fractions to evaluate.  Na content is pinned to the charge-balanced
        value at each, which is the physically accessible line; see
        :func:`pba_autoworkflow.thermo.structures.composition_grid` for the wider rectangle.
    use_sqs:
        Choose each decoration by SQS rather than at random.  Strongly recommended:
        a random decoration's energy carries configurational scatter that is easily
        as large as the hull features being resolved.
    n_config_samples:
        If > 0, also relax this many random decorations per composition and report
        their standard deviation.  This is how you find out whether a hull depth is
        larger than the configurational noise -- worth doing once for a given cell
        size, then not again.
    """
    from .structures import min_supercell_for

    if mu_water_eV is None and auto_water_reference and any(
            y > 0 for y in vacancy_values):
        mu_water_eV = water_reference_eV(model)

    if supercell is None:
        na_targets = tuple(charge_balanced_na(y) / 2.0 for y in vacancy_values)
        try:
            supercell = min_supercell_for(vacancy_values, na_targets)
        except ValueError:
            supercell = (2, 2, 2)

    rows: list[tuple[PBAComposition, EnergyResult, float | None]] = []
    for i, y in enumerate(vacancy_values):
        comp = PBAComposition(metal, charge_balanced_na(y), y)
        atoms = _decoration(comp, supercell, use_sqs, seed + i)
        res = model.relax(atoms, fmax=fmax, steps=steps)

        std: float | None = None
        if n_config_samples > 0:
            from .sqs import sample_decorations

            energies = [
                model.relax(a, fmax=fmax, steps=steps).energy_per_fu_eV
                for a in sample_decorations(comp, n_samples=n_config_samples,
                                            supercell=supercell, seed=seed + 100 + i)
            ]
            std = float(np.std(energies, ddof=1)) if len(energies) > 1 else None

        rows.append((comp, res, std))
        if progress:
            print(f"  y={y:.3f} E/fu={res.energy_per_fu_eV:.4f} "
                  f"conv={res.converged} warn={len(res.extrapolation_warnings)}",
                  flush=True)

    # Mixing energy against the two framework endpoints.  Endpoints are whichever
    # evaluated compositions have the least and most vacancy, so the construction
    # works on a partial series too.
    ys = np.array([c.vacancy_fraction for c, _, _ in rows])
    es = np.array([r.energy_per_fu_eV for _, r, _ in rows])
    i_lo, i_hi = int(np.argmin(ys)), int(np.argmax(ys))
    y_lo, y_hi = ys[i_lo], ys[i_hi]
    if math.isclose(y_lo, y_hi):
        raise ValueError("hull needs at least two distinct vacancy fractions")

    n_w = np.array([n_water_per_fu(float(y)) for y in ys])
    omega = es - (mu_water_eV * n_w if mu_water_eV is not None else 0.0)
    frac = (ys - y_lo) / (y_hi - y_lo)
    e_mix = omega - ((1.0 - frac) * omega[i_lo] + frac * omega[i_hi])

    verts = lower_hull(ys, e_mix)
    above = _hull_distance(ys, e_mix, verts)
    vertex_set = set(verts.tolist())

    points = [
        HullPoint(
            na_per_fu=float(r.relaxed.info.get("realized_na_per_fu", c.na_per_fu))
            if r.relaxed is not None else c.na_per_fu,
            vacancy_fraction=float(c.vacancy_fraction),
            energy_per_fu_eV=float(r.energy_per_fu_eV),
            grand_potential_per_fu_eV=float(omega[k]),
            mixing_energy_eV=float(e_mix[k]),
            e_above_hull_eV=float(max(0.0, above[k])),
            on_hull=bool(k in vertex_set or above[k] <= hull_tol_eV),
            lattice_a_A=float(r.lattice_a_A), converged=bool(r.converged),
            warnings=list(r.extrapolation_warnings), config_std_eV=std,
            formula=c.formula,
        )
        for k, (c, r, std) in enumerate(rows)
    ]

    t_kelvin = temperature_C + 273.15
    return HullResult(
        metal=metal, points=points, model_name=getattr(model, "name", "unknown"),
        supercell=tuple(supercell), temperature_C=temperature_C,
        max_hull_depth_eV=_signed_feature(e_mix),
        entropy_scale_eV=float(K_B_EV * t_kelvin * math.log(2.0)),
        mu_water_eV=mu_water_eV,
        path_warnings=check_path_linearity(
            tuple(float(c.vacancy_fraction) for c, _, _ in rows),
            tuple(float(c.na_per_fu) for c, _, _ in rows)),
    )


def water_reference_eV(model: "EnergyModel", model_size: str = "small") -> float:
    """Energy of one isolated water molecule under the same force field.

    ``model`` is REQUIRED.  It used to default to ``None`` and silently build a
    MACE water molecule, while ``getattr(None, "name", ...)`` filed the result
    under a ``mace-mp-*`` cache key -- so a hull computed with UMA or PET-MAD
    could be referenced against a MACE water without anything saying so.  This
    function's own contract is that the reference must come from the same model
    as the framework energies, because a cross-model reference introduces an
    error larger than the features being resolved.  A contract that matters
    that much is enforced rather than documented.
    """
    if model is None:
        raise ValueError(
            "water_reference_eV requires the same model that produced the "
            "framework energies; a cross-model water reference does not cancel "
            "in the grand potential."
        )
    global _WATER_CACHE
    key = getattr(model, "name", None)
    if key is None:
        raise ValueError(
            f"model {type(model).__name__} has no .name, so its water reference "
            "cannot be cached without risking a collision with another model"
        )
    if key in _WATER_CACHE:
        return _WATER_CACHE[key]

    from ase import Atoms as _Atoms

    # Gas-phase geometry; the relaxation refines it under the model.
    mol = _Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0],
                                   [-0.2400, 0.9266, 0.0]],
                 cell=np.eye(3) * 12.0, pbc=True)
    mol.info["n_formula_units"] = 1

    res = model.relax(mol, fmax=0.02, steps=200, relax_cell=False)
    _WATER_CACHE[key] = float(res.energy_eV)
    return _WATER_CACHE[key]


_WATER_CACHE: dict[str, float] = {}


def n_water_per_fu(vacancy_fraction: float, waters_per_vacancy: int = 6) -> float:
    """Aqua ligands per formula unit implied by the vacancy content.

    Six per vacancy: the missing hexacyanoferrate leaves six under-coordinated M
    sites, each completed by one water.  This mirrors what
    :func:`pba_autoworkflow.thermo.structures.build_pba` actually places, and the two must
    agree or the correction removes the wrong amount.
    """
    return waters_per_vacancy * vacancy_fraction


def check_path_linearity(vacancy_values: tuple[float, ...],
                         na_values: tuple[float, ...]) -> list[str]:
    """Verify the composition path is linear in every species.

    A pseudo-binary mixing energy subtracts a straight line between two
    endpoints.  That is only meaningful if the *composition* also moves along a
    straight line: otherwise the subtraction leaves an uncancelled chemical
    potential, and the residual -- which can be an electron-volt per formula unit
    for a species as strongly bound as sodium -- is reported as a mixing energy.

    On the charge-balanced line Na = 2 - 4y, so every count (Fe, C, N, Na, O, H)
    is linear in y -- *until* Na reaches zero at y = 0.5, where
    :func:`~pba_autoworkflow.thermo.structures.charge_balanced_na` clamps.  Beyond that
    point Na stays flat while the vacancy count keeps rising, the path bends, and
    the hull silently becomes uninterpretable.  This function is what makes that
    failure visible instead of producing a plausible-looking 2 eV feature.
    """
    y = np.asarray(vacancy_values, dtype=float)
    x = np.asarray(na_values, dtype=float)
    if y.size < 3:
        return ["fewer than three compositions: there is no interior point, so "
                "every point is trivially a hull vertex and no mixing tendency "
                "can be resolved"]
    if y.size == 3:
        # One interior point is one degree of freedom: the "hull" is a single
        # number and its shape carries no information about where a miscibility
        # gap opens or closes.
        return ["only one interior composition: the hull has a single degree of "
                "freedom, so no composition dependence can be resolved.  Add "
                "points -- at least two interior vacancy fractions -- before "
                "reading a mixing tendency from this series"]
    order = np.argsort(y)
    y, x = y[order], x[order]

    # Straight line through the two endpoints; interior points must lie on it.
    interp = np.interp(y, [y[0], y[-1]], [x[0], x[-1]])
    dev = np.abs(x - interp)
    bad = dev > 1e-6
    if not bad.any():
        return []
    worst = int(np.argmax(dev))
    return [
        f"composition path is not linear: at y={y[worst]:.3f} the sodium content "
        f"is {x[worst]:.3f} but the endpoint tie-line gives {interp[worst]:.3f} "
        f"(deviation {dev[worst]:.3f} Na/f.u.).  Mixing energies along this path "
        "carry an uncancelled sodium chemical potential and are not "
        "interpretable; restrict the series to y <= 0.5, where Na = 2 - 4y is "
        "linear, or subtract a Na reference explicitly."
    ]


def hull_from_energies(
    metal: str, rows: list[dict], model_name: str = "unknown",
    supercell: tuple[int, int, int] = (1, 1, 1), temperature_C: float = 25.0,
    hull_tol_eV: float = 1e-4, mu_water_eV: float | None = None,
) -> HullResult:
    """Build a hull from already-computed energies.

    Separating the construction from the expensive relaxations means a hull can be
    re-drawn, re-plotted or re-analysed at a different temperature without paying
    for the force field again -- and it keeps :func:`compute_hull` free to be the
    thin wrapper that it is.

    ``rows`` are dicts as written by ``scripts/run_hull.py``: ``vacancy_fraction``,
    ``energy_per_fu_eV``, ``lattice_a_A``, ``converged``, and optionally
    ``na_per_fu``, ``warnings``, ``config_std_eV``, ``formula``.
    """
    rows = sorted(rows, key=lambda r: r["vacancy_fraction"])
    ys = np.array([float(r["vacancy_fraction"]) for r in rows])
    es = np.array([float(r["energy_per_fu_eV"]) for r in rows])
    if ys.size < 2 or math.isclose(ys.min(), ys.max()):
        raise ValueError("hull needs at least two distinct vacancy fractions")

    # Subtract the water reservoir before anything else: the vacancy axis changes
    # the water count, and without this the "mixing energy" is dominated by the
    # energy of forming that water.
    n_w = np.array([n_water_per_fu(float(r["vacancy_fraction"])) for r in rows])
    omega = es - (mu_water_eV * n_w if mu_water_eV is not None else 0.0)

    i_lo, i_hi = int(np.argmin(ys)), int(np.argmax(ys))
    frac = (ys - ys[i_lo]) / (ys[i_hi] - ys[i_lo])
    e_mix = omega - ((1.0 - frac) * omega[i_lo] + frac * omega[i_hi])

    verts = lower_hull(ys, e_mix)
    above = _hull_distance(ys, e_mix, verts)
    vertex_set = set(verts.tolist())

    points = [
        HullPoint(
            na_per_fu=float(r.get("na_per_fu", float("nan"))),
            vacancy_fraction=float(r["vacancy_fraction"]),
            energy_per_fu_eV=float(r["energy_per_fu_eV"]),
            grand_potential_per_fu_eV=float(omega[k]),
            mixing_energy_eV=float(e_mix[k]),
            e_above_hull_eV=float(max(0.0, above[k])),
            on_hull=bool(k in vertex_set or above[k] <= hull_tol_eV),
            lattice_a_A=float(r.get("lattice_a_A", float("nan"))),
            converged=bool(r.get("converged", False)),
            warnings=list(r.get("warnings", [])),
            config_std_eV=(float(r["config_std_eV"])
                           if r.get("config_std_eV") is not None else None),
            formula=str(r.get("formula", "")),
        )
        for k, r in enumerate(rows)
    ]

    na = [float(r.get("na_per_fu", float("nan"))) for r in rows]
    path_warnings = ([] if any(math.isnan(v) for v in na)
                     else check_path_linearity(tuple(ys.tolist()), tuple(na)))

    return HullResult(
        metal=metal, points=points, model_name=model_name,
        supercell=tuple(supercell), temperature_C=temperature_C,
        max_hull_depth_eV=_signed_feature(e_mix),
        entropy_scale_eV=float(K_B_EV * (temperature_C + 273.15) * math.log(2.0)),
        mu_water_eV=mu_water_eV,
        path_warnings=path_warnings,
    )


def _signed_feature(e_mix: np.ndarray) -> float:
    """Largest deviation of mixing energy from zero, keeping its sign.

    Ordering (negative E_mix) and demixing (positive E_mix, i.e. points above the
    endpoint tie-line) are both real features of comparable importance to a
    synthesis campaign, so the magnitude reported is whichever is larger.
    """
    lo, hi = float(np.min(e_mix)), float(np.max(e_mix))
    return lo if abs(lo) >= abs(hi) else hi


def _decoration(comp: PBAComposition, supercell, use_sqs: bool, seed: int):
    """SQS decoration when available, random otherwise -- and say which."""
    if use_sqs:
        try:
            from .sqs import generate_sqs

            return generate_sqs(comp, supercell=supercell, seed=seed).atoms
        except ImportError:
            # icet absent: fall back, but leave a marker so the provenance of the
            # energy is visible downstream rather than being silently different.
            atoms = build_pba(comp, supercell=supercell,
                              rng=np.random.default_rng(seed))
            atoms.info["sqs"] = False
            atoms.info["sqs_unavailable"] = True
            return atoms
    return build_pba(comp, supercell=supercell, rng=np.random.default_rng(seed))


def hull_report(result: HullResult) -> str:
    """Human-readable summary that states its own limitations.

    Written as a function rather than a template so that the caveats travel with
    the numbers: a hull depth quoted without the entropy scale and the model
    identity invites exactly the over-reading this module is trying to prevent.
    """
    lines = [
        f"Na/vacancy pseudo-binary hull -- {result.metal} analogue",
        f"  force field      : {result.model_name}",
        f"  supercell        : {result.supercell[0]}x{result.supercell[1]}x{result.supercell[2]}",
        f"  water reference  : "
        + (f"{result.mu_water_eV:.4f} eV/H2O (cancels: n_H2O is linear in y)"
           if result.mu_water_eV is not None else "not applied"),
        f"  temperature      : {result.temperature_C:.0f} C "
        f"(kT ln2 = {result.entropy_scale_eV * 1000:.1f} meV/f.u.)",
        f"  largest feature  : {result.max_hull_depth_eV * 1000:+.1f} meV/f.u. "
        f"({'demixing' if result.max_hull_depth_eV > 0 else 'ordering'})",
        "",
    ]
    for w in result.path_warnings:
        lines.append(f"  !! PATH: {w}")
    if result.path_warnings:
        lines.append("")
    unconverged = [p for p in result.points if not p.converged]
    warned = [p for p in result.points if p.warnings]
    if unconverged:
        lines.append(f"  !! {len(unconverged)} composition(s) did not converge; "
                     "their energies are not usable")
    if warned:
        lines.append(f"  !! {len(warned)} composition(s) tripped a structure check "
                     "(model likely extrapolating)")

    lines.append("  y      Na/f.u.  E_mix (meV)  above hull (meV)  on hull  a (A)")
    for p in sorted(result.points, key=lambda q: q.vacancy_fraction):
        lines.append(
            f"  {p.vacancy_fraction:<6.3f} {p.na_per_fu:<8.2f} "
            f"{p.mixing_energy_eV * 1000:>+11.1f}  {p.e_above_hull_eV * 1000:>15.1f}  "
            f"{'yes' if p.on_hull else 'no ':>7}  {p.lattice_a_A:.3f}"
        )

    lines.append("")
    # Two independent reasons a hull can be unusable, and they need different
    # verdicts: a broken composition path means the numbers are not interpretable
    # at all, while a shallow feature means they are interpretable but erased by
    # thermal disorder.  Reporting the entropy comparison for a path failure
    # produced the self-contradictory line "largest feature (995.9 meV) is below
    # the entropy scale (17.8 meV)".
    if result.path_warnings:
        lines.append(
            "  The composition path or sampling is unsound (see PATH above), so "
            "these mixing energies are not interpretable regardless of their "
            "size.  Fix the path before comparing against any energy scale."
        )
    elif abs(result.max_hull_depth_eV) <= result.entropy_scale_eV:
        lines.append(
            f"  Largest feature ({abs(result.max_hull_depth_eV) * 1000:.1f} meV) is below the "
            f"configurational entropy scale ({result.entropy_scale_eV * 1000:.1f} meV) at "
            f"{result.temperature_C:.0f} C: thermal disorder erases this structure, and "
            "the hull should not be used to exclude compositions."
        )
    elif abs(result.max_hull_depth_eV) > IMPLAUSIBLE_FEATURE_EV:
        lines.append(
            f"  Largest feature ({abs(result.max_hull_depth_eV) * 1000:.0f} meV/f.u.) is "
            f"{abs(result.max_hull_depth_eV) / result.entropy_scale_eV:.0f}x the thermal "
            "scale.  Vacancy mixing in a framework this open does not reach that "
            "magnitude; a feature this large indicates the force field, not the "
            "chemistry.  Run `pba_autoworkflow.thermo validate` and treat these energies as "
            "untrustworthy until the model reproduces the measured lattice series."
        )
    else:
        character = ("demixing: intermediates lie above the endpoint tie-line"
                     if result.max_hull_depth_eV > 0 else
                     "ordering: a stable intermediate phase lies below the tie-line")
        lines.append(
            f"  Largest feature exceeds the entropy scale ({character}), so the "
            "tendency is plausible at this temperature -- subject to the "
            "force-field caveats in the module docstring."
        )
    spreads = [p.config_std_eV for p in result.points if p.config_std_eV is not None]
    if spreads:
        lines.append(f"  Configurational spread: up to {max(spreads) * 1000:.1f} meV/f.u. "
                     "over random decorations at fixed composition.")
    else:
        lines.append("  Configurational spread not measured (n_config_samples=0): "
                     "hull features smaller than that unknown scatter are not resolved.")
    return "\n".join(lines)
