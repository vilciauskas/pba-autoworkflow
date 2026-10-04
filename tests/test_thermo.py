# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the thermodynamics module.

Each test is named for the symptom it prevents, not the API it calls.  Several
encode defects found while building this module -- a framework built with the two
metal sublattices 9 A apart, aqua hydrogens pointing into their coordinating
metal, an SQS cutoff that excluded the first vacancy-vacancy shell, and a
correlation metric that reported a good SQS as bad.
"""

from __future__ import annotations

import math

import pathlib

import numpy as np

from pba_autoworkflow.thermo import cli
import pytest

ase = pytest.importorskip("ase")
from ase.neighborlist import neighbor_list  # noqa: E402

from pba_autoworkflow.thermo.structures import (FU_PER_CELL, LATTICE_A0_A,  # noqa: E402
                                       PBAComposition, build_pba,
                                       charge_balanced_na, composition_grid,
                                       min_supercell_for)
from pba_autoworkflow.thermo.hull import (K_B_EV, HullPoint, HullResult,  # noqa: E402
                                 compute_hull, hull_report, lower_hull)
from pba_autoworkflow.thermo.mlff import AnalyticModel, check_structure  # noqa: E402


def _bonds(atoms):
    i, j, d = neighbor_list("ijd", atoms, cutoff=2.7)
    sym = atoms.get_chemical_symbols()
    out: dict[tuple[str, str], list[float]] = {}
    for a, b, dd in zip(i, j, d):
        out.setdefault(tuple(sorted((sym[a], sym[b]))), []).append(float(dd))
    return out


# --------------------------------------------------------------------------- #
# structure geometry
# --------------------------------------------------------------------------- #

def test_framework_is_rock_salt_not_body_centred():
    """M and Fe must be a/2 apart along <100>, bridged by one cyanide.

    An earlier version put one M at the origin and one Fe at the body centre of
    the *conventional* cell, leaving them 9.1 A apart with no M-N bond at all --
    a structure that is not a Prussian blue analogue, and whose energy means
    nothing.  It also gave 1 formula unit per cell instead of 4, so every Na
    count was off by a factor of four.
    """
    atoms = build_pba(PBAComposition("Mn", 2.0, 0.0), supercell=(1, 1, 1))
    assert atoms.info["n_formula_units"] == FU_PER_CELL == 4
    b = _bonds(atoms)
    assert ("Mn", "N") in b, "M-N bond absent: sublattices are not adjacent"
    assert 2.0 < np.mean(b[("Mn", "N")]) < 2.4


def test_cyanide_bond_lengths_are_physical():
    """C-N and Fe-C are stiff and must not scale with the cell.

    Placing the light atoms at fixed *fractions* of the lattice parameter (an
    earlier version) stretched C-N to 1.26 A and Fe-C to 2.00 A, and made both
    vary from Mn to Cu -- distributing the lattice mismatch across every bond
    instead of the M-N bond that actually flexes.
    """
    for metal in LATTICE_A0_A:
        b = _bonds(build_pba(PBAComposition(metal, 2.0, 0.0), supercell=(1, 1, 1)))
        cn = np.array(b[("C", "N")])
        fec = np.array(b[tuple(sorted(("C", "Fe")))])
        assert cn.min() > 1.05 and cn.max() < 1.25, f"{metal}: C-N {cn.mean():.3f}"
        assert 1.80 < fec.mean() < 2.00, f"{metal}: Fe-C {fec.mean():.3f}"
        assert np.ptp(cn) < 0.01, "C-N should not vary within a structure"


def test_aqua_hydrogens_point_away_from_metal():
    """A hydrogen 1.74 A from the coordinating metal is a close contact.

    The water that completes a vacancy coordination site must have its H pointing
    into the cavity.  The sign error that put them on the metal side produced a
    structure the force field relaxes violently, which then shows up as a spurious
    vacancy formation energy.
    """
    atoms = build_pba(PBAComposition("Mn", 1.0, 0.25), supercell=(1, 1, 1),
                      rng=np.random.default_rng(1))
    b = _bonds(atoms)
    assert ("H", "Mn") not in b, "H within 2.7 A of Mn: water is pointing inwards"
    oh = np.array(b[("H", "O")])
    assert 0.90 < oh.mean() < 1.02


def test_water_geometry_is_water():
    atoms = build_pba(PBAComposition("Mn", 1.0, 0.25), supercell=(1, 1, 1),
                      rng=np.random.default_rng(1))
    sym = atoms.get_chemical_symbols()
    o = sym.index("O")
    hs = [k for k, s in enumerate(sym) if s == "H"][:2]
    assert 100.0 < atoms.get_angle(hs[0], o, hs[1]) < 110.0


def test_composition_is_realized_exactly_or_reported():
    """A cell that cannot hold the requested composition must say so.

    4 formula units cannot represent y = 0.125; rounding it silently to 0.0 and
    labelling the point y = 0.125 attributes a vacancy-free energy to a defective
    composition -- the same class of defect as splitting total Fe 50/50.
    """
    small = build_pba(PBAComposition("Ni", 1.75, 0.125), supercell=(1, 1, 1))
    assert small.info["realized_vacancy_fraction"] == 0.0
    assert small.info["composition_error"] > 0.1

    sc = min_supercell_for((0.0, 0.125, 0.25, 0.375, 0.5))
    assert sc == (2, 2, 2)
    for y in (0.0, 0.125, 0.25, 0.375, 0.5):
        a = build_pba(PBAComposition("Ni", charge_balanced_na(y), y), supercell=sc)
        assert a.info["composition_error"] < 1e-9, f"y={y} not represented exactly"


def test_charge_balance_ties_sodium_to_vacancies():
    """Each vacancy removes 4- of framework charge and so expels two Na."""
    assert charge_balanced_na(0.0) == pytest.approx(2.0)
    assert charge_balanced_na(0.25) == pytest.approx(1.0)
    assert charge_balanced_na(0.5) == pytest.approx(0.0)


def test_composition_grid_rejects_overfilled_cages():
    for c in composition_grid("Mn", vacancy_values=(0.0, 0.25, 0.5)):
        assert c.na_per_fu <= 2.0 * (1.0 - c.vacancy_fraction) + 1e-9


# --------------------------------------------------------------------------- #
# structure validation
# --------------------------------------------------------------------------- #

def test_structure_check_catches_broken_framework():
    """A torn structure must be reported, not scored."""
    atoms = build_pba(PBAComposition("Mn", 2.0, 0.0), supercell=(1, 1, 1))
    assert check_structure(atoms) == []
    broken = atoms.copy()
    pos = broken.get_positions()
    sym = broken.get_chemical_symbols()
    for k, s in enumerate(sym):
        if s == "N":
            pos[k] += np.array([0.45, 0.0, 0.0])   # stretch every C-N well past 1.30
    broken.set_positions(pos)
    assert check_structure(broken), "stretched cyanide not detected"


# --------------------------------------------------------------------------- #
# convex hull construction
# --------------------------------------------------------------------------- #

def test_lower_hull_finds_lower_branch_only():
    """The lower hull must ignore points above the tie-line."""
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y = np.array([0.0, 0.30, -0.20, 0.30, 0.0])   # one point below, two above
    verts = lower_hull(x, y)
    assert set(x[verts].tolist()) == {0.0, 0.5, 1.0}


def test_lower_hull_handles_two_points():
    verts = lower_hull(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert len(verts) == 2


def test_convex_series_puts_every_point_on_the_hull():
    x = np.linspace(0, 1, 5)
    y = (x - 0.5) ** 2 - 0.25
    verts = lower_hull(x, y)
    assert len(verts) == 5


def test_hull_endpoints_have_zero_mixing_energy():
    """Mixing energy is defined against the endpoints, so they must vanish."""
    res = compute_hull("Mn", AnalyticModel(seed=1), vacancy_values=(0.0, 0.25, 0.5),
                       supercell=(2, 2, 2), use_sqs=False, fmax=1.0, steps=1)
    by_y = {round(p.vacancy_fraction, 3): p for p in res.points}
    assert abs(by_y[0.0].mixing_energy_eV) < 1e-9
    assert abs(by_y[0.5].mixing_energy_eV) < 1e-9


def test_demixing_system_has_interior_points_above_hull():
    """A positive interaction parameter must produce a two-phase region.

    The analytic model carries a deliberate positive (demixing) term, so the
    interior compositions must sit above the endpoint tie-line.  If they do not,
    the mixing-energy or hull construction is wrong.
    """
    res = compute_hull("Mn", AnalyticModel(seed=1, interaction_eV=0.5),
                       vacancy_values=(0.0, 0.25, 0.5), supercell=(2, 2, 2),
                       use_sqs=False, fmax=1.0, steps=1)
    interior = [p for p in res.points if 0.0 < p.vacancy_fraction < 0.5]
    assert interior
    assert all(p.e_above_hull_eV > 0 for p in interior)
    assert not any(p.on_hull for p in interior)


def test_hull_depth_is_compared_against_entropy_scale():
    """A 5 meV hull feature is erased by thermal disorder and must be flagged."""
    shallow = HullResult(
        metal="Mn", points=[], model_name="test", supercell=(2, 2, 2),
        temperature_C=25.0, max_hull_depth_eV=-0.005,
        entropy_scale_eV=K_B_EV * 298.15 * math.log(2),
    )
    assert not shallow.resolvable
    deep = HullResult(
        metal="Mn", points=[], model_name="test", supercell=(2, 2, 2),
        temperature_C=25.0, max_hull_depth_eV=-0.25,
        entropy_scale_eV=K_B_EV * 298.15 * math.log(2),
    )
    assert deep.resolvable


def test_report_states_its_limitations():
    """The report must carry the model identity and the entropy comparison.

    A hull depth quoted without them invites exactly the over-reading this module
    is built to prevent.
    """
    res = compute_hull("Mn", AnalyticModel(seed=2), vacancy_values=(0.0, 0.25, 0.5),
                       supercell=(2, 2, 2), use_sqs=False, fmax=1.0, steps=1)
    txt = hull_report(res)
    assert "analytic" in txt
    assert "kT ln2" in txt
    assert "spread not measured" in txt or "Configurational spread" in txt


def test_unmeasured_configurational_spread_is_not_reported_as_zero():
    res = compute_hull("Mn", AnalyticModel(seed=3), vacancy_values=(0.0, 0.25, 0.5),
                       supercell=(2, 2, 2), use_sqs=False, n_config_samples=0,
                       fmax=1.0, steps=1)
    assert all(p.config_std_eV is None for p in res.points)
    assert "not measured" in hull_report(res)


# --------------------------------------------------------------------------- #
# the campaign prior
# --------------------------------------------------------------------------- #

def test_cubic_cell_relaxation_uses_one_degree_of_freedom():
    """The cell scan must respect cubic symmetry and report its own reliability.

    Relaxing the cell with a general cell filter ran 200 steps without converging
    on this system (max force 1.96 eV/A after 630 s) while the internal
    coordinates alone converged in 6 steps -- six spurious degrees of freedom for
    a lattice that has exactly one.  The scan replaces them, and flags a minimum
    that falls at the edge of the scanned range instead of reporting the edge
    value as an equilibrium.
    """
    pytest.importorskip("mace")
    from pba_autoworkflow.thermo.mlff import MACEModel

    model = MACEModel(model="small")
    atoms = build_pba(PBAComposition("Mn", 2.0, 0.0), supercell=(1, 1, 1))
    res = model.relax(atoms, fmax=0.1, steps=30)
    assert res.eos, "no equation-of-state diagnostics recorded"
    # The window must end up centred on a real interior minimum, re-centring if
    # the first attempt put the minimum at a shoulder (it does: the equilibrium
    # lattice parameter is ~5 % below the literature starting value for this model).
    assert not res.eos["at_scan_edge"], res.eos
    assert res.eos["curvature_eV"] > 0
    assert res.converged and res.max_force_eV_A <= 0.1


def test_atomic_overlap_is_rejected_not_fitted():
    """A cell squeezed into atomic overlap must not be fitted as an equilibrium.

    A PBA cell scaled to 0.55 of its edge (~83 % volume reduction) gives energies
    of order 1e6 eV -- overlapping
    atoms, far outside anything the model was trained on.  Including such a point
    in the parabola fit would drag the reported lattice parameter towards the
    overlap while still returning a plausible-looking number.
    """
    pytest.importorskip("mace")
    from pba_autoworkflow.thermo.mlff import MACEModel

    model = MACEModel(model="small")
    atoms = build_pba(PBAComposition("Mn", 2.0, 0.0), supercell=(1, 1, 1))
    squeezed = atoms.copy()
    squeezed.set_cell(atoms.get_cell() * 0.55, scale_atoms=True)
    res = model.relax(squeezed, fmax=0.1, steps=10)
    e_per_atom = res.energy_eV / res.n_atoms
    assert (res.eos.get("nonphysical") or res.eos.get("at_scan_edge")
            or e_per_atom < model.NONPHYSICAL_E_PER_ATOM_EV), res.eos


def test_relaxed_cell_stays_cubic():
    """A cubic scan must not shear the cell."""
    pytest.importorskip("mace")
    from pba_autoworkflow.thermo.mlff import MACEModel

    model = MACEModel(model="small")
    atoms = build_pba(PBAComposition("Ni", 2.0, 0.0), supercell=(1, 1, 1))
    res = model.relax(atoms, fmax=0.15, steps=20)
    cell = np.array(res.relaxed.get_cell())
    off_diagonal = cell - np.diag(np.diag(cell))
    assert np.abs(off_diagonal).max() < 1e-8
    assert np.ptp(np.diag(cell)) < 1e-8


def test_nonlinear_composition_path_is_rejected():
    """A bent composition path makes mixing energies uninterpretable.

    On the charge-balanced line Na = 2 - 4y, so every species count is linear in y
    -- but only up to y = 0.5, where Na hits zero and clamps.  Past that the path
    bends: Na stays flat while vacancies keep rising, so subtracting the endpoint
    tie-line leaves an uncancelled sodium chemical potential.  Measured on this
    system that residual was +2.2 eV/f.u., which is two orders of magnitude larger
    than any real vacancy mixing energy and would have been reported as a
    spectacular miscibility gap.
    """
    from pba_autoworkflow.thermo.hull import check_path_linearity

    straight = (0.0, 0.125, 0.25, 0.375, 0.5)
    assert check_path_linearity(
        straight, tuple(charge_balanced_na(y) for y in straight)) == []

    bent = (0.0, 0.25, 0.5, 0.75)
    problems = check_path_linearity(bent, tuple(charge_balanced_na(y) for y in bent))
    assert problems and "not linear" in problems[0]


def test_bent_path_marks_hull_unresolvable():
    """A hull on a bent path must refuse to be used, not just carry a note."""
    from pba_autoworkflow.thermo.hull import hull_from_energies

    rows = [{"vacancy_fraction": y, "na_per_fu": charge_balanced_na(y),
             "energy_per_fu_eV": -120.0 + 30.0 * y, "lattice_a_A": 10.2,
             "converged": True}
            for y in (0.0, 0.25, 0.5, 0.75, 1.0)]
    res = hull_from_energies("Mn", rows)
    assert any("not linear" in w for w in res.path_warnings)
    assert not res.resolvable
    assert "PATH" in hull_report(res)

    straight = [{"vacancy_fraction": y, "na_per_fu": charge_balanced_na(y),
                 "energy_per_fu_eV": -120.0 + 30.0 * y, "lattice_a_A": 10.2,
                 "converged": True}
                for y in (0.0, 0.125, 0.25, 0.375, 0.5)]
    assert not hull_from_energies("Mn", straight).path_warnings


def test_hydrogen_bonds_are_not_flagged_as_broken_water():
    """An H...O contact at 2.0-2.3 A is a hydrogen bond, not a stretched bond.

    Checking every H-O pair in a neighbour list flags all hydrated structures,
    because adjacent waters in a cavity hydrogen-bond to each other -- exactly what
    they should do.  Only each hydrogen's *nearest* oxygen is its covalent partner.
    """
    from pba_autoworkflow.thermo.mlff import _nearest_oh_distances

    atoms = build_pba(PBAComposition("Mn", 1.0, 0.25), supercell=(1, 1, 1),
                      rng=np.random.default_rng(1))
    assert check_structure(atoms) == [], check_structure(atoms)

    oh = _nearest_oh_distances(atoms)
    assert oh.size == sum(1 for s in atoms.get_chemical_symbols() if s == "H")
    assert oh.max() < 1.10

    # A genuinely dissociated water must still be caught.
    broken = atoms.copy()
    pos = broken.get_positions()
    sym = broken.get_chemical_symbols()
    first_h = sym.index("H")
    pos[first_h] += np.array([1.6, 0.0, 0.0])
    broken.set_positions(pos)
    assert any("dissociating" in w for w in check_structure(broken))


def test_water_reference_cancels_and_is_documented_as_such():
    """The water reference is recorded but provably cannot change this hull.

    A vacancy brings six aqua ligands, so n_H2O = 6y -- linear in y.  The mixing
    energy subtracts a straight line in y, so any term linear in y cancels
    identically, for any number of compositions.  This test exists because an
    earlier version of this module claimed the correction rescued a -1 eV feature;
    it does not, and the large features are a force-field problem rather than a
    bookkeeping one.
    """
    from pba_autoworkflow.thermo.hull import hull_from_energies, n_water_per_fu

    assert n_water_per_fu(0.25) == pytest.approx(1.5)
    assert n_water_per_fu(0.0) == 0.0

    ys = (0.0, 0.125, 0.25, 0.375, 0.5)
    rows = [{"vacancy_fraction": y, "na_per_fu": charge_balanced_na(y),
             "energy_per_fu_eV": -120.0 + 30.0 * y - 14.05 * 6 * y + 0.4 * y * (1 - y),
             "lattice_a_A": 10.2, "converged": True} for y in ys]

    bare = hull_from_energies("Mn", rows)
    corrected = hull_from_energies("Mn", rows, mu_water_eV=-14.05)

    assert bare.mu_water_eV is None
    assert corrected.mu_water_eV == pytest.approx(-14.05)
    # Bit-for-bit identical: this is the point.
    assert corrected.max_hull_depth_eV == pytest.approx(bare.max_hull_depth_eV,
                                                        abs=1e-12)
    for a, b in zip(sorted(bare.points, key=lambda p: p.vacancy_fraction),
                    sorted(corrected.points, key=lambda p: p.vacancy_fraction)):
        assert a.mixing_energy_eV == pytest.approx(b.mixing_energy_eV, abs=1e-12)
    # But the grand potential itself does shift, and is recorded.
    assert any(a.grand_potential_per_fu_eV != a.energy_per_fu_eV
               for a in corrected.points)
    assert "cancels" in hull_report(corrected)


def test_three_point_series_cannot_resolve_a_tendency():
    """One interior point is one degree of freedom; say so rather than imply more."""
    from pba_autoworkflow.thermo.hull import check_path_linearity

    ys = (0.0, 0.25, 0.5)
    problems = check_path_linearity(ys, tuple(charge_balanced_na(y) for y in ys))
    assert problems and "one interior composition" in problems[0]

    ys5 = (0.0, 0.125, 0.25, 0.375, 0.5)
    assert check_path_linearity(ys5, tuple(charge_balanced_na(y) for y in ys5)) == []


def test_prior_is_uninformative_where_it_has_no_hull():
    from pba_autoworkflow.thermo.prior import HullPrior

    res = compute_hull("Mn", AnalyticModel(seed=4), vacancy_values=(0.0, 0.25, 0.5),
                       supercell=(2, 2, 2), use_sqs=False, fmax=1.0, steps=1)
    prior = HullPrior.from_hulls(res)
    assert prior.stability_score("Cu", 0.2) == 0.5      # no hull for Cu
    # An uninformative prior must cost exactly what no prior costs.  This
    # assertion previously read `prior.weight * 0.5`, encoding a real defect:
    # an uncomputed metal was charged 0.075 while a computed-and-on-hull metal
    # paid 0.000, so absence of evidence silently demoted a candidate inside
    # the scalarizer.  See tests/test_thermo_advisor.py for the full case.
    assert prior.penalty("Cu", 0.2) == 0.0
    assert not prior.is_informative("Cu")


def test_prior_reports_when_it_contradicts_measurement():
    """The prior must be able to tell you the computation was misleading."""
    from pba_autoworkflow.thermo.prior import HullPrior

    res = compute_hull("Mn", AnalyticModel(seed=5, interaction_eV=0.8),
                       vacancy_values=(0.0, 0.125, 0.25, 0.375, 0.5),
                       supercell=(2, 2, 2), use_sqs=False, fmax=1.0, steps=1)
    prior = HullPrior.from_hulls(res)
    # Measurements that get *better* exactly where the prior says less stable.
    measured = [("Mn", p.vacancy_fraction, -prior.stability_score("Mn", p.vacancy_fraction))
                for p in res.points]
    out = prior.disagreement(measured)
    assert out["correlation"] is not None
    assert out["correlation"] < -0.3
    assert "misleading" in out["verdict"]


def test_prior_stays_advisory():
    """Default weight must be small enough that measurement dominates."""
    from pba_autoworkflow.thermo.prior import HullPrior

    assert HullPrior(hulls={}).weight <= 0.2
    assert HullPrior(hulls={}).penalty("Mn", 0.3) <= 0.2


def test_every_backend_shares_one_relaxation_protocol():
    """A cross-model comparison is only meaningful if the protocol is identical.

    If each backend carried its own ``relax``, a difference between MACE and
    PET-MAD could be the protocol rather than the force field.  The shared base
    class is what rules that out, so assert the inheritance rather than trusting
    it to survive a refactor.
    """
    from pba_autoworkflow.thermo.mlff import (ASEForceFieldModel, MACEModel,
                                     PETMADModel, UMAModel)

    for backend in (MACEModel, PETMADModel, UMAModel):
        assert issubclass(backend, ASEForceFieldModel)
        # the backend must not shadow the shared implementation
        assert "relax" not in vars(backend), (
            f"{backend.__name__} overrides relax(); the comparison would no "
            "longer be protocol-controlled"
        )
    assert "relax" in vars(ASEForceFieldModel)


def test_validate_verdict_does_not_promise_error_cancellation():
    """The verdict must not claim fixed-metal series are rescued by cancellation.

    It said so for several versions.  Measured on the identical Mn series,
    mace-mp-small gives E_mix(0.25) = -996 meV/f.u. and uma-s-1p1-omat gives
    >= +12 meV/f.u. -- a 1.0 eV disagreement that flips sign.  Error does not
    cancel along the composition axis, and a verdict that says otherwise invites
    exactly the mistake this module exists to prevent.
    """
    src = pathlib.Path(cli.__file__).read_text()
    assert "systematic error cancels along a" not in src
    assert "NOT rescued by error cancellation" in src
