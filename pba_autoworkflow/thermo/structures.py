# SPDX-License-Identifier: GPL-3.0-or-later
"""Build PBA crystal structures across the Na/vacancy composition space.

The parent is the cubic Prussian blue analogue Na_x M[Fe(CN)6]_(1-y), space group
Fm-3m.  Two interpenetrating metal sublattices are bridged by cyanide:

    M  (N-coordinated) at (0, 0, 0)
    Fe (C-coordinated) at (1/2, 1/2, 1/2)
    C, N along <100> between them
    Na in the 8c interstitial cages at (1/4, 1/4, 1/4) and equivalents

Two substitutions generate the composition space:

``y`` -- **hexacyanoferrate vacancies.**  A vacancy removes a whole [Fe(CN)6]
    unit, not just the iron: the six cyanides go with it, and in a real material
    coordinated water fills the open coordination sites on the six neighbouring
    M.  A vacancy should therefore cost far less than its formula suggests, and an
    anhydrous calculation should overestimate the penalty by leaving six metals
    under-coordinated.  That is the physical argument for hydrating by default
    (``hydrate_vacancies``, on unless disabled); it has *not* been quantified in
    this module, so treat it as reasoning rather than as a measured result.

``x`` -- **sodium content.**  Na occupies the 8c cages, up to two per formula
    unit.  Charge balance ties x to y: a vacancy removes 4- of framework charge
    and therefore expels two Na.  Sodium inventory and framework integrity are
    consequently *not* independent objectives, which is the whole reason this
    hull is drawn in both axes at once.

Nothing here imports the simulator or the analysis layer: these are structures
built from crystallography, and they are the input to the force field.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from ase import Atoms

#: Cubic lattice parameter of the fully-loaded framework, by N-site metal (Angstrom).
#: Deliberately duplicated from the analysis layer rather than imported: this module
#: must stay independent of measurement, so that a hull prediction and an indexed
#: pattern are two separate claims that can disagree.
LATTICE_A0_A: dict[str, float] = {
    "Mn": 10.53, "Fe": 10.28, "Co": 10.30, "Ni": 10.24, "Cu": 10.11,
}

#: Bond lengths held fixed across the series (Angstrom).  The cyanide bridge is
#: stiff -- Fe-C and C-N barely move between analogues -- so the lattice parameter
#: difference from Mn to Cu is absorbed almost entirely by the M-N bond.  Placing
#: the light atoms at *fixed fractions* of the cell instead (an earlier version)
#: stretches C-N to 1.26 A, which no cyanide does, and distributes the error
#: across every bond rather than the one that actually flexes.
_D_FE_C = 1.90
_D_C_N = 1.15
_D_M_O = 2.15        # M-O for aqua ligands completing a vacancy
_D_O_H = 0.96

#: The two interpenetrating FCC sublattices of the rock-salt framework, in
#: fractional coordinates of the conventional cubic cell.  M and Fe alternate on a
#: simple-cubic net of edge a/2, each M-Fe pair bridged by one cyanide.  The
#: conventional cell therefore contains FOUR formula units, and eight Na cage
#: sites -- i.e. the familiar maximum of two Na per formula unit.
_M_SITES = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
_FE_SITES = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5))
_NA_SITES = tuple(itertools.product((0.25, 0.75), repeat=3))
#: Formula units per conventional cubic cell.
FU_PER_CELL = 4


@dataclass(frozen=True)
class PBAComposition:
    """A point in the Na/vacancy composition space.

    ``na_per_fu`` is x and ``vacancy_fraction`` is y in Na_x M[Fe(CN)6]_(1-y),
    both per formula unit -- the same quantities and units the analysis layer
    reports, so a hull point and a measured sample compare directly.
    """

    metal: str
    na_per_fu: float
    vacancy_fraction: float

    def __post_init__(self) -> None:
        if self.metal not in LATTICE_A0_A:
            raise ValueError(f"unknown metal {self.metal!r}")
        if not 0.0 <= self.vacancy_fraction <= 1.0:
            raise ValueError(f"vacancy fraction {self.vacancy_fraction} outside [0, 1]")
        if not 0.0 <= self.na_per_fu <= 2.0 + 1e-9:
            raise ValueError(f"Na content {self.na_per_fu} outside [0, 2]")

    @property
    def formula(self) -> str:
        return (f"Na{self.na_per_fu:.2f}{self.metal}"
                f"[Fe(CN)6]{1.0 - self.vacancy_fraction:.2f}")


def charge_balanced_na(vacancy_fraction: float, m_oxidation: int = 2,
                       fe_oxidation: int = 2) -> float:
    """Sodium content making Na_x M[Fe(CN)6]_(1-y) charge neutral.

        x = (1 - y) * (6 - fe_oxidation) - m_oxidation

    M(II) with Fe(II) and no vacancies gives the familiar x = 2.
    """
    x = (1.0 - vacancy_fraction) * (6 - fe_oxidation) - m_oxidation
    return float(max(0.0, min(2.0, x)))


def build_pba(
    comp: PBAComposition,
    supercell: tuple[int, int, int] = (2, 2, 2),
    hydrate_vacancies: bool = True,
    rng: np.random.Generator | None = None,
    lattice_a_A: float | None = None,
) -> Atoms:
    """Construct an ``ase.Atoms`` for one composition by random decoration.

    A randomly decorated cell is *one sample* from a configurational ensemble and
    its energy carries that scatter.  For hull points use
    :func:`pba_autoworkflow.thermo.sqs.generate_sqs`, which picks a decoration whose
    short-range order matches the random alloy; this function is the building
    block that both it and the cluster expansion call.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    a = lattice_a_A if lattice_a_A is not None else LATTICE_A0_A[comp.metal]
    nx, ny, nz = supercell
    n_fu = FU_PER_CELL * nx * ny * nz

    symbols: list[str] = []
    positions: list[np.ndarray] = []

    # Enumerate every hexacyanoferrate site in the supercell, then choose which to
    # vacate.  Vacancies live on the Fe sublattice; the M sublattice stays complete.
    fe_all = [(np.array([i, j, k], dtype=float) + np.array(s)) * a
              for i, j, k in itertools.product(range(nx), range(ny), range(nz))
              for s in _FE_SITES]
    n_vac = int(round(comp.vacancy_fraction * len(fe_all)))
    vacant = set(rng.choice(len(fe_all), size=n_vac, replace=False).tolist()) if n_vac else set()

    # The N-site metal sublattice.
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        for s in _M_SITES:
            symbols.append(comp.metal)
            positions.append((np.array([i, j, k], dtype=float) + np.array(s)) * a)

    # Hexacyanoferrate units, or hydrated cavities where they are missing.  Each
    # Fe has six M neighbours along +-x, +-y, +-z at a/2; the cyanide bridges that
    # vector with real bond lengths, leaving M-N to absorb the lattice mismatch.
    for idx, fe_pos in enumerate(fe_all):
        if idx not in vacant:
            symbols.append("Fe")
            positions.append(fe_pos.copy())
            for axis, sign in itertools.product(range(3), (+1, -1)):
                u = np.zeros(3)
                u[axis] = sign
                symbols.append("C")
                positions.append(fe_pos + u * _D_FE_C)
                symbols.append("N")
                positions.append(fe_pos + u * (_D_FE_C + _D_C_N))
        elif hydrate_vacancies:
            # Six aqua ligands, one on each M that lost a cyanide, pointing from
            # that metal back into the empty cavity.
            for axis, sign in itertools.product(range(3), (+1, -1)):
                u = np.zeros(3)
                u[axis] = sign
                o_pos = fe_pos + u * (0.5 * a - _D_M_O)
                symbols.append("O")
                positions.append(o_pos)
                # The two H must point AWAY from the coordinating metal.  ``u`` runs
                # from the vacant Fe site towards that metal, and the oxygen sits
                # between them, so "away from the metal" is -u -- back towards the
                # cavity centre.  Getting this sign wrong puts H 1.74 A from the
                # metal, a close contact the force field then relaxes violently.
                perp = np.zeros(3)
                perp[(axis + 1) % 3] = _D_O_H * math.sin(math.radians(52.25))
                for h_sign in (-1, +1):
                    symbols.append("H")
                    positions.append(o_pos - u * (_D_O_H * math.cos(math.radians(52.25)))
                                     + h_sign * perp)

    # Sodium on the 8c cage sites: eight per conventional cell, two per formula unit.
    na_sites = [(np.array([i, j, k], dtype=float) + np.array(c)) * a
                for i, j, k in itertools.product(range(nx), range(ny), range(nz))
                for c in _NA_SITES]
    n_na = min(int(round(comp.na_per_fu * n_fu)), len(na_sites))
    for idx in rng.choice(len(na_sites), size=n_na, replace=False):
        symbols.append("Na")
        positions.append(na_sites[idx])

    atoms = Atoms(symbols=symbols, positions=np.array(positions),
                  cell=np.eye(3) * (a * np.array([nx, ny, nz])), pbc=True)

    # A finite cell can only represent compositions on a 1/n_fu grid.  Record what
    # was actually built alongside what was asked for, so a hull point is never
    # silently attributed to a composition the cell cannot hold: n_fu = 4 rounds a
    # requested y = 0.125 to 0.0, and a caller reading back `comp` rather than
    # `realized_*` would plot a vacancy-free energy at y = 0.125.
    realized_y = n_vac / len(fe_all)
    realized_x = n_na / n_fu
    atoms.info.update(
        metal=comp.metal,
        na_per_fu=comp.na_per_fu, vacancy_fraction=comp.vacancy_fraction,
        realized_na_per_fu=realized_x, realized_vacancy_fraction=realized_y,
        composition_error=max(abs(realized_x - comp.na_per_fu),
                              abs(realized_y - comp.vacancy_fraction)),
        n_formula_units=n_fu, hydrated=bool(hydrate_vacancies),
        formula=comp.formula,
    )
    return atoms


def min_supercell_for(vacancy_values: tuple[float, ...],
                      na_values: tuple[float, ...] = (),
                      max_repeat: int = 3) -> tuple[int, int, int]:
    """Smallest cubic supercell that represents every requested composition exactly.

    Vacancy fractions live on a 1/(4 n^3) grid and Na on a 1/(4 n^3) grid too (the
    eight cage sites per cell give two per formula unit).  Rather than let the
    caller discover the rounding after the fact, pick a cell that has no rounding.
    """
    for n in range(1, max_repeat + 1):
        n_fu = FU_PER_CELL * n ** 3
        ok = all(abs(v * n_fu - round(v * n_fu)) < 1e-6
                 for v in tuple(vacancy_values) + tuple(na_values))
        if ok:
            return (n, n, n)
    raise ValueError(
        f"no supercell up to {max_repeat}^3 represents {vacancy_values} and "
        f"{na_values} exactly; choose compositions on a 1/{FU_PER_CELL * max_repeat**3} grid"
    )


def composition_grid(
    metal: str,
    vacancy_values: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5),
    na_values: tuple[float, ...] | None = None,
    charge_balanced_only: bool = False,
) -> list[PBAComposition]:
    """Enumerate composition points for the hull.

    ``charge_balanced_only`` traces the neutral line through the (x, y) plane
    instead of the full rectangle.  Off-line points are still useful -- real
    samples sit off it because protons and hydronium also occupy the cages -- but
    they should be read as what they are: hypothetical, and formally charged.
    """
    out: list[PBAComposition] = []
    for y in vacancy_values:
        if charge_balanced_only:
            xs: tuple[float, ...] = (charge_balanced_na(y),)
        elif na_values is None:
            xs = tuple(float(v) for v in np.round(np.linspace(0.0, charge_balanced_na(y), 3), 4))
        else:
            xs = na_values
        for x in xs:
            if x > 2.0 * (1.0 - y) + 1e-9:
                continue
            out.append(PBAComposition(metal, float(x), float(y)))
    return out
