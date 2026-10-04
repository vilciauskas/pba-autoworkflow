# SPDX-License-Identifier: GPL-3.0-or-later
"""Special quasirandom structures for the PBA vacancy and Na sublattices.

A composition like Na_1.5 Mn[Fe(CN)6]_0.875 does not name a structure -- it names
an ensemble of decorations, and their energies span a range.  Taking one random
decoration and calling its energy "the energy of that composition" conflates a
sample with an expectation; the scatter is not small, because vacancy-vacancy
interactions in a framework this open are long-ranged.

Two ways out, both here:

:func:`generate_sqs`
    A *special quasirandom structure* -- the decoration whose cluster correlation
    functions best match the infinitely-large random alloy, out to a chosen cutoff.
    One structure, one energy, and it approximates the ensemble average of the
    ideal solid solution.  This is the right input for a hull that assumes
    configurational disorder, which is what a room-temperature precipitated PBA is.

:func:`sample_decorations`
    Brute-force sampling of random decorations, which gives the *spread* rather
    than a single number.  Slower, and the honest choice when you want to report
    an uncertainty on a hull point instead of a bare value.

The sublattices are handled jointly.  icet's :class:`ClusterSpace` is built on the
parent lattice with the Fe sites allowed to be Fe or vacancy and the cage sites
allowed to be Na or vacancy, so the SQS matches correlations in *both* variables
at once -- Na and vacancies are correlated in a real material (Na avoids the
water-filled cavities), and optimizing them independently would miss that.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from .structures import (FU_PER_CELL, LATTICE_A0_A, PBAComposition, _FE_SITES,
                         _M_SITES, _NA_SITES, build_pba)

#: Placeholder species standing in for a vacancy on each sublattice.  icet refuses
#: to share one vacancy symbol across two active sublattices, and rightly so: a
#: shared symbol would let the Monte Carlo swap an empty Na cage with a missing
#: hexacyanoferrate, which is not a physical rearrangement.  Two distinct
#: placeholders keep the sublattices independent.  Both are chemically inert noble
#: gases, chosen only because they are valid ASE symbols that cannot occur in a PBA
#: and so can never collide with a real species.
VAC_FRAMEWORK = "Xe"
VAC_CAGE = "Kr"


@dataclass
class SQSResult:
    """An SQS decoration and how well it matched the random-alloy target."""

    atoms: Atoms
    #: Mean absolute deviation of cluster correlations from the random-alloy
    #: target, over the clusters in the cluster space.  Zero is a perfect match;
    #: values below ~0.05 are conventionally considered good for a cell this size.
    correlation_mismatch: float
    n_atoms: int
    cutoffs_A: tuple[float, ...]
    method: str
    #: Composition actually realized (a finite cell cannot hit every target).
    realized_na_per_fu: float
    realized_vacancy_fraction: float


def _parent_lattice(metal: str, supercell: tuple[int, int, int],
                    lattice_a_A: float | None = None) -> tuple[Atoms, list[list[str]]]:
    """Build the icet parent structure: framework skeleton plus allowed species.

    The parent contains only the *substitutional* sites -- the N-site metal, the
    Fe/vacancy sites and the Na/vacancy cage sites.  Cyanide and water are left
    out: they are not independent degrees of freedom (they follow their Fe or
    their vacancy), and including 12 fixed light atoms per formula unit in the
    cluster space would make it enormous for no gain.  They are put back by
    :func:`_decorate` once the occupations are chosen.
    """
    import itertools

    a = lattice_a_A if lattice_a_A is not None else LATTICE_A0_A[metal]
    nx, ny, nz = supercell
    symbols: list[str] = []
    positions: list[list[float]] = []
    chemical_symbols: list[list[str]] = []

    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        base = np.array([i, j, k], dtype=float)
        for s in _M_SITES:
            symbols.append(metal)
            positions.append(((base + np.array(s)) * a).tolist())
            chemical_symbols.append([metal])          # not substitutional here
        for s in _FE_SITES:
            symbols.append("Fe")
            positions.append(((base + np.array(s)) * a).tolist())
            chemical_symbols.append(["Fe", VAC_FRAMEWORK])
        for s in _NA_SITES:
            symbols.append("Na")
            positions.append(((base + np.array(s)) * a).tolist())
            chemical_symbols.append(["Na", VAC_CAGE])

    parent = Atoms(symbols=symbols, positions=np.array(positions),
                   cell=np.eye(3) * (a * np.array([nx, ny, nz])), pbc=True)
    return parent, chemical_symbols


def _decorate(occupied: Atoms, metal: str, supercell: tuple[int, int, int],
              hydrate_vacancies: bool = True,
              lattice_a_A: float | None = None) -> Atoms:
    """Turn an icet occupation back into a full structure with cyanide and water.

    Reuses :func:`build_pba`'s geometry by rebuilding at the chosen composition and
    then overwriting the vacancy and Na *placement* with the SQS occupations, so
    the bond lengths and water orientation stay in one place rather than being
    re-derived here.
    """
    import itertools

    a = lattice_a_A if lattice_a_A is not None else LATTICE_A0_A[metal]
    nx, ny, nz = supercell
    n_cells = nx * ny * nz
    sym = occupied.get_chemical_symbols()

    # Walk the parent in the same order it was built to recover which sites are
    # occupied.  Per cell: len(_M_SITES) metals, then Fe sites, then Na cages.
    per_cell = len(_M_SITES) + len(_FE_SITES) + len(_NA_SITES)
    fe_occ: list[bool] = []
    na_occ: list[bool] = []
    for c in range(n_cells):
        off = c * per_cell + len(_M_SITES)
        fe_occ.extend(sym[off + t] == "Fe" for t in range(len(_FE_SITES)))
        off2 = off + len(_FE_SITES)
        na_occ.extend(sym[off2 + t] == "Na" for t in range(len(_NA_SITES)))

    y = 1.0 - (sum(fe_occ) / len(fe_occ))
    x = sum(na_occ) / (n_cells * FU_PER_CELL)

    # Rebuild with the exact counts, then permute placement to match the SQS.
    comp = PBAComposition(metal, na_per_fu=x, vacancy_fraction=y)
    atoms = _build_with_placement(comp, supercell, fe_occ, na_occ,
                                  hydrate_vacancies, a)
    atoms.info.update(sqs=True)
    return atoms


def _build_with_placement(comp: PBAComposition, supercell: tuple[int, int, int],
                          fe_occ: list[bool], na_occ: list[bool],
                          hydrate_vacancies: bool, a: float) -> Atoms:
    """:func:`build_pba` with vacancy/Na placement supplied instead of sampled."""
    # build_pba samples placement from an rng; here the placement is given, so it
    # is passed through a fixed-permutation rng shim.  Cleaner than duplicating the
    # geometry code, and keeps bond lengths defined in exactly one place.
    vacant_idx = [i for i, occ in enumerate(fe_occ) if not occ]
    na_idx = [i for i, occ in enumerate(na_occ) if occ]

    class _FixedRNG:
        """Returns the prescribed site lists in place of random choices."""

        def __init__(self) -> None:
            self._calls = 0

        def choice(self, n, size, replace=False):  # noqa: D102 - rng shim
            self._calls += 1
            if self._calls == 1:
                return np.array(vacant_idx[:size], dtype=int)
            return np.array(na_idx[:size], dtype=int)

    return build_pba(comp, supercell=supercell,
                     hydrate_vacancies=hydrate_vacancies,
                     rng=_FixedRNG(), lattice_a_A=a)


def default_cutoffs(lattice_a_A: float) -> tuple[float, float]:
    """Pair and triplet cutoffs that actually include the relevant shells.

    The sublattice spacings are set by the cubic edge ``a``:

        Fe-Na (cross-sublattice)   a * sqrt(3) / 4  ~ 4.56 A
        Na-Na (first)              a / 2            ~ 5.27 A
        Fe-Fe (first)              a * sqrt(2) / 2  ~ 7.45 A
        Na-Na (second)             a * sqrt(2) / 2  ~ 7.45 A

    A pair cutoff below the Fe-Fe distance excludes the *first* vacancy-vacancy
    shell -- which is the single most important interaction for how vacancies
    arrange -- leaving the SQS search nothing to optimize on that sublattice while
    still appearing to succeed.  The default therefore reaches past it, to the
    second Na-Na shell, and the triplet cutoff covers the nearest mixed triangle.
    """
    pair = 0.78 * lattice_a_A      # ~8.2 A at a = 10.5: clears Fe-Fe and Na-Na(2nd)
    triplet = 0.55 * lattice_a_A   # ~5.8 A: Fe-Na and Na-Na(1st) triangles
    return (round(pair, 2), round(triplet, 2))


def generate_sqs(
    comp: PBAComposition,
    supercell: tuple[int, int, int] = (2, 2, 2),
    cutoffs_A: tuple[float, ...] | None = None,
    n_steps: int = 3000,
    seed: int = 0,
    hydrate_vacancies: bool = True,
    lattice_a_A: float | None = None,
) -> SQSResult:
    """Find the decoration best matching the random alloy at this composition.

    ``cutoffs_A`` gives the pair and triplet cutoffs for the cluster space; when
    omitted they are derived from the lattice parameter by :func:`default_cutoffs`
    so that the first vacancy-vacancy shell is always included.

    Raises
    ------
    ImportError
        If icet is not installed.  This is deliberately not caught: a caller who
        asked for an SQS should not silently receive a random decoration.
    """
    from icet import ClusterSpace
    from icet.tools.structure_generation import generate_sqs_from_supercells

    a = lattice_a_A if lattice_a_A is not None else LATTICE_A0_A[comp.metal]
    if cutoffs_A is None:
        cutoffs_A = default_cutoffs(a)
    parent, chemical_symbols = _parent_lattice(comp.metal, (1, 1, 1), lattice_a_A)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = ClusterSpace(structure=parent, cutoffs=list(cutoffs_A),
                          chemical_symbols=chemical_symbols)

    # icet keys target concentrations by sublattice *letter* ('A', 'B', ...), not
    # by species, and the letter assignment depends on the order icet discovers the
    # orbits in.  Resolve it from the cluster space rather than hardcoding, so a
    # future change to the parent lattice cannot silently swap the two targets and
    # optimize Na concentration onto the vacancy sublattice.
    frac_fe = 1.0 - comp.vacancy_fraction
    frac_na = comp.na_per_fu / 2.0
    wanted = {"Fe": {"Fe": frac_fe, VAC_FRAMEWORK: 1.0 - frac_fe},
              "Na": {"Na": frac_na, VAC_CAGE: 1.0 - frac_na}}

    target: dict[str, dict[str, float]] = {}
    for sub in cs.get_sublattices(parent).active_sublattices:
        species = set(sub.chemical_symbols)
        for host, conc in wanted.items():
            if host in species:
                target[sub.symbol] = conc
                break
        else:  # pragma: no cover - would mean the parent lattice changed shape
            raise RuntimeError(
                f"active sublattice {sub.symbol!r} with species {sorted(species)} "
                "matches neither the Fe/vacancy nor the Na/cage sublattice"
            )

    # A sublattice at concentration 0 or 1 is fully ordered: there is nothing to
    # anneal, and icet correctly refuses ("no canonical swaps possible").  When
    # *both* are ordered the composition has a single decoration, which is the
    # exact answer rather than an approximation to it -- so build it directly and
    # report a zero mismatch, which is true.
    def _ordered(frac: float) -> bool:
        return frac < 1e-9 or frac > 1.0 - 1e-9

    if _ordered(frac_fe) and _ordered(frac_na):
        atoms = build_pba(comp, supercell=supercell,
                          hydrate_vacancies=hydrate_vacancies,
                          rng=np.random.default_rng(seed), lattice_a_A=lattice_a_A)
        atoms.info.update(sqs=True, sqs_method="ordered-endpoint")
        return SQSResult(
            atoms=atoms, correlation_mismatch=0.0, n_atoms=len(atoms),
            cutoffs_A=tuple(cutoffs_A), method="ordered-endpoint",
            realized_na_per_fu=float(atoms.info["realized_na_per_fu"]),
            realized_vacancy_fraction=float(atoms.info["realized_vacancy_fraction"]),
        )

    supercells = [parent.repeat(supercell)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sqs = generate_sqs_from_supercells(
            cluster_space=cs, supercells=supercells,
            target_concentrations=target, n_steps=n_steps,
            random_seed=seed,
        )

    decorated = _decorate(sqs, comp.metal, supercell, hydrate_vacancies, lattice_a_A)
    mismatch = _correlation_mismatch(cs, sqs, target)
    return SQSResult(
        atoms=decorated, correlation_mismatch=mismatch, n_atoms=len(decorated),
        cutoffs_A=tuple(cutoffs_A), method="icet-sqs",
        realized_na_per_fu=float(decorated.info["realized_na_per_fu"]),
        realized_vacancy_fraction=float(decorated.info["realized_vacancy_fraction"]),
    )


def _correlation_mismatch(cs, structure, target) -> float:
    """Mean |cluster vector - ideal random-alloy vector| over the cluster space.

    The ideal vector is obtained from icet itself rather than reconstructed here.
    An earlier version estimated the pair target as the square of the point
    correlation, which is only valid for a *single* binary sublattice: with two
    coupled sublattices it silently reports a large mismatch for a perfectly good
    SQS, because cross-sublattice pair clusters do not follow that rule.

    Reported rather than assumed -- a caller comparing hull points computed in
    different supercells needs to know whether both reached the random limit, and
    a stalled SQS search shows up here as a large value.
    """
    from icet.tools.structure_generation import (_get_sqs_cluster_vector,
                                                 _validate_concentrations)

    try:
        cv = np.array(cs.get_cluster_vector(structure))
        conc = _validate_concentrations(concentrations=target, cluster_space=cs)
        ideal = np.array(_get_sqs_cluster_vector(cluster_space=cs,
                                                target_concentrations=conc))
    except Exception:
        return float("nan")
    if cv.shape != ideal.shape:
        return float("nan")
    # Skip the zerolet, which is identically 1 and carries no information.
    return float(np.mean(np.abs(cv[1:] - ideal[1:])))


def sample_decorations(
    comp: PBAComposition,
    n_samples: int = 8,
    supercell: tuple[int, int, int] = (2, 2, 2),
    seed: int = 0,
    hydrate_vacancies: bool = True,
) -> list[Atoms]:
    """Random decorations at one composition, for measuring configurational spread.

    Use this to put an error bar on a hull point.  A single SQS gives the ensemble
    mean; this gives the distribution the mean was drawn from, which is what tells
    you whether a 20 meV hull depth is meaningful at your cell size.
    """
    rng = np.random.default_rng(seed)
    return [
        build_pba(comp, supercell=supercell, hydrate_vacancies=hydrate_vacancies,
                  rng=np.random.default_rng(int(rng.integers(1 << 31))))
        for _ in range(n_samples)
    ]
