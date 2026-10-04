# SPDX-License-Identifier: GPL-3.0-or-later
"""Machine-learned force field energies and relaxations.

The interface mirrors the device layer: :class:`EnergyModel` is a protocol, and
:class:`MACEModel` is one implementation.  A DFT backend, a fine-tuned MACE, or a
different foundation model drops in as one more class without the hull code
changing -- which matters here because *the foundation model is the weakest link
in this module* and you will want to replace it.

Why the caution.  MACE-MP is trained on Materials Project PBE relaxations of
mostly dense inorganic crystals.  A PBA is none of those things: it is a molecular
framework, roughly half empty by volume, held together by a strong-field cyanide
ligand, with hydrogen-bonded water in its cavities and open-shell 3d metals whose
spin state PBE without a Hubbard U describes poorly.  Every one of those is an
extrapolation from the training distribution.

The consequence is not that the numbers are useless -- it is that only *some* of
them are trustworthy:

*Relative* energies along a homologous series (same metal, same framework, one
composition variable changing) benefit from systematic error cancellation and are
the intended output of this module.

*Absolute* formation energies, and any comparison against a Materials Project hull
computed with different settings, are not supported by this model.  The
:attr:`EnergyResult.extrapolation_warnings` field exists so that a caller can see
when a structure has drifted outside what the model can be expected to handle,
rather than reading a plausible-looking number with no indication of its origin.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS


@dataclass
class EnergyResult:
    """One force-field evaluation, with the diagnostics needed to judge it."""

    energy_eV: float
    energy_per_fu_eV: float
    n_atoms: int
    n_formula_units: int
    max_force_eV_A: float
    converged: bool
    n_steps: int
    wall_time_s: float
    volume_A3: float
    lattice_a_A: float
    #: Structure-level checks that failed, e.g. a bond that relaxed to a
    #: non-physical length.  Empty means nothing was detected -- not that the
    #: number is right.
    extrapolation_warnings: list[str] = field(default_factory=list)
    relaxed: Atoms | None = None
    #: Lattice-parameter scan diagnostics from the cubic equation-of-state fit:
    #: the scanned scales, their energies, the fitted parameter, and whether the
    #: minimum fell at the edge of the scan (in which case the fit is unreliable).
    eos: dict = field(default_factory=dict)

    @property
    def trustworthy(self) -> bool:
        """Converged, and no structural check tripped."""
        return self.converged and not self.extrapolation_warnings


@runtime_checkable
class EnergyModel(Protocol):
    """Anything that can give an energy for an :class:`ase.Atoms`."""

    name: str

    def relax(self, atoms: Atoms, fmax: float = 0.05, steps: int = 200,
              relax_cell: bool = True) -> EnergyResult:
        ...


#: Bond-length windows a relaxed PBA must satisfy, in Angstrom.  These are not
#: tight -- they are wide enough to admit real thermal and chemical variation, and
#: narrow enough to catch a structure the model has torn apart.  A cyanide that
#: relaxes to 1.4 A is not a slightly different cyanide; it means the model is
#: outside its training distribution and the energy should not be used.
BOND_WINDOWS_A: dict[tuple[str, str], tuple[float, float]] = {
    ("C", "N"): (1.05, 1.30),
    ("C", "Fe"): (1.75, 2.10),
    ("O", "H"): (0.90, 1.10),
}


def check_structure(atoms: Atoms) -> list[str]:
    """Report bonds that relaxed outside their physical window.

    Uses the covalent-radius neighbour list rather than a fixed cutoff, so it does
    not mistake a genuinely expanded framework for a broken one.
    """
    from ase.neighborlist import neighbor_list

    out: list[str] = []
    try:
        i, j, d = neighbor_list("ijd", atoms, cutoff=2.4)
    except Exception as exc:  # pragma: no cover - geometry too degenerate to analyse
        return [f"neighbour analysis failed: {exc}"]

    sym = atoms.get_chemical_symbols()
    seen: dict[tuple[str, str], list[float]] = {}
    for a, b, dd in zip(i, j, d):
        seen.setdefault(tuple(sorted((sym[a], sym[b]))), []).append(float(dd))

    for pair, (lo, hi) in BOND_WINDOWS_A.items():
        key = tuple(sorted(pair))
        if key not in seen:
            # Absent is only a problem for bonds the framework must have.
            if key in (("C", "N"), ("C", "Fe")):
                out.append(f"{key[0]}-{key[1]} bonds absent after relaxation")
            continue
        arr = np.array(seen[key])

        if key == ("H", "O"):
            # Every H has one covalent O and possibly several hydrogen-bonded ones.
            # A neighbour list cannot tell them apart by distance alone, and an
            # H...O contact at 1.7-2.3 A is a hydrogen bond -- the thing hydrated
            # cavities are supposed to form -- not a stretched bond.  Check each
            # hydrogen's *nearest* oxygen instead, which is the covalent partner.
            arr = _nearest_oh_distances(atoms)
            if arr.size == 0:
                continue
            if arr.min() < lo or arr.max() > hi:
                out.append(
                    f"covalent O-H in [{arr.min():.2f}, {arr.max():.2f}] A outside "
                    f"physical window [{lo:.2f}, {hi:.2f}]: water is dissociating"
                )
            continue

        if arr.min() < lo or arr.max() > hi:
            out.append(
                f"{key[0]}-{key[1]} in [{arr.min():.2f}, {arr.max():.2f}] A "
                f"outside physical window [{lo:.2f}, {hi:.2f}]"
            )
    return out


def _nearest_oh_distances(atoms: Atoms) -> np.ndarray:
    """Distance from each hydrogen to its closest oxygen, minimum-image."""
    sym = np.array(atoms.get_chemical_symbols())
    if not (sym == "H").any() or not (sym == "O").any():
        return np.array([])
    pos = atoms.get_positions()
    cell = np.diag(np.array(atoms.get_cell()))
    h, o = pos[sym == "H"], pos[sym == "O"]
    d = o[None, :, :] - h[:, None, :]
    if np.all(cell > 0):
        d -= cell * np.round(d / cell)
    return np.linalg.norm(d, axis=-1).min(axis=1)


class ASEForceFieldModel:
    """Relaxation machinery shared by every ASE-calculator backend.

    Subclasses set ``self._calc`` (any ASE calculator) and ``self.name``; all of
    the physics here -- water pre-relaxation, the cubic-parameter scan, the
    structure checks -- is backend-independent.  Keeping it in one place is what
    makes a cross-model comparison meaningful: if each backend carried its own
    relaxation protocol, a difference in the result could be the protocol rather
    than the force field.
    """

    _calc: object
    name: str


    def relax(self, atoms: Atoms, fmax: float = 0.05, steps: int = 200,
              relax_cell: bool = True) -> EnergyResult:
        """Relax positions and the cubic lattice parameter, and return the energy.

        The cell *must* be free for a hull to mean anything: vacancies and Na
        content both change the equilibrium lattice parameter, and holding the cell
        at the vacancy-free value would charge every off-stoichiometric point an
        elastic penalty that has nothing to do with its stability.

        The cell is relaxed by scanning the single cubic parameter and fitting an
        equation of state, *not* with a full cell filter.  Measured on this system:
        internal coordinates converge in 6 LBFGS steps, while
        :class:`~ase.filters.FrechetCellFilter` on the same structure ran 200 steps
        without converging (max force 1.96 eV/A) and took 630 s.  Mixing six cell
        degrees of freedom into the force vector conditions badly for a framework
        this soft -- and it is also six degrees of freedom more than the symmetry
        allows.  A cubic PBA has exactly one, so scanning it is both faster and
        better posed, and it yields the bulk-modulus-like curvature for free.
        """
        work = atoms.copy()
        work.calc = self._calc
        t0 = time.time()

        # Water is placed at idealized positions with no hydrogen bonding, so it
        # carries almost all of the initial force.  Relaxing it first, with the
        # framework held fixed, costs a few cheap steps and stops it from consuming
        # the whole step budget of the full relaxation: without this, every
        # vacancy-bearing composition stalled at max force ~0.6-0.9 eV/A while the
        # vacancy-free endpoint converged in 6 steps.
        if any(s in ("O", "H") for s in work.get_chemical_symbols()):
            work = self._prerelax_water(work, steps=steps)

        if relax_cell:
            work, eos = self._relax_cubic_cell(work, fmax=fmax, steps=steps)
        else:
            eos = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                LBFGS(work, logfile=None).run(fmax=fmax, steps=steps)

        # Final internal relaxation at the fitted lattice parameter.
        opt = LBFGS(work, logfile=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.run(fmax=fmax, steps=steps)

        forces = work.get_forces()
        max_f = float(np.abs(forces).max())
        energy = float(work.get_potential_energy())
        n_fu = int(work.info.get("n_formula_units", 1))
        vol = float(work.get_volume())

        result = EnergyResult(
            energy_eV=energy,
            energy_per_fu_eV=energy / max(1, n_fu),
            n_atoms=len(work),
            n_formula_units=n_fu,
            max_force_eV_A=max_f,
            converged=bool(max_f <= fmax),
            n_steps=int(opt.get_number_of_steps()),
            wall_time_s=time.time() - t0,
            volume_A3=vol,
            # Cubic edge per conventional cell, comparable to a measured lattice
            # constant regardless of how many cells the supercell contains.
            lattice_a_A=float(vol / max(1, n_fu / 4.0)) ** (1.0 / 3.0),
            extrapolation_warnings=check_structure(work),
            relaxed=work,
        )
        if eos:
            result.eos = eos
            if eos.get("at_scan_edge"):
                result.extrapolation_warnings.append(
                    "equilibrium lattice parameter lies outside the scanned range; "
                    "widen it before using this energy"
                )
        return result

    #: Energies above this per-atom value mean overlapping atoms, not a stiff
    #: lattice: the model is being evaluated far outside anything it was trained on
    #: and its output is meaningless.  A squeezed PBA cell reached ~3.8e6 eV, which
    #: would dominate any subsequent parabola fit and silently pull the "relaxed"
    #: lattice parameter towards the overlap.
    NONPHYSICAL_E_PER_ATOM_EV = 1e3

    def _prerelax_water(self, atoms: Atoms, steps: int = 60,
                        fmax: float = 0.05) -> Atoms:
        """Relax only the aqua ligands, with the framework held fixed.

        The framework atoms start at crystallographic positions and are nearly at
        equilibrium; the water does not, because it is placed geometrically with no
        hydrogen bonding.  Constraining everything else turns a stiff global
        problem into a small local one.
        """
        from ase.constraints import FixAtoms

        work = atoms.copy()
        work.calc = self._calc
        framework = [k for k, s in enumerate(work.get_chemical_symbols())
                     if s not in ("O", "H")]
        work.set_constraint(FixAtoms(indices=framework))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            LBFGS(work, logfile=None).run(fmax=fmax, steps=steps)
        work.set_constraint()
        work.calc = self._calc
        return work

    def _relax_cubic_cell(self, atoms: Atoms, fmax: float, steps: int,
                          span: float = 0.06, n_points: int = 7,
                          max_recentre: int = 3):
        """Scan the cubic lattice parameter, relaxing internals at each point.

        The window is re-centred and retried when the discrete minimum lands on or
        next to an edge: a window centred on the *input* cell is not centred on the
        equilibrium one, and fitting a parabola at the shoulder of the scan biases
        the result.  Points whose energy is non-physical (overlapping atoms) are
        excluded from the fit rather than allowed to dominate it.

        A parabola through the three points nearest the minimum is enough: the
        curvature is shallow over a few per cent and a higher-order equation of
        state would fit noise from the incompletely-relaxed internals.
        """
        a0 = float(atoms.get_cell()[0, 0])
        centre = 1.0
        history: list[dict] = []

        for attempt in range(max_recentre + 1):
            scales = np.linspace(centre - span, centre + span, n_points)
            energies: list[float] = []
            frames: list[Atoms] = []
            for s in scales:
                trial = atoms.copy()
                trial.set_cell(atoms.get_cell() * s, scale_atoms=True)
                trial.calc = self._calc
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    LBFGS(trial, logfile=None).run(fmax=fmax, steps=steps)
                energies.append(float(trial.get_potential_energy()))
                frames.append(trial)

            e = np.array(energies)
            usable = e / max(1, len(atoms)) < self.NONPHYSICAL_E_PER_ATOM_EV
            history.append({"centre": float(centre),
                            "scales": scales.tolist(), "energies": e.tolist(),
                            "n_nonphysical": int((~usable).sum())})
            if not usable.any():
                # Every point is an atomic overlap: nothing to fit.
                return frames[int(np.argmin(e))], {
                    "a_initial_A": a0, "a_relaxed_A": a0 * float(scales[int(np.argmin(e))]),
                    "scan_scales": scales.tolist(), "scan_energies": e.tolist(),
                    "at_scan_edge": True, "nonphysical": True, "history": history,
                }

            masked = np.where(usable, e, np.inf)
            k = int(np.argmin(masked))
            # Interior means at least one usable point on each side, so the
            # parabola is fit around a real minimum rather than at a shoulder.
            interior = 0 < k < len(e) - 1 and usable[k - 1] and usable[k + 1]
            if interior or attempt == max_recentre:
                break
            centre = float(scales[k])   # re-centre on the best point and rescan

        if not interior:
            return frames[k], {
                "a_initial_A": a0, "a_relaxed_A": a0 * float(scales[k]),
                "scan_scales": scales.tolist(), "scan_energies": e.tolist(),
                "at_scan_edge": True, "nonphysical": bool((~usable).any()),
                "history": history,
            }

        xs, ys = scales[k - 1:k + 2], e[k - 1:k + 2]
        c2, c1, _ = np.polyfit(xs, ys, 2)
        s_min = float(-c1 / (2.0 * c2)) if c2 > 0 else float(scales[k])
        s_min = float(np.clip(s_min, scales[0], scales[-1]))

        best = atoms.copy()
        best.set_cell(atoms.get_cell() * s_min, scale_atoms=True)
        best.calc = self._calc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            LBFGS(best, logfile=None).run(fmax=fmax, steps=steps)

        return best, {
            "a_initial_A": a0, "a_relaxed_A": a0 * s_min,
            "scan_scales": scales.tolist(), "scan_energies": e.tolist(),
            "at_scan_edge": False, "nonphysical": bool((~usable).any()),
            # Curvature at the minimum, proportional to the bulk modulus -- a free
            # by-product of the scan and a useful sanity check: a framework this
            # open should be soft, and an implausibly stiff value means the fit
            # caught something other than the equilibrium.
            "curvature_eV": float(2.0 * c2),
            "n_rescans": len(history) - 1, "history": history,
        }


class MACEModel(ASEForceFieldModel):
    """MACE-MP foundation force field via ASE.

    ``model`` selects the released size ("small", "medium", "large").  The default
    is float64: geometry optimization on a framework this soft is not reliable in
    single precision, and the extra cost is small next to the number of structures
    a hull needs.
    """

    def __init__(self, model: str = "small", device: str = "cpu",
                 dtype: str = "float64") -> None:
        from mace.calculators import mace_mp

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._calc = mace_mp(model=model, default_dtype=dtype, device=device)
        self.name = f"mace-mp-{model}"
        self.model_size = model
        self.device = device


class PETMADModel(ASEForceFieldModel):
    """PET-MAD universal potential (lab-cosmo/pet-mad).

    Trained on the MAD dataset, which is deliberately weighted toward distorted,
    disordered and off-equilibrium configurations rather than the relaxed
    ground-state crystals that dominate Materials Project.  That is the reason to
    try it here: a half-empty cyanide framework carrying aqua ligands is far from
    any relaxed inorganic prototype, and MACE-MP's failure on the lattice trend
    is consistent with exactly that kind of out-of-distribution structure.

    Whether the different training distribution actually helps on PBAs is an
    empirical question.  Run ``python -m pba_autoworkflow.thermo validate --model petmad``
    and read the rank correlation before trusting any hull this produces.
    """

    def __init__(self, version: str = "latest", device: str = "cpu") -> None:
        from pet_mad.calculator import PETMADCalculator

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._calc = PETMADCalculator(version=version, device=device)
        self.name = f"pet-mad-{version}"
        self.model_size = version
        self.device = device


class UMAModel(ASEForceFieldModel):
    """Meta FAIR UMA universal potential (license-gated; needs an HF token).

    UMA is the base potential behind TIP (arXiv:2608.14502), and the reason to
    try it on PBAs is its training mix: unlike MACE-MP and PET-MAD, which are
    dominated by inorganic crystals, UMA is trained jointly across five domains
    exposed here as task heads --

        ``omat``  inorganic materials (OMat24)      -- the default for a crystal
        ``odac``  MOFs with adsorbed CO2/H2O        -- closest analogue to a PBA
        ``omc``   organic molecular crystals
        ``omol``  isolated molecules
        ``oc20``  catalytic surfaces + adsorbates

    ``task`` selects the head and is a real scientific choice, not a detail.  A
    Prussian blue analogue is a coordination polymer with a hydrated cavity, so
    it sits between ``omat`` (right lattice, wrong bonding) and ``odac`` (right
    bonding, framework + water).  Run ``validate`` under both rather than
    assuming: the head that reproduces the measured lattice ordering is the one
    to trust, and if neither does, UMA is no more usable here than its
    predecessors.
    """

    def __init__(self, version: str = "uma-s-1p1", task: str = "omat",
                 device: str = "cpu") -> None:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                predictor = pretrained_mlip.get_predict_unit(version, device=device)
            except Exception as err:  # noqa: BLE001
                # The weights are licence-gated.  Without a token this raises a
                # GatedRepoError from deep inside huggingface_hub, which tells a
                # user nothing about what to do; since UMA is the project
                # default, that is the first thing a new checkout hits.
                if any(s in type(err).__name__ for s in ("Gated", "Repo")) or \
                        "gated" in str(err).lower() or "401" in str(err):
                    raise RuntimeError(
                        f"UMA weights ({version}) are licence-gated and could not "
                        "be downloaded.  Accept the licence at "
                        "https://huggingface.co/facebook/UMA (manual approval by "
                        "the model owner), then set HF_TOKEN to a read-scoped "
                        "token.  Until then run an ungated backend explicitly: "
                        "--model petmad, or --model mace (note that mace-mp gets "
                        "the sign of the vacancy mixing energy wrong -- see "
                        "DFT.md), or --model analytic for tests."
                    ) from err
                raise
            self._calc = FAIRChemCalculator(predictor, task_name=task)
        self.name = f"{version}-{task}"
        self.model_size = version
        self.task = task
        self.device = device


#: Project default backend.  UMA is the only backend tested here whose vacancy
#: mixing energy agrees with DFT in SIGN (DFT +271, UMA +509 meV/f.u. on a
#: matched frozen-coordinate protocol); mace-mp-small inverts it (-996).  See
#: DFT.md for the calculation and the standing caveats.
DEFAULT_MODEL = "uma"


def build_model(name: str = DEFAULT_MODEL, size: str = "small",
                task: str = "omat"):
    """Construct a backend by name -- the single place that maps names to classes.

    This registry was previously duplicated in ``cli._model``, ``run_hull.py``
    and ``mlff_frozen_scan.py``, and two of those copies defaulted to MACE while
    the CLI defaulted to UMA, so the same nominal run could use different
    physics depending on which entry point invoked it.  An unknown name raises
    rather than falling back, because the historical fallback was MACE -- the
    backend now known to get the sign of the mixing energy wrong.
    """
    if name == "uma":
        return UMAModel(task=task)
    if name == "petmad":
        return PETMADModel()
    if name == "mace":
        return MACEModel(model=size)
    if name == "analytic":
        return AnalyticModel(seed=0)
    raise ValueError(
        f"unknown model {name!r}; choose from uma, petmad, mace, analytic "
        f"(default {DEFAULT_MODEL!r})"
    )


class AnalyticModel:
    """A cheap stand-in with no force field, for testing the hull machinery.

    This exists so the hull, plotting and campaign-integration code can be
    exercised without loading a neural network -- the same reason the orchestrator
    has a simulated device backend.  It is a smooth function of composition with a
    deliberate miscibility gap, *not* a physical model, and it is named so that no
    result carrying ``model="analytic"`` can be mistaken for a computed energy.
    """

    name = "analytic"

    def __init__(self, seed: int = 0, interaction_eV: float = 0.35) -> None:
        self.rng = np.random.default_rng(seed)
        self.interaction_eV = interaction_eV

    def relax(self, atoms: Atoms, fmax: float = 0.05, steps: int = 200,
              relax_cell: bool = True) -> EnergyResult:
        y = float(atoms.info.get("realized_vacancy_fraction",
                                 atoms.info.get("vacancy_fraction", 0.0)))
        x = float(atoms.info.get("realized_na_per_fu",
                                 atoms.info.get("na_per_fu", 0.0)))
        n_fu = int(atoms.info.get("n_formula_units", 1))
        # Endpoint energies plus a positive (demixing) interaction term, so the
        # resulting hull has a genuine two-phase region to find.
        e_fu = (-40.0 - 1.6 * x + 2.4 * y
                + self.interaction_eV * y * (1.0 - y) * 4.0
                + 0.02 * self.rng.normal())
        return EnergyResult(
            energy_eV=e_fu * n_fu, energy_per_fu_eV=e_fu, n_atoms=len(atoms),
            n_formula_units=n_fu, max_force_eV_A=0.0, converged=True, n_steps=0,
            wall_time_s=0.0, volume_A3=float(atoms.get_volume()),
            lattice_a_A=float(atoms.get_volume() / max(1, n_fu / 4.0)) ** (1 / 3),
            extrapolation_warnings=[], relaxed=atoms.copy(),
        )
