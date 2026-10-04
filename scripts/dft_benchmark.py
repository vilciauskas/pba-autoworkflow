# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure what one PBA SCF iteration actually costs, so a cluster ask is sized
from a measurement rather than a guess.

Runs a deliberately small number of SCF iterations on the real y = 0 cell and
reports seconds per iteration.  It does NOT converge anything -- the point is
the per-iteration cost and the memory high-water mark, which is what determines
node count and wall-clock request.  Settings are labelled loose or production so
the extrapolation is honest about which is which.
"""

from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from ase.units import Bohr  # noqa: E402  (after path insert)
from gpaw import GPAW, PW  # noqa: E402

from pba_autoworkflow.thermo.structures import PBAComposition, build_pba  # noqa: E402


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def main() -> None:
    ecut = float(sys.argv[1]) if len(sys.argv) > 1 else 340.0
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    comp = PBAComposition("Mn", 2.0, 0.0)
    atoms = build_pba(comp, supercell=(1, 1, 1), rng=np.random.default_rng(0))
    # High-spin starting moments on the divalent site; Fe(II) low-spin at the
    # carbon end starts near zero.  A bad initial guess costs iterations, so
    # this is part of the cost being measured.
    moments = [5.0 if s == "Mn" else (0.0 if s == "Fe" else 0.0)
               for s in atoms.get_chemical_symbols()]
    atoms.set_initial_magnetic_moments(moments)

    n_atoms = len(atoms)
    n_elec = sum(atoms.get_atomic_numbers())
    print(f"cell            : {atoms.cell.lengths()[0]:.3f} A cubic, "
          f"{n_atoms} atoms, Z_total = {n_elec}")
    print(f"settings        : PW({ecut:.0f}) eV, gamma-point only, spin-polarized, "
          f"maxiter = {maxiter}")

    atoms.calc = GPAW(mode=PW(ecut), xc="PBE", spinpol=True,
                      kpts=(1, 1, 1), maxiter=maxiter,
                      txt="runs/dft/bench.txt")

    t0 = time.time()
    try:
        atoms.get_potential_energy()
        converged = True
    except Exception as err:  # KohnShamConvergenceError is expected at maxiter=3
        converged = False
        note = type(err).__name__
    wall = time.time() - t0

    per_iter = wall / max(1, maxiter)
    payload = {
        "n_atoms": n_atoms, "z_total": int(n_elec), "ecut_eV": ecut,
        "maxiter": maxiter, "wall_s": wall, "s_per_scf_iteration": per_iter,
        "peak_rss_gb": peak_rss_gb(), "converged": converged,
        "note": "" if converged else note,
        "gamma_only": True, "spinpol": True,
    }
    print(f"wall            : {wall:.0f} s for {maxiter} iterations")
    print(f"per iteration   : {per_iter:.1f} s")
    print(f"peak RSS        : {payload['peak_rss_gb']:.2f} GB")
    print()
    # A converged magnetic PBA SCF typically needs 40-80 iterations from a
    # rough moment guess; a relaxation needs tens of SCFs on top of that.
    for label, n_scf, n_ionic in (("single-point SCF", 60, 1),
                                  ("relaxation (fixed cell)", 60, 40)):
        hours = per_iter * n_scf * n_ionic / 3600
        print(f"  extrapolated {label:26} ~{hours:8.1f} h  "
              f"(assumes {n_scf} SCF x {n_ionic} ionic steps)")
    print("\n  Extrapolations are linear in the measured per-iteration cost and\n"
          "  assume iteration counts typical of magnetic 3d systems.  They are\n"
          "  estimates, not measurements: only the per-iteration number above\n"
          "  and the memory figure were measured here.")

    with open("runs/dft/benchmark.json", "w") as fh:
        json.dump(payload, fh, indent=1)
    print("\nwrote runs/dft/benchmark.json")
    _ = Bohr


if __name__ == "__main__":
    main()
