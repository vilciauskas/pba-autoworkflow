# SPDX-License-Identifier: GPL-3.0-or-later
"""DFT lattice scan at two vacancy fractions: is MACE's 0.350 A collapse real?

The force fields disagree about what removing framework units does to the cell.
Over y = 0 -> 0.25 on the Mn series, mace-mp-small contracts the cubic edge by
0.350 A (~10 % of volume) while uma-s-1p1-omat expands it by 0.060 A.  That
disagreement is what makes their mixing energies differ by 1.0 eV, so settling
it settles whether either hull can inform a campaign.

**Frozen internal coordinates.**  At each trial lattice constant the fractional
coordinates are held fixed and only the cell is scaled.  No ionic relaxation --
that would cost ~9 h per composition instead of ~1 h for the whole scan.  The
consequence is that this measures the *framework's* volume preference, not the
fully relaxed one, so the comparison against a force field is only fair if the
force field is scanned the same way.  ``scripts/mlff_frozen_scan.py`` does
exactly that, and the two are compared on identical footing.

Everything is checkpointed to JSON after each lattice point, so a wall-clock
kill leaves usable partial results rather than nothing.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from gpaw import GPAW, PW  # noqa: E402

from pba_autoworkflow.thermo.structures import PBAComposition, build_pba  # noqa: E402

OUT = "runs/dft/lattice_scan.json"
ECUT = 340.0            # eV; converged enough for a volume trend, not for absolute energies
SCF_MAXITER = 90


def initial_moments(atoms):
    """High-spin on the divalent N-coordinated metal, low-spin Fe at the C end.

    A PBA puts M(II) in a weak field (nitrogen end of cyanide) and Fe(II) in a
    strong field (carbon end).  Starting Fe near zero and M near its high-spin
    value is the physically motivated guess; a bad guess costs SCF iterations
    and can converge to the wrong spin state entirely.
    """
    hs = {"Mn": 5.0, "Fe": 0.0, "Co": 3.0, "Ni": 2.0, "Cu": 1.0}
    syms = atoms.get_chemical_symbols()
    # The framework Fe is the one bonded to carbon; every Fe in this structure
    # is a framework Fe, so it starts low-spin.
    return [hs.get(s, 0.0) if s != "Fe" else 0.0 for s in syms]


def scan_one(metal: str, y: float, a_values, results: dict) -> None:
    comp = PBAComposition(metal, 2.0 - 4.0 * y, y)
    base = build_pba(comp, supercell=(1, 1, 1), rng=np.random.default_rng(0))
    a0 = float(base.cell.lengths()[0])
    frac = base.get_scaled_positions().copy()
    key = f"{metal}_y{y:.3f}"
    results.setdefault(key, {"metal": metal, "vacancy_fraction": y,
                             "n_atoms": len(base), "a0_built_A": a0,
                             "ecut_eV": ECUT, "points": []})

    for a in a_values:
        done = {p["a_A"] for p in results[key]["points"]}
        if any(abs(a - d) < 1e-6 for d in done):
            continue
        atoms = base.copy()
        atoms.set_cell([a, a, a], scale_atoms=False)
        atoms.set_scaled_positions(frac)      # freeze fractional coordinates
        atoms.set_initial_magnetic_moments(initial_moments(atoms))
        atoms.calc = GPAW(mode=PW(ECUT), xc="PBE", spinpol=True, kpts=(1, 1, 1),
                          maxiter=SCF_MAXITER,
                          txt=f"runs/dft/scf_{key}_a{a:.3f}.txt")
        t0 = time.time()
        try:
            e = float(atoms.get_potential_energy())
            mom = float(atoms.calc.get_magnetic_moment())
            ok, err = True, ""
        except Exception as exc:  # noqa: BLE001
            e, mom, ok, err = float("nan"), float("nan"), False, type(exc).__name__
        results[key]["points"].append(
            {"a_A": float(a), "energy_eV": e, "total_moment_muB": mom,
             "converged": ok, "error": err, "wall_s": time.time() - t0})
        print(f"  {key}  a={a:6.3f}  E={e:12.4f} eV  moment={mom:6.2f}  "
              f"{'ok' if ok else err}  ({time.time()-t0:.0f}s)", flush=True)
        with open(OUT, "w") as fh:
            json.dump(results, fh, indent=1)


def fit_minimum(points):
    """Parabola through the converged points; returns (a_min, curvature)."""
    good = [(p["a_A"], p["energy_eV"]) for p in points
            if p["converged"] and np.isfinite(p["energy_eV"])]
    if len(good) < 3:
        return float("nan"), float("nan"), len(good)
    a = np.array([g[0] for g in good])
    e = np.array([g[1] for g in good])
    c2, c1, c0 = np.polyfit(a, e, 2)
    if c2 <= 0:            # opening downward: no interior minimum
        return float("nan"), float(c2), len(good)
    return float(-c1 / (2 * c2)), float(c2), len(good)


def main() -> None:
    os.makedirs("runs/dft", exist_ok=True)
    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))
        print(f"resuming from {OUT}")

    # Centred on the measured Mn constant (10.53 A), wide enough to bracket
    # both force-field predictions (9.685 and 10.043 A).
    grid_y0 = [9.90, 10.15, 10.40, 10.65, 10.90, 11.15, 11.40, 11.65]
    grid_y25 = [9.65, 9.95, 10.25, 10.55, 10.85, 11.15, 11.45, 11.75]

    print(f"DFT lattice scan, PW({ECUT:.0f}) eV, gamma-point, spin-polarized, "
          f"frozen fractional coordinates\n")
    scan_one("Mn", 0.0, grid_y0, results)
    scan_one("Mn", 0.25, grid_y25, results)
    # y = 0.5 is the far endpoint of the charge-balanced path (Na -> 0).
    # Needed for a mixing energy: E_mix(0.25) is measured against the
    # y=0 and y=0.5 endpoints, so without it there is no reference line.
    grid_y50 = [9.95, 10.25, 10.55, 10.85, 11.15, 11.45, 11.75]
    scan_one("Mn", 0.5, grid_y50, results)

    print("\n" + "=" * 68)
    mins = {}
    for key, blk in results.items():
        if key.startswith("_"):      # the summary block written by a prior run
            continue
        a_min, curv, n = fit_minimum(blk["points"])
        mins[blk["vacancy_fraction"]] = a_min
        print(f"{key}: parabola minimum a = {a_min:.3f} A "
              f"(curvature {curv:+.3f} eV/A^2, {n} converged points)")

    if 0.0 in mins and 0.25 in mins and np.isfinite(mins[0.0]) and np.isfinite(mins[0.25]):
        d_dft = mins[0.25] - mins[0.0]
        print(f"\nDFT  delta_a (y=0.25 - y=0) = {d_dft:+.3f} A")
        print(f"MACE (relaxed internals)     = {-0.350:+.3f} A")
        print(f"UMA  (relaxed internals)     = {+0.060:+.3f} A")
        print("\n  Note: the force-field numbers above relaxed internal coordinates;\n"
              "  this scan froze them.  scripts/mlff_frozen_scan.py produces the\n"
              "  matched frozen-coordinate force-field numbers for a fair comparison.")
        results["_summary"] = {"a_min_y0_A": mins[0.0], "a_min_y25_A": mins[0.25],
                               "delta_a_dft_A": d_dft,
                               "delta_a_mace_relaxed_A": -0.350,
                               "delta_a_uma_relaxed_A": 0.060}
        with open(OUT, "w") as fh:
            json.dump(results, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
