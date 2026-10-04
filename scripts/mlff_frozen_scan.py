# SPDX-License-Identifier: GPL-3.0-or-later
"""Force-field lattice scan with FROZEN internal coordinates.

The DFT scan in ``scripts/dft_lattice_scan.py`` cannot afford ionic relaxation,
so it holds fractional coordinates fixed and scales only the cell.  Comparing
that against a force-field number obtained WITH internal relaxation would
confound two different quantities: a genuine disagreement about volume
preference, and the extra contraction that relaxing water and Na into a vacancy
buys.  This script runs the force field the same frozen way, on the same grid,
so the difference that remains is attributable to the force field.

Usage:
    python scripts/mlff_frozen_scan.py uma      # or petmad, mace, analytic
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from pba_autoworkflow.thermo.structures import PBAComposition, build_pba  # noqa: E402

GRIDS = {0.0: [9.90, 10.15, 10.40, 10.65, 10.90, 11.15, 11.40, 11.65],
         0.25: [9.65, 9.95, 10.25, 10.55, 10.85, 11.15, 11.45, 11.75],
         0.5: [9.95, 10.25, 10.55, 10.85, 11.15, 11.45, 11.75]}


def build_model(name: str):
    from pba_autoworkflow.thermo.mlff import build_model as _bm

    return _bm(name, task="omat")


def fit_minimum(a, e):
    a, e = np.asarray(a, float), np.asarray(e, float)
    m = np.isfinite(e)
    if m.sum() < 3:
        return float("nan"), float("nan")
    c2, c1, _ = np.polyfit(a[m], e[m], 2)
    if c2 <= 0:
        return float("nan"), float(c2)
    return float(-c1 / (2 * c2)), float(c2)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "uma"
    model = build_model(name)
    os.makedirs("runs/dft", exist_ok=True)
    out = f"runs/dft/mlff_frozen_scan_{name}.json"

    res = {"model": getattr(model, "name", name), "series": {}}
    mins = {}
    for y, grid in GRIDS.items():
        comp = PBAComposition("Mn", 2.0 - 4.0 * y, y)
        base = build_pba(comp, supercell=(1, 1, 1), rng=np.random.default_rng(0))
        frac = base.get_scaled_positions().copy()
        n_fu = len(base) // 16 if len(base) >= 16 else 1
        energies = []
        for a in grid:
            atoms = base.copy()
            atoms.set_cell([a, a, a], scale_atoms=False)
            atoms.set_scaled_positions(frac)
            atoms.calc = model._calc
            e = float(atoms.get_potential_energy())
            energies.append(e)
            print(f"  y={y:.2f}  a={a:6.3f}  E={e:12.4f} eV", flush=True)
        a_min, curv = fit_minimum(grid, energies)
        mins[y] = a_min
        res["series"][f"y{y:.3f}"] = {"a_A": grid, "energy_eV": energies,
                                      "a_min_A": a_min, "curvature_eV_A2": curv,
                                      "n_atoms": len(base), "n_fu": n_fu}
        print(f"  -> y={y:.2f} minimum a = {a_min:.3f} A "
              f"(curvature {curv:+.3f} eV/A^2)\n", flush=True)

    if all(np.isfinite(v) for v in mins.values()):
        res["delta_a_A"] = mins[0.25] - mins[0.0]
        print(f"{res['model']} frozen-coordinate delta_a "
              f"(y=0.25 - y=0) = {res['delta_a_A']:+.3f} A")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
