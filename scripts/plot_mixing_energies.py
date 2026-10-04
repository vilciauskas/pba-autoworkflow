# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot the energetics along the Mn pseudo-binary path: DFT against force fields.

Terminology, because the distinction decides what can be plotted.  A *formation*
energy is referenced to elemental standard states (bulk Mn, bulk Fe, graphite,
N2, ...), and none of those were computed at this level of theory, so no
formation energy is available and none is claimed here.  What the hull -- and
therefore ``HullPrior`` -- actually uses is the *mixing* energy: the energy of an
intermediate composition measured against the straight line joining the two
endpoints of the charge-balanced path, y = 0 and y = 0.5.  Elemental references
cancel in that difference, which is why it is the well-posed quantity.

Two consequences:

* The y = 0.5 endpoint is not optional.  Without it there is no reference line
  and no mixing energy at all, only total energies at different compositions
  that cannot be compared.
* Water must be referenced out.  A vacancy is filled by 6 H2O, so the cells
  differ in water content; ``n_water_per_fu`` is linear in y (0, 1.5, 3.0), and
  the grand potential subtracts ``mu_water * n_water``.  The chemical potential
  must come from the same functional and cutoff as the framework, or it will not
  cancel -- here a PBE/PW(340) water molecule, -9.9031 eV.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

FU_PER_CELL = 4          # 1x1x1 conventional cubic cell holds 4 formula units
DFT = "runs/dft/lattice_scan.json"
UMA = "runs/dft/mlff_frozen_scan_uma.json"
WATER = "runs/dft/water_reference.json"


def fit_min(a, e, k=5):
    """Parabola through the k points centred on the lowest-energy point.

    Returns (a_min, e_min).  A wide-range fit is a poor model for an anharmonic
    E(a) curve, so the window is deliberately local; ``k`` is varied by the
    caller to expose how much the answer depends on it.
    """
    a, e = np.asarray(a, float), np.asarray(e, float)
    order = np.argsort(a); a, e = a[order], e[order]
    if len(a) < 3:
        return np.nan, np.nan
    i = int(np.argmin(e))
    # The lowest sampled point must have sampled neighbours on BOTH sides.  If
    # the minimum sits at an end of the grid the curve is still descending
    # there, and a parabola minimum computed from such a set is an
    # extrapolation beyond the data, not a measurement of it.  This exact
    # mistake produced a spurious y=0 lattice constant earlier in this project
    # (grid 9.90-10.90 A, energy still falling at the last point), so it is
    # refused here rather than reported with a plausible-looking number.
    if i == 0 or i == len(a) - 1:
        return np.nan, np.nan
    k = min(k, len(a))
    lo = max(0, min(i - k // 2, len(a) - k)); hi = lo + k
    c2, c1, c0 = np.polyfit(a[lo:hi], e[lo:hi], 2)
    if c2 <= 0:
        return np.nan, np.nan
    am = -c1 / (2 * c2)
    return float(am), float(c2 * am**2 + c1 * am + c0)


def dft_series(spin_tol=3.0):
    """Per-formula-unit minimum energy for each DFT composition.

    Points whose total moment is far from the high-spin value of 4 Mn x 5 muB
    are a different electronic solution, not a point on the same E(a) curve, and
    are dropped rather than averaged in.
    """
    d = json.load(open(DFT))
    out = {}
    for key, blk in d.items():
        if key.startswith("_"):
            continue
        pts = [p for p in blk["points"] if p["converged"]
               and abs(p["total_moment_muB"] - 20.0) < spin_tol]
        if len(pts) < 3:
            out[blk["vacancy_fraction"]] = (np.nan, np.nan, len(pts))
            continue
        a = [p["a_A"] for p in pts]; e = [p["energy_eV"] for p in pts]
        am, em = fit_min(a, e)
        out[blk["vacancy_fraction"]] = (am, em / FU_PER_CELL, len(pts))
    return out


def uma_series():
    j = json.load(open(UMA))
    out = {}
    for key, blk in j["series"].items():
        y = float(key[1:])
        am, em = fit_min(blk["a_A"], blk["energy_eV"])
        out[y] = (am, em / FU_PER_CELL, len(blk["a_A"]))
    return out


def mixing(series, mu_water):
    """E_mix per f.u. against the y=0 and y=0.5 endpoints, water referenced out."""
    from pba_autoworkflow.thermo.hull import n_water_per_fu

    need = (0.0, 0.25, 0.5)
    if not all(y in series and np.isfinite(series[y][1]) for y in need):
        return None
    omega = {y: series[y][1] - mu_water * n_water_per_fu(y) for y in need}
    line = 0.5 * (omega[0.0] + omega[0.5])
    return omega[0.25] - line


def main() -> None:
    mu_water = json.load(open(WATER))["mu_water_eV"]
    dft, uma = dft_series(), uma_series()

    print(f"water chemical potential (PBE/PW(340)) : {mu_water:.4f} eV")
    print(f"\n{'y':>6}  {'DFT a_min':>10} {'DFT E/f.u.':>12} {'n':>3}   "
          f"{'UMA a_min':>10} {'UMA E/f.u.':>12} {'n':>3}")
    for y in (0.0, 0.25, 0.5):
        da, de, dn = dft.get(y, (np.nan, np.nan, 0))
        ua, ue, un = uma.get(y, (np.nan, np.nan, 0))
        print(f"{y:6.2f}  {da:10.3f} {de:12.4f} {dn:3d}   "
              f"{ua:10.3f} {ue:12.4f} {un:3d}")

    e_dft, e_uma = mixing(dft, mu_water), mixing(uma, mu_water)
    print()
    if e_dft is None:
        missing = [y for y in (0.0, 0.25, 0.5)
                   if y not in dft or not np.isfinite(dft[y][1])]
        print(f"DFT  E_mix(0.25): NOT AVAILABLE -- endpoint(s) {missing} incomplete. "
              "Without both endpoints there is no reference line.")
    else:
        print(f"DFT  E_mix(0.25) = {e_dft*1000:+8.1f} meV/f.u.")
    if e_uma is not None:
        print(f"UMA  E_mix(0.25) = {e_uma*1000:+8.1f} meV/f.u.  (frozen coords, matched)")
    print(f"MACE E_mix(0.25) = {-995.9:+8.1f} meV/f.u.  (relaxed coords, stored run)")
    print(f"UMA  E_mix(0.25) = {+12.0:+8.1f} meV/f.u.  (relaxed coords, stored run)")

    json.dump({"mu_water_eV": mu_water,
               "dft": {str(k): {"a_min_A": v[0], "e_per_fu_eV": v[1], "n_points": v[2]}
                       for k, v in dft.items()},
               "uma_frozen": {str(k): {"a_min_A": v[0], "e_per_fu_eV": v[1],
                                       "n_points": v[2]} for k, v in uma.items()},
               "e_mix_dft_eV": e_dft, "e_mix_uma_frozen_eV": e_uma,
               "e_mix_mace_relaxed_eV": -0.9959, "e_mix_uma_relaxed_eV": 0.0120},
              open("runs/dft/mixing_energies.json", "w"), indent=1)
    print("\nwrote runs/dft/mixing_energies.json")


if __name__ == "__main__":
    os.makedirs("runs/dft", exist_ok=True)
    main()
