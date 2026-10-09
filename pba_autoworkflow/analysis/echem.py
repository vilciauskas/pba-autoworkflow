# SPDX-License-Identifier: GPL-3.0-or-later
"""Electrochemistry reduction: formal potentials, mechanism, and ion selectivity.

From single-ion cycling (one cation per electrolyte):

* **formal potential** E°' -- midpoint of the charge- and discharge-averaged
  potentials of the second cycle (the first charge also extracts the as-made
  A cations, so it is not used);
* **capacity** -- second-cycle discharge, mAh/g; **retention** -- last over
  second discharge;
* **plateau fraction** -- share of the second discharge capacity delivered
  where |dE/dq| is below 10 % of its median over the half-cycle mid-range: near 1
  for a two-phase (conversion) reaction, low for a solid solution.

From insertion out of a mixed electrolyte followed by digestion of the electrode:

* **inserted amounts** per Fe from the digest -- except for the framework's own
  metal (Zn into zinc hexacyanoferrate), whose inserted share is far below the
  assay error of the framework content.  That ion is obtained by coulometry
  instead: charge equivalents passed in the final discharge minus those carried
  by the other (assayed) ions, divided by its charge.  Requires the formula
  weight of the active material;

* **separation factor** alpha_A/B = (x_A / x_B)_solid / (c_A / c_B)_electrolyte
  (concentrations stand in for activities -- use matched ionic strength);
* **predicted separation factor** from the single-ion formal potentials, per
  electron:  ln K_A/B = F (E°'_A - E°'_B) / RT with each ion's concentration raised
  to 1/z.  For a monovalent pair this is the familiar 59 mV per decade at 25 °C.
  Expressed in the same alpha convention:
  alpha_A/B = exp(F(E_A - E_B)/RT) * c_A^(1/z_A - 1) * c_B^(1 - 1/z_B) * z_B / z_A,
  where the z ratio converts charge equivalents to ions.
"""

from __future__ import annotations

import math

import numpy as np

from ..schema import FARADAY_C_PER_MOL, EchemCycleData, EchemDescriptors, ICPResult

RT_F_25C = 8.314462618 * 298.15 / FARADAY_C_PER_MOL
CHARGE = {"Zn": 2, "Na": 1, "K": 1, "Li": 1, "Mg": 2, "Ca": 2}


def _avg_potential(q: np.ndarray, E: np.ndarray) -> float:
    q = np.asarray(q, float); E = np.asarray(E, float)
    if q.size < 2 or q[-1] <= q[0]:
        return float("nan")
    return float(np.trapezoid(E, q) / (q[-1] - q[0]))


def plateau_fraction(q: np.ndarray, E: np.ndarray) -> float:
    q = np.asarray(q, float); E = np.asarray(E, float)
    if q.size < 10 or q[-1] <= 0:
        return float("nan")
    # Flat = |dE/dq| below a fixed 1 mV per (mAh/g): a two-phase plateau is
    # flat to within the cell's noise, a solid solution slopes by ~0.1 V over
    # tens of mAh/g (~3-5 mV per mAh/g).  Scale-free thresholds fail on curves
    # that are entirely sloped, so this is absolute.
    E_s = np.convolve(E, np.ones(5) / 5.0, mode="same")
    slope = np.abs(np.gradient(E_s, q))
    flat = slope < 1e-3
    flat[:3] = flat[-3:] = False
    dq = np.gradient(q)
    return float(np.sum(dq[flat]) / np.sum(dq))


def single_ion_descriptors(data: EchemCycleData) -> dict[str, float]:
    k = 1 if len(data.discharge_q) > 1 else 0
    e_dis = _avg_potential(data.discharge_q[k], data.discharge_E[k])
    e_chg = _avg_potential(data.charge_q[k], data.charge_E[k])
    q2 = float(data.discharge_q[k][-1])
    q_last = float(data.discharge_q[-1][-1])
    return dict(
        formal_potential_V=0.5 * (e_dis + e_chg),
        capacity_mAh_g=q2,
        retention=q_last / q2 if q2 > 0 else float("nan"),
        plateau_fraction=plateau_fraction(data.discharge_q[k], data.discharge_E[k]),
    )


def predicted_separation_factor(e_a: float, e_b: float, a: str, b: str,
                                electrolyte_M: dict[str, float]) -> float:
    za, zb = CHARGE[a], CHARGE[b]
    ca, cb = electrolyte_M[a], electrolyte_M[b]
    ln = ((e_a - e_b) / RT_F_25C + (1.0 / za - 1.0) * math.log(ca)
          + (1.0 - 1.0 / zb) * math.log(cb) + math.log(zb / za))
    return float(math.exp(ln))


def inserted_per_fe(data: EchemCycleData, electrode: ICPResult,
                    framework_metal: str | None, formula_weight: float | None,
                    fe_per_fu: float | None) -> dict[str, float]:
    """Inserted cations per Fe after the final discharge.

    Assayed directly for every ion except ``framework_metal``, which is taken by
    coulometric difference (needs ``formula_weight`` and ``fe_per_fu``).
    """
    conc = electrode.concentrations_mol_L
    fe = conc.get("Fe", 0.0)
    ions = [i for i, c in data.electrolyte_M.items() if c > 0]
    out = {}
    for i in ions:
        if i != framework_metal:
            out[i] = conc.get(i, 0.0) / fe if fe > 0 else float("nan")
    if framework_metal in ions:
        if not formula_weight or not fe_per_fu:
            out[framework_metal] = float("nan")
        else:
            q = float(data.discharge_q[-1][-1])                     # mAh/g
            e_per_fe = q * 3.6 * formula_weight / FARADAY_C_PER_MOL / fe_per_fu
            rest = sum(CHARGE[i] * v for i, v in out.items() if math.isfinite(v))
            out[framework_metal] = max(e_per_fe - rest, 0.0) / CHARGE[framework_metal]
    return out


def separation_from_inserted(x: dict[str, float], a: str, b: str,
                             electrolyte_M: dict[str, float]) -> float:
    ca, cb = electrolyte_M.get(a, 0.0), electrolyte_M.get(b, 0.0)
    xa, xb = x.get(a, float("nan")), x.get(b, float("nan"))
    if not (math.isfinite(xa) and math.isfinite(xb)) or ca <= 0 or cb <= 0:
        return float("nan")
    if xb <= 0:
        return float("inf") if xa > 0 else float("nan")
    return float(max(xa, 0.0) / xb / (ca / cb))


def build_echem_descriptors(single: dict[str, EchemCycleData],
                            mixed: list[tuple[EchemCycleData, ICPResult, ICPResult | None]],
                            framework_metal: str | None = None,
                            formula_weight: float | None = None,
                            fe_per_fu: float | None = None) -> EchemDescriptors:
    """``single``: ion -> cycling data; ``mixed``: (cycling, electrode digest,
    electrolyte assay) per mixed electrolyte."""
    d = EchemDescriptors()
    for ion, data in single.items():
        s = single_ion_descriptors(data)
        d.formal_potential_V[ion] = s["formal_potential_V"]
        d.capacity_mAh_g[ion] = s["capacity_mAh_g"]
        d.retention[ion] = s["retention"]
        d.plateau_fraction[ion] = s["plateau_fraction"]
        d.n_cycles = max(d.n_cycles, len(data.discharge_q))
    dissolved = []
    for data, electrode, electrolyte in mixed:
        ions = sorted((i for i, c in data.electrolyte_M.items() if c > 0),
                      key=lambda i: (CHARGE[i], i))
        for i, a in enumerate(ions):
            for b in ions[i + 1:]:
                # lower charge first (alpha_K/Zn, alpha_Na/Zn); same charge:
                # larger cation first (alpha_K/Na)
                if CHARGE[a] == CHARGE[b]:
                    hi, lo = (a, b) if a == "K" else (b, a)
                else:
                    hi, lo = a, b
                pair = f"{hi}/{lo}"
                x = inserted_per_fe(data, electrode, framework_metal, formula_weight, fe_per_fu)
                d.separation_factor[pair] = separation_from_inserted(x, hi, lo, data.electrolyte_M)
                if hi in d.formal_potential_V and lo in d.formal_potential_V:
                    d.predicted_separation_factor[pair] = predicted_separation_factor(
                        d.formal_potential_V[hi], d.formal_potential_V[lo], hi, lo,
                        data.electrolyte_M)
        fe_el = electrode.concentrations_mol_L.get("Fe", 0.0) * electrode.digest_volume_mL
        if electrolyte is not None and fe_el > 0:
            dissolved.append(electrolyte.concentrations_mol_L.get("Fe", 0.0)
                             * electrolyte.digest_volume_mL / fe_el)
    if dissolved:
        d.dissolved_fe_fraction = float(np.mean(dissolved))
    return d
