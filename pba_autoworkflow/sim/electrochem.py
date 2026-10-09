# SPDX-License-Identifier: GPL-3.0-or-later
"""Simulated electrochemistry: insertion of Zn2+, Na+ and K+ into the framework.

Illustrative model for exercising the selectivity workflow, not a fit to data.

* Each framework phase has a formal potential per inserting cation (V vs
  Zn2+/Zn).  K+ > Na+ > Zn2+ throughout, as for PBAs generally; the K-Zn gap is
  smaller in the disordered cubic Zn framework than in R-3c, which encodes the
  *hypothesis* that vacancy-related sites make it more Zn2+-tolerant.
* Insertion from a mixed electrolyte is competitive.  Per electron,
  A+ + 1/2 Zn(host) <=> A(host) + 1/2 Zn2+, so the charge-equivalent share of
  ion i is proportional to c_i^(1/z_i) exp(F E_i / RT).
* R-3c (and monoclinic) phases insert by two-phase reaction (flat plateau);
  cubic ones by solid solution (sloping curve).  Capacity fades exponentially;
  Zn2+ fades faster in R-3c (Pilipavicius et al.: 24.9 % vs 53.5 % retention
  after 200 cycles for R- vs DC-ZnHCF).
* Zinc frameworks dissolve when K+ dominates the electrolyte (Fe released to
  the solution rises steeply above a K+ mole fraction of ~0.7).
"""

from __future__ import annotations

import math

import numpy as np

from ..schema import FARADAY_C_PER_MOL, formula_weight
from .ground_truth import LatentState

RT_F = 8.314462618 * 298.15 / FARADAY_C_PER_MOL
CHARGE = {"Zn": 2, "Na": 1, "K": 1}

#: formal potentials (V vs Zn2+/Zn) per phase class and ion
_E0 = {
    "r3c": {"Zn": 1.70, "Na": 1.80, "K": 1.92},
    "cubic_zn": {"Zn": 1.66, "Na": 1.76, "K": 1.88},
    "dc": {"Zn": 1.68, "Na": 1.75, "K": 1.85},
    "pba": {"Zn": 1.70, "Na": 1.85, "K": 1.95},
}
_METAL_SHIFT = {"Mn": 0.05, "Fe": -0.05, "Co": 0.0, "Ni": 0.02, "Cu": 0.04, "Zn": 0.0}
#: curve width (V): small = two-phase plateau, large = solid solution
_WIDTH = {"r3c": 0.006, "p21n": 0.008, "cubic_zn": 0.045, "dc": 0.06, "pba": 0.04}
#: share of the Fe sites each ion reaches
_UTIL = {"Zn": {"r3c": 0.75, "cubic_zn": 0.6, "dc": 0.55, "pba": 0.5, "p21n": 0.5},
         "Na": 0.85, "K": 0.9}
#: capacity fade per cycle
_FADE = {"Zn": {"r3c": 0.0070, "cubic_zn": 0.0040, "dc": 0.0032, "pba": 0.0050, "p21n": 0.0050},
         "Na": 0.0015, "K": 0.0012}


def _classes(latent: LatentState) -> list[tuple[str, float]]:
    """(phase class, share of framework) for the electrochemically active phases."""
    out = []
    for key, share in latent.polymorph_shares.items():
        pid = key.split("/")[0]
        if pid == "znhcf_r3c":
            out.append(("r3c", share))
        elif pid == "pba_p21n":
            out.append(("p21n", share))
        elif latent.metal == "Zn":
            dc = latent.disordered_cubic_share
            if dc > 0:
                out.append(("dc", share * dc))
            if dc < 1:
                out.append(("cubic_zn", share * (1.0 - dc)))
        else:
            out.append(("pba", share))
    return [(c, w) for c, w in out if w > 1e-4]


def formal_potential(cls: str, ion: str, metal: str, offsets: dict[str, float]) -> float:
    base = _E0.get(cls, _E0["pba"]) if cls != "p21n" else _E0["pba"]
    return base[ion] + (_METAL_SHIFT.get(metal, 0.0) if cls in ("pba", "p21n") else 0.0) \
        + offsets.get(ion, 0.0)


def _util(ion: str, cls: str) -> float:
    u = _UTIL[ion]
    return u.get(cls, 0.5) if isinstance(u, dict) else u


def _fade(ion: str, cls: str) -> float:
    f = _FADE[ion]
    return f.get(cls, 0.005) if isinstance(f, dict) else f


def ion_shares(cls: str, metal: str, electrolyte_M: dict[str, float],
               offsets: dict[str, float]) -> dict[str, float]:
    """Charge-equivalent share of each ion inserted into one phase class."""
    logw = {i: math.log(max(c, 1e-12)) / CHARGE[i] + formal_potential(cls, i, metal, offsets) / RT_F
            for i, c in electrolyte_M.items() if c > 0}
    m = max(logw.values())
    w = {i: math.exp(v - m) for i, v in logw.items()}
    tot = sum(w.values())
    return {i: v / tot for i, v in w.items()}


def mixed_potential(cls: str, metal: str, electrolyte_M: dict[str, float],
                    offsets: dict[str, float]) -> float:
    terms = [math.log(max(c, 1e-12)) / CHARGE[i] + formal_potential(cls, i, metal, offsets) / RT_F
             for i, c in electrolyte_M.items() if c > 0]
    m = max(terms)
    return RT_F * (m + math.log(sum(math.exp(t - m) for t in terms)))


def simulate_cycling(latent: LatentState, electrolyte_M: dict[str, float],
                     rng: np.random.Generator, offsets: dict[str, float] | None = None,
                     n_cycles: int = 20, current_mA_g: float = 100.0,
                     n_points: int = 160) -> dict:
    """Galvanostatic curves plus the electrode/electrolyte state after the run.

    Returns a dict with ``charge_q/E`` and ``discharge_q/E`` (lists per cycle),
    ``inserted`` (ions per formula unit in the electrode after the final,
    inserting half-cycle) and ``dissolved_fe_fraction``.
    """
    offsets = offsets or {}
    metal = latent.metal
    fw = formula_weight(metal, latent.na_per_fu, latent.vacancy_fraction,
                        latent.water_per_fu, latent.k_per_fu)
    q_sites = (1.0 - latent.vacancy_fraction) * FARADAY_C_PER_MOL / (3.6 * fw)   # mAh/g
    kinetic = float(np.clip(1.06 / (1.0 + (latent.domain_size_nm / 95.0) ** 1.7), 0.3, 1.0))
    order = max(latent.crystallinity, 1e-3) ** 0.35
    eta = 0.02 * current_mA_g / 100.0

    phases = []          # (E_eff, width, Q0, fade, shares)
    for cls, w in _classes(latent):
        sh = ion_shares(cls, metal, electrolyte_M, offsets)
        util = sum(sh[i] * _util(i, cls) for i in sh)
        fade = sum(sh[i] * _fade(i, cls) for i in sh)
        e_eff = mixed_potential(cls, metal, electrolyte_M, offsets)
        phases.append((e_eff, _WIDTH.get(cls, 0.04), w * q_sites * util * kinetic * order,
                       fade, sh, w * util * kinetic * order))
    if not phases:
        raise ValueError("no electrochemically active framework phase")

    e_lo = min(p[0] for p in phases) - 0.4
    e_hi = max(p[0] for p in phases) + 0.4
    grid = np.linspace(e_hi, e_lo, 2000)
    chg_q, chg_E, dis_q, dis_E = [], [], [], []
    for n in range(n_cycles):
        noise = 1.0 + rng.normal(0.0, 0.004)
        q_dis = np.zeros_like(grid)
        q_chg = np.zeros_like(grid)
        for e_eff, width, q0, fade, _sh, _ in phases:
            qn = q0 * math.exp(-fade * n) * noise
            q_dis += qn / (1.0 + np.exp(-(e_eff - eta - grid) / width))
            q_chg += qn / (1.0 + np.exp(-(grid[::-1] - e_eff - eta) / width))
        for q, E, out_q, out_E in ((q_dis, grid, dis_q, dis_E), (q_chg, grid[::-1], chg_q, chg_E)):
            target = np.linspace(0.0, q[-1], n_points)
            out_q.append(target)
            out_E.append(np.interp(target, q, E) + rng.normal(0.0, 0.001, n_points))

    # Electrode composition after the last (inserting) half-cycle, per f.u.
    inserted: dict[str, float] = {}
    for _e, _w, _q0, fade, sh, sites in phases:
        for ion, chi in sh.items():
            inserted[ion] = inserted.get(ion, 0.0) + (1.0 - latent.vacancy_fraction) * sites \
                * math.exp(-fade * (n_cycles - 1)) * chi / CHARGE[ion]
    cations = {i: c for i, c in electrolyte_M.items() if c > 0}
    x_k = cations.get("K", 0.0) / sum(cations.values())
    per_cycle = (0.0004 + 0.01 / (1.0 + math.exp(-(x_k - 0.7) / 0.05))) if metal == "Zn" else 0.0002
    dissolved = float(min(1.0, per_cycle * n_cycles * (1.0 + rng.normal(0.0, 0.05))))
    return dict(charge_q=chg_q, charge_E=chg_E, discharge_q=dis_q, discharge_E=dis_E,
                inserted=inserted, dissolved_fe_fraction=dissolved, formula_weight=fw)
