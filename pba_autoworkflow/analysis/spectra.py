# SPDX-License-Identifier: GPL-3.0-or-later
"""Elemental-assay reduction and the capacity estimate.

The ICP routine converts digest concentrations into a per-formula-unit
composition, propagating the assay error into the reported Na content and
correcting for the residual surface sodium that survives washing.
"""

from __future__ import annotations

import math

import numpy as np

from ..schema import (
    ATOMIC_WEIGHT,
    CompositionDescriptors,
    ICPResult,
    SynthesisParameters,
    theoretical_capacity_mAh_g,
)

#: Empirical surface-sodium correction determined from wash-cycle calibration.
SURFACE_NA_CORRECTION = 0.035


def analyze_icp(icp: ICPResult, params: SynthesisParameters) -> CompositionDescriptors:
    """Convert digest concentrations to Na_n M[Fe(CN)6]_(1-x) . zH2O stoichiometry.

    Water content is obtained by mass balance: whatever dry mass the balance saw
    that the framework cations and cyanide cannot account for is assigned to
    coordinated and interstitial water.
    """
    conc = icp.concentrations_mol_L
    vol_L = icp.digest_volume_mL * 1e-3
    metal = params.metal

    n_na = conc.get("Na", 0.0) * vol_L

    # Hexacyanoferrate content from CHN carbon: six C per intact [Fe(CN)6].  This
    # is the measurement that resolves the C-site sublattice; it is used in
    # preference to iron for *every* analogue, and it is the only option for the
    # Fe analogue, where ICP reports a single indistinguishable iron pool.
    n_hcf_carbon: float | None = None
    if math.isfinite(icp.carbon_wt_pct) and icp.carbon_wt_pct > 0:
        c_mol = (icp.carbon_wt_pct / 100.0) * (icp.digest_mass_mg * 1e-3) / ATOMIC_WEIGHT["C"]
        n_hcf_carbon = c_mol / 6.0

    if metal == "Fe":
        # ICP sees total Fe = N-site + C-site.  Carbon fixes the C-site count, and
        # the N-site metal is the remainder.  Splitting total Fe 50/50 instead --
        # as an earlier version did -- forces Fe/M to exactly 1.000 by construction,
        # so Prussian blue is reported vacancy-free no matter what was made, and the
        # optimizer is told this analogue is always perfect.
        n_fe_total = conc.get("Fe", 0.0) * vol_L
        if n_hcf_carbon is None:
            return CompositionDescriptors(
                na_per_fu=0.0, fe_per_metal=float("nan"), vacancy_fraction=float("nan"),
                water_per_fu=0.0,
                formula="undetermined (Fe analogue requires a carbon assay)",
            )
        n_fe_c = n_hcf_carbon
        n_metal = n_fe_total - n_fe_c
    else:
        n_metal = conc.get(metal, 0.0) * vol_L
        # Carbon when available, iron as the fallback for a deck without CHN.
        n_fe_c = n_hcf_carbon if n_hcf_carbon is not None else conc.get("Fe", 0.0) * vol_L

    if n_metal <= 0:
        return CompositionDescriptors(
            na_per_fu=0.0, fe_per_metal=0.0, vacancy_fraction=0.0,
            water_per_fu=0.0, formula="undetermined",
        )

    fe_per_metal = n_fe_c / n_metal
    # Deliberately not clipped at zero: see CompositionDescriptors.  A nearly
    # vacancy-free sample assays above Fe/M = 1 half the time, and flattening
    # those to exactly 0.000 would erase the ordering the optimizer needs.
    vacancy = float(np.clip(1.0 - fe_per_metal, -0.15, 0.9))
    na_raw = n_na / n_metal
    na_per_fu = float(np.clip(na_raw / (1.0 + SURFACE_NA_CORRECTION), 0.0, 2.0))

    # Water by mass balance on the digested aliquot.
    hcf_mass = ATOMIC_WEIGHT["Fe"] + 6 * (ATOMIC_WEIGHT["C"] + ATOMIC_WEIGHT["N"])
    anhydrous_fw = (
        na_per_fu * ATOMIC_WEIGHT["Na"]
        + ATOMIC_WEIGHT[metal]
        + (1.0 - vacancy) * hcf_mass
    )
    measured_fw = (icp.digest_mass_mg * 1e-3) / n_metal if n_metal > 0 else 0.0
    water_mass = max(0.0, measured_fw - anhydrous_fw)
    water_per_fu = float(np.clip(
        water_mass / (2 * ATOMIC_WEIGHT["H"] + ATOMIC_WEIGHT["O"]), 0.0, 8.0
    ))

    formula = (
        f"Na{na_per_fu:.2f}{metal}[Fe(CN)6]{1 - vacancy:.2f}"
        f"·{water_per_fu:.1f}H2O"
    )
    return CompositionDescriptors(
        na_per_fu=na_per_fu,
        fe_per_metal=float(fe_per_metal),
        vacancy_fraction=vacancy,
        water_per_fu=water_per_fu,
        formula=formula,
    )


def estimate_capacity_mAh_g(comp: CompositionDescriptors, domain_size_nm: float,
                            crystallinity: float, metal: str) -> float:
    """Practical first-cycle capacity estimate from measured descriptors.

    This is the surrogate for an electrochemical measurement the platform does not
    have: the theoretical Na inventory discounted by the three factors that are
    known to control utilization -- vacancy content (dead A-sites and trapped
    water), diffusion length (large domains do not fully de-sodiate at rate), and
    loss of long-range order.
    """
    q_theo = theoretical_capacity_mAh_g(
        metal, comp.na_per_fu, comp.vacancy_fraction, comp.water_per_fu
    )
    # Apply redox cap: Ni and Cu PBAs have only 1 electrochemically active site (Fe)
    if metal in ("Ni", "Cu") and comp.na_per_fu > 0:
        active_na = min(comp.na_per_fu, 1.0 - comp.vacancy_fraction)
        q_theo *= (active_na / comp.na_per_fu)

    if not math.isfinite(domain_size_nm) or domain_size_nm <= 0:
        return float("nan")
    kinetic = float(np.clip(1.06 / (1.0 + (domain_size_nm / 95.0) ** 1.7), 0.30, 1.0))
    defect = float(np.clip(1.0 - 1.15 * comp.vacancy_fraction, 0.15, 1.0))
    hydration = float(np.clip(1.0 - 0.055 * comp.water_per_fu, 0.55, 1.0))
    return float(q_theo * kinetic * defect * hydration * max(crystallinity, 1e-3) ** 0.35)
