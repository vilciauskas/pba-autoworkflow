# SPDX-License-Identifier: GPL-3.0-or-later
"""Elemental-assay reduction, Fe valence from IR, and the capacity estimate.

The ICP routine converts digest concentrations into a per-formula-unit
composition (Na and K on the A sites), correcting for the residual surface
salt that survives washing.  The IR routine estimates the Fe(II) share of the
hexacyanoferrate from the cyanide-stretch bands, and :func:`charge_balance_residual`
combines both into the check that decides whether A cations alone balance
the framework charge.
"""

from __future__ import annotations

import math

import numpy as np

from scipy.optimize import least_squares

from ..schema import (
    ATOMIC_WEIGHT,
    CompositionDescriptors,
    ICPResult,
    IRDescriptors,
    IRSpectrum,
    SynthesisParameters,
    charge_balanced_a,
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
    n_k = conc.get("K", 0.0) * vol_L

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
    k_per_fu = float(np.clip((n_k / n_metal) / (1.0 + SURFACE_NA_CORRECTION), 0.0, 2.0))
    cn_deficit = float("nan")
    if metal != "Fe" and n_hcf_carbon is not None and conc.get("Fe", 0.0) > 0:
        cn_deficit = float(6.0 * (conc["Fe"] * vol_L - n_hcf_carbon) / n_metal)

    # Water by mass balance on the digested aliquot.
    hcf_mass = ATOMIC_WEIGHT["Fe"] + 6 * (ATOMIC_WEIGHT["C"] + ATOMIC_WEIGHT["N"])
    anhydrous_fw = (
        na_per_fu * ATOMIC_WEIGHT["Na"]
        + k_per_fu * ATOMIC_WEIGHT["K"]
        + ATOMIC_WEIGHT[metal]
        + (1.0 - vacancy) * hcf_mass
    )
    measured_fw = (icp.digest_mass_mg * 1e-3) / n_metal if n_metal > 0 else 0.0
    water_mass = max(0.0, measured_fw - anhydrous_fw)
    water_per_fu = float(np.clip(
        water_mass / (2 * ATOMIC_WEIGHT["H"] + ATOMIC_WEIGHT["O"]), 0.0, 8.0
    ))

    k_txt = f"K{k_per_fu:.2f}" if k_per_fu > 0.005 else ""
    formula = (
        f"Na{na_per_fu:.2f}{k_txt}{metal}[Fe(CN)6]{1 - vacancy:.2f}"
        f"·{water_per_fu:.1f}H2O"
    )
    return CompositionDescriptors(
        na_per_fu=na_per_fu,
        fe_per_metal=float(fe_per_metal),
        vacancy_fraction=vacancy,
        water_per_fu=water_per_fu,
        formula=formula,
        k_per_fu=k_per_fu,
        cn_deficit_per_fu=cn_deficit,
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
        metal, comp.na_per_fu, comp.vacancy_fraction, comp.water_per_fu, comp.k_per_fu
    )
    if not math.isfinite(domain_size_nm) or domain_size_nm <= 0:
        return float("nan")
    kinetic = float(np.clip(1.06 / (1.0 + (domain_size_nm / 95.0) ** 1.7), 0.30, 1.0))
    defect = float(np.clip(1.0 - 1.15 * comp.vacancy_fraction, 0.15, 1.0))
    hydration = float(np.clip(1.0 - 0.055 * comp.water_per_fu, 0.55, 1.0))
    return float(q_theo * kinetic * defect * hydration * max(crystallinity, 1e-3) ** 0.35)


#: Assumed ratio of integrated absorptivities eps(Fe(III)-CN) / eps(Fe(II)-CN).
#: The Fe(III) band is several times weaker; the exact ratio depends on the
#: framework and is the main calibration uncertainty of the IR Fe(II) estimate.
IR_ABSORPTIVITY_RATIO = 0.25
_FE2_WINDOW = (2040.0, 2125.0)
_FE3_WINDOW = (2125.0, 2210.0)


def analyze_ir(ir: IRSpectrum, absorptivity_ratio: float = IR_ABSORPTIVITY_RATIO
               ) -> IRDescriptors:
    """Fe(II) share of the hexacyanoferrate from the cyanide-stretch bands.

    Fits a linear baseline plus one Gaussian in each window (Fe(II)-CN-M near
    2070-2100 cm-1, Fe(III)-CN-M near 2150-2175 cm-1), then
    f = A2 / (A2 + A3 / ratio).  Bulk-sensitive (ATR samples microns), so it can
    differ from XPS; with the default ratio it is a relative measure.
    """
    wn = np.asarray(ir.wavenumber_cm1, float)
    a = np.asarray(ir.absorbance, float)

    def model(p):
        b0, b1, A2, nu2, w2, A3, nu3, w3 = p
        g = lambda A, nu, w: A * np.exp(-0.5 * ((wn - nu) / w) ** 2) / (w * math.sqrt(2 * math.pi))  # noqa: E731
        return b0 + b1 * (wn - wn[0]) + g(A2, nu2, w2) + g(A3, nu3, w3)

    i2 = (wn > _FE2_WINDOW[0]) & (wn < _FE2_WINDOW[1])
    i3 = (wn > _FE3_WINDOW[0]) & (wn < _FE3_WINDOW[1])
    if not i2.any() or not i3.any():
        return IRDescriptors(fe2_fraction=float("nan"), nu_fe2_cm1=float("nan"),
                             nu_fe3_cm1=float("nan"), fit_residual=float("inf"))
    base = float(np.median(np.r_[a[:20], a[-20:]]))
    nu2_0 = float(wn[i2][np.argmax(a[i2])]); nu3_0 = float(wn[i3][np.argmax(a[i3])])
    p0 = [base, 0.0, max(np.ptp(a[i2]), 1e-3) * 40, nu2_0, 20.0,
          max(np.ptp(a[i3]), 1e-3) * 40, nu3_0, 25.0]
    lo = [-np.inf, -np.inf, 0.0, _FE2_WINDOW[0], 6.0, 0.0, _FE3_WINDOW[0], 6.0]
    hi = [np.inf, np.inf, np.inf, _FE2_WINDOW[1], 60.0, np.inf, _FE3_WINDOW[1], 60.0]
    r = least_squares(lambda p: model(p) - a, p0, bounds=(lo, hi))
    _b0, _b1, A2, nu2, _w2, A3, nu3, _w3 = r.x
    n2, n3 = A2, A3 / absorptivity_ratio
    f2 = float(n2 / (n2 + n3)) if n2 + n3 > 0 else float("nan")
    return IRDescriptors(fe2_fraction=f2, nu_fe2_cm1=float(nu2), nu_fe3_cm1=float(nu3),
                         fit_residual=float(np.sqrt(np.mean(r.fun ** 2))))


def charge_balance_residual(comp: CompositionDescriptors, fe2_fraction: float) -> float:
    """A cations measured minus those the framework charge requires.

    Required: ``(1 - y)(3 + f) - 2`` (see :func:`~pba_autoworkflow.schema.charge_balanced_a`).
    A clearly negative value means Na + K cannot be all that balances the
    charge: decyanation, cations the assay does not cover (NH4+, H3O+), or an
    over-estimated Fe(II) share.  NaN when an input is missing.
    """
    if not (math.isfinite(fe2_fraction) and math.isfinite(comp.vacancy_fraction)):
        return float("nan")
    return float(comp.a_per_fu - charge_balanced_a(comp.vacancy_fraction, fe2_fraction))
