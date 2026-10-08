# SPDX-License-Identifier: GPL-3.0-or-later
"""Supernatant spectroscopy and elemental-assay reduction.

The UV-Vis routine quantifies unreacted hexacyanoferrate against the turbidity
and intervalence-charge-transfer background contributed by colloidal product.
Fitting all three contributions simultaneously matters: reading absorbance at
322 nm alone systematically over-reports residual ferrocyanide whenever the
supernatant is cloudy, which is exactly when the run is worst behaved.

The ICP routine converts digest concentrations into a per-formula-unit
composition, propagating the assay error into the reported Na content and
correcting for the residual surface sodium that survives washing.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import nnls

from ..schema import (
    ATOMIC_WEIGHT,
    CompositionDescriptors,
    ICPResult,
    SynthesisParameters,
    UVVisDescriptors,
    UVVisSpectrum,
    theoretical_capacity_mAh_g,
)

#: Molar extinction of Na4[Fe(CN)6] at its 322 nm maximum (L mol-1 cm-1).
EPS_HCF_322 = 1.0e4
_HCF_CENTRE_NM = 322.0
_HCF_SIGMA_NM = 26.0

#: Empirical surface-sodium correction determined from wash-cycle calibration.
SURFACE_NA_CORRECTION = 0.035


def _basis(wl: np.ndarray, metal: str) -> np.ndarray:
    """Design matrix: [ferrocyanide band, IVCT/d-d band, Rayleigh turbidity, baseline]."""
    hcf = np.exp(-0.5 * ((wl - _HCF_CENTRE_NM) / _HCF_SIGMA_NM) ** 2)
    metal_bands = {
        "Mn": (420.0, 60.0),
        "Fe": (690.0, 95.0),
        "Co": (530.0, 80.0),
        "Ni": (400.0, 70.0),
        "Cu": (480.0, 85.0),
    }
    center, sigma = metal_bands.get(metal, (690.0, 95.0))
    ivct = np.exp(-0.5 * ((wl - center) / sigma) ** 2)
    turbidity = (450.0 / wl) ** 4
    baseline = np.ones_like(wl)
    return np.column_stack([hcf, ivct, turbidity, baseline])


def analyze_uvvis(spectrum: UVVisSpectrum, params: SynthesisParameters
                  ) -> UVVisDescriptors:
    """Deconvolute the supernatant spectrum and compute conversion."""
    wl = np.asarray(spectrum.wavelength_nm, dtype=float)
    a = np.asarray(spectrum.absorbance, dtype=float)

    # Exclude saturated points; they carry no quantitative information.
    keep = a < 3.9
    A = _basis(wl[keep], params.metal)
    coeffs, _ = nnls(A, a[keep])
    c_hcf_band, c_ivct, _c_turb, _c_base = (float(v) for v in coeffs)

    c_cuvette = c_hcf_band / (EPS_HCF_322 * spectrum.path_length_cm)
    residual_hcf_M = c_cuvette * spectrum.dilution_factor

    # Conversion against the hexacyanoferrate actually charged, on a
    # total-volume basis (the supernatant is the whole reaction liquid).
    charged_M = params.c_hcf_M * params.volume_B_mL / params.total_volume_mL
    conversion = float(np.clip(1.0 - residual_hcf_M / max(charged_M, 1e-9), 0.0, 1.0))

    a_420 = float(np.interp(420.0, wl, a))
    lambda_max = None
    if c_ivct > 0.02:
        vis = (wl > 550.0) & (wl < 800.0)
        fitted_ivct = c_ivct * np.exp(-0.5 * ((wl[vis] - 690.0) / 95.0) ** 2)
        lambda_max = float(wl[vis][int(np.argmax(fitted_ivct))])

    return UVVisDescriptors(
        residual_hcf_M=float(max(residual_hcf_M, 0.0)),
        a_420=a_420,
        conversion=conversion,
        ivct_lambda_max_nm=lambda_max,
    )


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
