# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic instrument traces generated from the latent state.

These functions are the only bridge between :mod:`pba_autoworkflow.sim.ground_truth` and
the rest of the package.  They emit the same objects a real driver would return
(:class:`~pba_autoworkflow.schema.XRDPattern`, :class:`~pba_autoworkflow.schema.UVVisSpectrum`,
:class:`~pba_autoworkflow.schema.ICPResult`) so that the analysis layer can be developed,
unit-tested and later pointed at real hardware without changing a line.

The XRD generator is a forward model: it places the allowed reflections of the
face-centred cubic PBA framework at Bragg positions computed from the latent
lattice constant, broadens them by Scherrer plus a fixed instrumental term, adds
a rhombohedral/monoclinic splitting when the latent phase calls for it, and lays
the whole thing on a decaying amorphous background with Poisson counting noise.
Extracting the lattice constant back out of that pattern is a genuine fitting
problem.
"""

from __future__ import annotations

import math

import numpy as np

from ..schema import ICPResult, SynthesisParameters, UVVisSpectrum, XRDPattern
from .ground_truth import LatentState

#: Allowed reflections of the Fm-3m PBA framework with relative structure
#: factors typical of a sodium-rich hexacyanoferrate.
_HKL: tuple[tuple[tuple[int, int, int], float], ...] = (
    ((2, 0, 0), 1.00),
    ((2, 2, 0), 0.62),
    ((4, 0, 0), 0.28),
    ((4, 2, 0), 0.44),
    ((4, 2, 2), 0.30),
    ((4, 4, 0), 0.18),
    ((6, 0, 0), 0.12),
    ((6, 2, 0), 0.16),
    ((6, 4, 2), 0.10),
)

_INSTRUMENT_FWHM_DEG = 0.085  # Caglioti-like constant term of the diffractometer


def _bragg_two_theta(d_A: float, wavelength_A: float) -> float | None:
    ratio = wavelength_A / (2.0 * d_A)
    if ratio >= 1.0:
        return None
    return 2.0 * math.degrees(math.asin(ratio))


def _scherrer_fwhm_deg(domain_nm: float, two_theta_deg: float,
                       wavelength_A: float, K: float = 0.9) -> float:
    theta = math.radians(two_theta_deg / 2.0)
    beta_rad = K * (wavelength_A * 0.1) / (domain_nm * max(math.cos(theta), 1e-3))
    return math.degrees(beta_rad)


def simulate_xrd(
    latent: LatentState,
    rng: np.random.Generator,
    two_theta_range: tuple[float, float] = (10.0, 60.0),
    step_deg: float = 0.02,
    wavelength_A: float = 1.5406,
    exposure_s: float = 120.0,
    peak_counts: float = 9000.0,
) -> XRDPattern:
    """Forward-model a powder pattern from the latent structural state."""
    tt = np.arange(two_theta_range[0], two_theta_range[1] + step_deg, step_deg)
    signal = np.zeros_like(tt)

    a = latent.lattice_a_A
    # Amorphous / poorly-ordered fraction shows up as a broad hump near the
    # strongest framework correlation.
    order = latent.crystallinity

    for (h, k, l), rel in _HKL:
        d = a / math.sqrt(h * h + k * k + l * l)
        centre = _bragg_two_theta(d, wavelength_A)
        if centre is None or not (tt[0] < centre < tt[-1]):
            continue
        fwhm = math.hypot(
            _scherrer_fwhm_deg(latent.domain_size_nm, centre, wavelength_A),
            _INSTRUMENT_FWHM_DEG,
        )
        sigma = fwhm / 2.3548
        amp = peak_counts * rel * order
        # Lorentz-polarization and thermal fall-off with angle.
        amp *= 1.0 / (1.0 + (centre / 55.0) ** 2)

        if latent.phase in ("rhombohedral", "monoclinic") and (h + k + l) % 4 != 0:
            # Distortion splits the affected reflections.
            split = 0.16 if latent.phase == "rhombohedral" else 0.31
            for offset, weight in ((-split / 2, 0.55), (split / 2, 0.45)):
                signal += amp * weight * np.exp(
                    -0.5 * ((tt - (centre + offset)) / sigma) ** 2
                )
        else:
            signal += amp * np.exp(-0.5 * ((tt - centre) / sigma) ** 2)

    # Amorphous hump + air-scatter background.
    hump_centre = _bragg_two_theta(a / 2.0, wavelength_A) or 25.0
    signal += peak_counts * 0.55 * (1.0 - order) * np.exp(
        -0.5 * ((tt - hump_centre) / 5.5) ** 2
    )
    background = 120.0 + 2200.0 * np.exp(-(tt - tt[0]) / 9.0)
    counts = rng.poisson(np.clip(signal + background, 1.0, None)).astype(float)
    return XRDPattern(two_theta_deg=tt, intensity=counts,
                      wavelength_A=wavelength_A, exposure_s=exposure_s)


def simulate_uvvis(
    latent: LatentState,
    params: SynthesisParameters,
    rng: np.random.Generator,
    wavelength_range: tuple[float, float] = (300.0, 800.0),
    step_nm: float = 1.0,
    dilution_factor: float = 10.0,
) -> UVVisSpectrum:
    """Absorbance of the diluted supernatant.

    Two chromophores matter: unreacted [Fe(CN)6]4- (a band near 320 nm with a
    shoulder at 420 nm from partially oxidized species) and colloidal PBA that
    escaped the pellet, which contributes the broad Fe(2+)-Fe(3+) intervalence
    charge-transfer band near 690 nm plus Rayleigh turbidity.
    """
    wl = np.arange(wavelength_range[0], wavelength_range[1] + step_nm, step_nm)
    c_hcf = latent.residual_hcf_M / dilution_factor

    # Molar extinction coefficients (L mol-1 cm-1), ferrocyanide.
    a = 1.0e4 * c_hcf * np.exp(-0.5 * ((wl - 322.0) / 26.0) ** 2)
    a += 3.1e2 * c_hcf * np.exp(-0.5 * ((wl - 420.0) / 48.0) ** 2)

    # Colloidal carry-over scales with how poorly the solid settles.
    colloid = 0.055 * math.exp(-latent.domain_size_nm / 14.0) * latent.conversion
    ivct_centre = 690.0 + 22.0 * (latent.na_per_fu / 2.0) - 30.0 * latent.vacancy_fraction
    a += colloid * 9.0 * np.exp(-0.5 * ((wl - ivct_centre) / 95.0) ** 2)
    a += colloid * 6.0 * (450.0 / wl) ** 4  # turbidity

    a += 0.004 + rng.normal(0.0, 0.0016, size=wl.size)  # baseline + detector noise
    a = np.clip(a, 0.0, 4.0)  # detector saturation
    return UVVisSpectrum(wavelength_nm=wl, absorbance=a,
                         path_length_cm=1.0, dilution_factor=dilution_factor)


def simulate_icp(
    latent: LatentState,
    params: SynthesisParameters,
    rng: np.random.Generator,
    digest_volume_mL: float = 25.0,
    target_digest_mass_mg: float = 5.0,
) -> ICPResult:
    """Elemental assay of an aliquot of the washed, dried powder.

    Reported concentrations carry a 2 % relative calibration error, and Na is
    additionally biased high by residual surface salt that survived washing --
    a real and well-known artefact when quantifying A-site occupancy.
    """
    from ..schema import ATOMIC_WEIGHT, formula_weight

    mass_mg = min(target_digest_mass_mg, max(latent.solid_mass_mg * 0.6, 0.2))
    fw = formula_weight(params.metal, latent.na_per_fu, latent.vacancy_fraction,
                        latent.water_per_fu)
    n_fu_mol = (mass_mg * 1e-3) / fw

    surface_na = 1.0 + max(0.0, rng.normal(0.035, 0.02))  # unwashed NaCl
    moles = {
        "Na": n_fu_mol * latent.na_per_fu * surface_na,
        params.metal: n_fu_mol,
        "Fe": n_fu_mol * (1.0 - latent.vacancy_fraction),
    }
    if params.metal == "Fe":  # N-site and C-site iron are indistinguishable
        moles = {
            "Na": moles["Na"],
            "Fe": n_fu_mol * (2.0 - latent.vacancy_fraction),
        }

    conc = {}
    for el, mol in moles.items():
        c = mol / (digest_volume_mL * 1e-3)
        conc[el] = float(max(0.0, c * (1.0 + rng.normal(0.0, 0.02))))

    # CHN combustion on the same aliquot.  Six carbons per intact hexacyanoferrate,
    # so carbon measures the C-site sublattice independently of the metals -- the
    # only way to resolve vacancies in the Fe analogue, where ICP sees one iron
    # pool.  1 % relative, typical for a well-run combustion analyzer.
    c_moles = n_fu_mol * 6.0 * (1.0 - latent.vacancy_fraction)
    carbon_wt_pct = 100.0 * c_moles * ATOMIC_WEIGHT["C"] / (mass_mg * 1e-3)
    carbon_wt_pct *= 1.0 + rng.normal(0.0, 0.01)

    return ICPResult(
        concentrations_mol_L=conc,
        digest_mass_mg=float(mass_mg * (1.0 + rng.normal(0.0, 0.01))),
        digest_volume_mL=digest_volume_mL,
        dry_mass_mg=float(latent.solid_mass_mg),
        carbon_wt_pct=float(max(0.0, carbon_wt_pct)),
    )


def simulate_gravimetric_yield(
    latent: LatentState, params: SynthesisParameters, rng: np.random.Generator
) -> tuple[float, float]:
    """Return (weighed dry mass in mg, theoretical mass in mg).

    The balance is good to 0.1 mg; the theoretical mass is computed from the
    limiting reagent and the *measured* formula, so an isolated yield derived
    from this pair inherits the composition error -- as it does in practice.
    """
    from ..schema import formula_weight

    limiting_mol = min(
        params.c_metal_M * params.volume_A_mL, params.c_hcf_M * params.volume_B_mL
    ) * 1e-3
    fw = formula_weight(params.metal, latent.na_per_fu, latent.vacancy_fraction,
                        latent.water_per_fu)
    theo_mg = limiting_mol * fw * 1e3
    weighed = latent.solid_mass_mg + rng.normal(0.0, 0.1)
    return float(max(0.0, weighed)), float(theo_mg)
