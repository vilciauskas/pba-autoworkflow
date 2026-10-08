# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic instrument traces generated from the latent state.

These functions are the only bridge between :mod:`pba_autoworkflow.sim.ground_truth` and
the rest of the package.  They emit the same objects a real driver would return
(:class:`~pba_autoworkflow.schema.XRDPattern` and
:class:`~pba_autoworkflow.schema.ICPResult`) so that the analysis layer can be developed,
unit-tested and later pointed at real hardware without changing a line.

The XRD generator is a forward model: it places the allowed reflections of the
face-centred cubic PBA framework at Bragg positions computed from the latent
lattice constant, broadens them by Scherrer plus a fixed instrumental term, adds
a rhombohedral/monoclinic splitting when the latent phase calls for it, adds the
lines of crystalline secondary phases (NaCl residue, metal hydroxide or CuO at
high pH), and lays the whole thing on a decaying amorphous background with
Poisson counting noise.
Extracting the lattice constant back out of that pattern is a genuine fitting
problem.
"""

from __future__ import annotations

import math

import numpy as np

from ..schema import ICPResult, SynthesisParameters, XRDPattern
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


_SQRT_2PI = math.sqrt(2.0 * math.pi)
_IMPURITY_DOMAIN_NM = 45.0
_REFERENCE_DOMAIN_NM = 40.0
_HALO_SIGMA_DEG = 5.5


def _hexagonal_d(a: float, c: float, h: int, k: int, l: int) -> float:
    return 1.0 / math.sqrt(4.0 / 3.0 * (h * h + h * k + k * k) / a ** 2 + l * l / c ** 2)


def _brucite_lines(a: float, c: float) -> tuple[tuple[float, float], ...]:
    """(d, relative intensity) of the main brucite-type M(OH)2 reflections."""
    hkl_rel = (((0, 0, 1), 1.0), ((1, 0, 0), 0.35), ((1, 0, 1), 0.9),
               ((1, 0, 2), 0.25), ((1, 1, 0), 0.2))
    return tuple((_hexagonal_d(a, c, *hkl), r) for hkl, r in hkl_rel)


#: Rock-salt NaCl, a = 5.640 A: (111), (200), (220), (311), (222).
_NACL_LINES: tuple[tuple[float, float], ...] = tuple(
    (5.640 / math.sqrt(m), r) for m, r in ((3, 0.13), (4, 1.0), (8, 0.55), (11, 0.02), (12, 0.15))
)

#: Secondary phase formed at high pH, per metal.  Approximate lattice
#: parameters / d-spacings; Cu is represented by tenorite CuO.
_HYDROXIDE_LINES: dict[str, tuple[tuple[float, float], ...]] = {
    "Mn": _brucite_lines(3.322, 4.734),
    "Fe": _brucite_lines(3.258, 4.605),
    "Co": _brucite_lines(3.183, 4.652),
    "Ni": _brucite_lines(3.126, 4.605),
    "Cu": ((2.523, 1.0), (2.323, 0.96), (1.866, 0.25), (1.581, 0.14), (1.505, 0.14)),
}


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
    ref_area = 0.0   # integrated Bragg intensity of a fully ordered PBA pattern

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
        # Lorentz-polarization and thermal fall-off with angle.  Broadening
        # conserves integrated intensity: ``peak_counts`` is the height a
        # reflection would have at the reference domain size.
        sigma_ref = math.hypot(_scherrer_fwhm_deg(_REFERENCE_DOMAIN_NM, centre, wavelength_A),
                               _INSTRUMENT_FWHM_DEG) / 2.3548
        amp_full = peak_counts * rel / (1.0 + (centre / 55.0) ** 2) * sigma_ref / sigma
        ref_area += amp_full * sigma * _SQRT_2PI
        amp = amp_full * order

        if latent.phase in ("rhombohedral", "monoclinic") and (h + k + l) % 4 != 0:
            # Distortion splits the affected reflections.
            split = 0.16 if latent.phase == "rhombohedral" else 0.31
            for offset, weight in ((-split / 2, 0.55), (split / 2, 0.45)):
                signal += amp * weight * np.exp(
                    -0.5 * ((tt - (centre + offset)) / sigma) ** 2
                )
        else:
            signal += amp * np.exp(-0.5 * ((tt - centre) / sigma) ** 2)

    # Secondary crystalline phases.  Each gets ``fraction * ref_area`` of
    # integrated intensity, so the PBA share of all Bragg intensity is
    # order / (order + sum of fractions) -- what an intensity-based phase purity
    # should recover.
    impurities = ((_NACL_LINES, latent.nacl_fraction),
                  (_HYDROXIDE_LINES.get(latent.metal, ()), latent.hydroxide_fraction))
    for lines, fraction in impurities:
        visible = [(c, r) for c, r in ((_bragg_two_theta(d, wavelength_A), r) for d, r in lines)
                   if c is not None and tt[0] < c < tt[-1]]
        if fraction <= 0 or not visible:
            continue
        total_rel = sum(r for _, r in visible)
        for centre, rel in visible:
            fwhm = math.hypot(_scherrer_fwhm_deg(_IMPURITY_DOMAIN_NM, centre, wavelength_A),
                              _INSTRUMENT_FWHM_DEG)
            sigma = fwhm / 2.3548
            area = ref_area * fraction * rel / total_rel
            signal += area / (sigma * _SQRT_2PI) * np.exp(-0.5 * ((tt - centre) / sigma) ** 2)

    # Amorphous hump + air-scatter background.
    hump_centre = _bragg_two_theta(a / 2.0, wavelength_A) or 25.0
    # The halo carries the scattering the disordered fraction does not put into
    # Bragg peaks, so the PBA intensity crystallinity equals ``order``.
    signal += ref_area * (1.0 - order) / (_HALO_SIGMA_DEG * _SQRT_2PI) * np.exp(
        -0.5 * ((tt - hump_centre) / _HALO_SIGMA_DEG) ** 2
    )
    background = 120.0 + 2200.0 * np.exp(-(tt - tt[0]) / 9.0)
    counts = rng.poisson(np.clip(signal + background, 1.0, None)).astype(float)
    return XRDPattern(two_theta_deg=tt, intensity=counts,
                      wavelength_A=wavelength_A, exposure_s=exposure_s)


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
