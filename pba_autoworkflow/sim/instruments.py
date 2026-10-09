# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic instrument traces generated from the latent state.

These functions are the only bridge between :mod:`pba_autoworkflow.sim.ground_truth` and
the rest of the package.  They emit the same objects a real driver would return
(:class:`~pba_autoworkflow.schema.XRDPattern` and
:class:`~pba_autoworkflow.schema.ICPResult`) so that the analysis layer can be developed,
unit-tested and later pointed at real hardware without changing a line.

The XRD generator is a forward model over the latent phase assemblage: every
crystalline phase (framework polymorphs and secondary phases) contributes the
reflections of its reference structure from the phase library at its latent
lattice, broadened by Scherrer plus a fixed instrumental term, with a random
per-reflection texture factor; the disordered framework share adds a broad
halo, and the whole thing sits on a decaying background with Poisson counting
noise.
Extracting the lattice constant back out of that pattern is a genuine fitting
problem.
"""

from __future__ import annotations

import math

import numpy as np

from ..schema import ICPResult, IRSpectrum, SynthesisParameters, XRDPattern
from .ground_truth import LatentState

_INSTRUMENT_FWHM_DEG = 0.085  # Caglioti-like constant term of the diffractometer
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_IMPURITY_DOMAIN_NM = 45.0
_REFERENCE_DOMAIN_NM = 40.0
_HALO_SIGMA_DEG = 5.5
#: coherence length of the short-range-ordered (diffuse) framework component
_DIFFUSE_DOMAIN_NM = 2.0
#: Per-reflection log-normal intensity scatter (texture, counting statistics of
#: the powder): keeps the forward model from being an exact copy of the
#: analysis templates.
_TEXTURE_SIGMA = 0.08


def _bragg_two_theta(d_A: float, wavelength_A: float) -> float | None:
    ratio = wavelength_A / (2.0 * d_A)
    if ratio >= 1.0:
        return None
    return 2.0 * math.degrees(math.asin(ratio))


def _scherrer_fwhm_deg(domain_nm, two_theta_deg, wavelength_A: float, K: float = 0.9):
    theta = np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)
    beta_rad = K * (wavelength_A * 0.1) / (domain_nm * np.maximum(np.cos(theta), 1e-3))
    return np.degrees(beta_rad)


def _phase_scale(latent: LatentState, ref) -> tuple[float, ...]:
    if ref.key in latent.lattice_scale:
        return tuple(latent.lattice_scale[ref.key])
    if ref.phase_id == "pba_fm3m":
        return (latent.lattice_a_A / ref.lattice[0],)
    return (1.0, 1.0) if ref.strain == "ac" else (1.0,)


def simulate_xrd(
    latent: LatentState,
    rng: np.random.Generator,
    two_theta_range: tuple[float, float] = (10.0, 60.0),
    step_deg: float = 0.02,
    wavelength_A: float = 1.5406,
    exposure_s: float = 120.0,
    peak_counts: float = 9000.0,
) -> XRDPattern:
    """Forward-model a powder pattern from the latent phase assemblage.

    Every crystalline phase contributes its library reflections (``|F|^2`` times
    Lorentz-polarization) with a Hill-Howard scale ``S_p ~ w_p / (Z M V)_p``, so
    the weight fractions of :meth:`LatentState.xrd_weight_fractions` are what a
    perfect quantification would recover.  The disordered share of the framework
    scatters into a broad halo near the strongest low-angle framework line.
    Intensities are normalised so that a fully ordered, single-phase Fm-3m
    framework of the same metal has its strongest line at ``peak_counts``.
    """
    from ..analysis.phases import load_library

    lib = load_library()
    tt = np.arange(two_theta_range[0], two_theta_range[1] + step_deg, step_deg)
    ext = (two_theta_range[0] - 1.0, two_theta_range[1] + 1.0)

    ref0 = lib[f"pba_fm3m/{latent.metal}"]
    tt0, I0 = ref0.lines((1.0,), wavelength_A, ext)
    i0 = int(np.argmax(I0))
    fwhm0 = math.hypot(float(_scherrer_fwhm_deg(_REFERENCE_DOMAIN_NM, tt0[i0], wavelength_A)),
                       _INSTRUMENT_FWHM_DEG)
    K = peak_counts * (fwhm0 / 2.3548) * _SQRT_2PI / (I0[i0] / (ref0.cell_mass_amu * ref0.cell_volume_A3))

    signal = np.zeros_like(tt)
    halo_area = 0.0
    framework_mass = max(1.0 - latent.nacl_fraction - latent.hydroxide_fraction, 0.0)
    dominant = max(latent.polymorph_shares, key=latent.polymorph_shares.get,
                   default=f"pba_fm3m/{latent.metal}")
    hump_centre = 25.0

    def add_phase(ref, mass: float, domain_nm: float, texture: bool = True) -> float:
        scale = _phase_scale(latent, ref)
        ltt, lI = ref.lines(scale, wavelength_A, ext)
        if ltt.size == 0:
            return 0.0
        c = K * mass / (ref.cell_mass_amu * ref.volume(scale))
        area = c * lI * (np.exp(rng.normal(0.0, _TEXTURE_SIGMA, size=lI.size)) if texture else 1.0)
        sigma = np.hypot(_scherrer_fwhm_deg(domain_nm, ltt, wavelength_A), _INSTRUMENT_FWHM_DEG) / 2.3548
        nonlocal signal
        signal = signal + (np.exp(-0.5 * ((tt[:, None] - ltt[None, :]) / sigma[None, :]) ** 2)
                           / (sigma[None, :] * _SQRT_2PI)) @ area
        return float(c * lI.sum())

    for key, share in latent.polymorph_shares.items():
        ref = lib[key]
        mass = framework_mass * share
        dshare = latent.diffuse_share.get(key, 0.0)
        ordered = mass * latent.crystallinity
        full = add_phase(ref, ordered * (1.0 - dshare), latent.domain_size_nm)
        if dshare > 0:
            # Short-range-ordered framework: the same reflections, broadened to
            # a ~2 nm coherence length (diffuse scattering around Bragg positions).
            full += add_phase(ref, ordered * dshare, _DIFFUSE_DOMAIN_NM, texture=False)
        if latent.crystallinity > 0:
            halo_area += full / latent.crystallinity * (1.0 - latent.crystallinity)
        if key == dominant:
            ltt, lI = ref.lines(_phase_scale(latent, ref), wavelength_A, (10.0, 30.0))
            if ltt.size:
                hump_centre = float(ltt[int(np.argmax(lI))])
    from .ground_truth import HIGH_PH_PHASE
    for key, mass in (("nacl", latent.nacl_fraction),
                      (HIGH_PH_PHASE.get(latent.metal, ""), latent.hydroxide_fraction)):
        if mass > 0 and key in lib:
            add_phase(lib[key], mass, _IMPURITY_DOMAIN_NM)

    signal += halo_area / (_HALO_SIGMA_DEG * _SQRT_2PI) * np.exp(
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
    elements=None,
) -> ICPResult:
    """Elemental assay of an aliquot of the washed, dried powder.

    Reported concentrations carry a 2 % relative calibration error, and Na is
    additionally biased high by residual surface salt that survived washing --
    a real and well-known artefact when quantifying A-site occupancy.
    """
    from ..schema import ATOMIC_WEIGHT, formula_weight

    mass_mg = min(target_digest_mass_mg, max(latent.solid_mass_mg * 0.6, 0.2))
    fw = formula_weight(params.metal, latent.na_per_fu, latent.vacancy_fraction,
                        latent.water_per_fu, latent.k_per_fu)
    n_fu_mol = (mass_mg * 1e-3) / fw

    surface_na = 1.0 + max(0.0, rng.normal(0.035, 0.02))  # unwashed NaCl
    surface_k = 1.0 + max(0.0, rng.normal(0.035, 0.02))   # unwashed K salt
    moles = {
        "Na": n_fu_mol * latent.na_per_fu * surface_na,
        "K": n_fu_mol * latent.k_per_fu * surface_k,
        params.metal: n_fu_mol,
        "Fe": n_fu_mol * (1.0 - latent.vacancy_fraction),
    }
    if params.metal == "Fe":  # N-site and C-site iron are indistinguishable
        moles = {
            "Na": moles["Na"],
            "K": moles["K"],
            "Fe": n_fu_mol * (2.0 - latent.vacancy_fraction),
        }
    if elements is not None:
        moles = {el: m for el, m in moles.items() if el in elements}

    conc = {}
    for el, mol in moles.items():
        c = mol / (digest_volume_mL * 1e-3)
        conc[el] = float(max(0.0, c * (1.0 + rng.normal(0.0, 0.02))))

    # CHN combustion on the same aliquot.  Six carbons per intact hexacyanoferrate,
    # so carbon measures the C-site sublattice independently of the metals -- the
    # only way to resolve vacancies in the Fe analogue, where ICP sees one iron
    # pool.  1 % relative, typical for a well-run combustion analyzer.
    # Reductive decyanation during drying removes one CN per reduced Fe.
    c_moles = n_fu_mol * (6.0 * (1.0 - latent.vacancy_fraction) - latent.cn_loss_per_fu)
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
                        latent.water_per_fu, latent.k_per_fu)
    theo_mg = limiting_mol * fw * 1e3
    weighed = latent.solid_mass_mg + rng.normal(0.0, 0.1)
    return float(max(0.0, weighed)), float(theo_mg)


#: Cyanide-stretch band positions (cm-1) of Fe(II)-CN-M and Fe(III)-CN-M.
_NU_FE2 = {"Mn": 2072.0, "Fe": 2085.0, "Co": 2090.0, "Ni": 2096.0, "Cu": 2098.0, "Zn": 2097.0}
_NU_FE3 = {"Mn": 2150.0, "Fe": 2160.0, "Co": 2165.0, "Ni": 2168.0, "Cu": 2170.0, "Zn": 2172.0}


def simulate_ir(latent: LatentState, rng: np.random.Generator, absorptivity_ratio: float = 0.25,
                wn_range: tuple[float, float] = (1950.0, 2300.0), step_cm1: float = 1.0
                ) -> IRSpectrum:
    """ATR-IR absorbance of the cyanide stretch.

    Two Gaussian bands, Fe(II)-CN-M and Fe(III)-CN-M, with integrated areas
    proportional to the bulk Fe(II)/Fe(III) amounts times their absorptivities
    (``absorptivity_ratio`` = eps(Fe(III)) / eps(Fe(II)); the Fe(III) band is the
    weaker one).  Bands broaden with vacancies; baseline drift and noise added.
    """
    wn = np.arange(wn_range[0], wn_range[1] + step_cm1, step_cm1)
    m = latent.metal or "Mn"
    f2 = float(np.clip(latent.fe2_fraction, 0.0, 1.0))
    w2 = 16.0 + 30.0 * latent.vacancy_fraction
    w3 = 20.0 + 30.0 * latent.vacancy_fraction
    nu2 = _NU_FE2.get(m, 2085.0) + rng.normal(0.0, 1.5)
    nu3 = _NU_FE3.get(m, 2165.0) + rng.normal(0.0, 1.5)
    amp = 0.6 * (1.0 - latent.vacancy_fraction)
    g = lambda nu, w: np.exp(-0.5 * ((wn - nu) / w) ** 2) / (w * _SQRT_2PI)  # noqa: E731
    a = amp * 40.0 * (f2 * g(nu2, w2) + (1.0 - f2) * absorptivity_ratio * g(nu3, w3))
    base = 0.02 + 0.00004 * (wn - wn[0]) + rng.normal(0.0, 0.004)
    a = a + base + rng.normal(0.0, 0.0025, size=wn.size)
    return IRSpectrum(wavenumber_cm1=wn, absorbance=a)
