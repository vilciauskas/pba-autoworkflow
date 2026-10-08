# SPDX-License-Identifier: GPL-3.0-or-later
"""Automated powder-pattern reduction.

Given a raw diffractogram this module does what an operator would do at the
instrument, without an operator:

1. Strip the background with an iterative morphological "rolling ball" filter
   (SNIP-style), which handles the steep low-angle air-scatter tail without
   biasing peak areas.
2. Find candidate peaks by prominence on the stripped pattern.
3. Fit each candidate with a pseudo-Voigt profile to get position and FWHM at
   sub-step precision.
4. Index the peaks against the face-centred-cubic PBA framework by least squares
   on ``sin^2(theta) = (lambda/2a)^2 (h^2+k^2+l^2)``, which gives the lattice
   constant from *all* indexed reflections rather than from one line.
5. Estimate the volume-weighted coherent domain size from the Scherrer equation
   after deconvoluting the instrumental broadening, using a Williamson-Hall style
   linear fit when enough reflections are available so that microstrain does not
   contaminate the size estimate.
6. Classify the phase from the presence of distortion-induced splitting and from
   the crystalline fraction.

Everything here operates on the array pair alone, so it works identically on
simulated and real patterns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks, savgol_filter

from ..schema import XRDDescriptors, XRDPattern

#: Instrumental FWHM (deg) used to deconvolute the observed broadening.  On a
#: real platform this comes from a LaB6 or Si standard measured in the same
#: geometry; here it matches the value the simulator convolutes in.
INSTRUMENT_FWHM_DEG = 0.085

def allowed_reflections(m_max: int = 56, parity: str = "even"
                        ) -> dict[int, str]:
    """Allowed reflections of the F-centred framework, as ``{h^2+k^2+l^2: "hkl"}``.

    Generated from the reflection condition rather than tabulated by hand -- a
    typed table is where an ``m`` silently acquires the wrong ``hkl`` label, and a
    mislabelled reflection in a report is indistinguishable from a mis-indexed
    pattern to whoever reads it.

    ``parity="even"`` restricts to the all-even reflections.  For a sodium-rich
    hexacyanoferrate these carry essentially all of the scattered intensity: the
    all-odd reflections (111), (311) depend on the difference between the two
    metal sublattice contributions and are weak in the analogues considered here,
    so including them among the indexing candidates adds near-degenerate
    alternatives that can only degrade the assignment.  Pass ``"all"`` if the
    framework under study does show them.
    """
    out: dict[int, str] = {}
    limit = int(math.isqrt(m_max)) + 1
    for h in range(limit + 1):
        for k in range(h + 1):
            for l in range(k + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                m = h * h + k * k + l * l
                if m > m_max:
                    continue
                parities = {h % 2, k % 2, l % 2}
                if len(parities) != 1:
                    continue  # mixed parity: extinct in an F lattice
                if parity == "even" and h % 2 != 0:
                    continue
                # Among permutations sharing an m, the conventional label is the
                # one with the largest leading index.
                label = f"{h}{k}{l}"
                if m not in out or label > out[m]:
                    out[m] = label
    return dict(sorted(out.items()))


#: Reflection labels and, ascending, the multiplicities used for indexing.
_HKL_LABELS: dict[int, str] = allowed_reflections()
_ALLOWED_M = np.array(sorted(_HKL_LABELS), dtype=float)


def snip_background(y: np.ndarray, max_window: int = 60, iterations: int = 24
                    ) -> np.ndarray:
    """SNIP background estimate on a log-log-transformed spectrum."""
    v = np.log(np.log(np.sqrt(np.clip(y, 0.0, None) + 1.0) + 1.0) + 1.0)
    n = v.size
    for p in np.linspace(1, max_window, iterations).astype(int):
        if p < 1 or 2 * p >= n:
            continue
        left = np.concatenate([v[:p], v[:-p]])[:n]
        right = np.concatenate([v[p:], v[-p:]])[:n]
        v = np.minimum(v, 0.5 * (left + right))
    return (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0


def pseudo_voigt(x: np.ndarray, amp: float, centre: float, fwhm: float,
                 eta: float) -> np.ndarray:
    """Normalized pseudo-Voigt: eta=0 pure Gaussian, eta=1 pure Lorentzian."""
    fwhm = max(fwhm, 1e-4)
    sigma = fwhm / 2.3548200450309493
    gauss = np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    gamma = fwhm / 2.0
    lorentz = gamma ** 2 / ((x - centre) ** 2 + gamma ** 2)
    return amp * ((1.0 - eta) * gauss + eta * lorentz)


@dataclass
class FittedPeak:
    two_theta_deg: float
    fwhm_deg: float
    amplitude: float
    eta: float
    area: float
    ok: bool = True


def _fit_peak(tt: np.ndarray, y: np.ndarray, idx: int, half_width_pts: int
              ) -> FittedPeak | None:
    lo = max(0, idx - half_width_pts)
    hi = min(tt.size, idx + half_width_pts + 1)
    if hi - lo < 7:
        return None
    x_w, y_w = tt[lo:hi], y[lo:hi]
    step = float(tt[1] - tt[0])
    p0 = [float(y[idx]), float(tt[idx]), max(4.0 * step, 0.12), 0.5]
    bounds = (
        [0.0, float(x_w[0]), 1.5 * step, 0.0],
        [10.0 * max(float(y[idx]), 1.0), float(x_w[-1]), float(x_w[-1] - x_w[0]), 1.0],
    )
    try:
        popt, _ = curve_fit(pseudo_voigt, x_w, y_w, p0=p0, bounds=bounds, maxfev=6000)
    except (RuntimeError, ValueError):
        return None
    amp, centre, fwhm, eta = (float(v) for v in popt)
    area = float(np.trapezoid(pseudo_voigt(x_w, amp, centre, fwhm, eta), x_w))
    return FittedPeak(centre, fwhm, amp, eta, area)


def amorphous_halo_area(pattern: XRDPattern, lattice_a_A: float | None = None) -> float:
    """Integrated intensity of the broad amorphous halo.

    SNIP (window ~1 deg) keeps the Bragg peaks but strips the halo (sigma ~5 deg)
    into its background, so the halo is recovered from that background curve by
    fitting ``c0 + c1 exp(-(2theta - 2theta_0)/L) + Gaussian``: a flat term for
    fluorescence/detector floor, an exponential for air and low-angle scatter,
    and the halo.  The halo is held near the (200) position of the refined
    lattice (+/- 2 deg; 14-24 deg when the pattern did not index) and to a width
    of 3-8 deg -- left free, a wide Gaussian at the low-angle edge imitates the
    air-scatter exponential and inflates the halo several-fold.  This is a
    profile model, so its numbers are only as good as that assumption for the
    instrument at hand.
    """
    if lattice_a_A is not None and math.isfinite(lattice_a_A):
        ratio = pattern.wavelength_A / lattice_a_A          # d(200) = a / 2
        c = 2.0 * math.degrees(math.asin(min(ratio, 0.999)))
        centre_bounds = (c - 2.0, c + 2.0)
    else:
        centre_bounds = (14.0, 24.0)
    tt = np.asarray(pattern.two_theta_deg, dtype=float)
    bg = snip_background(np.asarray(pattern.intensity, dtype=float))
    t0 = float(tt[0])

    def model(t, c0, c1, L, A, mu, s):
        return c0 + c1 * np.exp(-(t - t0) / L) + A * np.exp(-0.5 * ((t - mu) / s) ** 2)

    span = float(np.max(bg) - np.min(bg)) or 1.0
    p0 = (float(np.min(bg)), span, 8.0, 0.2 * span, float(np.mean(centre_bounds)), 5.0)
    lo = (0.0, 0.0, 1.0, 0.0, centre_bounds[0], 3.0)
    hi = (np.inf, np.inf, 40.0, np.inf, centre_bounds[1], 8.0)
    try:
        (_c0, _c1, _L, A, _mu, s), _ = curve_fit(model, tt, bg, p0=p0, bounds=(lo, hi), maxfev=20000)
    except (RuntimeError, ValueError):
        return 0.0
    return float(A * s * math.sqrt(2.0 * math.pi))


def find_and_fit_peaks(pattern: XRDPattern, min_prominence_frac: float = 0.035
                       ) -> tuple[list[FittedPeak], np.ndarray, np.ndarray]:
    """Background-strip, locate and profile-fit the reflections."""
    tt = np.asarray(pattern.two_theta_deg, dtype=float)
    y_raw = np.asarray(pattern.intensity, dtype=float)
    bg = snip_background(y_raw)
    y = y_raw - bg
    win = min(21, (y.size // 2) * 2 - 1)
    if win >= 5:
        y_s = savgol_filter(y, win, 3)
    else:
        y_s = y

    span = float(np.nanmax(y_s) - np.nanmin(y_s))
    # A reflection must rise above the counting noise of the background it sits
    # on, not merely above the dynamic range of a featureless trace.  Without this
    # absolute floor a flat pattern yields spurious sub-count "peaks" that the
    # indexer will happily fit a lattice to.
    noise_floor = 3.0 * float(np.sqrt(max(np.median(bg), 1.0)))
    if span <= noise_floor:
        return [], tt, y
    step = float(tt[1] - tt[0])
    idx, props = find_peaks(
        y_s,
        prominence=min_prominence_frac * span,
        distance=max(3, int(round(0.12 / step))),
        width=max(2, int(round(0.05 / step))),
    )
    peaks: list[FittedPeak] = []
    for i, width_pts in zip(idx, props.get("widths", np.full(idx.size, 6.0))):
        hw = int(max(4, min(2.5 * width_pts, 0.9 / step)))
        fp = _fit_peak(tt, y_s, int(i), hw)
        if fp is not None and fp.amplitude > max(0.02 * span, noise_floor):
            peaks.append(fp)
    peaks.sort(key=lambda p: p.two_theta_deg)
    return peaks, tt, y


def _sin2theta(two_theta_deg: np.ndarray | float) -> np.ndarray:
    return np.sin(np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)) ** 2


def index_cubic(peaks: list[FittedPeak], wavelength_A: float,
                a_bounds: tuple[float, float] = (9.0, 11.2)
                ) -> tuple[float, dict[int, FittedPeak], float]:
    """Assign FCC indices and refine the cubic lattice constant.

    A coarse scan over ``a`` scores each candidate by how well every observed
    peak lands on an allowed reflection; the best candidate is then refined by
    least squares on ``sin^2(theta)``, which is linear in ``1/a^2`` and therefore
    well conditioned.  Returns ``(a, {m: peak}, rms_residual_deg)``.
    """
    if not peaks:
        return float("nan"), {}, float("inf")
    obs = np.array([p.two_theta_deg for p in peaks])
    weights = np.array([max(p.area, 1e-9) for p in peaks])
    weights = weights / weights.sum()

    best = (np.inf, np.nan, {})
    for a in np.arange(a_bounds[0], a_bounds[1], 0.002):
        d = a / np.sqrt(_ALLOWED_M)
        ratio = wavelength_A / (2.0 * d)
        valid = ratio < 1.0
        calc = 2.0 * np.degrees(np.arcsin(np.clip(ratio, 0, 1 - 1e-12)))
        calc = calc[valid]
        m_valid = _ALLOWED_M[valid]
        if calc.size == 0:
            continue
        diff = np.abs(obs[:, None] - calc[None, :])
        j = np.argmin(diff, axis=1)
        resid = diff[np.arange(obs.size), j]
        # Truncated loss: a line that indexes to nothing (a secondary phase)
        # costs a fixed penalty instead of dragging ``a`` towards a wrong
        # solution that half-fits it.
        score = float(np.sum(weights * np.minimum(resid, 0.45) ** 2))
        # Require the strongest observed line to index within 0.4 deg.
        if resid[int(np.argmax([p.area for p in peaks]))] > 0.4:
            score += 10.0
        if score < best[0]:
            assign = {}
            for k, (pk, jj, r) in enumerate(zip(peaks, j, resid)):
                if r < 0.45:
                    m = int(m_valid[jj])
                    if m not in assign or pk.area > assign[m].area:
                        assign[m] = pk
            best = (score, float(a), assign)

    _, a_coarse, assign = best
    if len(assign) < 1 or not np.isfinite(a_coarse):
        return float("nan"), {}, float("inf")

    m_arr = np.array(sorted(assign), dtype=float)
    tt_arr = np.array([assign[int(m)].two_theta_deg for m in m_arr])
    w = np.array([assign[int(m)].area for m in m_arr])
    w = w / w.sum()

    def resid_fn(theta: np.ndarray) -> np.ndarray:
        a = theta[0]
        return np.sqrt(w) * (_sin2theta(tt_arr) - (wavelength_A / (2.0 * a)) ** 2 * m_arr)

    sol = least_squares(resid_fn, x0=[a_coarse], bounds=([8.5], [12.0]))
    a_ref = float(sol.x[0])

    calc_tt = 2.0 * np.degrees(
        np.arcsin(np.clip(wavelength_A * np.sqrt(m_arr) / (2.0 * a_ref), 0, 1 - 1e-12))
    )
    rms = float(np.sqrt(np.mean((tt_arr - calc_tt) ** 2)))
    return a_ref, assign, rms


def scherrer_domain_size_nm(peaks_by_m: dict[int, FittedPeak], wavelength_A: float,
                            K: float = 0.9) -> tuple[float, float]:
    """Volume-weighted domain size, and microstrain, by Williamson-Hall.

    Williamson-Hall plots ``beta*cos(theta)`` against ``4*sin(theta)``: the
    intercept gives ``K*lambda/D`` and the slope gives the strain.  With fewer
    than three indexed reflections the plot is under-determined, so we fall back
    to the plain Scherrer size of the strongest line.
    """
    items = [(m, p) for m, p in peaks_by_m.items() if p.fwhm_deg > INSTRUMENT_FWHM_DEG]
    if not items:
        # Broadening at or below the instrumental limit: report the resolution cap.
        strongest = max(peaks_by_m.values(), key=lambda p: p.area, default=None)
        if strongest is None:
            return float("nan"), 0.0
        theta = math.radians(strongest.two_theta_deg / 2.0)
        cap = K * (wavelength_A * 0.1) / (
            math.radians(INSTRUMENT_FWHM_DEG) * max(math.cos(theta), 1e-3)
        )
        return float(cap), 0.0

    theta = np.array([math.radians(p.two_theta_deg / 2.0) for _, p in items])
    beta = np.radians(
        np.sqrt(np.clip(
            np.array([p.fwhm_deg for _, p in items]) ** 2 - INSTRUMENT_FWHM_DEG ** 2,
            1e-8, None,
        ))
    )
    lam_nm = wavelength_A * 0.1

    if len(items) >= 3:
        x = 4.0 * np.sin(theta)
        y = beta * np.cos(theta)
        slope, intercept = np.polyfit(x, y, 1)
        if intercept > 1e-6:
            return float(K * lam_nm / intercept), float(max(slope, 0.0))

    m_max, p_max = max(items, key=lambda kv: kv[1].area)
    th = math.radians(p_max.two_theta_deg / 2.0)
    b = math.radians(
        math.sqrt(max(p_max.fwhm_deg ** 2 - INSTRUMENT_FWHM_DEG ** 2, 1e-8))
    )
    return float(K * lam_nm / (b * max(math.cos(th), 1e-3))), 0.0


def classify_phase(peaks: list[FittedPeak], assign: dict[int, FittedPeak],
                   crystallinity: float) -> str:
    """Cubic vs distorted vs amorphous, from splitting of the odd-parity lines."""
    if crystallinity < 0.18 or len(assign) < 2:
        return "amorphous"
    # Look for doublets: two fitted peaks close together straddling an indexed one.
    splits: list[float] = []
    for m, ref in assign.items():
        if m % 16 == 0:  # (400), (440) etc. are unsplit by the distortion
            continue
        near = [p for p in peaks if abs(p.two_theta_deg - ref.two_theta_deg) < 0.55
                and p is not ref]
        if near:
            partner = min(near, key=lambda p: abs(p.two_theta_deg - ref.two_theta_deg))
            splits.append(abs(partner.two_theta_deg - ref.two_theta_deg))
    # A distortion splits the affected reflections systematically; a secondary
    # phase line that happens to sit next to one or two PBA reflections does not.
    eligible = sum(1 for m in assign if m % 16 != 0)
    if len(splits) < 2 or len(splits) < 0.4 * eligible:
        return "cubic"
    mean_split = float(np.mean(splits))
    if mean_split > 0.24:
        return "monoclinic"
    return "rhombohedral"


#: A fitted peak belongs to the PBA if it lies this close to an allowed cubic
#: reflection at the refined lattice constant...
PBA_LINE_TOL_DEG = 0.45
#: ...or this close to an indexed reflection (a component of a rhombohedral or
#: monoclinic split; same window :func:`classify_phase` uses).
SPLIT_PARTNER_TOL_DEG = 0.55


def attribute_peaks(peaks: list[FittedPeak], a: float, assign: dict[int, FittedPeak],
                    wavelength_A: float) -> tuple[list[FittedPeak], list[FittedPeak]]:
    """Split fitted peaks into PBA reflections and unindexed (impurity) lines.

    An impurity line that happens to coincide with a PBA reflection is counted
    as PBA, so phase purity is an upper bound; secondary phases whose strong
    lines all overlap the framework pattern are not detectable this way.
    """
    ratio = wavelength_A / (2.0 * a / np.sqrt(_ALLOWED_M))
    calc = 2.0 * np.degrees(np.arcsin(ratio[ratio < 1.0]))
    indexed = [p.two_theta_deg for p in assign.values()]
    pba, impurity = [], []
    for p in peaks:
        near_line = calc.size and float(np.min(np.abs(calc - p.two_theta_deg))) < PBA_LINE_TOL_DEG
        near_indexed = any(abs(p.two_theta_deg - t) < SPLIT_PARTNER_TOL_DEG for t in indexed)
        (pba if near_line or near_indexed else impurity).append(p)
    return pba, impurity


def analyze_pattern(pattern: XRDPattern) -> XRDDescriptors:
    """Full reduction of one diffractogram to structural descriptors."""
    peaks, tt, y = find_and_fit_peaks(pattern)
    a, assign, rms = index_cubic(peaks, pattern.wavelength_A)
    halo = amorphous_halo_area(pattern, a if assign else None)
    if not assign:
        peak_area = float(sum(p.area for p in peaks))
        return XRDDescriptors(
            lattice_a_A=float("nan"), domain_size_nm=float("nan"),
            crystallinity_index=float(peak_area / (peak_area + halo)) if peak_area + halo > 0 else 0.0,
            fwhm_200_deg=float("nan"), phase="amorphous", n_peaks_indexed=0,
            fit_residual=float("inf"), phase_purity=0.0, n_impurity_peaks=len(peaks),
        )
    pba, impurity = attribute_peaks(peaks, a, assign, pattern.wavelength_A)
    pba_area = float(sum(p.area for p in pba))
    imp_area = float(sum(p.area for p in impurity))
    purity = pba_area / (pba_area + imp_area) if pba_area + imp_area > 0 else 0.0
    # Crystallinity of the PBA itself: its Bragg intensity against Bragg plus
    # amorphous halo.  Impurity lines are left out, so a crystalline secondary
    # phase cannot raise it.
    crystallinity = pba_area / (pba_area + halo) if pba_area + halo > 0 else 0.0
    d_nm, _strain = scherrer_domain_size_nm(assign, pattern.wavelength_A)
    fwhm_200 = assign[4].fwhm_deg if 4 in assign else min(
        assign.values(), key=lambda p: p.two_theta_deg
    ).fwhm_deg
    phase = classify_phase(peaks, assign, crystallinity)
    return XRDDescriptors(
        lattice_a_A=float(a),
        domain_size_nm=float(d_nm),
        crystallinity_index=crystallinity,
        fwhm_200_deg=float(fwhm_200),
        phase=phase,  # type: ignore[arg-type]
        n_peaks_indexed=len(assign),
        fit_residual=float(rms),
        phase_purity=float(np.clip(purity, 0.0, 1.0)),
        n_impurity_peaks=len(impurity),
    )


def indexed_reflection_table(pattern: XRDPattern) -> list[dict[str, float | str]]:
    """Per-reflection detail, for the QC report."""
    peaks, _, _ = find_and_fit_peaks(pattern)
    a, assign, _ = index_cubic(peaks, pattern.wavelength_A)
    rows: list[dict[str, float | str]] = []
    for m in sorted(assign):
        p = assign[m]
        d_obs = pattern.wavelength_A / (2.0 * math.sin(math.radians(p.two_theta_deg / 2)))
        rows.append({
            "hkl": _HKL_LABELS.get(m, str(m)),
            "two_theta_obs": round(p.two_theta_deg, 4),
            "d_obs_A": round(d_obs, 5),
            "d_calc_A": round(a / math.sqrt(m), 5),
            "fwhm_deg": round(p.fwhm_deg, 4),
            "area": round(p.area, 1),
        })
    return rows
