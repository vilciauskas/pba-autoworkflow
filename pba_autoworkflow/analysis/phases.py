# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-phase identification and quantification from a powder pattern.

The reference library (``data/phase_library.json``, built by
``scripts/build_phase_library.py`` from COD structures) holds, for every
candidate phase, its reflections as ``multiplicity * |F|^2`` per unit cell.
Quantification then follows the standard whole-pattern route, without a full
Rietveld refinement:

1. **Lattice.**  Each candidate's lattice is refined against the fitted peak
   positions within a window around its reference cell -- one scale for cubic
   and monoclinic phases, separate *a* and *c* scales for hexagonal/trigonal
   ones -- by a grid search on a truncated, intensity-weighted position misfit,
   followed by least squares on the matched lines.
2. **Profile.**  Each phase's pattern is synthesised at its refined lattice with
   ``|F|^2`` times the Lorentz-polarization factor and a pseudo-Voigt whose
   width follows the phase's own Scherrer size (from its matched peaks).
3. **Scale.**  The background-stripped pattern is fitted as a non-negative sum
   of the phase patterns plus one free profile per peak that no phase explains.
4. **Weight fractions** follow from the Hill-Howard relation
   ``w_p ~ S_p (Z M V)_p``, using the library's cell mass and the refined cell
   volume.  They are fractions of the *crystalline* material; the amorphous
   share is reported separately as crystallinity.

Limits that matter on real data: intensities come from fixed reference
structures (Na/water occupancy, preferred orientation and microabsorption are
not refined), so weight fractions are typically good to a few percent for
well-separated phases and worse where strong lines overlap.  A phase not in the
library shows up only as *unidentified* intensity.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, nnls

LIBRARY_PATH = Path(__file__).resolve().parents[1] / "data" / "phase_library.json"

#: Framework phase ids that can be a campaign's target phase.
FRAMEWORK_PHASES: tuple[str, ...] = ("pba_fm3m", "pba_p21n", "znhcf_r3c")

#: Candidate phases considered for a recipe with a given N-site metal.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "Mn": ("pba_fm3m/Mn", "pba_p21n/Mn", "nacl", "m_oh2/Mn"),
    "Fe": ("pba_fm3m/Fe", "pba_p21n/Fe", "nacl", "m_oh2/Fe"),
    "Co": ("pba_fm3m/Co", "nacl", "m_oh2/Co"),
    "Ni": ("pba_fm3m/Ni", "nacl", "m_oh2/Ni"),
    "Cu": ("pba_fm3m/Cu", "nacl", "cuo"),
    "Zn": ("pba_fm3m/Zn", "znhcf_r3c/Zn", "nacl", "zno"),
}

INSTRUMENT_FWHM_DEG = 0.085
#: A calculated line is matched to an observed peak within this distance.
MATCH_TOL_DEG = 0.12
#: An observed peak further than this from every line of the identified phases
#: is "unidentified".
UNEXPLAINED_TOL_DEG = 0.25
#: Phases explaining less than this share of the Bragg intensity are dropped.
MIN_INTENSITY_SHARE = 0.01
#: Smallest coherent domain a phase profile may refine to.  Lines broader than
#: this behave like a smooth background, so a free width lets a phase absorb
#: intensity it has no claim to (at low crystallinity this inflated the R-3c
#: fraction of a mixed Zn sample from 0.30 to 0.67).
MIN_DOMAIN_NM = 5.0


@dataclass(frozen=True)
class PhaseRef:
    key: str
    phase_id: str
    variant: str
    label: str
    framework: bool
    strain: str                 # "iso" | "ac"
    window: float               # allowed fractional lattice change
    lattice: tuple[float, ...]  # a, b, c, alpha, beta, gamma
    cell_mass_amu: float
    cell_volume_A3: float
    source: str
    hkl: np.ndarray = field(repr=False)
    f2m: np.ndarray = field(repr=False)

    def lattice_at(self, scale: tuple[float, ...]) -> tuple[float, ...]:
        a, b, c, al, be, ga = self.lattice
        if self.strain == "ac":
            sa, sc = scale
            return (a * sa, b * sa, c * sc, al, be, ga)
        (s,) = scale
        return (a * s, b * s, c * s, al, be, ga)

    def d_spacings(self, scale: tuple[float, ...]) -> np.ndarray:
        return _d_spacings(self.hkl, self.lattice_at(scale))

    def volume(self, scale: tuple[float, ...]) -> float:
        if self.strain == "ac":
            return self.cell_volume_A3 * scale[0] ** 2 * scale[1]
        return self.cell_volume_A3 * scale[0] ** 3

    def lines(self, scale: tuple[float, ...], wavelength_A: float,
              tt_range: tuple[float, float], return_index: bool = False):
        """(2theta, intensity[, reflection index]) of the reflections inside ``tt_range``."""
        d = self.d_spacings(scale)
        x = wavelength_A / (2.0 * d)
        ok = x < 1.0
        th = np.arcsin(x[ok])
        tt = 2.0 * np.degrees(th)
        lp = (1.0 + np.cos(2.0 * th) ** 2) / (np.sin(th) ** 2 * np.cos(th))
        inten = self.f2m[ok] * lp
        sel = (tt >= tt_range[0]) & (tt <= tt_range[1])
        if return_index:
            return tt[sel], inten[sel], np.nonzero(ok)[0][sel]
        return tt[sel], inten[sel]


def _d_spacings(hkl: np.ndarray, lat: tuple[float, ...]) -> np.ndarray:
    a, b, c, al, be, ga = lat
    al, be, ga = (math.radians(v) for v in (al, be, ga))
    G = np.array([[a * a, a * b * math.cos(ga), a * c * math.cos(be)],
                  [a * b * math.cos(ga), b * b, b * c * math.cos(al)],
                  [a * c * math.cos(be), b * c * math.cos(al), c * c]])
    Gi = np.linalg.inv(G)
    inv_d2 = np.einsum("ij,jk,ik->i", hkl, Gi, hkl)
    return 1.0 / np.sqrt(inv_d2)


@functools.lru_cache(maxsize=4)
def load_library(path: str | None = None) -> dict[str, PhaseRef]:
    raw = json.loads(Path(path or LIBRARY_PATH).read_text())["phases"]
    lib = {}
    for key, e in raw.items():
        refl = e["reflections"]
        lib[key] = PhaseRef(
            key=key, phase_id=e["phase_id"], variant=e["variant"], label=e["label"],
            framework=bool(e["framework"]), strain=e["strain"], window=float(e["window"]),
            lattice=tuple(e["lattice"]), cell_mass_amu=float(e["cell_mass_amu"]),
            cell_volume_A3=float(e["cell_volume_A3"]), source=e["source"],
            hkl=np.array([r["hkl"] for r in refl], dtype=float),
            f2m=np.array([r["f2m"] for r in refl], dtype=float),
        )
    return lib


def candidates(metal: str | None) -> list[PhaseRef]:
    lib = load_library()
    if metal is None:
        keys = sorted({k for ks in CANDIDATES.values() for k in ks})
    else:
        keys = CANDIDATES[metal]
    return [lib[k] for k in keys]


# --------------------------------------------------------------------------- #
# Lattice refinement against the fitted peak list
# --------------------------------------------------------------------------- #

def _misfit(ref: PhaseRef, scale, obs_tt: np.ndarray, wavelength_A: float,
            tt_range, tol: float = 0.30) -> float:
    tt, inten = ref.lines(scale, wavelength_A, tt_range)
    if tt.size == 0 or obs_tt.size == 0:
        return 1.0
    w = inten / inten.sum()
    strong = w > 0.004
    dist = np.min(np.abs(tt[strong][:, None] - obs_tt[None, :]), axis=1)
    return float(np.sum(w[strong] * np.minimum(dist / tol, 1.0) ** 2) / np.sum(w[strong]))


def refine_lattice(ref: PhaseRef, peaks, wavelength_A: float,
                   tt_range: tuple[float, float]) -> tuple[tuple[float, ...], float]:
    """Best lattice scale for ``ref`` given the observed peaks, and its misfit."""
    obs = np.array([p.two_theta_deg for p in peaks])
    if obs.size == 0:
        return ((1.0, 1.0) if ref.strain == "ac" else (1.0,)), 1.0
    win = ref.window
    if ref.strain == "ac":
        coarse = np.arange(1.0 - win, 1.0 + win + 1e-9, 0.002)
        best = min(((_misfit(ref, (sa, sc), obs, wavelength_A, tt_range), (sa, sc))
                    for sa in coarse for sc in coarse), key=lambda t: t[0])
        sa0, sc0 = best[1]
        fine = np.arange(-0.002, 0.00201, 0.0004)
        best = min(((_misfit(ref, (sa0 + da, sc0 + dc), obs, wavelength_A, tt_range),
                     (sa0 + da, sc0 + dc)) for da in fine for dc in fine), key=lambda t: t[0])
    else:
        grid = np.arange(1.0 - win, 1.0 + win + 1e-9, 0.0004)
        best = min(((_misfit(ref, (s,), obs, wavelength_A, tt_range), (s,)) for s in grid),
                   key=lambda t: t[0])
    scale = tuple(float(v) for v in best[1])
    return _ls_polish(ref, scale, obs, wavelength_A, tt_range), _misfit(ref, scale, obs, wavelength_A, tt_range)


def _ls_polish(ref: PhaseRef, scale, obs, wavelength_A, tt_range):
    win = ref.window

    # Least squares on the strong lines that matched an observed peak.
    tt, inten = ref.lines(scale, wavelength_A, tt_range)
    if tt.size:
        w = inten / inten.sum()
        j = np.argmin(np.abs(tt[:, None] - obs[None, :]), axis=1)
        m = (np.abs(tt - obs[j]) < MATCH_TOL_DEG) & (w > 0.004)
        if m.sum() >= len(scale) + 1:
            d_all = ref.d_spacings(scale)
            x = wavelength_A / (2.0 * d_all)
            in_rng = np.zeros(d_all.size, bool)
            th_all = 2.0 * np.degrees(np.arcsin(np.clip(x, 0, 1 - 1e-12)))
            in_rng[(x < 1.0) & (th_all >= tt_range[0]) & (th_all <= tt_range[1])] = True
            idx_rng = np.nonzero(in_rng)[0][m]
            target = obs[j[m]]
            sw = np.sqrt(w[m])

            def resid(s):
                d = _d_spacings(ref.hkl[idx_rng], ref.lattice_at(tuple(s)))
                calc = 2.0 * np.degrees(np.arcsin(np.clip(wavelength_A / (2.0 * d), 0, 1 - 1e-12)))
                return sw * (calc - target)
            lo = [1.0 - win] * len(scale)
            hi = [1.0 + win] * len(scale)
            sol = least_squares(resid, x0=np.clip(scale, lo, hi), bounds=(lo, hi))
            scale = tuple(float(v) for v in sol.x)
    return scale


# --------------------------------------------------------------------------- #
# Profile synthesis and scale fit
# --------------------------------------------------------------------------- #

def _scherrer_fwhm(domain_nm: float, tt: np.ndarray, wavelength_A: float) -> np.ndarray:
    theta = np.radians(tt / 2.0)
    return np.degrees(0.9 * wavelength_A * 0.1 / (domain_nm * np.maximum(np.cos(theta), 1e-3)))


def _domain_from_peaks(matched) -> float:
    est = []
    for p in matched:
        b2 = p.fwhm_deg ** 2 - INSTRUMENT_FWHM_DEG ** 2
        if b2 <= 1e-6:
            continue
        th = math.radians(p.two_theta_deg / 2.0)
        est.append(0.9 * 0.15406 / (math.radians(math.sqrt(b2)) * math.cos(th)))
    return float(np.clip(np.median(est), MIN_DOMAIN_NM, 300.0)) if est else 40.0


def _pv_area_normalised(x: np.ndarray, centres: np.ndarray, fwhm: np.ndarray,
                        eta: float) -> np.ndarray:
    """Columns: unit-area pseudo-Voigt per centre."""
    dx = x[:, None] - centres[None, :]
    sigma = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * (dx / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
    gamma = fwhm / 2.0
    lor = gamma / (math.pi * (dx ** 2 + gamma ** 2))
    return (1.0 - eta) * g + eta * lor


#: A phase stays in the fit only if this many of its *distinctive* lines (not
#: overlapped by another kept phase) are observed as peaks.
MIN_DISTINCTIVE_LINES = 2
DISTINCT_SEP_DEG = 0.25
EXPLAINED_SHARE = 0.3
#: Residual ratio (without / with) above which an overlapped phase is kept.
OVERLAP_RSS_RATIO = 1.5


def _distinctive_matches(it, others) -> int:
    """Observed peaks on lines of ``it`` that the other kept phases cannot explain.

    A peak counts as explained by another phase when that phase's *fitted*
    line intensity within the separation window is at least
    :data:`EXPLAINED_SHARE` of the peak's area: a dense set of weak lines from a
    low-symmetry phase does not mask a strong line of a high-symmetry one.
    """
    rel_max = it["lI"].max()
    t_strongest = float(it["ltt"][int(np.argmax(it["lI"]))])
    n = 0
    strongest = False
    for _hkl, t, pk in it["matched"]:
        k = int(np.argmin(np.abs(it["ltt"] - t)))
        if it["lI"][k] < 0.03 * rel_max:
            continue
        sep = max(DISTINCT_SEP_DEG, 0.6 * pk.fwhm_deg)
        explained = sum(float(o["c1"] * o["lI"][np.abs(o["ltt"] - t) <= sep].sum()) for o in others)
        if explained < EXPLAINED_SHARE * pk.area:
            n += 1
            strongest |= abs(t - t_strongest) < 1e-6
    # The phase's own strongest reflection, observed and not explained by any
    # other phase, is enough on its own (a minor cubic Zn phase next to R-3c:
    # only its (200) line is clear of the dense R-3c pattern).
    # Framework phases only: a secondary phase's strongest line (e.g. M(OH)2
    # (001) near 19 deg) can sit under the amorphous halo, where background
    # residue alone produces a peak.
    return max(n, MIN_DISTINCTIVE_LINES) if strongest and it["ref"].framework else n


def _supported(kept: list, area: dict, rss=None) -> list:
    """Drop phases whose intensity is not backed by reflections of their own.

    Without this, a low-symmetry phase whose many lines sit close to another
    phase's reflections soaks up per-reflection intensity mismatch (texture,
    occupancy differences) and is reported although nothing in the pattern
    requires it.  Weakest phases are tested first; when two phases cannot be
    told apart at all, the one with the larger fitted intensity is kept.
    """
    kept = sorted(kept, key=lambda it: area[id(it)])
    changed = True
    while changed and len(kept) > 1:
        changed = False
        base = rss(kept) if rss is not None else None
        for it in kept:
            if it is kept[-1]:
                continue
            others = [o for o in kept if o is not it]
            if _distinctive_matches(it, others) >= MIN_DISTINCTIVE_LINES:
                continue
            # Fully overlapped phase: keep it only if the pattern cannot be
            # fitted nearly as well without it (e.g. a cubic + monoclinic
            # mixture, where every cubic line sits inside a monoclinic doublet).
            if base is not None and base > 0 and rss(others) / base > OVERLAP_RSS_RATIO:
                continue
            kept = others
            changed = True
            break
    return kept


def _column(tt, ltt, lI, domain_nm, eta, wavelength_A):
    fwhm = np.hypot(INSTRUMENT_FWHM_DEG, _scherrer_fwhm(domain_nm, ltt, wavelength_A))
    return _pv_area_normalised(tt, ltt, fwhm, eta) @ lI


@dataclass
class PhaseResult:
    key: str
    phase_id: str
    label: str
    framework: bool
    lattice: tuple[float, ...]
    domain_nm: float
    bragg_area: float
    weight_fraction: float
    n_matched: int
    rms_deg: float
    lines_tt: np.ndarray = field(repr=False)
    lines_I: np.ndarray = field(repr=False)
    matched: list = field(default_factory=list, repr=False)


@dataclass
class QuantResult:
    phases: list[PhaseResult]
    unidentified_area: float
    unidentified_peaks: list = field(repr=False)
    model: np.ndarray = field(repr=False)

    @property
    def bragg_total(self) -> float:
        return sum(p.bragg_area for p in self.phases) + self.unidentified_area

    @property
    def unidentified_fraction(self) -> float:
        t = self.bragg_total
        return self.unidentified_area / t if t > 0 else 0.0

    def fraction(self, phase_id: str) -> float:
        return float(sum(p.weight_fraction for p in self.phases if p.phase_id == phase_id))

    def by_id(self) -> dict[str, PhaseResult]:
        return {p.phase_id: p for p in self.phases}


def quantify_phases(tt: np.ndarray, y: np.ndarray, peaks, wavelength_A: float,
                    metal: str | None = None, eta: float | None = None) -> QuantResult:
    """Identify and quantify the crystalline phases in a background-stripped pattern."""
    tt = np.asarray(tt, float)
    y = np.clip(np.asarray(y, float), 0.0, None)
    rng = (float(tt[0]), float(tt[-1]))
    if eta is None:
        etas = [p.eta for p in peaks if 0.0 <= p.eta <= 1.0]
        eta = float(np.median(etas)) if etas else 0.3
    obs = np.array([p.two_theta_deg for p in peaks])

    def build(ref, scale, dom=None):
        ltt, lI, lidx = ref.lines(scale, wavelength_A, rng, return_index=True)
        if ltt.size == 0:
            return None
        w = lI / lI.sum()
        matched = []          # (hkl, calculated 2theta, observed peak)
        if obs.size:
            j = np.argmin(np.abs(ltt[:, None] - obs[None, :]), axis=1)
            for k in np.nonzero((np.abs(ltt - obs[j]) < MATCH_TOL_DEG) & (w > 0.01))[0]:
                matched.append((tuple(int(v) for v in ref.hkl[lidx[k]]), float(ltt[k]), peaks[j[k]]))
        if dom is None:
            dom = _domain_from_peaks([m[2] for m in matched])
        return dict(ref=ref, scale=tuple(scale), ltt=ltt, lI=lI, dom=dom,
                    col=_column(tt, ltt, lI, dom, eta, wavelength_A), matched=matched)

    prepared = []
    for ref in candidates(metal):
        scale, mis = refine_lattice(ref, peaks, wavelength_A, rng)
        it = build(ref, scale) if mis <= 0.85 else None
        if it is not None:
            prepared.append(it)

    def solve(items, extra_peaks):
        cols = [it["col"] for it in items]
        for p in extra_peaks:
            cols.append(_pv_area_normalised(tt, np.array([p.two_theta_deg]),
                                            np.array([max(p.fwhm_deg, 0.05)]), eta)[:, 0])
        if not cols:
            return np.zeros(0), np.zeros_like(tt)
        A = np.column_stack(cols)
        coef, _ = nnls(A, y)
        return coef, A @ coef

    # pass 1: phases only; drop phases that explain (almost) nothing
    coef, _ = solve(prepared, [])
    areas = [c * it["lI"].sum() for c, it in zip(coef, prepared)]
    total = sum(areas)
    for it, c in zip(prepared, coef):
        it["c1"] = float(c)
    kept = [it for it, a in zip(prepared, areas) if total > 0 and a / total >= MIN_INTENSITY_SHARE]
    kept = _supported(kept, dict(zip((id(it) for it in prepared), areas)),
                      rss=lambda items: float(np.sum((y - solve(items, [])[1]) ** 2)))

    # Profile width per phase: coordinate descent on the NNLS residual over a
    # few domain sizes around the Scherrer estimate from the matched peaks.
    for _sweep in range(2):
        for it in kept:
            best = None
            d0 = it["dom"]
            for f in (0.6, 0.8, 1.0, 1.25, 1.6):
                if d0 * f < MIN_DOMAIN_NM:
                    continue
                col = _column(tt, it["ltt"], it["lI"], d0 * f, eta, wavelength_A)
                it_col, it["col"] = it["col"], col
                _, mdl = solve(kept, [])
                r = float(np.sum((y - mdl) ** 2))
                if best is None or r < best[0]:
                    best = (r, d0 * f, col)
                it["col"] = it_col
            if best is not None:
                it["dom"], it["col"] = best[1], best[2]

    # unexplained peaks: far from every reasonably strong line of a kept phase
    lines_all = np.concatenate([it["ltt"][it["lI"] / it["lI"].max() > 0.01] for it in kept]) \
        if kept else np.zeros(0)
    unexplained = [p for p in peaks
                   if lines_all.size == 0 or np.min(np.abs(lines_all - p.two_theta_deg)) > UNEXPLAINED_TOL_DEG]

    coef, model = solve(kept, unexplained)
    n_ph = len(kept)
    results = []
    raw_w = []
    for c, it in zip(coef[:n_ph], kept):
        ref, scale = it["ref"], it["scale"]
        raw_w.append(c * ref.cell_mass_amu * ref.volume(scale))
    wsum = sum(raw_w)
    for c, it, rw in zip(coef[:n_ph], kept, raw_w):
        ref, scale = it["ref"], it["scale"]
        if it["matched"]:
            rms = float(np.sqrt(np.mean([(pk.two_theta_deg - t) ** 2 for _, t, pk in it["matched"]])))
        else:
            rms = float("inf")
        results.append(PhaseResult(
            key=ref.key, phase_id=ref.phase_id, label=ref.label, framework=ref.framework,
            lattice=tuple(round(v, 5) for v in ref.lattice_at(scale)), domain_nm=it["dom"],
            bragg_area=float(c * it["lI"].sum()),
            weight_fraction=float(rw / wsum) if wsum > 0 else 0.0,
            n_matched=len(it["matched"]), rms_deg=rms, lines_tt=it["ltt"], lines_I=it["lI"] * c,
            matched=it["matched"],
        ))
    results.sort(key=lambda r: -r.weight_fraction)
    unid_area = float(sum(c for c in coef[n_ph:]))
    return QuantResult(phases=results, unidentified_area=unid_area,
                       unidentified_peaks=unexplained, model=model)
