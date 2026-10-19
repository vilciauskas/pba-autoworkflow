# SPDX-License-Identifier: GPL-3.0-or-later
"""Zinc polymorph quantification: line evidence, minor phases, disordered
cubic, weight-fraction uncertainty, and its use as surrogate noise."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from pba_autoworkflow.analysis.objectives import compute_objectives
from pba_autoworkflow.analysis.xrd import analyze_pattern
from pba_autoworkflow.optimize.surrogate import MixedGP
from pba_autoworkflow.schema import (
    Objectives,
    SampleDescriptors,
    SynthesisParameters,
    XRDDescriptors,
)
from pba_autoworkflow.sim.ground_truth import DC_DIFFUSE_SHARE, GroundTruth
from pba_autoworkflow.sim.instruments import simulate_xrd

PAPER = dict(metal="Zn", hcf_precursor="K3FeCN6", c_metal_M=0.1, c_hcf_M=0.05, c_nacl_M=0.0,
             c_citrate_M=0.0, ph=5.5, temperature_C=25.0, addition_rate_mL_min=1.0,
             aging_time_h=1.0, stir_rate_rpm=600)


def _mixture(s: float, order: float, dc: float, domain_nm: float):
    """Dried ZnHCF with R-3c share ``s`` of the framework, no secondary phases.

    ``dc``: disordered-cubic share of the cubic phase (contracted cell plus a
    diffuse component, as the simulator models fast drying).
    """
    gt = GroundTruth(seed=1, reproducibility=0.0, failure_rate=0.0)
    p = SynthesisParameters(**PAPER, dry_temperature_C=20.0, dry_pressure_mbar=1013.0,
                            dry_gas="ambient")
    lat = copy.deepcopy(gt.precipitate(p, np.random.default_rng(5)))
    lat.dried = True
    a_hyd = lat.lattice_a_A
    lat.polymorph_shares = {k: v for k, v in {"pba_fm3m/Zn": 1 - s, "znhcf_r3c/Zn": s}.items()
                            if v > 1e-6}
    lat.crystallinity, lat.domain_size_nm = order, domain_nm
    lat.nacl_fraction = lat.hydroxide_fraction = 0.0
    lat.diffuse_share = {}
    lat.disordered_cubic_share = dc if s < 1 else 0.0
    if lat.disordered_cubic_share:
        lat.diffuse_share["pba_fm3m/Zn"] = DC_DIFFUSE_SHARE * dc
        lat.lattice_a_A = a_hyd * (1 - dc * (1 - 0.984 ** (1 / 3)))
    return lat


def _measure(lat, seed):
    return analyze_pattern(simulate_xrd(lat, np.random.default_rng(seed)), "Zn")


@pytest.mark.parametrize("dc", [0.0, 1.0])
@pytest.mark.parametrize("order", [0.35, 0.5])
def test_poorly_ordered_cubic_is_not_read_as_r3c(order, dc):
    """The documented failure: a cubic framework with weak, broad lines used to
    be read as 0.1-0.3 R-3c, because R-3c's many lines absorbed intensity."""
    lat = _mixture(0.0, order, dc, 25.0)
    for seed in (1, 2):
        x = _measure(lat, seed)
        assert x.phase_fractions.get("znhcf_r3c", 0.0) <= 0.05, x.phase_fractions


def test_minor_cubic_next_to_r3c_is_found():
    lat = _mixture(0.75, 0.7, 0.0, 25.0)
    truth = lat.xrd_weight_fractions()
    x = _measure(lat, 1)
    assert x.phase_fractions.get("pba_fm3m", 0.0) == pytest.approx(truth["pba_fm3m"], abs=0.08)


@pytest.mark.parametrize("T, P", [(20.0, 0.02), (120.0, 8.0)])
def test_paper_fast_drying_routes_read_cubic(T, P):
    """Disordered cubic ZnHCF (paper's fast-drying routes, true R-3c <= 0.05)
    used to read 0.21-0.26 R-3c."""
    for k in range(3):
        gt = GroundTruth(seed=k, failure_rate=0.0)
        p = SynthesisParameters(**PAPER, dry_temperature_C=T, dry_pressure_mbar=P, dry_gas="ambient")
        lat = gt.latent(p, np.random.default_rng(100 + k))
        x = _measure(lat, 200 + k)
        assert x.phase_fractions.get("znhcf_r3c", 0.0) <= 0.10, (k, x.phase_fractions)


def test_weight_fraction_uncertainty_covers_the_error():
    """|error| <= 2 sd for most samples, over resolved and unresolved patterns,
    and the sd is larger where the polymorphs are not resolved (8 nm)."""
    errs, sds, small, large = [], [], [], []
    for s in (0.0, 0.25, 0.5, 0.75):
        for order in (0.35, 0.7):
            for d in (8.0, 25.0):
                lat = _mixture(s, order, 0.0, d)
                x = _measure(lat, 7)
                e = abs(x.phase_fractions.get("znhcf_r3c", 0.0) - lat.xrd_weight_fractions().get("znhcf_r3c", 0.0))
                sd = x.phase_fraction_sd["znhcf_r3c"]      # reported even when absent
                errs.append(e); sds.append(sd)
                (small if d == 8.0 else large).append(sd)
    errs, sds = np.array(errs), np.array(sds)
    assert np.mean(errs <= 2 * sds) >= 0.85, list(zip(errs.round(3), sds.round(3)))
    assert np.median(small) > 2 * np.median(large)


def test_target_uncertainty_reaches_objectives_and_survives_reload():
    lat = _mixture(0.5, 0.7, 0.0, 25.0)
    x = _measure(lat, 3)
    desc = SampleDescriptors(xrd=x)
    p = SynthesisParameters(**PAPER, dry_temperature_C=20.0, dry_pressure_mbar=1013.0)
    o = compute_objectives(desc, p, target_phase="znhcf_r3c")
    assert 0.0 < o.uncertainty["target_phase_fraction"] < 0.3
    o2 = Objectives.model_validate_json(o.model_dump_json())
    assert o2.uncertainty == o.uncertainty
    x2 = XRDDescriptors.model_validate_json(x.model_dump_json())
    assert x2.phase_fraction_sd == x.phase_fraction_sd
    # records written before the field existed
    old = x.model_dump(); old.pop("phase_fraction_sd")
    assert XRDDescriptors.model_validate(old).phase_fraction_sd == {}
    assert "xrd_wsd_znhcf_r3c" in desc.flat()


def test_gp_measurement_noise_downweights_uncertain_points():
    """A wrong observation with a large reported sd pulls the fit less."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 1))
    y = np.sin(4 * X[:, 0])
    y_bad = y.copy(); y_bad[:6] += 1.5                   # six wrong readings
    sd = np.zeros(30); sd[:6] = 1.0                      # ... that say so
    Xq = X[:6]
    plain = MixedGP(1); plain.fit(X, y_bad, rng=np.random.default_rng(1))
    noisy = MixedGP(1); noisy.fit(X, y_bad, rng=np.random.default_rng(1), noise_sd=sd)
    err_plain = np.abs(plain.predict(Xq, return_std=False) - y[:6]).mean()
    err_noisy = np.abs(noisy.predict(Xq, return_std=False) - y[:6]).mean()
    assert err_noisy < 0.6 * err_plain, (err_plain, err_noisy)
    with pytest.raises(ValueError):
        MixedGP(1).fit(X, y, noise_sd=np.ones(3))
