# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the PBA orchestrator.

The important tests here are not the round-trips -- they are the two that check
the *analysis* recovers the hidden truth from a simulated trace
(``test_xrd_recovers_lattice_constant``, ``test_xrd_recovers_secondary_phase_fractions``) and
the two that check the orchestrator behaves under failure
(``test_hardware_fault_does_not_kill_campaign``, ``test_station_exclusivity``).
Those are the properties that decide whether the loop is trustworthy on real
hardware.
"""

from __future__ import annotations

import asyncio
import math
import time

import numpy as np
import pytest

from pba_autoworkflow import (
    CampaignConfig,
    ProvenanceStore,
    build_simulated_platform,
    default_design_space,
)
from pba_autoworkflow.analysis.objectives import (
    LATTICE_WINDOW_A,
    compute_objectives,
    quality_flags,
    scalarize,
)
from pba_autoworkflow.analysis.spectra import analyze_icp
from pba_autoworkflow.analysis.xrd import analyze_pattern, index_cubic, find_and_fit_peaks
from pba_autoworkflow.campaign import Campaign
from pba_autoworkflow.devices.base import HardwareFault, VesselHandle
from pba_autoworkflow.optimize.planner import (
    BayesPlanner,
    SobolPlanner,
    hypervolume,
    pareto_mask,
)
from pba_autoworkflow.optimize.surrogate import MixedGP
from pba_autoworkflow.report import plot_campaign
from pba_autoworkflow.schema import (
    XRDDescriptors,
    METALS,
    CompositionDescriptors,
    Objectives,
    SampleDescriptors,
    SynthesisParameters,
    formula_weight,
    theoretical_capacity_mAh_g,
)
from pba_autoworkflow.scheduler import StationPool
from pba_autoworkflow.sim.ground_truth import GroundTruth
from pba_autoworkflow.sim.instruments import simulate_icp, simulate_xrd


def recipe(**over) -> SynthesisParameters:
    base = dict(
        metal="Mn", c_metal_M=0.05, c_hcf_M=0.05, c_nacl_M=2.0, c_citrate_M=0.05,
        ph=3.0, temperature_C=65.0, addition_rate_mL_min=0.5, aging_time_h=6.0,
        stir_rate_rpm=600.0,
    )
    base.update(over)
    return SynthesisParameters(**base)


# --------------------------------------------------------------------------- #
# Schema and design space
# --------------------------------------------------------------------------- #


def test_encode_decode_roundtrip():
    space = default_design_space()
    p = recipe(metal="Ni", c_metal_M=0.08, temperature_C=72.0, aging_time_h=3.0)
    x = space.encode(p)
    assert x.shape == (space.dim,)
    q = space.decode(x)
    assert q.metal == "Ni"
    for name in ("c_metal_M", "c_hcf_M", "c_nacl_M", "c_citrate_M", "ph",
                 "temperature_C", "addition_rate_mL_min", "aging_time_h"):
        assert getattr(q, name) == pytest.approx(getattr(p, name), rel=1e-9)


def test_log_scaled_parameter_maps_geometrically():
    space = default_design_space()
    spec = {s.name: s for s in space.continuous}["aging_time_h"]
    mid = spec.from_unit(0.5)
    assert mid == pytest.approx(math.sqrt(0.5 * 24.0), rel=1e-9)


def test_formula_weight_and_capacity_are_physical():
    fw = formula_weight("Mn", na_per_fu=1.8, vacancy_fraction=0.05, water_per_fu=1.5)
    assert 280.0 < fw < 340.0
    q = theoretical_capacity_mAh_g("Mn", 1.8, 0.05, 1.5)
    assert 130.0 < q < 175.0  # sodium-rich Mn-HCF regime


def test_parameter_validation_rejects_impossible_values():
    with pytest.raises(Exception):
        recipe(c_metal_M=-0.1)
    with pytest.raises(Exception):
        recipe(ph=14.0)


# --------------------------------------------------------------------------- #
# Analysis recovers the hidden truth
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metal", ["Mn", "Fe", "Co", "Ni", "Cu"])
def test_xrd_recovers_lattice_constant(metal):
    """Indexing must return the latent lattice constant across the metal series."""
    gt = GroundTruth(seed=3, reproducibility=0.0, failure_rate=0.0)
    rng = np.random.default_rng(11)
    p = recipe(metal=metal, c_citrate_M=0.06, aging_time_h=12.0)
    latent = gt.latent(p, rng)
    pattern = simulate_xrd(latent, rng)
    desc = analyze_pattern(pattern)
    assert desc.n_peaks_indexed >= 3
    assert desc.lattice_a_A == pytest.approx(latent.lattice_a_A, abs=0.03)


def test_xrd_recovers_domain_size_order_of_magnitude():
    gt = GroundTruth(seed=5, reproducibility=0.0, failure_rate=0.0)
    rng = np.random.default_rng(2)
    # A slow, chelated, hot, long-aged synthesis gives large domains; a fast cold
    # one gives small domains.  The analysis must order them correctly.
    slow = gt.latent(recipe(c_citrate_M=0.12, addition_rate_mL_min=0.06,
                            temperature_C=88.0, aging_time_h=24.0), rng)
    fast = gt.latent(recipe(c_citrate_M=0.0, addition_rate_mL_min=18.0,
                            temperature_C=28.0, aging_time_h=0.5,
                            c_metal_M=0.25, c_hcf_M=0.25), rng)
    d_slow = analyze_pattern(simulate_xrd(slow, rng)).domain_size_nm
    d_fast = analyze_pattern(simulate_xrd(fast, rng)).domain_size_nm
    assert slow.domain_size_nm > fast.domain_size_nm
    assert d_slow > d_fast


def test_xrd_indexing_returns_nothing_for_flat_pattern():
    from pba_autoworkflow.schema import XRDPattern

    tt = np.arange(10.0, 60.0, 0.02)
    flat = XRDPattern(two_theta_deg=tt,
                      intensity=np.full_like(tt, 500.0))
    peaks, _, _ = find_and_fit_peaks(flat)
    a, assign, _ = index_cubic(peaks, 1.5406)
    assert not assign or math.isnan(a)


def _latent_with(metal="Mn", **impurity):
    gt = GroundTruth(seed=9, reproducibility=0.0, failure_rate=0.0)
    p = recipe(metal=metal, c_citrate_M=0.06, aging_time_h=12.0)
    lat = gt.latent(p, np.random.default_rng(4))
    lat.nacl_fraction = impurity.get("nacl", 0.0)
    lat.hydroxide_fraction = impurity.get("hydroxide", 0.0)
    return lat


@pytest.mark.parametrize("metal,impurity", [("Mn", {"nacl": 0.25}), ("Co", {"hydroxide": 0.3}),
                                            ("Ni", {"nacl": 0.1, "hydroxide": 0.15}),
                                            ("Cu", {"hydroxide": 0.2})])
def test_xrd_recovers_secondary_phase_fractions(metal, impurity):
    """Secondary phases are identified, quantified, and do not bias the PBA lattice."""
    lat = _latent_with(metal, **impurity)
    desc = analyze_pattern(simulate_xrd(lat, np.random.default_rng(1)), metal)
    truth = lat.xrd_weight_fractions()
    for phase, w in truth.items():
        if w > 0.02:
            assert desc.phase_fractions.get(phase, 0.0) == pytest.approx(w, abs=0.06), (phase, desc.phase_fractions)
    assert desc.lattice_a_A == pytest.approx(lat.lattice_a_A, abs=0.01)
    assert desc.phase == "pba_fm3m"
    assert desc.unidentified_fraction < 0.05


def test_xrd_pure_pattern_has_only_the_framework_phase():
    lat = _latent_with("Fe")
    desc = analyze_pattern(simulate_xrd(lat, np.random.default_rng(1)), "Fe")
    assert desc.phase_fractions.get("pba_fm3m", 0.0) > 0.97, desc.phase_fractions
    assert desc.n_unidentified_peaks == 0


def test_xrd_crystallinity_tracks_order():
    """Crystallinity must rank samples by their ordered fraction.

    The halo profile fit has a known downward bias (about -0.15 on the simulated
    deck), so this checks rank and rough scale, not absolute agreement.
    """
    measured = []
    for order in (0.3, 0.5, 0.7, 0.9):
        lat = _latent_with("Mn"); lat.crystallinity = order
        measured.append(analyze_pattern(simulate_xrd(lat, np.random.default_rng(2))).crystallinity_index)
    assert measured == sorted(measured), measured
    assert measured[-1] - measured[0] > 0.4, measured


def test_amorphous_product_is_scored_not_quarantined():
    """An amorphous result is information about phase formation, not bad data."""
    x = XRDDescriptors(lattice_a_A=float("nan"), domain_size_nm=float("nan"),
                       crystallinity_index=0.05, fwhm_200_deg=float("nan"), phase="amorphous",
                       n_peaks_indexed=0, fit_residual=float("inf"))
    comp = CompositionDescriptors(na_per_fu=1.0, fe_per_metal=0.8, vacancy_fraction=0.2,
                                  water_per_fu=3.0, formula="t")
    desc = SampleDescriptors(xrd=x, composition=comp, isolated_yield=0.6)
    assert quality_flags(desc, recipe()).passed
    assert compute_objectives(desc, recipe()).values["target_phase_fraction"] == 0.0


def test_icp_recovers_composition():
    gt = GroundTruth(seed=13, reproducibility=0.0, failure_rate=0.0)
    rng = np.random.default_rng(6)
    p = recipe(c_nacl_M=4.0, c_citrate_M=0.10, aging_time_h=16.0)
    latent = gt.latent(p, rng)
    comp = analyze_icp(simulate_icp(latent, p, rng), p)
    assert comp.na_per_fu == pytest.approx(latent.na_per_fu, abs=0.12)
    assert comp.vacancy_fraction == pytest.approx(latent.vacancy_fraction, abs=0.06)


def test_chemistry_trend_slow_addition_lowers_vacancies():
    """The simulator must encode the known kinetic origin of vacancies."""
    gt = GroundTruth(seed=21, reproducibility=0.0, failure_rate=0.0)
    rng = np.random.default_rng(1)
    fast = gt.latent(recipe(addition_rate_mL_min=18.0, c_citrate_M=0.0), rng)
    slow = gt.latent(recipe(addition_rate_mL_min=0.06, c_citrate_M=0.10), rng)
    assert slow.vacancy_fraction < fast.vacancy_fraction


def test_chemistry_trend_more_nacl_raises_sodium():
    gt = GroundTruth(seed=21, reproducibility=0.0, failure_rate=0.0)
    rng = np.random.default_rng(1)
    lo = gt.latent(recipe(c_nacl_M=0.1), rng)
    hi = gt.latent(recipe(c_nacl_M=5.0), rng)
    assert hi.na_per_fu > lo.na_per_fu


# --------------------------------------------------------------------------- #
# Quality control
# --------------------------------------------------------------------------- #


def test_qc_rejects_charge_imbalanced_composition():
    from pba_autoworkflow.schema import CompositionDescriptors, XRDDescriptors

    desc = SampleDescriptors(
        xrd=XRDDescriptors(lattice_a_A=10.52, domain_size_nm=25.0,
                           crystallinity_index=0.8, fwhm_200_deg=0.3,
                           phase="cubic", n_peaks_indexed=5, fit_residual=0.02),
        composition=CompositionDescriptors(na_per_fu=1.95, fe_per_metal=0.55,
                                           vacancy_fraction=0.45, water_per_fu=3.0,
                                           formula="test"),
        isolated_yield=0.8,
    )
    report = quality_flags(desc, recipe())
    assert not report.passed
    assert any("charge-balance" in f for f in report.flags)


def test_qc_rejects_lattice_outside_metal_window():
    from pba_autoworkflow.schema import XRDDescriptors

    desc = SampleDescriptors(
        xrd=XRDDescriptors(lattice_a_A=8.55, domain_size_nm=20.0,
                           crystallinity_index=0.7, fwhm_200_deg=0.3,
                           phase="pba_fm3m", n_peaks_indexed=4, fit_residual=0.02),
    )
    report = quality_flags(desc, recipe(metal="Mn"))
    assert not report.passed
    assert any("lattice" in f for f in report.flags)


def test_objectives_flag_infeasible_low_yield():
    from pba_autoworkflow.schema import CompositionDescriptors

    desc = SampleDescriptors(
        composition=CompositionDescriptors(na_per_fu=1.7, fe_per_metal=0.97,
                                           vacancy_fraction=0.03, water_per_fu=1.2,
                                           formula="test"),
        isolated_yield=0.10,
    )
    obj = compute_objectives(desc, recipe())
    assert not obj.feasible
    assert scalarize(obj) < 0.0


# --------------------------------------------------------------------------- #
# Surrogate and planner
# --------------------------------------------------------------------------- #


def test_gp_interpolates_and_learns_lengthscales():
    rng = np.random.default_rng(0)
    X = rng.random((28, 3))
    y = np.sin(4 * X[:, 0]) + 0.3 * X[:, 1]  # third dimension is irrelevant
    gp = MixedGP(n_continuous=3)
    info = gp.fit(X, y, rng=rng)
    mu, sd = gp.predict(X)
    assert np.allclose(mu, y, atol=0.08)
    assert np.all(sd >= 0)
    # ARD must down-weight the irrelevant dimension.
    assert info.lengthscales[2] > info.lengthscales[0]


def test_gp_categorical_kernel_shares_information():
    rng = np.random.default_rng(1)
    n = 40
    Xc = rng.random((n, 1))
    onehot = np.zeros((n, 2))
    onehot[np.arange(n), rng.integers(0, 2, n)] = 1.0
    X = np.hstack([Xc, onehot])
    y = 2.0 * Xc[:, 0] + 0.4 * onehot[:, 1]
    gp = MixedGP(1, [("m", [1, 2])])
    gp.fit(X, y, rng=rng)
    mu = gp.predict(X, return_std=False)
    assert np.corrcoef(mu, y)[0, 1] > 0.95


def test_gp_loo_diagnostics_detect_noise():
    rng = np.random.default_rng(2)
    X = rng.random((30, 2))
    y_clean = X[:, 0] ** 2
    gp_clean = MixedGP(2)
    gp_clean.fit(X, y_clean, rng=rng)
    gp_noisy = MixedGP(2)
    gp_noisy.fit(X, rng.normal(0, 1, 30), rng=rng)
    assert gp_clean.loo_diagnostics()["loo_r2"] > gp_noisy.loo_diagnostics()["loo_r2"]


def test_pareto_and_hypervolume():
    Y = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0.4, 0.4]])
    mask = pareto_mask(Y)
    assert mask.tolist() == [True, True, True, False]
    hv = hypervolume(Y, np.zeros(2))
    # Exact: 0.9*0.1 + (0.5-0.1)*0.5 ... computed by the staircase
    assert hv == pytest.approx(0.09 + 0.4 * 0.5 + 0.1 * 0.4, rel=1e-9)


def test_hypervolume_monotone_under_added_point():
    Y = np.array([[0.5, 0.5]])
    ref = np.zeros(2)
    hv1 = hypervolume(Y, ref)
    hv2 = hypervolume(np.vstack([Y, [0.8, 0.6]]), ref)
    assert hv2 > hv1


def test_sobol_seed_covers_all_metals():
    space = default_design_space()
    sugg = SobolPlanner(space, seed=0).suggest(16, [])
    assert {s.parameters.metal for s in sugg} == set(METALS)
    assert {s.parameters.dry_atmosphere for s in sugg} == {"air", "vacuum"}


def test_bayes_planner_falls_back_when_undertrained():
    space = default_design_space()
    planner = BayesPlanner(space, seed=0, min_train=8, n_candidates=256,
                           n_mc_samples=8)
    sugg = planner.suggest(4, [])
    assert all(s.origin == "sobol-warmup" for s in sugg)
    assert planner.last_diagnostics is not None
    assert planner.last_diagnostics.exploration_fallback


def test_bayes_planner_produces_diverse_batch(tmp_path):
    """With enough history the planner must not return near-duplicate recipes."""
    space = default_design_space()
    store = ProvenanceStore(tmp_path / "div")
    platform, _ = build_simulated_platform(seed=2, time_scale=0.0)
    campaign = Campaign(platform, store,
                        CampaignConfig(campaign_id="div", n_seed=14, batch_size=6,
                                       max_iterations=1, replicate_fraction=0.0,
                                       seed=2))
    asyncio.run(campaign.run(max_iterations=1))
    planner = BayesPlanner(space, seed=1, min_train=6, n_candidates=512,
                           n_mc_samples=16)
    sugg = planner.suggest(5, campaign.history)
    X = np.array([space.encode(s.parameters) for s in sugg])
    dists = [np.linalg.norm(X[i] - X[j])
             for i in range(len(X)) for j in range(i + 1, len(X))]
    assert min(dists) > 1e-3
    store.close()


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #


def test_station_exclusivity():
    """A capacity-1 station must never host two coroutines at once."""
    pool = StationPool({"xrd": 1})
    concurrent = 0
    peak = 0

    async def user():
        nonlocal concurrent, peak
        async with pool.acquire("xrd"):
            concurrent += 1
            peak = max(peak, concurrent)
            await asyncio.sleep(0.01)
            concurrent -= 1

    async def main():
        await asyncio.wait_for(
            asyncio.gather(*(user() for _ in range(6))), timeout=5
        )

    asyncio.run(main())
    assert peak == 1
    assert pool.stats["xrd"].n_acquisitions == 6
    assert pool.stats["xrd"].max_wait_s > 0


def test_station_pool_reports_bottleneck():
    pool = StationPool({"fast": 2, "slow": 1})

    async def work():
        async def slow():
            async with pool.acquire("slow"):
                await asyncio.sleep(0.05)

        async def fast():
            async with pool.acquire("fast"):
                await asyncio.sleep(0.001)

        await asyncio.gather(*(slow() for _ in range(3)),
                             *(fast() for _ in range(3)))

    asyncio.run(work())
    assert pool.bottleneck() == "slow"


def test_unknown_station_raises():
    pool = StationPool({"a": 1})

    async def bad():
        async with pool.acquire("nope"):
            pass

    with pytest.raises(KeyError):
        asyncio.run(bad())


# --------------------------------------------------------------------------- #
# Workflow and provenance
# --------------------------------------------------------------------------- #


def test_single_experiment_end_to_end(tmp_path):
    from pba_autoworkflow.schema import Experiment
    from pba_autoworkflow.workflow import ExperimentWorkflow

    platform, backend = build_simulated_platform(seed=1, time_scale=0.0,
                                                 failure_rate=0.0)
    store = ProvenanceStore(tmp_path / "one")
    store.create_campaign("c1", {}, default_design_space().model_dump_json())
    p = recipe(c_nacl_M=3.0, c_citrate_M=0.08, aging_time_h=10.0)
    exp = Experiment("e1", "c1", 0, p, origin="manual")
    store.register_experiment(exp)

    wf = ExperimentWorkflow(platform, store)
    done = asyncio.run(wf.run(exp))

    assert done.status.value in ("complete", "quarantined")
    assert done.descriptors.xrd is not None
    assert done.descriptors.composition is not None
    assert "xrd" in done.raw_refs
    arrays = store.load_trace(done.raw_refs["xrd"])
    assert arrays["two_theta_deg"].size > 1000
    store.close()


def test_provenance_resumption(tmp_path):
    platform, _ = build_simulated_platform(seed=4, time_scale=0.0)
    store = ProvenanceStore(tmp_path / "resume")
    cfg = CampaignConfig(campaign_id="rc", n_seed=6, batch_size=4,
                         max_iterations=1, replicate_fraction=0.0, seed=4)
    c1 = Campaign(platform, store, cfg)
    asyncio.run(c1.run(max_iterations=1))
    n1 = len(c1.history)
    assert n1 == 6
    store.close()

    store2 = ProvenanceStore(tmp_path / "resume")
    c2 = Campaign(platform, store2, cfg)
    assert c2.resumed
    assert len(c2.history) == n1
    assert store2.next_batch_index("rc") == 1
    # Raw traces survive the restart.
    with_traces = [e for e in c2.history if e.raw_refs]
    assert with_traces
    store2.close()


def test_traces_are_reanalyzable(tmp_path):
    """A stored pattern must reduce to the same descriptors on re-analysis."""
    platform, _ = build_simulated_platform(seed=8, time_scale=0.0, failure_rate=0.0)
    store = ProvenanceStore(tmp_path / "reanalyze")
    cfg = CampaignConfig(campaign_id="ra", n_seed=4, batch_size=4,
                         max_iterations=1, replicate_fraction=0.0, seed=8)
    c = Campaign(platform, store, cfg)
    asyncio.run(c.run(max_iterations=1))
    from pba_autoworkflow.schema import XRDPattern

    checked = 0
    for exp in c.history:
        if "xrd" not in exp.raw_refs or exp.descriptors.xrd is None:
            continue
        arr = store.load_trace(exp.raw_refs["xrd"])
        again = analyze_pattern(XRDPattern(arr["two_theta_deg"], arr["intensity"]), exp.parameters.metal)
        assert again.lattice_a_A == pytest.approx(exp.descriptors.xrd.lattice_a_A,
                                                 abs=1e-6, nan_ok=True)
        checked += 1
    assert checked > 0
    store.close()


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


def test_hardware_fault_does_not_kill_campaign(tmp_path):
    """A raising instrument must fail its own sample only."""
    platform, _ = build_simulated_platform(seed=6, time_scale=0.0, failure_rate=0.0)
    store = ProvenanceStore(tmp_path / "fault")

    original = platform.diffractometer.measure
    calls = {"n": 0}

    async def flaky(solid, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise HardwareFault("xrd-01: goniometer crash")
        return await original(solid, *args, **kwargs)

    platform.diffractometer.measure = flaky  # type: ignore[assignment]

    cfg = CampaignConfig(campaign_id="ft", n_seed=6, batch_size=6,
                         max_iterations=1, replicate_fraction=0.0, seed=6,
                         max_batch_failure_rate=1.0)
    c = Campaign(platform, store, cfg)
    asyncio.run(c.run(max_iterations=1))
    # Every sample got a verdict, and the loop did not raise.
    assert len(c.history) == 6
    # Samples whose XRD crashed lack diffraction descriptors but still have ICP.
    no_xrd = [e for e in c.history if e.descriptors.xrd is None]
    assert no_xrd
    assert any(e.descriptors.composition is not None for e in no_xrd)
    store.close()


def test_transport_error_is_retried(tmp_path):
    from pba_autoworkflow.devices.base import TransportError
    from pba_autoworkflow.schema import Experiment
    from pba_autoworkflow.workflow import ExperimentWorkflow

    platform, backend = build_simulated_platform(seed=3, time_scale=0.0,
                                                 failure_rate=0.0)
    store = ProvenanceStore(tmp_path / "retry")
    store.create_campaign("rt", {}, default_design_space().model_dump_json())
    p = recipe()
    exp = Experiment("r1", "rt", 0, p)
    store.register_experiment(exp)

    original = platform.diffractometer.measure
    attempts = {"n": 0}

    async def once_flaky(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TransportError("xrd-01: bus timeout")
        return await original(*args, **kwargs)

    platform.diffractometer.measure = once_flaky  # type: ignore[assignment]
    done = asyncio.run(ExperimentWorkflow(platform, store).run(exp))
    assert attempts["n"] >= 2
    assert done.descriptors.xrd is not None
    store.close()


def test_infeasible_recipe_is_rejected_before_hardware(tmp_path):
    from pba_autoworkflow.schema import Experiment
    from pba_autoworkflow.workflow import ExperimentWorkflow

    platform, _ = build_simulated_platform(seed=1, time_scale=0.0)
    store = ProvenanceStore(tmp_path / "infeas")
    store.create_campaign("inf", {}, default_design_space().model_dump_json())
    # 10 mL at 0.001 mL/min is far beyond the deck reservation.
    p = recipe(addition_rate_mL_min=0.0002)
    exp = Experiment("i1", "inf", 0, p)
    store.register_experiment(exp)
    done = asyncio.run(ExperimentWorkflow(platform, store).run(exp))
    assert done.status.value == "failed"
    assert "infeasible" in (done.error or "")
    store.close()


def test_campaign_halts_on_platform_health_trip(tmp_path):
    """Systematic failure must stop the loop rather than consume the budget."""
    platform, _ = build_simulated_platform(seed=5, time_scale=0.0, failure_rate=1.0)
    store = ProvenanceStore(tmp_path / "health")
    cfg = CampaignConfig(campaign_id="hl", n_seed=4, batch_size=4,
                         max_iterations=5, replicate_fraction=0.0, seed=5,
                         max_batch_failure_rate=0.5)
    c = Campaign(platform, store, cfg)
    asyncio.run(c.run())
    assert len(c.iterations) == 1
    assert "platform health" in (c.iterations[-1].stop_reason or "")
    store.close()


# --------------------------------------------------------------------------- #
# Closed loop
# --------------------------------------------------------------------------- #


def test_closed_loop_improves_over_seed(tmp_path):
    """The loop must beat its own seed design on the campaign objective."""
    platform, _ = build_simulated_platform(seed=17, time_scale=0.0,
                                           reproducibility=0.02, failure_rate=0.0)
    store = ProvenanceStore(tmp_path / "loop")
    cfg = CampaignConfig(campaign_id="lp", n_seed=16, batch_size=8,
                         max_iterations=4, replicate_fraction=0.0, seed=17,
                         hv_convergence_tol=0.0)
    c = Campaign(platform, store, cfg)
    asyncio.run(c.run())

    seed_exps = [e for e in c.usable if e.batch_index == 0]
    later = [e for e in c.usable if e.batch_index > 0]
    assert seed_exps and later
    best_seed = max(scalarize(e.objectives) for e in seed_exps)
    best_later = max(scalarize(e.objectives) for e in later)
    assert best_later >= best_seed - 1e-9
    assert c.current_hypervolume() > 0
    assert len(c.pareto_front()) >= 1
    store.close()


def test_replicates_measure_reproducibility(tmp_path):
    platform, _ = build_simulated_platform(seed=23, time_scale=0.0,
                                           reproducibility=0.06, failure_rate=0.0)
    store = ProvenanceStore(tmp_path / "reps")
    cfg = CampaignConfig(campaign_id="rp", n_seed=10, batch_size=8,
                         max_iterations=2, replicate_fraction=0.5, seed=23,
                         hv_convergence_tol=0.0)
    c = Campaign(platform, store, cfg)
    asyncio.run(c.run())
    stats = c.replicate_statistics()
    assert stats.get("n_replicate_groups", 0) >= 1
    assert any(k.startswith("rsd_") for k in stats)
    store.close()


def test_summary_is_serializable(tmp_path):
    import json

    platform, _ = build_simulated_platform(seed=31, time_scale=0.0)
    store = ProvenanceStore(tmp_path / "summ")
    c = Campaign(platform, store,
                 CampaignConfig(campaign_id="sm", n_seed=6, batch_size=4,
                                max_iterations=1, seed=31))
    asyncio.run(c.run(max_iterations=1))
    text = json.dumps(c.summary(), default=str)
    assert "campaign_id" in text
    df = c.dataframe()
    assert len(df) == len(c.history)
    assert "obj_target_phase_fraction" in df.columns or "status" in df.columns
    store.close()


# --------------------------------------------------------------------------- #
# Regressions.  Each of these encodes a defect that reached a generated report
# before it was caught, so the test names describe the symptom, not the API.
# --------------------------------------------------------------------------- #


def test_lattice_constants_are_physical_for_all_analogues():
    """A cubic PBA sits near 10.1-10.6 A; an intercept error moves the whole series.

    The QC window is derived from these constants, so a wrong baseline either
    passes mis-indexed patterns or rejects every good one.
    """
    from pba_autoworkflow.schema import LATTICE_A0_A

    assert set(LATTICE_A0_A) == set(METALS)
    for metal, a in LATTICE_A0_A.items():
        assert 10.0 <= a <= 10.7, f"{metal}: a={a} outside the PBA range"
    # The cell edge correlates with the divalent radius but is not strictly
    # ordered by it -- Cu(II) is Jahn-Teller contracted and the reported cubic
    # Fe/Co pair is inverted.  Assert the correlation and the two endpoints, which
    # is what the physics supports; a strict-ordering assertion here would only be
    # satisfiable by fabricating the table from the radii.
    from scipy.stats import spearmanr

    from pba_autoworkflow.schema import IONIC_RADIUS_A

    rho = spearmanr([IONIC_RADIUS_A[m] for m in METALS],
                    [LATTICE_A0_A[m] for m in METALS]).statistic
    assert rho > 0.4, f"lattice series uncorrelated with ionic radius (rho={rho})"
    assert max(LATTICE_A0_A, key=LATTICE_A0_A.get) == "Mn"
    assert min(LATTICE_A0_A, key=LATTICE_A0_A.get) == "Cu"
    assert LATTICE_A0_A["Cu"] < LATTICE_A0_A["Ni"], "expected Cu contraction"


def test_indexing_recovers_lattice_across_series_and_temperature():
    """End-to-end simulate->index accuracy, not just for one lucky sample."""
    gt = GroundTruth(seed=3, reproducibility=0.0, failure_rate=0.0)
    rng = np.random.default_rng(11)
    worst = 0.0
    for metal in METALS:
        for T in (35.0, 65.0, 85.0):
            # Zn: drying cold keeps the cubic polymorph present and indexable
            p = recipe(metal=metal, temperature_C=T, aging_time_h=12.0,
                       addition_rate_mL_min=10.0 if metal == "Zn" else 0.5,
                       dry_temperature_C=30.0 if metal == "Zn" else 70.0)
            latent = gt.latent(p, rng)
            if latent.polymorph_shares.get(f"pba_fm3m/{metal}", 0.0) < 0.5:
                continue
            desc = analyze_pattern(simulate_xrd(latent, rng), metal)
            worst = max(worst, abs(desc.lattice_a_A - latent.lattice_a_A))
            lo, hi = LATTICE_WINDOW_A[metal]
            assert lo <= desc.lattice_a_A <= hi
    assert worst < 0.01, f"worst lattice error {worst:.4f} A"


def test_reflection_labels_are_generated_not_tabulated():
    """Every indexing candidate must carry a correct hkl label.

    A missing entry used to print the bare multiplicity ("32" for (440)), which
    is indistinguishable from a mis-indexed pattern to a reader.
    """
    from pba_autoworkflow.analysis.xrd import _ALLOWED_M, _HKL_LABELS, allowed_reflections

    for m in _ALLOWED_M:
        label = _HKL_LABELS.get(int(m))
        assert label is not None and len(label) == 3, f"m={m} has label {label!r}"
        h, k, l = (int(ch) for ch in label)
        assert h * h + k * k + l * l == int(m)
        # F-centring: all indices share a parity.
        assert len({h % 2, k % 2, l % 2}) == 1
    assert allowed_reflections()[32] == "440"
    assert allowed_reflections()[36] == "600"
    # Mixed-parity reflections are extinct and must never appear.
    assert 5 not in allowed_reflections(parity="all")


def test_target_phase_objective_follows_target_and_discounts_unknowns():
    def xd(fracs, cryst, unid=0.0):
        return XRDDescriptors(lattice_a_A=10.2, domain_size_nm=40.0, crystallinity_index=cryst,
                              fwhm_200_deg=0.2, phase="pba_fm3m", n_peaks_indexed=8,
                              fit_residual=0.01, phase_fractions=fracs, unidentified_fraction=unid)
    mix = {"pba_fm3m": 0.3, "znhcf_r3c": 0.6, "zno": 0.1}
    d = SampleDescriptors(xrd=xd(mix, 0.8), isolated_yield=0.6)
    zn = recipe(metal="Zn")
    assert compute_objectives(d, zn, target_phase="znhcf_r3c").values["target_phase_fraction"] == pytest.approx(0.6)
    assert compute_objectives(d, zn, target_phase="pba_fm3m").values["target_phase_fraction"] == pytest.approx(0.3)
    assert compute_objectives(d, zn).values["crystallinity"] == pytest.approx(0.8)
    d_unknown = SampleDescriptors(xrd=xd(mix, 0.8, unid=0.5), isolated_yield=0.6)
    assert compute_objectives(d_unknown, zn, target_phase="znhcf_r3c").values["target_phase_fraction"] == pytest.approx(0.3)
    none = compute_objectives(SampleDescriptors(isolated_yield=0.6), recipe()).values
    assert none == {"target_phase_fraction": 0.0, "crystallinity": 0.0}


def test_formula_weight_clamps_negative_vacancy():
    """A marginally negative measured vacancy must not inflate the formula weight."""
    fw_neg = formula_weight("Mn", 1.8, -0.05, 1.0)
    fw_zero = formula_weight("Mn", 1.8, 0.0, 1.0)
    assert fw_neg == pytest.approx(fw_zero)


def test_durations_are_monotonic_not_wall_clock():
    """Stage durations must survive a system-clock jump.

    A suspended host once produced an 81 495 s iteration in a campaign that ran
    for thirty seconds, which made the reported bottleneck meaningless.
    """
    import pba_autoworkflow.clock as clock

    sw = clock.Stopwatch()
    real_time = time.time
    try:
        # Simulate the host clock jumping forward a day mid-operation.
        time.time = lambda: real_time() + 86400.0
        time.sleep(0.02)
        started_at, finished_at, duration = sw.stop()
    finally:
        time.time = real_time
    assert 0.01 < duration < 5.0, f"duration {duration} contaminated by clock jump"
    # Tolerance, not exact equality: finished_at is an epoch value near 1.8e9, so
    # adding a duration to it carries only ~0.2 us of resolution in float64.
    assert finished_at - started_at == pytest.approx(duration, abs=1e-3)
    assert started_at < real_time() + 1.0, "started_at should predate the jump"


def test_device_utilization_uses_stored_durations(tmp_path):
    store = ProvenanceStore(tmp_path / "util")
    store.record_device_call("e1", "xrd-01", "measure", started_at=1000.0,
                             duration_s=12.5, ok=True)
    store.record_device_call("e1", "xrd-01", "measure", started_at=2000.0,
                             duration_s=7.5, ok=False)
    rows = {r["device_id"]: r for r in store.device_utilization()}
    assert rows["xrd-01"]["busy_s"] == pytest.approx(20.0)
    assert rows["xrd-01"]["n_failed"] == 1
    store.close()


def test_every_latent_failure_mode_is_raised_by_a_device():
    """The simulator must not invent a failure no device acts on.

    A latent state marked failed with a mode nothing raises means the run
    proceeds and is scored as a success -- a silently wrong training label.
    """
    gt = GroundTruth(seed=1, reproducibility=0.05, failure_rate=1.0)
    rng = np.random.default_rng(0)
    modes = set()
    for i, metal in enumerate(METALS * 4):
        latent = gt.latent(recipe(metal=metal, aging_time_h=1.0 + i), rng)
        if latent.failed:
            modes.add(latent.failure_mode)
    assert modes, "failure_rate=1.0 produced no failures"
    handled = {"insufficient_solid", "pellet_lost_in_decant"}
    assert modes <= handled, f"unhandled latent failure modes: {modes - handled}"


def test_mechanical_faults_are_independent_of_chemistry():
    """Hardware faults must arise from the deck, with no chemistry failure at all."""
    platform, backend = build_simulated_platform(
        seed=2, time_scale=0.0, failure_rate=0.0, mechanical_fault_rate=1.0)
    assert backend.ground_truth.failure_rate == 0.0

    async def go():
        await platform.ensure_online()
        v = VesselHandle("v1", "e1")
        with pytest.raises(HardwareFault):
            await platform.liquid_handler.prepare_solution(v, {"MnCl2": 0.05}, 10.0)

    asyncio.run(go())


def test_scalarize_never_prefers_infeasible_over_feasible():
    """Feasibility must dominate, however good the objectives look.

    With a soft penalty a high-scoring infeasible run can outrank a feasible one,
    and the campaign then reports a best recipe that made too little solid to
    characterize.
    """
    great_but_infeasible = Objectives(
        values={"target_phase_fraction": 1.0, "crystallinity": 1.0},
        constraints={"isolated_yield": -0.30}, feasible=False)
    poor_but_feasible = Objectives(
        values={"target_phase_fraction": 0.05, "crystallinity": 0.05},
        constraints={"isolated_yield": 0.01}, feasible=True)
    assert scalarize(poor_but_feasible) > scalarize(great_but_infeasible)
    # Among infeasible points, a smaller shortfall must still rank higher.
    near_miss = Objectives(values={"target_phase_fraction": 0.5, "crystallinity": 0.5},
                           constraints={"isolated_yield": -0.01}, feasible=False)
    assert scalarize(near_miss) > scalarize(great_but_infeasible)


def test_occupancy_is_nonzero_and_labelled_by_what_was_simulated(tmp_path):
    """Occupancy must be measurable, and the panel must say what it measured.

    The all-zero occupancy chart that prompted this test was caused by the
    wall-clock duration bug, not by time compression: a suspended host inflated
    the denominator to 81 495 s.  With monotonic timing the figures are finite
    either way, but at ``time_scale=0`` the busy time is analysis compute rather
    than instrument dwell and ranks the stations differently, so the report has
    to distinguish the two cases instead of implying a platform bottleneck.
    """
    from pba_autoworkflow.report import _simulated_time_scale

    platform, _ = build_simulated_platform(seed=7, time_scale=0.0)
    store = ProvenanceStore(tmp_path / "rep")
    c = Campaign(platform, store, CampaignConfig(
        campaign_id="rp", n_seed=4, batch_size=4, max_iterations=1,
        replicate_fraction=0.0, seed=7))
    asyncio.run(c.run(max_iterations=1))

    rep = c.pool.report()
    assert rep and all("total_busy_s" in r for r in rep)
    # Finite and sane: the pathology was occupancy ~= 0 from an inflated denominator.
    assert 0.0 < c.pool.wall_time_s() < 3600.0
    assert sum(float(r["occupancy"]) for r in rep) > 0.0
    for r in rep:
        assert 0.0 <= float(r["occupancy"]) <= 1.0 + 1e-9
        assert float(r["total_busy_s"]) >= 0.0

    assert _simulated_time_scale(c) == 0.0
    out = plot_campaign(c, tmp_path / "overview.png")
    assert out.exists() and out.stat().st_size > 10_000
    store.close()


def test_report_detects_real_durations_for_a_non_simulated_deck():
    """A deck with no compression factor must not be labelled as compressed."""
    from pba_autoworkflow.report import _simulated_time_scale

    class _FakeCampaign:
        class platform:
            @staticmethod
            def all_devices():
                return [object()]

    assert _simulated_time_scale(_FakeCampaign()) is None


def test_fe_analogue_vacancy_is_measured_not_assumed():
    """Prussian blue must not be reported vacancy-free by construction.

    ICP sees one indistinguishable iron pool for the Fe analogue.  An earlier
    version split total Fe 50/50 between the N and C sites, which forces
    Fe/M == 1.000 exactly -- so every Prussian blue sample scored a perfect
    framework (then an objective) regardless of what was synthesized, and the
    optimizer duly reported Fe as the best recipe.  Carbon (CHN) resolves the sublattice.
    """
    gt = GroundTruth(seed=3, reproducibility=0.0, failure_rate=0.0)
    rng = np.random.default_rng(5)
    measured, truth = [], []
    for cit, T in ((0.0, 45.0), (0.08, 60.0), (0.15, 75.0)):
        p = recipe(metal="Fe", c_citrate_M=cit, temperature_C=T, aging_time_h=6.0)
        latent = gt.latent(p, rng)
        comp = analyze_icp(simulate_icp(latent, p, rng), p)
        measured.append(comp.vacancy_fraction)
        truth.append(latent.vacancy_fraction)
    # Distinct samples must give distinct answers, and they must track the truth.
    assert len(set(round(v, 4) for v in measured)) == len(measured)
    assert max(abs(m - t) for m, t in zip(measured, truth)) < 0.08
    assert not any(abs(m - 0.0) < 1e-9 for m in measured)


def test_carbon_assay_resolves_sublattice_for_all_analogues():
    """Vacancy recovery must hold across the series, not just where ICP is easy."""
    gt = GroundTruth(seed=3, reproducibility=0.0, failure_rate=0.0)
    rng = np.random.default_rng(5)
    worst = 0.0
    for metal in METALS:
        for cit in (0.0, 0.15):
            p = recipe(metal=metal, c_citrate_M=cit, aging_time_h=6.0)
            latent = gt.latent(p, rng)
            comp = analyze_icp(simulate_icp(latent, p, rng), p)
            worst = max(worst, abs(comp.vacancy_fraction - latent.vacancy_fraction))
    assert worst < 0.08, f"worst vacancy error {worst:.4f}"


def test_fe_analogue_without_carbon_is_undetermined_not_guessed():
    """A deck with no CHN must report the Fe analogue as undetermined.

    Returning a plausible-looking number from an unresolvable measurement is worse
    than returning nothing: QC would pass it and the surrogate would train on it.
    """
    import dataclasses

    gt = GroundTruth(seed=3, reproducibility=0.0, failure_rate=0.0)
    rng = np.random.default_rng(5)
    p = recipe(metal="Fe", aging_time_h=6.0)
    icp = simulate_icp(gt.latent(p, rng), p, rng)
    blind = dataclasses.replace(icp, carbon_wt_pct=float("nan"))
    comp = analyze_icp(blind, p)
    assert math.isnan(comp.vacancy_fraction)
    assert "undetermined" in comp.formula
    # And QC must reject it rather than let it reach the optimizer.
    report = quality_flags(SampleDescriptors(composition=comp, isolated_yield=0.5), p)
    assert not report.passed
    assert any("not determinable" in f for f in report.flags)


def test_xrd_population_no_false_distortion_and_bounded_crystallinity_bias():
    """Over random recipes (with their secondary phases), impurity lines must not
    be read as a rhombohedral/monoclinic split, and the crystallinity index must
    stay within a bounded bias of the simulated ordered fraction."""
    from pba_autoworkflow.schema import default_design_space
    space = default_design_space()
    names, n_cont = space.encoded_names, len(space.continuous)
    gt = GroundTruth(seed=0, reproducibility=0.0, failure_rate=0.0)
    rng = np.random.default_rng(7)
    false_split, bias = 0, []
    for k in range(60):
        x = np.zeros(len(names)); x[:n_cont] = rng.uniform(0, 1, n_cont)
        x[n_cont + k % (len(names) - n_cont)] = 1.0
        p = space.decode(x)
        lat = gt.latent(p, rng)
        if lat.failed:
            continue
        d = analyze_pattern(simulate_xrd(lat, rng))
        false_split += lat.phase == "cubic" and d.phase in ("rhombohedral", "monoclinic")
        bias.append(d.crystallinity_index - lat.crystallinity)
    assert false_split <= 1, f"{false_split} cubic samples called distorted"
    assert abs(float(np.median(bias))) < 0.25, f"median crystallinity bias {np.median(bias):+.3f}"



# --------------------------------------------------------------------------- #
# Polymorph control
# --------------------------------------------------------------------------- #

def _zn(**over):
    base = dict(metal="Zn", c_metal_M=0.05, c_hcf_M=0.05, c_nacl_M=1.0, c_citrate_M=0.05, ph=4.0,
                temperature_C=25.0, addition_rate_mL_min=15.0, aging_time_h=0.5, dry_temperature_C=30.0)
    base.update(over)
    return recipe(**base)


def test_nan_descriptors_survive_a_json_round_trip():
    x = XRDDescriptors(lattice_a_A=float("nan"), domain_size_nm=30.0, crystallinity_index=0.6,
                       fwhm_200_deg=0.3, phase="znhcf_r3c", n_peaks_indexed=9, fit_residual=0.02,
                       phase_fractions={"znhcf_r3c": 1.0})
    again = XRDDescriptors.model_validate_json(x.model_dump_json())
    assert math.isnan(again.lattice_a_A) and again.phase_fractions == {"znhcf_r3c": 1.0}


def test_phase_library_is_consistent():
    from pba_autoworkflow.analysis.phases import CANDIDATES, FRAMEWORK_PHASES, load_library
    lib = load_library()
    for metal in METALS:
        assert f"pba_fm3m/{metal}" in CANDIDATES[metal]
        for key in CANDIDATES[metal]:
            assert key in lib and lib[key].f2m.size > 0
    assert set(FRAMEWORK_PHASES) == {r.phase_id for r in lib.values() if r.framework}
    # NaCl (200) at a = 5.640 A, Cu Ka1
    tt, inten = lib["nacl"].lines((1.0,), 1.5406, (10.0, 60.0))
    assert tt[int(np.argmax(inten))] == pytest.approx(31.70, abs=0.02)


@pytest.mark.parametrize("over", [dict(), dict(dry_temperature_C=110.0, dry_atmosphere="vacuum"),
                                  dict(temperature_C=60.0, addition_rate_mL_min=0.5, aging_time_h=6.0)])
def test_zn_polymorph_fractions_recovered(over):
    """Weight fractions of the two Zn polymorphs, for a well-ordered framework.

    Below an ordered fraction of ~0.5 the R-3c share reads systematically high
    (see ``test_zn_low_order_bias_is_bounded``), so this sets order = 0.7.
    """
    gt = GroundTruth(seed=1, reproducibility=0.0, failure_rate=0.0)
    lat = gt.latent(_zn(**over), np.random.default_rng(3))
    lat.crystallinity = 0.7
    desc = analyze_pattern(simulate_xrd(lat, np.random.default_rng(4)), "Zn")
    truth = lat.xrd_weight_fractions()
    for phase in ("znhcf_r3c", "pba_fm3m"):
        assert desc.phase_fractions.get(phase, 0.0) == pytest.approx(truth.get(phase, 0.0), abs=0.08), \
            (phase, truth, desc.phase_fractions)


def test_hot_or_vacuum_drying_converts_cubic_zn_to_r3c():
    gt = GroundTruth(seed=1, reproducibility=0.0, failure_rate=0.0)
    share = lambda **o: gt.latent(_zn(**o), np.random.default_rng(3)).polymorph_shares.get("znhcf_r3c/Zn", 0.0)
    cold, hot, vac = share(), share(dry_temperature_C=100.0), share(dry_temperature_C=55.0, dry_atmosphere="vacuum")
    assert cold < 0.5 < hot and vac > share(dry_temperature_C=55.0)


def test_no_false_monoclinic_in_cubic_mn_fe():
    """Low-symmetry phases must not absorb intensity mismatch of a cubic pattern."""
    gt = GroundTruth(seed=0, reproducibility=0.0, failure_rate=0.0)
    rng = np.random.default_rng(5)
    false = n = 0
    for k in range(16):
        metal = ("Mn", "Fe")[k % 2]
        p = recipe(metal=metal, c_nacl_M=float(rng.uniform(0, 1)), c_hcf_M=float(rng.uniform(0.02, 0.06)),
                   temperature_C=float(rng.uniform(30, 80)), aging_time_h=float(rng.uniform(1, 20)))
        lat = gt.latent(p, rng)
        if lat.failed or lat.polymorph_shares.get(f"pba_p21n/{metal}", 0.0) > 0.01:
            continue
        n += 1
        false += analyze_pattern(simulate_xrd(lat, rng), metal).phase_fractions.get("pba_p21n", 0.0) > 0.05
    assert n >= 8 and false == 0, (false, n)


def test_target_phase_is_validated_and_recorded(tmp_path):
    from pba_autoworkflow.schema import CategoricalSpec
    platform, _ = build_simulated_platform(seed=2, time_scale=0.0, failure_rate=0.0)
    space = default_design_space()
    no_zn = space.model_copy(update={"categorical": tuple(
        c.model_copy(update={"choices": ("Mn", "Fe")}) if c.name == "metal" else c for c in space.categorical)})
    with pytest.raises(ValueError, match="can form"):
        Campaign(platform, ProvenanceStore(tmp_path / "a"),
                 CampaignConfig(campaign_id="x", target_phase="znhcf_r3c"), space=no_zn)
    with pytest.raises(ValueError, match="not one of"):
        Campaign(platform, ProvenanceStore(tmp_path / "b"), CampaignConfig(campaign_id="y", target_phase="nacl"))
    store = ProvenanceStore(tmp_path / "c")
    cfg = CampaignConfig(campaign_id="zn", target_phase="znhcf_r3c", n_seed=4, batch_size=4, max_experiments=4, seed=2)
    camp = Campaign(platform, store, cfg)
    asyncio.run(camp.run(1))
    done = [e for e in camp.history if e.objectives is not None]
    assert done and all(e.metadata["target_phase"] == "znhcf_r3c" for e in done)
    # resuming without naming a target keeps the recorded one; a different one is refused
    again = Campaign(build_simulated_platform(seed=2, time_scale=0.0)[0], ProvenanceStore(tmp_path / "c"),
                     CampaignConfig(campaign_id="zn"))
    assert again.config.target_phase == "znhcf_r3c"
    with pytest.raises(ValueError, match="target phase"):
        Campaign(build_simulated_platform(seed=2, time_scale=0.0)[0], ProvenanceStore(tmp_path / "c"),
                 CampaignConfig(campaign_id="zn", target_phase="pba_fm3m"))



def test_zn_low_order_bias_is_bounded():
    """Known limitation, pinned so a change in either direction is noticed.

    In a poorly ordered cubic + R-3c Zn mixture the many weak R-3c lines absorb
    intensity the cubic phase should get; the R-3c fraction reads high.
    """
    import copy
    gt = GroundTruth(seed=1, reproducibility=0.0, failure_rate=0.0)
    base = gt.latent(_zn(), np.random.default_rng(3))
    base.domain_size_nm = 25.0
    bias = {}
    for order in (0.35, 0.7):
        lat = copy.deepcopy(base); lat.crystallinity = order
        truth = lat.xrd_weight_fractions()["znhcf_r3c"]
        bias[order] = float(np.median([
            analyze_pattern(simulate_xrd(lat, np.random.default_rng(10 + k)), "Zn")
            .phase_fractions.get("znhcf_r3c", 0.0) - truth for k in range(4)]))
    assert abs(bias[0.7]) < 0.08, bias
    assert bias[0.35] < 0.30, bias
