# SPDX-License-Identifier: GPL-3.0-or-later
"""The hull touches the campaign only after measurement, and never for free."""

from __future__ import annotations

import math

import pytest

from pba_autoworkflow.schema import (CompositionDescriptors, Experiment, Objectives,
                            SampleDescriptors, SynthesisParameters, XRDDescriptors)
from pba_autoworkflow.thermo.advisor import MIN_SERIES, ThermoAdvisor
from pba_autoworkflow.thermo.hull import hull_from_energies
from pba_autoworkflow.thermo.prior import HullPrior
from pba_autoworkflow.thermo.structures import charge_balanced_na


def _hull(metal="Mn", interaction=0.6, ys=(0.0, 0.125, 0.25, 0.375, 0.5)):
    return hull_from_energies(metal, [
        {"vacancy_fraction": y, "na_per_fu": charge_balanced_na(y),
         "energy_per_fu_eV": -120.0 + 30.0 * y + interaction * y * (1 - y),
         "lattice_a_A": 10.2, "converged": True} for y in ys])


def _exp(metal, y, na, a=10.2, yield_=0.6):
    p = SynthesisParameters(metal=metal, c_metal_M=0.05, c_hcf_M=0.05,
                            c_nacl_M=1.0, c_citrate_M=0.02, ph=3.0,
                            temperature_C=60.0, addition_rate_mL_min=2.0,
                            aging_time_h=4.0, stir_rate_rpm=400.0)
    e = Experiment(experiment_id=f"x-{metal}-{y}", campaign_id="c", batch_index=0,
                   parameters=p)
    e.descriptors = SampleDescriptors(
        composition=CompositionDescriptors(
            na_per_fu=na, fe_per_metal=1.0 - y, vacancy_fraction=y,
            water_per_fu=6.0 * y, formula="x"),
        xrd=XRDDescriptors(lattice_a_A=a, domain_size_nm=30.0,
                           crystallinity_index=0.9, fwhm_200_deg=0.2,
                           phase="cubic", n_peaks_indexed=8, fit_residual=0.05),
        isolated_yield=yield_)
    e.objectives = Objectives(
        values={"na_inventory": na / 2.0, "framework_integrity": 1.0 - y},
        constraints={"isolated_yield": yield_}, feasible=True)
    return e


# --------------------------------------------------------------------------- #
# The penalty regression -- this is the one that matters most.
# --------------------------------------------------------------------------- #

def test_uninformative_prior_costs_exactly_as_much_as_no_prior():
    """An uncomputed metal must not be ranked below a computed-and-stable one.

    ``penalty`` used to return ``weight * (1 - 0.5) = 0.075`` whenever the prior
    had nothing to say -- for a metal with no hull at all, and for a hull whose
    deepest feature sits below the thermal scale.  A metal that was computed and
    found on its hull paid 0.000.  So inside the scalarizer, never having run a
    calculation was worth 0.075 of objective against you, silently.  That turns
    absence of evidence into evidence of absence, which is precisely the failure
    this module exists to prevent.
    """
    prior = HullPrior.from_hulls(_hull("Ni", interaction=0.6))
    assert prior.is_informative("Ni")

    # a metal with no hull at all
    assert prior.penalty("Cu", 0.25) == 0.0
    assert not prior.is_informative("Cu")

    # a metal whose hull is real but sub-thermal: same treatment
    flat = HullPrior.from_hulls(_hull("Mn", interaction=0.002))
    assert not flat.is_informative("Mn"), "a sub-kT feature must read as uninformative"
    assert flat.penalty("Mn", 0.25) == 0.0

    # and an informative prior still charges something off its own hull
    assert prior.penalty("Ni", 0.25) > 0.0


def test_stability_score_stays_neutral_where_penalty_is_zero():
    """0.5 from ``stability_score`` is 'no opinion', not 'middling'."""
    prior = HullPrior.from_hulls(_hull("Ni"))
    assert prior.stability_score("Cu", 0.3) == pytest.approx(0.5)
    assert prior.penalty("Cu", 0.3) == 0.0


# --------------------------------------------------------------------------- #
# The advisor
# --------------------------------------------------------------------------- #

def test_advisor_reports_uninformative_rather_than_inventing_a_verdict():
    """A sub-thermal hull yields no stability verdict, and says so."""
    flat = HullPrior.from_hulls(_hull("Mn", interaction=0.002))
    adv = ThermoAdvisor(flat).review(
        [_exp("Mn", y, charge_balanced_na(y)) for y in (0.0, 0.25, 0.375, 0.5)])
    assert adv.stability_verdict == "uninformative"
    assert any("nothing to falsify" in n for n in adv.notes)


def test_advisor_needs_enough_samples_before_it_will_speak():
    prior = HullPrior.from_hulls(_hull("Mn"))
    adv = ThermoAdvisor(prior).review([_exp("Mn", 0.25, 1.0)])
    assert adv.stability_verdict == "insufficient-data"
    assert adv.n_stability < MIN_SERIES


def test_advisor_catches_a_prior_that_ranks_backwards():
    """Feed measurements that improve exactly where the hull says less stable."""
    prior = HullPrior.from_hulls(_hull("Mn"))
    exps = []
    for y in (0.0, 0.125, 0.25, 0.375, 0.5):
        # objective deliberately anti-correlated with the prior's score
        obj = 1.0 - prior.stability_score("Mn", y)
        e = _exp("Mn", y, charge_balanced_na(y))
        e.objectives = Objectives(
            values={"na_inventory": obj, "framework_integrity": obj},
            constraints={"isolated_yield": 0.6}, feasible=True)
        exps.append(e)
    adv = ThermoAdvisor(prior).review(exps)
    assert adv.stability_verdict == "disagrees"
    assert adv.stability_correlation < 0


def test_lattice_check_compares_changes_not_absolute_edges():
    """Every model tested is 0.2-0.5 A off in absolute terms.

    A check on absolute cell edges would fail for all of them regardless of
    whether the *composition dependence* is right, which is the thing being
    tested.  Here the measured series is offset by a constant +0.5 A from the
    prediction but has identical shape, and must pass.
    """
    pred = {"Mn": {0.0: 9.98, 0.25: 10.04, 0.5: 10.02}}
    exps = [_exp("Mn", y, charge_balanced_na(y), a=pred["Mn"][y] + 0.5)
            for y in (0.0, 0.25, 0.5)]
    exps.append(_exp("Mn", 0.375, 0.5, a=10.53))
    adv = ThermoAdvisor(HullPrior.from_hulls(_hull("Mn")), pred).review(exps)
    assert adv.lattice_verdict == "agrees", adv.lattice_residual_A
    assert abs(adv.lattice_residual_A["Mn"]) <= 0.05


def test_lattice_check_flags_a_wrong_composition_response():
    """MACE's -0.35 A collapse against a flat measured series must be caught."""
    mace_like = {"Mn": {0.0: 10.035, 0.25: 9.685, 0.5: 9.914}}
    flat_measured = [_exp("Mn", y, charge_balanced_na(y), a=10.50)
                     for y in (0.0, 0.25, 0.375, 0.5)]
    adv = ThermoAdvisor(HullPrior.from_hulls(_hull("Mn")),
                        mace_like).review(flat_measured)
    assert adv.lattice_verdict == "disagrees"
    assert abs(adv.lattice_residual_A["Mn"]) > 0.05


def test_samples_with_undetermined_vacancy_fraction_are_excluded():
    """NaN means the assay could not determine it -- notably Fe without CHN.

    Such a sample must be dropped, never imputed: a fabricated composition
    training the comparison is the same defect as a fabricated measurement.
    """
    prior = HullPrior.from_hulls(_hull("Mn"))
    good = [_exp("Mn", y, charge_balanced_na(y)) for y in (0.0, 0.25, 0.5)]
    bad = _exp("Mn", 0.375, 0.5)
    bad.descriptors.composition.vacancy_fraction = float("nan")
    adv = ThermoAdvisor(prior).review(good + [bad])
    assert adv.n_stability == 3, "the NaN sample must not be counted"


def test_advisor_never_alters_what_the_planner_proposes():
    """The advisor has no method that returns or mutates a suggestion."""
    adv = ThermoAdvisor(HullPrior.from_hulls(_hull("Mn")))
    public = {n for n in dir(adv) if not n.startswith("_")}
    assert public == {"review", "prior", "predicted_lattice"}, (
        "ThermoAdvisor grew a surface beyond read-only review; a hull must not "
        "reach the planner, because it is indexed by a measured outcome")


def test_campaign_survives_a_broken_advisor(tmp_path):
    """An advisory check must not kill a batch that already cost reagents."""
    import asyncio

    from pba_autoworkflow.campaign import Campaign, CampaignConfig
    from pba_autoworkflow.devices.simulated import build_simulated_platform
    from pba_autoworkflow.provenance import ProvenanceStore

    class Exploding:
        def review(self, experiments):
            raise RuntimeError("boom")

    # ProvenanceStore takes a DIRECTORY root, not a sqlite path: passing
    # ":memory:" silently creates a folder of that name that survives the
    # test and makes the next run resume a spent campaign.
    store = ProvenanceStore(tmp_path / "prov")
    cfg = CampaignConfig(campaign_id="advisor-fault", n_seed=2, batch_size=2,
                         max_iterations=1, max_experiments=8)
    platform, _backend = build_simulated_platform(time_scale=0.0)
    camp = Campaign(platform, store, cfg,
                    thermo_advisor=Exploding())
    rec = asyncio.run(camp.run_iteration(0))
    assert "error" in rec.thermo_advice
    assert "boom" in rec.thermo_advice["error"]
    assert rec.n_suggested == 2, "the batch still ran"
