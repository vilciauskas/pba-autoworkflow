# SPDX-License-Identifier: GPL-3.0-or-later
"""Ion table, generated objectives, electrolyte rules, composition template,
and the descriptor kernel."""

from __future__ import annotations

import asyncio
import math
import tempfile

import numpy as np
import pytest

from pba_autoworkflow import ions as I
from pba_autoworkflow.analysis.echem import inserted_per_fe, predicted_separation_factor
from pba_autoworkflow.analysis.objectives import (
    compute_objectives,
    echem_plan,
    objective_catalogue,
    parse_objective,
    validate_objectives,
)
from pba_autoworkflow.composition import PBA
from pba_autoworkflow.devices.simulated import build_simulated_platform
from pba_autoworkflow.optimize.surrogate import MixedGP
from pba_autoworkflow.provenance import ProvenanceStore
from pba_autoworkflow.schema import (
    CompositionDescriptors,
    EchemCycleData,
    EchemDescriptors,
    Experiment,
    ExperimentStatus,
    ICPResult,
    SampleDescriptors,
    SynthesisParameters,
    charge_balanced_a,
    default_design_space,
    formula_weight,
)
from pba_autoworkflow.sim.electrochem import CHARGE, formal_potential, ion_shares
from pba_autoworkflow.workflow import ExperimentWorkflow, WorkflowConfig


def test_ion_table_values():
    """Shannon (VI) radii and Marcus (1991) hydration Gibbs energies."""
    assert I.ion("Na").radius_A == 1.02 and I.ion("K").radius_A == 1.38
    assert I.ion("Li").hydration_G_kJ == -475 and I.ion("Zn").hydration_G_kJ == -1955
    assert I.ion("Al").charge == 3 and not I.ion("NH4").icp_assayable
    assert I.ion("NH4").mass == pytest.approx(18.04, abs=0.01)


def test_pair_name_is_canonical():
    assert I.pair_name("Zn", "K") == I.pair_name("K", "Zn") == "K/Zn"
    assert I.pair_name("Na", "K") == "K/Na"
    assert I.pair_name("Mg", "Li") == "Li/Mg"
    assert I.pair_name("Al", "Ca") == "Ca/Al"


def test_objective_names_are_generated_for_every_ion():
    cat = objective_catalogue()
    for a in ("li", "nh4", "mg", "ca", "al"):
        assert f"{a}_retention" in cat and f"{a}_capacity" in cat
    assert "li_na_selectivity" in cat and "na_li_selectivity" in cat
    assert parse_objective("zn_tolerance") == ("selectivity", "Zn", "K")
    assert parse_objective("k_zn_selectivity") == ("selectivity", "K", "Zn")
    for bad in ("na_na_selectivity", "xx_retention", "capacity", "k_zn_selectivityy"):
        with pytest.raises(ValueError):
            validate_objectives([bad])


def test_echem_plan_for_new_pairs():
    single, mixed = echem_plan(["li_na_selectivity", "mg_retention"])
    assert set(single) == {"Li", "Na", "Mg"}
    assert mixed == ({"Li": 1.0, "Na": 0.01},)       # Na expected preferred -> dilute
    # earlier tuned mixtures are kept
    assert echem_plan(["k_zn_selectivity"])[1] == ({"Zn": 1.0, "K": 0.005},)


def test_reversed_pair_scores_are_complementary():
    ec = EchemDescriptors(separation_factor={"K/Zn": 1000.0})
    d = SampleDescriptors(echem=ec)
    p = SynthesisParameters(metal="Zn", c_metal_M=0.05, c_hcf_M=0.05, c_nacl_M=0.0, c_citrate_M=0.0,
                            ph=4.0, temperature_C=25.0, addition_rate_mL_min=1.0, aging_time_h=1.0,
                            stir_rate_rpm=600.0)
    o = compute_objectives(d, p, objectives=("k_zn_selectivity", "zn_k_selectivity", "zn_tolerance"))
    assert o.values["k_zn_selectivity"] == pytest.approx(0.875)
    assert o.values["zn_k_selectivity"] == pytest.approx(0.125) == o.values["zn_tolerance"]


def test_electrolyte_rules():
    r = I.electrolyte_recipe({"Zn": 1.0, "K": 0.005})
    assert r.anion == "sulfate" and r.salts_M["ZnSO4"] == 1.0 and r.salts_M["K2SO4"] == 0.0025
    r = I.electrolyte_recipe({"Ca": 1.0, "Na": 0.01})
    assert r.anion == "nitrate" and "Ca(NO3)2" in r.salts_M           # CaSO4 insoluble
    r = I.electrolyte_recipe({"Al": 1.0, "K": 0.005})
    assert r.max_pH == 3.5 and any("Al" in n for n in r.notes)
    with pytest.raises(ValueError):
        I.electrolyte_recipe({"Xx": 1.0})


def test_composition_template_reproduces_old_formula_weight_and_charge():
    # Na1.6 Mn [Fe(CN)6]0.9 . 2 H2O, computed by hand from atomic weights
    from pba_autoworkflow.schema import ATOMIC_WEIGHT as A
    hand = 1.6 * A["Na"] + A["Mn"] + 0.9 * (A["Fe"] + 6 * (A["C"] + A["N"])) + 2 * (2 * A["H"] + A["O"])
    assert formula_weight("Mn", 1.6, 0.1, 2.0) == pytest.approx(hand, rel=1e-12)
    assert formula_weight("Mn", 1.0, 0.1, 2.0, a_site={"Na": 1.0, "Li": 0.6}) == pytest.approx(
        hand - 0.6 * A["Na"] + 0.6 * A["Li"], rel=1e-12)
    assert charge_balanced_a(0.0, 1.0) == pytest.approx(2.0)
    assert charge_balanced_a(1 / 3, 0.0) == pytest.approx(0.0)
    assert PBA.a_charge({"Zn": 0.5, "Na": 0.2}) == pytest.approx(1.2)


def test_old_composition_records_get_an_a_site():
    c = CompositionDescriptors.model_validate(dict(
        na_per_fu=0.8, k_per_fu=0.1, fe_per_metal=0.9, vacancy_fraction=0.1, water_per_fu=2.0,
        formula="x"))
    assert c.a_site == {"Na": 0.8, "K": 0.1} and c.a_per_fu == pytest.approx(0.9)
    c2 = CompositionDescriptors.model_validate_json(c.model_dump_json())
    assert c2.a_site == c.a_site


def _cycle(q_last: float, electrolyte: dict) -> EchemCycleData:
    q = np.linspace(0, q_last, 20)
    return EchemCycleData(electrolyte_M=electrolyte, current_mA_g=100.0, reference="Zn2+/Zn",
                          charge_q=[q], charge_E=[np.ones(20)], discharge_q=[q],
                          discharge_E=[np.ones(20)])


def test_coulometric_difference_for_nh4_and_two_hidden_ions():
    icp = ICPResult(concentrations_mol_L={"Fe": 1e-3, "Na": 2e-4}, digest_mass_mg=5.0,
                    digest_volume_mL=10.0, dry_mass_mg=5.0)
    fw, fe = 400.0, 0.9
    q = 0.5 * fe * 96485.33212 / (3.6 * fw)            # 0.5 e per Fe
    x = inserted_per_fe(_cycle(q, {"Na": 1.0, "NH4": 0.01}), icp, "Mn", fw, fe)
    assert x["Na"] == pytest.approx(0.2) and x["NH4"] == pytest.approx(0.3, abs=1e-9)
    x = inserted_per_fe(_cycle(q, {"Zn": 1.0, "NH4": 0.01}), icp, "Zn", fw, fe)
    assert math.isnan(x["Zn"]) and math.isnan(x["NH4"])


@pytest.mark.parametrize("pair", [("Li", "Na"), ("Mg", "K"), ("Ca", "Zn"), ("Al", "Na")])
def test_predicted_alpha_matches_simulated_insertion_for_new_pairs(pair):
    a, b = I.pair_name(*pair).split("/")
    el = I.competitive_electrolyte(a, b)
    sh = ion_shares("pba", "Mn", el, {})
    x = {i: sh[i] / CHARGE[i] for i in sh}
    alpha_sim = (x[a] / x[b]) / (el[a] / el[b])
    e = {i: formal_potential("pba", i, "Mn", {}) for i in el}
    assert predicted_separation_factor(e[a], e[b], a, b, el) == pytest.approx(alpha_sim, rel=1e-9)


def test_simulated_potentials_order_and_anchors():
    e = {i: formal_potential("r3c", i, "Zn", {}) for i in I.IONS}
    assert (e["Zn"], e["Na"], e["K"]) == (1.70, 1.80, 1.92)          # unchanged anchors
    assert e["NH4"] > e["K"] > e["Na"] > e["Li"] and e["Al"] < e["Zn"]


def test_li_na_selectivity_runs_end_to_end():
    objs = ("crystallinity", "li_na_selectivity", "li_retention")
    single, mixed = echem_plan(objs)
    platform, _ = build_simulated_platform(seed=2, time_scale=0.0, failure_rate=0.0,
                                           mechanical_fault_rate=0.0, transport_fault_rate=0.0)
    p = SynthesisParameters(metal="Ni", c_metal_M=0.05, c_hcf_M=0.05, c_nacl_M=1.0, c_citrate_M=0.05,
                            ph=4.0, temperature_C=60.0, addition_rate_mL_min=0.5, aging_time_h=6.0,
                            stir_rate_rpm=600.0)
    exp = Experiment(experiment_id="l-b00-e00", campaign_id="l", batch_index=0, parameters=p)
    wf = ExperimentWorkflow(platform, ProvenanceStore(tempfile.mkdtemp()), WorkflowConfig(
        objectives=objs, echem_single_ions=single, echem_mixed_electrolytes=mixed))
    asyncio.run(wf.run(exp))
    assert exp.status is ExperimentStatus.COMPLETE, exp.error
    ec = exp.descriptors.echem
    a = ec.separation_factor["Na/Li"]
    assert 0.3 < a / ec.predicted_separation_factor["Na/Li"] < 3.0
    assert 0.0 < exp.objectives.values["li_na_selectivity"] < 0.5   # Na preferred over Li
    assert 0.0 < exp.objectives.values["li_retention"] <= 1.0


def test_echem_gate_skips_wrong_phase_samples():
    objs = ("target_phase_fraction", "zn_retention")
    single, mixed = echem_plan(objs)
    platform, _ = build_simulated_platform(seed=0, time_scale=0.0, failure_rate=0.0,
                                           mechanical_fault_rate=0.0, transport_fault_rate=0.0)
    p = SynthesisParameters(metal="Zn", hcf_precursor="K3FeCN6", c_metal_M=0.1, c_hcf_M=0.05,
                            c_nacl_M=0.0, c_citrate_M=0.0, ph=5.5, temperature_C=25.0,
                            addition_rate_mL_min=1.0, aging_time_h=1.0, stir_rate_rpm=600.0,
                            dry_temperature_C=20.0, dry_pressure_mbar=0.02)   # disordered cubic
    exp = Experiment(experiment_id="g-b00-e00", campaign_id="g", batch_index=0, parameters=p)
    wf = ExperimentWorkflow(platform, ProvenanceStore(tempfile.mkdtemp()), WorkflowConfig(
        target_phase="znhcf_r3c", objectives=objs, echem_single_ions=single,
        echem_mixed_electrolytes=mixed,
        echem_gate={"target_phase_fraction": 0.5, "crystallinity": 0.25}))
    asyncio.run(wf.run(exp))
    assert exp.status is ExperimentStatus.COMPLETE
    assert exp.descriptors.echem is None and "target phase fraction" in exp.metadata["echem_skipped"]
    assert exp.objectives.values["zn_retention"] == 0.0


def test_design_space_descriptors_and_kernel():
    sp = default_design_space()
    D = sp.categorical_descriptors()
    assert D["metal"].shape == (6, 4) and np.allclose(D["metal"].mean(0), 0)
    nc, blocks = len(sp.continuous), sp.categorical_blocks()
    gp = MixedGP(nc, blocks, categorical_descriptors=D)
    rng = np.random.default_rng(0)
    X = np.zeros((2, sp.dim))
    for n, cols in blocks:
        X[0, cols[0]] = X[1, cols[0]] = 1.0
    X[:, :nc] = rng.uniform(0, 1, nc)
    K = gp._kernel(X, X, np.ones(nc), 1.0, {n: 0.3 for n, _ in blocks})
    assert K[0, 1] == pytest.approx(K[0, 0])          # same levels -> full correlation
    # metal correlation falls with descriptor distance: Mn-Fe closer than Mn-Ni
    metals = sp.categorical[0].choices
    mi = blocks[0][1]
    def corr(m1, m2):
        A = np.zeros((2, sp.dim)); A[:, :nc] = 0.5
        for n, cols in blocks[1:]:
            A[:, cols[0]] = 1
        A[0, mi[metals.index(m1)]] = A[1, mi[metals.index(m2)]] = 1
        return gp._kernel(A, A, np.ones(nc), 1.0, {n: 0.3 for n, _ in blocks})[0, 1]
    assert corr("Mn", "Fe") > corr("Mn", "Ni")
