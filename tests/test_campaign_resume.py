# SPDX-License-Identifier: GPL-3.0-or-later
"""A campaign reloaded from its store keeps its iteration history."""

from __future__ import annotations

import asyncio

from pba_autoworkflow.campaign import Campaign, CampaignConfig
from pba_autoworkflow.devices.simulated import build_simulated_platform
from pba_autoworkflow.provenance import ProvenanceStore


def _campaign(root, **kw):
    platform, _ = build_simulated_platform(seed=3, time_scale=0.0, failure_rate=0.0)
    cfg = CampaignConfig(campaign_id="resume", n_seed=4, batch_size=4, max_iterations=2,
                         max_experiments=40, max_batch_failure_rate=1.0, seed=3, **kw)
    return Campaign(platform, ProvenanceStore(root), cfg)


def test_reloaded_campaign_restores_iterations_and_continues_numbering(tmp_path):
    root = tmp_path / "prov"
    first = _campaign(root)
    ran = asyncio.run(first.run(2))
    assert len(ran) == 2
    hv = [r.hypervolume for r in ran]
    first.store.close()

    # Rebuilding from the store (what `report` and `resume` do) used to start
    # with an empty iteration list: an empty progress panel, an empty
    # iterations.csv, and convergence detection that forgot every earlier batch.
    again = _campaign(root)
    assert [r.iteration for r in again.iterations] == [0, 1]
    assert [r.hypervolume for r in again.iterations] == hv
    assert again._hv_history == hv

    more = asyncio.run(again.run(1))
    assert [r.iteration for r in more] == [0, 1, 2], "resume continues the numbering"


def test_reactor_faults_do_not_leak_positions(tmp_path):
    """A fault while a vessel is in the reactor must free its position.

    Before the fix, the scheduler's reactor slot was released but the vessel
    stayed loaded, so after a few faults a healthy experiment failed with
    "all 4 positions full" -- one fault silently cost a second sample.
    """
    platform, _ = build_simulated_platform(seed=11, time_scale=0.0, failure_rate=0.0,
                                           mechanical_fault_rate=0.12, reactor_capacity=4)
    cfg = CampaignConfig(campaign_id="leak", n_seed=8, batch_size=8, max_iterations=3,
                         max_experiments=40, max_batch_failure_rate=1.0, seed=11)
    camp = Campaign(platform, ProvenanceStore(tmp_path / "prov"), cfg)
    asyncio.run(camp.run(3))
    errors = [e.error or "" for e in camp.history]
    assert any("interlock" in m or "clogged" in m or "fault" in m.lower() for m in errors), \
        "the test needs at least one hardware fault to be meaningful"
    assert not [m for m in errors if "positions full" in m], errors
    assert len(platform.reactor._loaded) == 0, "no vessel left behind in the reactor"


def test_restore_keeps_chronological_order_for_legacy_numbering(tmp_path):
    """Older stores logged every resumed run as iteration 0 again.

    Restoring must follow the event log, not the stored numbers: with a fixed
    reference point the dominated hypervolume of a growing data set cannot fall,
    so the restored trajectory must be non-decreasing.
    """
    root = tmp_path / "prov"
    c = _campaign(root)
    asyncio.run(c.run(1))
    c.store.close()
    c = _campaign(root)
    c.iterations = []                      # emulate the old behaviour: forget, restart at 0
    asyncio.run(c.run(1))
    c.store.close()

    again = _campaign(root)
    hv = [r.hypervolume for r in again.iterations]
    assert [r.iteration for r in again.iterations] == [0, 1]
    assert all(b >= a - 1e-12 for a, b in zip(hv, hv[1:])), hv


def test_simulated_noise_is_independent_of_python_hash_salt():
    """Same seed -> same simulated data in every process.

    The per-experiment noise stream used to be seeded with hash(), which Python
    salts per process, so no simulated campaign could be reproduced.
    """
    import os
    import subprocess
    import sys

    code = ("from pba_autoworkflow.devices.simulated import build_simulated_platform;"
            "_, be = build_simulated_platform(seed=7);"
            "print(be.rng_for('demo-b00-e00').random())")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": s}).stdout for s in ("1", "2", "3")}
    assert len(outs) == 1, outs


def test_campaign_recorded_with_other_objectives_is_refused(tmp_path):
    """A store scored on different objectives must not be mixed into the new loop."""
    import json, pytest
    from pba_autoworkflow import CampaignConfig, ProvenanceStore, build_simulated_platform
    from pba_autoworkflow.campaign import Campaign
    platform, _ = build_simulated_platform(seed=1, time_scale=0.0)
    cfg = CampaignConfig(campaign_id="old", n_seed=4, batch_size=4, max_experiments=4, seed=1)
    store = ProvenanceStore(tmp_path / "prov")
    camp = Campaign(platform, store, cfg)
    asyncio.run(camp.run(1))
    exp = next(e for e in camp.history if e.objectives is not None)
    exp.objectives.values = {"na_inventory": 0.5, "framework_integrity": 0.9}
    store.finalize_experiment(exp)
    with pytest.raises(ValueError, match="start a new campaign id"):
        Campaign(build_simulated_platform(seed=1, time_scale=0.0)[0], ProvenanceStore(tmp_path / "prov"), cfg)
