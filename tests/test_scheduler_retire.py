# SPDX-License-Identifier: GPL-3.0-or-later
"""Retiring station slots: a vessel stuck in a reactor position.

The position is physically occupied until someone clears it, so the scheduler
must stop offering it -- without deadlocking the experiment that discovered the
fault, and without hanging the campaign once every position is gone.
"""
from __future__ import annotations

import asyncio

import pytest

from pba_autoworkflow import CampaignConfig, ProvenanceStore, build_simulated_platform
from pba_autoworkflow.campaign import Campaign
from pba_autoworkflow.devices.base import HardwareFault
from pba_autoworkflow.scheduler import StationOutOfService, StationPool


def _run(coro, timeout=5.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


def test_retire_while_holding_the_last_slot_does_not_deadlock():
    async def main():
        pool = StationPool({"rx": 1})
        async with pool.acquire("rx"):
            pool.retire_slot("rx")          # caller still holds the only slot
        assert pool.available_capacity("rx") == 0
        with pytest.raises(StationOutOfService):
            async with pool.acquire("rx"):
                pass
        assert pool.report()[0]["retired"] == 1
    _run(main())


def test_retired_slot_lowers_concurrency_and_queue_stays_fifo():
    async def main():
        pool = StationPool({"rx": 2})
        pool.retire_slot("rx")
        order, live, peak = [], 0, 0

        async def job(i):
            nonlocal live, peak
            async with pool.acquire("rx"):
                live += 1; peak = max(peak, live); order.append(i)
                await asyncio.sleep(0.01)
                live -= 1

        await asyncio.gather(*(job(i) for i in range(5)))
        assert peak == 1
        assert order == list(range(5))
    _run(main())


def test_waiters_fail_instead_of_hanging_when_last_slot_is_retired():
    async def main():
        pool = StationPool({"rx": 1})
        entered = asyncio.Event()

        async def holder():
            async with pool.acquire("rx"):
                entered.set()
                await asyncio.sleep(0.02)
                pool.retire_slot("rx")

        async def waiter():
            await entered.wait()
            async with pool.acquire("rx"):
                return "ran"

        results = await asyncio.gather(holder(), waiter(), return_exceptions=True)
        assert isinstance(results[1], StationOutOfService)
    _run(main())


def test_cancelled_waiter_does_not_leak_a_slot():
    async def main():
        pool = StationPool({"rx": 1})
        async with pool.acquire("rx"):
            t = asyncio.create_task(pool.acquire("rx").__aenter__())
            await asyncio.sleep(0)
            t.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t
        async with pool.acquire("rx"):      # must not block
            pass
        assert pool._slots["rx"].in_use == 0
    _run(main())


def test_stuck_vessels_retire_reactor_positions_and_campaign_terminates(tmp_path):
    """Every reactor run faults and the vessel cannot be ejected.

    Each stuck vessel must retire one position; once all are gone the remaining
    experiments fail fast with a clear reason, and the campaign returns.
    """
    platform, _ = build_simulated_platform(seed=3, time_scale=0.0, failure_rate=0.0,
                                           mechanical_fault_rate=0.0, reactor_capacity=2)

    async def hold(*_a, **_k):
        raise HardwareFault("rx-01: over-temperature interlock tripped")

    async def unload(*_a, **_k):
        raise HardwareFault("rx-01: gripper cannot release vessel")

    platform.reactor.hold = hold
    platform.reactor.unload = unload
    cfg = CampaignConfig(campaign_id="stuck", n_seed=6, batch_size=6, max_iterations=1,
                         max_experiments=6, max_batch_failure_rate=1.0, seed=3,
                         reactor_capacity=2)
    camp = Campaign(platform, ProvenanceStore(tmp_path / "prov"), cfg)
    _run(camp.run(1), timeout=120)

    errors = [e.error or "" for e in camp.history]
    assert len(errors) == 6
    assert sum(bool(e.metadata.get("reactor_vessel_stuck")) for e in camp.history) == 2
    assert camp.pool.stats["reactor"].n_retired == 2
    # the recipe check may reject some proposals before they reach the reactor;
    # every other experiment either stuck a vessel or was refused a position
    reached = [m for m in errors if not m.startswith("infeasible recipe")]
    assert sum("all positions out of service" in m for m in reached) == len(reached) - 2, errors
    assert len(reached) > 2, "the test needs experiments after the block is exhausted"


def test_analysis_version_follows_package_version():
    import pba_autoworkflow
    from pba_autoworkflow.provenance import ANALYSIS_VERSION
    assert ANALYSIS_VERSION == pba_autoworkflow.__version__ != "unknown"
