# SPDX-License-Identifier: GPL-3.0-or-later
"""Concurrency control for a shared instrument deck.

A batch of experiments cannot simply be launched with ``asyncio.gather``: the
platform has one liquid handler, four reactor positions and one diffractometer,
and two coroutines that both believe they own the diffractometer will produce two
patterns of one sample and none of the other.

:class:`StationPool` gives each station a capacity-bounded, FIFO-fair set of slots
and records queueing statistics, which is what turns "the campaign felt slow"
into "the diffractometer was the bottleneck at 78 % occupancy while the reactor
block sat at 20 %".

:class:`BatchScheduler` runs a batch with a bounded number of samples in flight.
The bound is not cosmetic: with N experiments all started at once, every one of
them queues at the first station and the last sample's ageing step does not begin
until the deck has cleared, which lengthens total campaign time and -- worse --
leaves aged slurries waiting for workup, which changes the chemistry.  Limiting
in-flight samples to roughly the reactor capacity keeps the pipeline full without
stranding material.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Sequence

from .clock import tick, timestamp
from .devices.base import HardwareFault
from .schema import Experiment, ExperimentStatus


@dataclass
class StationStats:
    name: str
    capacity: int
    n_acquisitions: int = 0
    total_wait_s: float = 0.0
    total_busy_s: float = 0.0
    max_wait_s: float = 0.0
    max_concurrent: int = 0
    #: slots permanently taken out of service (e.g. a vessel stuck in a reactor position)
    n_retired: int = 0

    @property
    def mean_wait_s(self) -> float:
        return self.total_wait_s / self.n_acquisitions if self.n_acquisitions else 0.0

    def occupancy(self, wall_s: float) -> float:
        if wall_s <= 0 or self.capacity <= 0:
            return 0.0
        return self.total_busy_s / (wall_s * self.capacity)


class StationOutOfService(HardwareFault):
    """Every slot of a station has been retired; nothing can run there until a human intervenes."""


class _Slots:
    """FIFO-fair counting slots whose capacity can be reduced while slots are held.

    ``asyncio.Semaphore`` cannot shrink: retiring a slot means acquiring one and
    never releasing it, which deadlocks if the caller already holds the last free
    slot, and leaves queued waiters blocked forever once capacity reaches zero.
    Here capacity is a plain counter; a retired slot simply is not handed on when
    it is released, and when capacity reaches zero every waiter is failed.
    """

    def __init__(self, name: str, capacity: int) -> None:
        self.name = name
        self.capacity = capacity
        self.in_use = 0
        self._waiters: collections.deque[asyncio.Future[None]] = collections.deque()

    def _out_of_service(self) -> StationOutOfService:
        return StationOutOfService(f"{self.name}: all positions out of service")

    async def acquire(self) -> None:
        if self.capacity <= 0:
            raise self._out_of_service()
        if self.in_use < self.capacity and not self._waiters:
            self.in_use += 1
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut                     # resolved by _dispatch with the slot already counted
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                self.release()            # slot was handed over just as we were cancelled
            raise

    def release(self) -> None:
        self.in_use -= 1
        self._dispatch()

    def retire(self) -> None:
        self.capacity = max(0, self.capacity - 1)
        self._dispatch()

    def _dispatch(self) -> None:
        while self._waiters and self.in_use < self.capacity:
            fut = self._waiters.popleft()
            if not fut.done():
                self.in_use += 1
                fut.set_result(None)
        if self.capacity <= 0:
            while self._waiters:
                fut = self._waiters.popleft()
                if not fut.done():
                    fut.set_exception(self._out_of_service())


class StationPool:
    """Named, capacity-limited stations with queueing telemetry."""

    def __init__(self, capacities: dict[str, int]) -> None:
        self._slots = {k: _Slots(k, v) for k, v in capacities.items()}
        self._in_use = {k: 0 for k in capacities}
        self.stats = {k: StationStats(k, v) for k, v in capacities.items()}
        self._t_start = tick()

    def retire_slot(self, name: str) -> None:
        """Permanently take one slot of ``name`` out of service.

        Safe to call while the caller holds a slot of that station: the slot is
        withdrawn when it is released rather than by acquiring another one.
        Once every slot is retired, waiting and future acquisitions raise
        :class:`StationOutOfService` instead of blocking.
        """
        if name not in self._slots:
            raise KeyError(f"unknown station {name!r}; have {sorted(self._slots)}")
        self._slots[name].retire()
        self.stats[name].n_retired += 1

    def available_capacity(self, name: str) -> int:
        return self._slots[name].capacity

    @contextlib.asynccontextmanager
    async def acquire(self, name: str) -> AsyncIterator[None]:
        if name not in self._slots:
            raise KeyError(f"unknown station {name!r}; have {sorted(self._slots)}")
        st = self.stats[name]
        t_queued = tick()
        slots = self._slots[name]
        await slots.acquire()
        try:
            wait = tick() - t_queued
            st.n_acquisitions += 1
            st.total_wait_s += wait
            st.max_wait_s = max(st.max_wait_s, wait)
            self._in_use[name] += 1
            st.max_concurrent = max(st.max_concurrent, self._in_use[name])
            t_busy = tick()
            try:
                yield
            finally:
                st.total_busy_s += tick() - t_busy
                self._in_use[name] -= 1
        finally:
            slots.release()

    def wall_time_s(self) -> float:
        return tick() - self._t_start

    def report(self) -> list[dict[str, float | str | int]]:
        wall = self.wall_time_s()
        return [
            {
                "station": s.name,
                "capacity": s.capacity,
                "n_acquisitions": s.n_acquisitions,
                "total_busy_s": round(s.total_busy_s, 4),
                "occupancy": round(s.occupancy(wall), 4),
                "mean_wait_s": round(s.mean_wait_s, 3),
                "max_wait_s": round(s.max_wait_s, 3),
                "max_concurrent": s.max_concurrent,
                "retired": s.n_retired,
            }
            for s in sorted(self.stats.values(), key=lambda x: -x.total_busy_s)
        ]

    def bottleneck(self) -> str | None:
        wall = self.wall_time_s()
        if wall <= 0:
            return None
        ranked = sorted(self.stats.values(), key=lambda s: -s.occupancy(wall))
        return ranked[0].name if ranked and ranked[0].occupancy(wall) > 0.05 else None


def default_capacities(reactor_capacity: int = 4) -> dict[str, int]:
    return {
        "liquid_handler": 1,
        "reactor": reactor_capacity,
        "workup": 2,
        "diffractometer": 1,
        "spectrophotometer": 1,
        "elemental": 1,
    }


@dataclass
class BatchOutcome:
    experiments: list[Experiment]
    wall_time_s: float
    station_report: list[dict[str, float | str | int]] = field(default_factory=list)
    bottleneck: str | None = None

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.experiments:
            out[e.status.value] = out.get(e.status.value, 0) + 1
        return out


class BatchScheduler:
    """Runs a batch of experiments with bounded concurrency and a wall-clock cap."""

    def __init__(self, pool: StationPool, max_in_flight: int = 4,
                 per_experiment_timeout_s: float | None = None) -> None:
        self.pool = pool
        self.max_in_flight = max(1, int(max_in_flight))
        self.per_experiment_timeout_s = per_experiment_timeout_s

    async def run_batch(
        self,
        experiments: Sequence[Experiment],
        runner: Callable[[Experiment], Awaitable[Experiment]],
        on_complete: Callable[[Experiment], None] | None = None,
    ) -> BatchOutcome:
        gate = asyncio.Semaphore(self.max_in_flight)
        t0 = tick()

        async def one(exp: Experiment) -> Experiment:
            async with gate:
                try:
                    if self.per_experiment_timeout_s:
                        result = await asyncio.wait_for(
                            runner(exp), self.per_experiment_timeout_s
                        )
                    else:
                        result = await runner(exp)
                except asyncio.TimeoutError:
                    exp.status = ExperimentStatus.FAILED
                    exp.error = (
                        f"exceeded {self.per_experiment_timeout_s:.0f} s wall-clock "
                        "budget; sample abandoned on deck"
                    )
                    exp.finished_at = timestamp()
                    result = exp
                if on_complete is not None:
                    on_complete(result)
                return result

        results = await asyncio.gather(*(one(e) for e in experiments))
        return BatchOutcome(
            experiments=list(results),
            wall_time_s=tick() - t0,
            station_report=self.pool.report(),
            bottleneck=self.pool.bottleneck(),
        )
