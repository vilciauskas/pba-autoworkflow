# SPDX-License-Identifier: GPL-3.0-or-later
"""Timekeeping, with timestamps and durations kept strictly apart.

A campaign is long-running: it spans suspends, resumes, NTP corrections and, on a
real platform, days.  Two different questions get asked of the clock and they need
different sources:

**"When did this happen?"** -- provenance timestamps, which must be comparable
across processes and meaningful to a human reading the database months later.
That is :func:`timestamp`, a wall-clock epoch value.

**"How long did this take?"** -- stage durations, station occupancy, queue waits.
These must never be computed from wall-clock differences.  A laptop that suspends
for a day mid-campaign, or an NTP step correction, makes wall-clock deltas
arbitrarily wrong: a batch that took thirty seconds is recorded as twenty-two
hours, the reported bottleneck becomes meaningless, and any timing-based
scheduling decision is corrupted.  :func:`tick` returns a monotonic counter that
cannot jump backwards or be adjusted, and :class:`Stopwatch` wraps the common
pattern.

This distinction is not hypothetical: it was introduced after a demonstration
campaign reported an 81 495 s iteration because the host suspended between two
batches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def timestamp() -> float:
    """Wall-clock epoch seconds.  For *recording when*, never for durations."""
    return time.time()


def tick() -> float:
    """Monotonic counter in seconds.  For *measuring how long*, never for records.

    The absolute value is meaningless and comparable only within this process.
    """
    return time.monotonic()


@dataclass
class Stopwatch:
    """Measures a duration monotonically while recording a wall-clock start.

    ``started_at`` is safe to store; ``duration_s`` is safe to compare.
    """

    started_at: float = field(default_factory=timestamp)
    _t0: float = field(default_factory=tick, repr=False)

    @property
    def duration_s(self) -> float:
        return tick() - self._t0

    def stop(self) -> tuple[float, float, float]:
        """Return ``(started_at, finished_at, duration_s)``.

        ``finished_at`` is derived by adding the monotonic duration to the
        wall-clock start rather than by reading the wall clock again, so a
        recorded interval is always internally consistent even if the system
        clock moved while the operation was running.
        """
        d = self.duration_s
        return self.started_at, self.started_at + d, d
