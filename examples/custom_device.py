# SPDX-License-Identifier: GPL-3.0-or-later
"""Template for a real instrument driver: a diffractometer behind a watch folder.

Many diffractometers are driven from vendor software that cannot be scripted
directly but can export each scan as a two-column ``.xy`` file.  This driver
treats the export folder as the instrument interface:

1. ``measure()`` asks for a scan of a given vessel (here: logs the request; on a
   real deck, send the job to the vendor software or a robot arm);
2. it waits for ``<vessel_id>.xy`` to appear in the export folder;
3. it parses the file into an ``XRDPattern`` and returns it.

Everything downstream (peak fitting, lattice constant, Williamson-Hall, QC
flags, provenance) is unchanged, because the orchestrator only sees the
``Diffractometer`` protocol.

    python examples/custom_device.py        # self-test with a synthetic scan
"""

from __future__ import annotations

import asyncio
import dataclasses
import tempfile
import time
from pathlib import Path

import numpy as np

from pba_autoworkflow.devices.base import (
    Device,
    DeviceState,
    HardwareFault,
    TransportError,
    VesselHandle,
)
from pba_autoworkflow.schema import XRDPattern


class WatchFolderDiffractometer(Device):
    """Diffractometer whose scans arrive as ``<vessel_id>.xy`` files."""

    capacity = 1                      # one sample stage

    def __init__(self, device_id: str, export_dir: str | Path,
                 wavelength_A: float = 1.5406, timeout_s: float = 3600.0,
                 poll_s: float = 2.0) -> None:
        super().__init__(device_id)
        self.export_dir = Path(export_dir)
        self.wavelength_A = wavelength_A
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    async def connect(self) -> None:
        # A missing share is a transport problem: worth retrying, not a sample failure.
        if not self.export_dir.is_dir():
            raise TransportError(f"{self.device_id}: export folder {self.export_dir} not reachable")
        await super().connect()

    async def measure(self, solid: VesselHandle, two_theta_range: tuple[float, float],
                      step_deg: float, exposure_s: float) -> XRDPattern:
        self._require_ready()
        self._state = DeviceState.BUSY
        try:
            self._request_scan(solid, two_theta_range, step_deg, exposure_s)
            path = await self._wait_for(self.export_dir / f"{solid.vessel_id}.xy")
            tt, inten = self._read_xy(path)
        finally:
            self._state = DeviceState.IDLE
        lo, hi = two_theta_range
        keep = (tt >= lo) & (tt <= hi)
        if keep.sum() < 50:
            raise HardwareFault(f"{self.device_id}: scan {path.name} has only "
                                f"{int(keep.sum())} points in {lo}-{hi} deg")
        return XRDPattern(two_theta_deg=tt[keep], intensity=inten[keep],
                          wavelength_A=self.wavelength_A, exposure_s=exposure_s)

    # -- instrument-specific parts: replace these for your hardware ----------- #
    def _request_scan(self, solid, two_theta_range, step_deg, exposure_s) -> None:
        print(f"[{self.device_id}] scan requested: {solid.vessel_id} "
              f"{two_theta_range[0]}-{two_theta_range[1]} deg, step {step_deg}, {exposure_s} s")

    async def _wait_for(self, path: Path) -> Path:
        t0 = time.monotonic()
        while not path.exists():
            if time.monotonic() - t0 > self.timeout_s:
                raise TransportError(f"{self.device_id}: no {path.name} after {self.timeout_s:.0f} s")
            await asyncio.sleep(self.poll_s)
        # wait until the exporting program has finished writing the file
        size = -1
        while path.stat().st_size != size:
            size = path.stat().st_size
            await asyncio.sleep(min(self.poll_s, 0.5))
        return path

    @staticmethod
    def _read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
        data = np.loadtxt(path, comments=("#", "'", ";"), usecols=(0, 1))
        if data.ndim != 2 or data.shape[0] < 2:
            raise HardwareFault(f"unreadable scan file {path.name}")
        return data[:, 0], data[:, 1]


def use_on_platform(platform, export_dir):
    """Return a copy of ``platform`` with this driver in place of its diffractometer."""
    return dataclasses.replace(
        platform, diffractometer=WatchFolderDiffractometer("xrd-01", export_dir))


async def _self_test() -> None:
    from pba_autoworkflow.analysis.xrd import analyze_pattern

    with tempfile.TemporaryDirectory() as d:
        dev = WatchFolderDiffractometer("xrd-01", d, poll_s=0.1)
        await dev.connect()
        vessel = VesselHandle("demo-b00-e00-S", "demo-b00-e00")

        async def vendor_software_exports_a_scan():
            # stands in for the instrument: a cubic PBA, a = 10.20 A, Cu K-alpha
            await asyncio.sleep(0.3)
            tt = np.arange(10.0, 60.0, 0.02)
            y = 2500.0 * np.exp(-tt / 18.0) + 100.0
            a, lam = 10.20, 1.5406
            for (h, k, l), amp in {(2, 0, 0): 6000, (2, 2, 0): 3500, (4, 0, 0): 1300,
                                   (4, 2, 0): 1900, (4, 2, 2): 1200, (4, 4, 0): 600}.items():
                d_hkl = a / np.sqrt(h * h + k * k + l * l)
                c = 2 * np.degrees(np.arcsin(lam / (2 * d_hkl)))
                y += amp * np.exp(-0.5 * ((tt - c) / 0.06) ** 2)
            y = np.random.default_rng(0).poisson(y).astype(float)
            np.savetxt(Path(d) / f"{vessel.vessel_id}.xy", np.c_[tt, y], header="2theta counts")

        pattern, _ = await asyncio.gather(dev.measure(vessel, (10.0, 60.0), 0.02, 1.0),
                                          vendor_software_exports_a_scan())
        desc = analyze_pattern(pattern)
        print(f"parsed {pattern.two_theta_deg.size} points -> a = {desc.lattice_a_A:.4f} A, "
              f"{desc.n_peaks_indexed} reflections indexed, phase {desc.phase}")


if __name__ == "__main__":
    asyncio.run(_self_test())
