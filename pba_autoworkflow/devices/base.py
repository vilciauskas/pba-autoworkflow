# SPDX-License-Identifier: GPL-3.0-or-later
"""Device abstraction layer.

Every instrument on the platform is reached through one of the protocols below.
The orchestrator holds only these types, so a simulated deck and a real deck are
interchangeable: to go live you implement the same methods against Opentrons HTTP,
SiLA2, Modbus, a vendor SDK or a serial line, register the class, and change one
line of configuration.

Design rules that the concrete drivers must honour:

* **Async.** Instrument calls are I/O bound and often minutes long.  Every method
  is a coroutine so the scheduler can overlap a 20-minute ageing step on one
  reactor with a 2-minute diffraction scan on another sample.
* **Idempotent status.** ``status()`` must be safe to call at any time, including
  mid-operation, and must never block on the instrument's own lock.
* **Explicit failure.** Hardware problems raise :class:`DeviceError`; they never
  return a sentinel value.  The scheduler distinguishes retryable transport
  faults from terminal ones by exception subclass.
* **No chemistry.** A driver moves liquid, sets a temperature, or returns a
  trace.  Interpretation of that trace belongs to :mod:`pba_autoworkflow.analysis`.
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

from ..clock import timestamp
from ..schema import EchemCycleData, ICPResult, IRSpectrum, SynthesisParameters, XRDPattern


class DeviceError(RuntimeError):
    """Base class for all instrument faults."""

    retryable: bool = False


class TransportError(DeviceError):
    """Lost connection, timeout, bus error -- worth retrying."""

    retryable = True


class HardwareFault(DeviceError):
    """Physical fault that a retry will not fix (clog, crash, interlock)."""

    retryable = False


class ConsumableExhausted(DeviceError):
    """Out of tips, plates, or a stock solution.  Needs a human."""

    retryable = False


class DeviceState(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class DeviceStatus:
    device_id: str
    state: DeviceState
    detail: str = ""
    current_task: str | None = None
    updated_at: float = field(default_factory=timestamp)


@dataclass
class VesselHandle:
    """Identifies a physical sample as it moves between stations."""

    vessel_id: str
    experiment_id: str
    station: str = "unassigned"
    contents_mL: float = 0.0
    solid_present: bool = False


class Device(abc.ABC):
    """Common lifecycle for every instrument."""

    #: Number of samples this device can hold simultaneously.
    capacity: int = 1

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self._state = DeviceState.OFFLINE
        self._detail = "not connected"
        self._lock = asyncio.Semaphore(self.capacity)

    async def connect(self) -> None:
        """Open the transport and verify the instrument answers."""
        self._state = DeviceState.IDLE
        self._detail = "ready"

    async def disconnect(self) -> None:
        self._state = DeviceState.OFFLINE
        self._detail = "disconnected"

    async def status(self) -> DeviceStatus:
        return DeviceStatus(self.device_id, self._state, self._detail)

    async def reset(self) -> None:
        """Clear a recoverable error state (home axes, purge lines)."""
        if self._state is DeviceState.ERROR:
            self._state = DeviceState.IDLE
            self._detail = "reset"

    def _require_ready(self) -> None:
        if self._state in (DeviceState.OFFLINE, DeviceState.MAINTENANCE):
            raise TransportError(f"{self.device_id} is {self._state.value}")


# --------------------------------------------------------------------------- #
# Capability protocols
# --------------------------------------------------------------------------- #


@runtime_checkable
class LiquidHandler(Protocol):
    """Prepares precursor solutions and meters one into the other."""

    device_id: str

    async def prepare_solution(
        self, vessel: VesselHandle, components: dict[str, float], volume_mL: float
    ) -> VesselHandle:
        """Dispense a solution of the given molar composition into ``vessel``."""

    async def meter_into(
        self, source: VesselHandle, destination: VesselHandle,
        volume_mL: float, rate_mL_min: float,
    ) -> VesselHandle:
        """Transfer ``volume_mL`` at a controlled rate (the addition step)."""

    async def adjust_ph(self, vessel: VesselHandle, target_ph: float) -> float:
        """Titrate to the setpoint; return the pH actually achieved."""


@runtime_checkable
class Reactor(Protocol):
    """Temperature- and stir-controlled vessel where precipitation and ageing run."""

    device_id: str

    async def load(self, vessel: VesselHandle) -> None: ...

    async def set_conditions(self, vessel: VesselHandle, temperature_C: float,
                             stir_rate_rpm: float) -> None: ...

    async def hold(self, vessel: VesselHandle, duration_s: float) -> None:
        """Maintain conditions for the ageing period."""

    async def unload(self, vessel: VesselHandle) -> VesselHandle: ...


@runtime_checkable
class Workup(Protocol):
    """Centrifuge / filter / wash / dry train that isolates the powder."""

    device_id: str

    async def separate(self, vessel: VesselHandle) -> tuple[VesselHandle, VesselHandle]:
        """Return (solid handle, supernatant handle)."""

    async def wash(self, solid: VesselHandle, cycles: int = 3,
                   solvent: str = "water") -> VesselHandle: ...

    async def dry(self, solid: VesselHandle, temperature_C: float = 70.0,
                  duration_s: float = 3600.0, pressure_mbar: float = 1013.0,
                  gas: str = "ambient") -> VesselHandle:
        """Dry the solid at ``temperature_C`` and total pressure ``pressure_mbar``.

        ``gas`` is ``"ambient"`` (lab air, or the residual gas of a vacuum oven)
        or ``"dry"`` (desiccant such as P2O5, or a dry-gas purge).  The rate of
        water removal can select the polymorph, so a driver must deliver the
        pressure and gas it is given or raise.
        """

    async def weigh(self, solid: VesselHandle) -> float:
        """Dry mass in mg."""


@runtime_checkable
class Diffractometer(Protocol):
    device_id: str

    async def measure(self, solid: VesselHandle, two_theta_range: tuple[float, float],
                      step_deg: float, exposure_s: float) -> XRDPattern: ...


@runtime_checkable
class ElementalAnalyzer(Protocol):
    device_id: str

    async def measure(self, solid: VesselHandle,
                      elements: Sequence[str]) -> ICPResult:
        """Digest and assay a sample: the dried powder, a cycled electrode
        (``EchemCycleData.electrode``) or a spent electrolyte."""


@runtime_checkable
class IRSpectrometer(Protocol):
    """ATR-IR on a few mg of the dried powder (non-destructive)."""

    device_id: str

    async def measure(self, solid: VesselHandle,
                      wn_range_cm1: tuple[float, float] = (1950.0, 2300.0)) -> IRSpectrum: ...


@runtime_checkable
class Potentiostat(Protocol):
    """Electrode preparation plus galvanostatic cycling.

    Each call casts a fresh electrode from the powder, cycles it in the given
    electrolyte (charge first, ending on a discharge, i.e. with the electrolyte
    cations inserted) and returns the curves with handles to the cycled
    electrode and the spent electrolyte for elemental analysis.
    """

    device_id: str

    async def cycle(self, solid: VesselHandle, electrolyte_M: dict[str, float],
                    current_mA_g: float, n_cycles: int) -> EchemCycleData: ...


@dataclass
class Platform:
    """The bound set of devices the orchestrator drives."""

    liquid_handler: LiquidHandler
    reactor: Reactor
    workup: Workup
    diffractometer: Diffractometer
    elemental: ElementalAnalyzer
    #: optional stations: run when present
    ir: IRSpectrometer | None = None
    electrochem: Potentiostat | None = None

    def all_devices(self) -> list[Device]:
        seen: dict[str, Device] = {}
        for obj in (self.liquid_handler, self.reactor, self.workup,
                    self.diffractometer, self.elemental, self.ir, self.electrochem):
            if isinstance(obj, Device):
                seen[obj.device_id] = obj
        return list(seen.values())

    async def connect_all(self) -> None:
        await asyncio.gather(*(d.connect() for d in self.all_devices()))

    async def ensure_online(self) -> list[str]:
        """Connect any device that is not already up; return the ones connected.

        Idempotent, so it is safe to call at the top of every experiment.  That is
        deliberate: a device that dropped its transport mid-campaign is brought
        back before the next sample rather than failing every remaining
        experiment, and a campaign resumed in a fresh process does not need the
        caller to remember a separate connect step.
        """
        offline = [d for d in self.all_devices() if d._state is DeviceState.OFFLINE]
        if offline:
            await asyncio.gather(*(d.connect() for d in offline))
        return [d.device_id for d in offline]

    async def disconnect_all(self) -> None:
        await asyncio.gather(*(d.disconnect() for d in self.all_devices()))

    async def status_all(self) -> dict[str, DeviceStatus]:
        devices = self.all_devices()
        statuses = await asyncio.gather(*(d.status() for d in devices))
        return {s.device_id: s for s in statuses}

    def synthesis_recipe_check(self, p: SynthesisParameters) -> list[str]:
        """Static feasibility check before anything is committed to hardware."""
        problems: list[str] = []
        if p.addition_time_min > 8 * 60:
            problems.append("addition step exceeds 8 h deck reservation")
        if p.c_citrate_M > 3.0 * p.c_metal_M + 0.3:
            problems.append("citrate exceeds stock solubility budget")
        if p.total_volume_mL > 50.0:
            problems.append("vessel capacity exceeded")
        return problems
