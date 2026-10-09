# SPDX-License-Identifier: GPL-3.0-or-later
"""Simulated drivers implementing the device protocols.

A :class:`SimulatedBackend` holds the hidden chemistry and the per-vessel state
that the individual drivers consult.  Nothing above the device layer touches it.

Wall-clock behaviour is compressed by ``time_scale``: an ageing step of 6 h is
executed as ``6*3600*time_scale`` seconds of real waiting, so a full campaign runs
in seconds during development and can be set to 1.0 for a timing rehearsal.  The
compression is applied uniformly, which means scheduler contention -- the reason
the platform needs a scheduler at all -- is preserved in miniature.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from ..clock import timestamp
from ..schema import (
    HCF_PRECURSORS,
    EchemCycleData,
    ICPResult,
    IRSpectrum,
    METALS,
    SynthesisParameters,
    XRDPattern,
)
from ..sim.electrochem import simulate_cycling
from ..sim.ground_truth import GroundTruth, LatentState
from ..sim.instruments import (
    simulate_gravimetric_yield,
    simulate_icp,
    simulate_ir,
    simulate_xrd,
)
from .base import (
    ConsumableExhausted,
    Device,
    DeviceState,
    HardwareFault,
    Platform,
    TransportError,
    VesselHandle,
)


@dataclass
class _VesselRecord:
    """What the deck has actually done to one vessel.

    The simulator reconstructs the effective recipe from these commanded values
    rather than being handed the intended recipe.  That is deliberate: if the
    workflow dispenses the wrong volume or forgets to ramp the reactor, the
    simulated chemistry responds to what was *done*, and the bug shows up as an
    anomalous result instead of being silently masked.
    """

    components: dict[str, float] = field(default_factory=dict)
    volume_A_mL: float = 0.0
    volume_B_mL: float = 0.0
    metal: str | None = None
    ph_actual: float | None = None
    temperature_C: float | None = None
    stir_rate_rpm: float | None = None
    addition_rate_mL_min: float | None = None
    aging_time_h: float | None = None
    hcf_precursor: str | None = None

    latent: LatentState | None = None
    supernatant_of: str | None = None
    weighed_mass_mg: float | None = None
    theoretical_mass_mg: float | None = None

    @property
    def params(self) -> SynthesisParameters | None:
        """Effective recipe implied by the commands issued so far."""
        if self.metal is None or self.volume_A_mL <= 0 or self.volume_B_mL <= 0:
            return None
        if self.temperature_C is None or self.aging_time_h is None:
            return None
        # Precursor from what was dispensed into this vessel (solution B's
        # components are merged into A by the metered addition).
        hcf = self.hcf_precursor or next((k for k in HCF_PRECURSORS if k in self.components),
                                         "Na4FeCN6")
        try:
            # Drying is not part of this: it is applied to the solid when the
            # workup's dry step runs, with the conditions that step was given.
            return SynthesisParameters(
                metal=self.metal,  # type: ignore[arg-type]
                hcf_precursor=hcf,  # type: ignore[arg-type]
                c_metal_M=self.components.get(self.metal, 0.0),
                c_hcf_M=self.components.get(hcf, 0.0),
                c_nacl_M=self.components.get("NaCl", 0.0),
                c_citrate_M=self.components.get("citrate", 0.0),
                ph=self.ph_actual if self.ph_actual is not None else 7.0,
                temperature_C=self.temperature_C,
                addition_rate_mL_min=self.addition_rate_mL_min or 1.0,
                aging_time_h=self.aging_time_h,
                stir_rate_rpm=self.stir_rate_rpm or 0.0,
                volume_A_mL=self.volume_A_mL,
                volume_B_mL=self.volume_B_mL,
            )
        except Exception:
            return None


@dataclass
class SimulatedBackend:
    """Shared hidden state for the simulated platform."""

    ground_truth: GroundTruth
    time_scale: float = 1e-3
    transport_fault_rate: float = 0.0
    mechanical_fault_rate: float = 0.0
    seed: int = 0
    _rng: np.random.Generator = field(init=False)
    vessels: dict[str, _VesselRecord] = field(default_factory=dict, init=False)
    tips_remaining: int = 4000
    call_log: list[tuple[float, str, str]] = field(default_factory=list, init=False)
    #: cycled electrodes and spent electrolytes, for the elemental analyzer
    echem_samples: dict[str, dict] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def rng_for(self, experiment_id: str) -> np.random.Generator:
        """Per-experiment stream: repeated characterization of one sample is
        consistent, but two experiments never share noise."""
        # A stable digest, not hash(): str hashing is salted per process
        # (PYTHONHASHSEED), which made the same seed give different simulated
        # data in every new process and broke run-to-run reproducibility.
        digest = hashlib.blake2b(f"{self.seed}:{experiment_id}".encode(), digest_size=8)
        return np.random.default_rng(int.from_bytes(digest.digest(), "little"))

    def record(self, vessel_id: str) -> _VesselRecord:
        return self.vessels.setdefault(vessel_id, _VesselRecord())

    async def dwell(self, nominal_s: float, device_id: str, task: str) -> None:
        self.call_log.append((timestamp(), device_id, task))
        await asyncio.sleep(max(0.0, nominal_s * self.time_scale))

    def maybe_transport_fault(self, device_id: str) -> None:
        """Retryable communication glitch: the operation itself never started."""
        if self.transport_fault_rate > 0 and self._rng.random() < self.transport_fault_rate:
            raise TransportError(f"{device_id}: simulated bus timeout")

    def maybe_mechanical_fault(self, device_id: str, mode: str) -> None:
        """Terminal hardware fault, raised by the device that would suffer it.

        Kept separate from the chemistry: a clogged tip or an over-temperature
        abort tells you nothing about the product, and the orchestrator must
        distinguish "this recipe is bad" from "this instrument needs attention".
        """
        if self.mechanical_fault_rate > 0 and self._rng.random() < self.mechanical_fault_rate:
            raise HardwareFault(f"{device_id}: {mode}")


class _SimDevice(Device):
    def __init__(self, device_id: str, backend: SimulatedBackend) -> None:
        super().__init__(device_id)
        self.backend = backend


class SimulatedLiquidHandler(_SimDevice):
    """Eight-channel dispensing deck."""

    capacity = 1

    async def prepare_solution(
        self, vessel: VesselHandle, components: dict[str, float], volume_mL: float
    ) -> VesselHandle:
        self._require_ready()
        self.backend.maybe_transport_fault(self.device_id)
        n_aspirations = max(1, len(components))
        if self.backend.tips_remaining < n_aspirations:
            raise ConsumableExhausted(f"{self.device_id}: tip box empty")
        self.backend.maybe_mechanical_fault(self.device_id, "tip clogged during aspiration")
        self.backend.tips_remaining -= n_aspirations
        self._state = DeviceState.BUSY
        try:
            await self.backend.dwell(18.0 * n_aspirations, self.device_id,
                                     f"prepare:{vessel.vessel_id}")
        finally:
            self._state = DeviceState.IDLE
        rec = self.backend.record(vessel.vessel_id)
        rec.components.update({k: float(v) for k, v in components.items()})
        rec.volume_A_mL = float(volume_mL)
        for key in components:
            if key in METALS:
                rec.metal = key
            if key in HCF_PRECURSORS:
                rec.hcf_precursor = key
        vessel.contents_mL = volume_mL
        return vessel

    async def meter_into(
        self, source: VesselHandle, destination: VesselHandle,
        volume_mL: float, rate_mL_min: float,
    ) -> VesselHandle:
        self._require_ready()
        self.backend.maybe_transport_fault(self.device_id)
        duration_s = 60.0 * volume_mL / max(rate_mL_min, 1e-6)
        self._state = DeviceState.BUSY
        try:
            await self.backend.dwell(duration_s, self.device_id,
                                     f"meter:{destination.vessel_id}")
        finally:
            self._state = DeviceState.IDLE
        src = self.backend.record(source.vessel_id)
        dst = self.backend.record(destination.vessel_id)
        # Solution B's solutes are now in the reaction vessel; concentrations stay
        # expressed per original solution, with the two volumes recorded so the
        # chemistry model can do its own mixing arithmetic.
        for key, value in src.components.items():
            dst.components.setdefault(key, float(value))
        dst.volume_B_mL = float(volume_mL)
        dst.addition_rate_mL_min = float(rate_mL_min)
        destination.contents_mL += volume_mL
        source.contents_mL = max(0.0, source.contents_mL - volume_mL)
        return destination

    async def adjust_ph(self, vessel: VesselHandle, target_ph: float) -> float:
        self._require_ready()
        self._state = DeviceState.BUSY
        try:
            await self.backend.dwell(45.0, self.device_id, f"ph:{vessel.vessel_id}")
        finally:
            self._state = DeviceState.IDLE
        # Titration overshoot: the achieved pH is what the chemistry actually sees.
        achieved = float(target_ph + self.backend.rng_for(vessel.experiment_id)
                         .normal(0.0, 0.06))
        self.backend.record(vessel.vessel_id).ph_actual = achieved
        return achieved


class SimulatedReactor(_SimDevice):
    """Four-position heater-stirrer block."""

    capacity = 4

    def __init__(self, device_id: str, backend: SimulatedBackend,
                 capacity: int = 4) -> None:
        self.capacity = capacity
        super().__init__(device_id, backend)
        self._loaded: dict[str, VesselHandle] = {}
        self._conditions: dict[str, tuple[float, float]] = {}

    async def load(self, vessel: VesselHandle) -> None:
        self._require_ready()
        if len(self._loaded) >= self.capacity:
            raise HardwareFault(f"{self.device_id}: all {self.capacity} positions full")
        await self.backend.dwell(20.0, self.device_id, f"load:{vessel.vessel_id}")
        self._loaded[vessel.vessel_id] = vessel
        vessel.station = self.device_id

    async def set_conditions(self, vessel: VesselHandle, temperature_C: float,
                             stir_rate_rpm: float) -> None:
        self._require_ready()
        if vessel.vessel_id not in self._loaded:
            raise HardwareFault(f"{self.device_id}: {vessel.vessel_id} not loaded")
        self.backend.maybe_mechanical_fault(
            self.device_id, "over-temperature interlock tripped during ramp")
        ramp_s = abs(temperature_C - 25.0) * 12.0
        await self.backend.dwell(ramp_s, self.device_id, f"ramp:{vessel.vessel_id}")
        self._conditions[vessel.vessel_id] = (temperature_C, stir_rate_rpm)
        rec = self.backend.record(vessel.vessel_id)
        rec.temperature_C = float(temperature_C)
        rec.stir_rate_rpm = float(stir_rate_rpm)

    async def hold(self, vessel: VesselHandle, duration_s: float) -> None:
        self._require_ready()
        self._state = DeviceState.BUSY if len(self._loaded) else DeviceState.IDLE
        await self.backend.dwell(duration_s, self.device_id, f"age:{vessel.vessel_id}")
        rec = self.backend.record(vessel.vessel_id)
        rec.aging_time_h = float(duration_s / 3600.0)
        effective = rec.params
        if effective is not None and rec.latent is None:
            # The chemistry resolves here: this is where the solid actually forms,
            # under the conditions the deck actually delivered.
            rec.latent = self.backend.ground_truth.precipitate(
                effective, self.backend.rng_for(vessel.experiment_id)
            )
            vessel.solid_present = not rec.latent.failed
        if len(self._loaded) <= 1:
            self._state = DeviceState.IDLE

    async def unload(self, vessel: VesselHandle) -> VesselHandle:
        self._loaded.pop(vessel.vessel_id, None)
        self._conditions.pop(vessel.vessel_id, None)
        await self.backend.dwell(20.0, self.device_id, f"unload:{vessel.vessel_id}")
        vessel.station = "transit"
        return vessel


class SimulatedWorkup(_SimDevice):
    """Centrifuge plus wash/dry train."""

    capacity = 2

    async def separate(self, vessel: VesselHandle) -> tuple[VesselHandle, VesselHandle]:
        self._require_ready()
        rec = self.backend.record(vessel.vessel_id)
        await self.backend.dwell(600.0, self.device_id, f"spin:{vessel.vessel_id}")
        if rec.latent is not None and rec.latent.failed:
            if rec.latent.failure_mode == "pellet_lost_in_decant":
                raise HardwareFault(f"{self.device_id}: pellet lost during decant")
            if rec.latent.failure_mode == "insufficient_solid":
                raise HardwareFault(
                    f"{self.device_id}: no recoverable solid "
                    f"({rec.latent.solid_mass_mg:.2f} mg)"
                )
        solid = VesselHandle(f"{vessel.vessel_id}-S", vessel.experiment_id,
                             station=self.device_id, solid_present=True)
        liquid = VesselHandle(f"{vessel.vessel_id}-L", vessel.experiment_id,
                              station=self.device_id,
                              contents_mL=vessel.contents_mL)
        # Solid and supernatant both inherit the reaction vessel's history, so the
        # effective recipe stays attached to each downstream measurement.
        self.backend.vessels[solid.vessel_id] = rec
        sup = replace(rec, supernatant_of=vessel.vessel_id,
                      components=dict(rec.components))
        self.backend.vessels[liquid.vessel_id] = sup
        return solid, liquid

    async def wash(self, solid: VesselHandle, cycles: int = 3,
                   solvent: str = "water") -> VesselHandle:
        self._require_ready()
        await self.backend.dwell(420.0 * cycles, self.device_id,
                                 f"wash:{solid.vessel_id}")
        return solid

    async def dry(self, solid: VesselHandle, temperature_C: float = 70.0,
                  duration_s: float = 3600.0, pressure_mbar: float = 1013.0,
                  gas: str = "ambient") -> VesselHandle:
        self._require_ready()
        await self.backend.dwell(duration_s, self.device_id, f"dry:{solid.vessel_id}")
        rec = self.backend.record(solid.vessel_id)
        if rec.latent is not None:
            # The solid changes here, under the conditions actually commanded.
            self.backend.ground_truth.dry(
                rec.latent, temperature_C, pressure_mbar, gas,
                self.backend.rng_for(solid.experiment_id + ":dry"))
        return solid

    async def weigh(self, solid: VesselHandle) -> float:
        self._require_ready()
        rec = self.backend.record(solid.vessel_id)
        if rec.latent is None or rec.params is None:
            raise HardwareFault(f"{self.device_id}: nothing to weigh")
        await self.backend.dwell(60.0, self.device_id, f"weigh:{solid.vessel_id}")
        mass, theo = simulate_gravimetric_yield(
            rec.latent, rec.params, self.backend.rng_for(solid.experiment_id)
        )
        rec.weighed_mass_mg = mass
        rec.theoretical_mass_mg = theo
        return mass


class SimulatedDiffractometer(_SimDevice):
    capacity = 1

    async def measure(self, solid: VesselHandle, two_theta_range: tuple[float, float],
                      step_deg: float, exposure_s: float) -> XRDPattern:
        self._require_ready()
        self.backend.maybe_transport_fault(self.device_id)
        rec = self.backend.record(solid.vessel_id)
        if rec.latent is None:
            raise HardwareFault(f"{self.device_id}: no sample mounted")
        n_steps = (two_theta_range[1] - two_theta_range[0]) / step_deg
        async with self._lock:
            self._state = DeviceState.BUSY
            try:
                await self.backend.dwell(n_steps * step_deg * 60.0 + exposure_s,
                                         self.device_id, f"xrd:{solid.vessel_id}")
            finally:
                self._state = DeviceState.IDLE
        return simulate_xrd(
            rec.latent, self.backend.rng_for(solid.experiment_id),
            two_theta_range=two_theta_range, step_deg=step_deg,
            exposure_s=exposure_s,
        )


class SimulatedElementalAnalyzer(_SimDevice):
    capacity = 1

    async def measure(self, solid: VesselHandle,
                      elements: Sequence[str]) -> ICPResult:
        self._require_ready()
        sample = self.backend.echem_samples.get(solid.vessel_id)
        if sample is not None:
            async with self._lock:
                await self.backend.dwell(900.0, self.device_id, f"icp:{solid.vessel_id}")
            return _icp_of_echem_sample(sample, elements,
                                        self.backend.rng_for(solid.vessel_id + ":icp"))
        rec = self.backend.record(solid.vessel_id)
        if rec.latent is None or rec.params is None:
            raise HardwareFault(f"{self.device_id}: no digest available")
        async with self._lock:
            await self.backend.dwell(900.0, self.device_id, f"icp:{solid.vessel_id}")
        return simulate_icp(rec.latent, rec.params,
                            self.backend.rng_for(solid.experiment_id), elements=elements)


def _icp_of_echem_sample(sample: dict, elements: Sequence[str],
                         rng: np.random.Generator) -> ICPResult:
    """Digest of a cycled electrode, or the spent electrolyte (Fe only matters)."""
    vol_mL = 10.0
    if sample["kind"] == "electrode":
        n_fu = sample["active_mass_mg"] * 1e-3 / sample["formula_weight"]
        mol = {"Fe": n_fu * sample["fe_per_fu"]}
        mol[sample["metal"]] = mol.get(sample["metal"], 0.0) + n_fu
        for ion, x in sample["inserted"].items():
            mol[ion] = mol.get(ion, 0.0) + n_fu * x
        mass = sample["active_mass_mg"]
    else:
        mol = {"Fe": sample["fe_mol"]}
        vol_mL = sample["volume_mL"]
        mass = float("nan")
    conc = {el: float(max(0.0, m / (vol_mL * 1e-3) * (1.0 + rng.normal(0.0, 0.02))))
            for el, m in mol.items() if el in elements}
    return ICPResult(concentrations_mol_L=conc, digest_mass_mg=mass,
                     digest_volume_mL=vol_mL, dry_mass_mg=mass)


class SimulatedIRSpectrometer(_SimDevice):
    capacity = 1

    async def measure(self, solid: VesselHandle,
                      wn_range_cm1: tuple[float, float] = (1950.0, 2300.0)) -> IRSpectrum:
        self._require_ready()
        self.backend.maybe_transport_fault(self.device_id)
        rec = self.backend.record(solid.vessel_id)
        if rec.latent is None:
            raise HardwareFault(f"{self.device_id}: no sample on the ATR crystal")
        async with self._lock:
            await self.backend.dwell(120.0, self.device_id, f"ir:{solid.vessel_id}")
        return simulate_ir(rec.latent, self.backend.rng_for(solid.experiment_id + ":ir"),
                           self.backend.ground_truth.ir_absorptivity_ratio, wn_range_cm1)


class SimulatedPotentiostat(_SimDevice):
    """Multichannel potentiostat with automated electrode casting (flooded cells)."""

    capacity = 4
    #: active mass per electrode and electrolyte per cell
    active_mass_mg = 3.0
    electrolyte_volume_mL = 20.0

    async def cycle(self, solid: VesselHandle, electrolyte_M: dict[str, float],
                    current_mA_g: float = 100.0, n_cycles: int = 20) -> EchemCycleData:
        self._require_ready()
        self.backend.maybe_transport_fault(self.device_id)
        rec = self.backend.record(solid.vessel_id)
        if rec.latent is None:
            raise HardwareFault(f"{self.device_id}: no powder to cast an electrode from")
        tag = "+".join(f"{k}{v:g}" for k, v in sorted(electrolyte_M.items()))
        rng = self.backend.rng_for(f"{solid.experiment_id}:ec:{tag}")
        self.backend.maybe_mechanical_fault(self.device_id, "cell short circuit")
        out = simulate_cycling(rec.latent, electrolyte_M, rng,
                               self.backend.ground_truth.echem_offsets,
                               n_cycles=n_cycles, current_mA_g=current_mA_g)
        q_mean = float(np.mean([q[-1] for q in out["discharge_q"]])) or 1.0
        async with self._lock:
            await self.backend.dwell(2 * n_cycles * 3600.0 * q_mean / max(current_mA_g, 1e-6),
                                     self.device_id, f"cycle:{solid.vessel_id}:{tag}")
        lat = rec.latent
        electrode = VesselHandle(f"{solid.vessel_id}-E-{tag}", solid.experiment_id,
                                 station=self.device_id, solid_present=True)
        electrolyte = VesselHandle(f"{solid.vessel_id}-L-{tag}", solid.experiment_id,
                                   station=self.device_id, contents_mL=self.electrolyte_volume_mL)
        fe_per_fu = 1.0 - lat.vacancy_fraction
        n_fu = self.active_mass_mg * 1e-3 / out["formula_weight"]
        self.backend.echem_samples[electrode.vessel_id] = dict(
            kind="electrode", metal=lat.metal, inserted=out["inserted"], fe_per_fu=fe_per_fu,
            active_mass_mg=self.active_mass_mg, formula_weight=out["formula_weight"])
        self.backend.echem_samples[electrolyte.vessel_id] = dict(
            kind="electrolyte", volume_mL=self.electrolyte_volume_mL,
            fe_mol=n_fu * fe_per_fu * out["dissolved_fe_fraction"])
        return EchemCycleData(
            electrolyte_M=dict(electrolyte_M), current_mA_g=current_mA_g,
            reference="Zn2+/Zn", charge_q=out["charge_q"], charge_E=out["charge_E"],
            discharge_q=out["discharge_q"], discharge_E=out["discharge_E"],
            electrode=electrode, electrolyte=electrolyte,
            active_mass_mg=self.active_mass_mg,
            electrolyte_volume_mL=self.electrolyte_volume_mL)


def build_simulated_platform(
    seed: int = 0,
    time_scale: float = 1e-3,
    reproducibility: float = 0.03,
    failure_rate: float = 0.02,
    transport_fault_rate: float = 0.0,
    mechanical_fault_rate: float = 0.0,
    reactor_capacity: int = 4,
    with_ir: bool = True,
    with_electrochem: bool = True,
) -> tuple[Platform, SimulatedBackend]:
    """Assemble a complete simulated platform and return it with its backend."""
    backend = SimulatedBackend(
        ground_truth=GroundTruth(seed=seed, reproducibility=reproducibility,
                                 failure_rate=failure_rate),
        time_scale=time_scale,
        transport_fault_rate=transport_fault_rate,
        mechanical_fault_rate=mechanical_fault_rate,
        seed=seed,
    )
    platform = Platform(
        liquid_handler=SimulatedLiquidHandler("lh-01", backend),
        reactor=SimulatedReactor("rx-01", backend, capacity=reactor_capacity),
        workup=SimulatedWorkup("wu-01", backend),
        diffractometer=SimulatedDiffractometer("xrd-01", backend),
        elemental=SimulatedElementalAnalyzer("icp-01", backend),
        ir=SimulatedIRSpectrometer("ir-01", backend) if with_ir else None,
        electrochem=SimulatedPotentiostat("ec-01", backend) if with_electrochem else None,
    )
    return platform, backend
