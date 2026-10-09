# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-experiment workflow: one recipe from deck to descriptors.

This is the imperative heart of the platform -- the sequence a technician would
follow -- expressed once, as an async coroutine, so that many samples can be in
different stages simultaneously.  Each stage acquires the station it needs from
the :class:`~pba_autoworkflow.scheduler.StationPool`, holds it only for as long as the
physical operation lasts, and releases it before waiting on anything else.

Retry policy is per-stage rather than per-experiment.  A transport timeout while
mounting a sample on the diffractometer should retry the mount, not re-synthesize
the material; a hardware fault during workup is terminal for that sample but must
not stop the campaign.  Sample-destroying stages are marked non-retryable
explicitly so that the retry logic can never silently repeat an operation that
consumed the material.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from .analysis.echem import build_echem_descriptors
from .analysis.objectives import OBJECTIVE_NAMES, compute_objectives, quality_flags
from .analysis.spectra import (
    analyze_icp,
    analyze_ir,
    charge_balance_residual,
    estimate_capacity_mAh_g,
)
from .analysis.xrd import analyze_pattern
from .devices.base import (
    ConsumableExhausted,
    DeviceError,
    HardwareFault,
    Platform,
    TransportError,
    VesselHandle,
)
from .clock import Stopwatch, tick, timestamp
from .provenance import ProvenanceStore
from .schema import (
    Experiment,
    ExperimentStatus,
    SampleDescriptors,
    SynthesisParameters,
)

T = TypeVar("T")


@dataclass
class WorkflowConfig:
    """Fixed protocol settings -- not optimized, but recorded."""

    xrd_range: tuple[float, float] = (10.0, 60.0)
    xrd_step_deg: float = 0.02
    xrd_exposure_s: float = 120.0
    wash_cycles: int = 3
    #: drying temperature, pressure and gas are recipe parameters
    #: (``SynthesisParameters.dry_temperature_C`` / ``dry_pressure_mbar`` / ``dry_gas``)
    dry_duration_s: float = 3600.0
    icp_elements: tuple[str, ...] = ("Na", "K", "Fe", "Mn", "Co", "Ni", "Cu", "Zn")
    #: framework phase whose weight fraction is the first objective
    target_phase: str = "pba_fm3m"
    #: objectives scored for every run (see ``analysis.objectives.OBJECTIVE_CATALOGUE``)
    objectives: tuple[str, ...] = OBJECTIVE_NAMES
    #: run ATR-IR (Fe(II) share) when the platform has a spectrometer
    run_ir: bool = True
    #: electrochemistry: single-ion electrolytes (1 mol/L of each cation) and
    #: mixed electrolytes for competitive insertion; empty = skip the stage
    echem_single_ions: tuple[str, ...] = ()
    echem_mixed_electrolytes: tuple[dict, ...] = ()
    echem_current_mA_g: float = 100.0
    echem_n_cycles: int = 20
    max_transport_retries: int = 2
    retry_backoff_s: float = 1.0


@dataclass
class StageResult:
    name: str
    ok: bool
    duration_s: float
    detail: str = ""


@dataclass
class WorkflowTrace:
    stages: list[StageResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, t0: float, detail: str = "") -> None:
        """Append a stage result; ``t0`` is a monotonic :func:`tick` reading."""
        self.stages.append(StageResult(name, ok, tick() - t0, detail))


class ExperimentWorkflow:
    """Runs one experiment end to end against a :class:`Platform`."""

    def __init__(self, platform: Platform, store: ProvenanceStore,
                 config: WorkflowConfig | None = None,
                 station_pool: "object | None" = None) -> None:
        self.platform = platform
        self.store = store
        self.config = config or WorkflowConfig()
        self.pool = station_pool  # StationPool | None; None = no contention control

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _station(self, name: str):
        """Acquire a station if a pool is configured, else a null context."""
        if self.pool is None:
            class _Null:
                async def __aenter__(self_inner): return None
                async def __aexit__(self_inner, *a): return False
            return _Null()
        return self.pool.acquire(name)

    async def _call(self, exp: Experiment, device_id: str, action: str,
                    fn: Callable[[], Awaitable[T]], retryable: bool = True) -> T:
        """Invoke a device operation with provenance and bounded retries."""
        attempts = self.config.max_transport_retries + 1 if retryable else 1
        last: Exception | None = None
        for attempt in range(attempts):
            sw = Stopwatch()
            try:
                result = await fn()
            except TransportError as err:
                self.store.record_device_call(exp.experiment_id, device_id, action,
                                              sw.started_at, sw.duration_s, False,
                                              f"attempt {attempt + 1}: {err}")
                last = err
                if attempt + 1 < attempts:
                    await asyncio.sleep(self.config.retry_backoff_s * (2 ** attempt))
                    continue
                raise
            except DeviceError as err:
                self.store.record_device_call(exp.experiment_id, device_id, action,
                                              sw.started_at, sw.duration_s, False,
                                              str(err))
                raise
            else:
                self.store.record_device_call(exp.experiment_id, device_id, action,
                                              sw.started_at, sw.duration_s, True)
                return result
        raise last if last else RuntimeError("unreachable")

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    async def _synthesize(self, exp: Experiment, trace: WorkflowTrace
                          ) -> tuple[VesselHandle, VesselHandle]:
        p: SynthesisParameters = exp.parameters
        lh = self.platform.liquid_handler
        rx = self.platform.reactor

        vessel_a = VesselHandle(f"{exp.experiment_id}-A", exp.experiment_id)
        vessel_b = VesselHandle(f"{exp.experiment_id}-B", exp.experiment_id)

        async with await self._station("liquid_handler"):
            t0 = tick()
            await self._call(exp, lh.device_id, "prepare_A", lambda: lh.prepare_solution(
                vessel_a,
                {p.metal: p.c_metal_M, "citrate": p.c_citrate_M},
                p.volume_A_mL,
            ))
            await self._call(exp, lh.device_id, "prepare_B", lambda: lh.prepare_solution(
                vessel_b,
                {p.hcf_precursor: p.c_hcf_M, "NaCl": p.c_nacl_M},
                p.volume_B_mL,
            ))
            ph_actual = await self._call(exp, lh.device_id, "adjust_pH",
                                         lambda: lh.adjust_ph(vessel_a, p.ph))
            exp.metadata["ph_actual"] = ph_actual
            trace.add("prepare_solutions", True, t0, f"pH={ph_actual:.2f}")

        async with await self._station("reactor"):
            loaded = False
            try:
                t0 = tick()
                await self._call(exp, rx.device_id, "load", lambda: rx.load(vessel_a))
                loaded = True
                await self._call(exp, rx.device_id, "set_conditions",
                                 lambda: rx.set_conditions(vessel_a, p.temperature_C,
                                                           p.stir_rate_rpm))
                trace.add("reactor_ramp", True, t0)

                # The addition step occupies both the liquid handler and the reactor.
                async with await self._station("liquid_handler"):
                    t0 = tick()
                    await self._call(exp, lh.device_id, "meter_addition",
                                     lambda: lh.meter_into(vessel_b, vessel_a,
                                                           p.volume_B_mL,
                                                           p.addition_rate_mL_min))
                    trace.add("addition", True, t0,
                              f"{p.addition_time_min:.1f} min")

                t0 = tick()
                await self._call(exp, rx.device_id, "age",
                                 lambda: rx.hold(vessel_a, p.aging_time_h * 3600.0),
                                 retryable=False)
                trace.add("ageing", True, t0, f"{p.aging_time_h:.2f} h")
                slurry = await self._call(exp, rx.device_id, "unload",
                                          lambda: rx.unload(vessel_a))
                loaded = False
            except BaseException:
                # The reactor slot in the StationPool is released when this block
                # exits, so the physical position must be freed too.  Without
                # this, a vessel abandoned by a fault stayed loaded, the device
                # ran out of positions while the scheduler believed it had free
                # ones, and a later, healthy experiment failed with "all N
                # positions full".  Best effort: if the eject itself fails, the
                # original error is still the one reported.
                if loaded:
                    try:
                        await rx.unload(vessel_a)
                        exp.metadata["reactor_vessel_ejected_after_fault"] = True
                    except Exception:  # noqa: BLE001
                        exp.metadata["reactor_vessel_stuck"] = True
                        # The position is physically occupied until someone clears
                        # it, so the scheduler must stop offering it.  The slot this
                        # experiment holds is withdrawn on release; once every
                        # position is retired, later experiments fail fast with
                        # StationOutOfService instead of the campaign hanging.
                        if self.pool is not None:
                            self.pool.retire_slot("reactor")
                raise

        async with await self._station("workup"):
            wu = self.platform.workup
            t0 = tick()
            solid, liquid = await self._call(exp, wu.device_id, "separate",
                                             lambda: wu.separate(slurry),
                                             retryable=False)
            await self._call(exp, wu.device_id, "wash",
                             lambda: wu.wash(solid, self.config.wash_cycles),
                             retryable=False)
            await self._call(exp, wu.device_id, "dry",
                             lambda: wu.dry(solid, exp.parameters.dry_temperature_C,
                                            self.config.dry_duration_s,
                                            pressure_mbar=exp.parameters.dry_pressure_mbar,
                                            gas=exp.parameters.dry_gas))
            mass_mg = await self._call(exp, wu.device_id, "weigh",
                                       lambda: wu.weigh(solid))
            exp.metadata["dry_mass_mg"] = mass_mg
            trace.add("workup", True, t0, f"{mass_mg:.2f} mg")
        return solid, liquid

    async def _characterize(self, exp: Experiment, solid: VesselHandle,
                            liquid: VesselHandle, trace: WorkflowTrace
                            ) -> SampleDescriptors:
        cfg = self.config
        desc = SampleDescriptors()

        async def do_xrd():
            xd = self.platform.diffractometer
            async with await self._station("diffractometer"):
                t0 = tick()
                pattern = await self._call(
                    exp, xd.device_id, "xrd_scan",
                    lambda: xd.measure(solid, cfg.xrd_range, cfg.xrd_step_deg,
                                       cfg.xrd_exposure_s),
                )
                ref = self.store.save_trace(
                    exp.experiment_id, "xrd", xd.device_id, pattern.as_arrays(),
                    {"wavelength_A": pattern.wavelength_A,
                     "exposure_s": pattern.exposure_s,
                     "step_deg": cfg.xrd_step_deg},
                )
                exp.raw_refs["xrd"] = ref
                trace.add("xrd", True, t0)
            return await asyncio.to_thread(analyze_pattern, pattern, exp.parameters.metal)

        async def do_icp():
            ea = self.platform.elemental
            async with await self._station("elemental"):
                t0 = tick()
                icp = await self._call(
                    exp, ea.device_id, "icp_assay",
                    lambda: ea.measure(solid, cfg.icp_elements),
                )
                self.store.log_event("icp_result", payload=icp.concentrations_mol_L,
                                     campaign_id=exp.campaign_id,
                                     experiment_id=exp.experiment_id)
                trace.add("icp", True, t0)
            return analyze_icp(icp, exp.parameters)

        async def do_ir():
            ir_dev = self.platform.ir
            if ir_dev is None or not cfg.run_ir:
                return None
            async with await self._station("ir"):
                t0 = tick()
                spec = await self._call(exp, ir_dev.device_id, "ir_scan",
                                        lambda: ir_dev.measure(solid))
                exp.raw_refs["ir"] = self.store.save_trace(
                    exp.experiment_id, "ir", ir_dev.device_id, spec.as_arrays(), {})
                trace.add("ir", True, t0)
            return analyze_ir(spec)

        # The characterizations are independent; run them concurrently and let a
        # failure in one leave the others usable.
        t_char = tick()
        results = await asyncio.gather(do_xrd(), do_icp(), do_ir(), return_exceptions=True)
        xrd_res, icp_res, ir_res = results
        if not isinstance(xrd_res, BaseException):
            desc.xrd = xrd_res
        else:
            trace.add("xrd", False, t_char, str(xrd_res))
        if not isinstance(icp_res, BaseException):
            desc.composition = icp_res
        else:
            trace.add("icp", False, t_char, str(icp_res))
        if not isinstance(ir_res, BaseException):
            desc.ir = ir_res
        else:
            trace.add("ir", False, t_char, str(ir_res))
        desc.drying_index = exp.parameters.drying_index
        if desc.composition is not None and desc.ir is not None:
            desc.charge_balance_residual = charge_balance_residual(
                desc.composition, desc.ir.fe2_fraction)

        if cfg.echem_single_ions or cfg.echem_mixed_electrolytes:
            # Errors here are terminal for the run: an electrochemical objective
            # without its measurement would be scored as the worst value.
            desc.echem = await self._electrochemistry(exp, solid, desc, trace)

        # Isolated yield from the weighed mass against the measured formula.
        mass = exp.metadata.get("dry_mass_mg")
        if mass is not None and desc.composition is not None:
            from .schema import formula_weight

            p = exp.parameters
            limiting_mol = min(p.c_metal_M * p.volume_A_mL,
                               p.c_hcf_M * p.volume_B_mL) * 1e-3
            fw = formula_weight(p.metal, desc.composition.na_per_fu,
                                desc.composition.vacancy_fraction,
                                desc.composition.water_per_fu,
                                desc.composition.k_per_fu)
            theo_mg = limiting_mol * fw * 1e3
            if theo_mg > 0:
                desc.isolated_yield = min(float(mass / theo_mg), 1.2)

        if desc.composition is not None and desc.xrd is not None:
            desc.capacity_mAh_g = estimate_capacity_mAh_g(
                desc.composition, desc.xrd.domain_size_nm,
                desc.xrd.crystallinity_index, exp.parameters.metal,
            )
        return desc

    async def _electrochemistry(self, exp: Experiment, solid: VesselHandle,
                                desc: SampleDescriptors, trace: WorkflowTrace):
        cfg = self.config
        ec = self.platform.electrochem
        if ec is None:
            raise DeviceError("electrochemistry requested but the platform has no potentiostat")
        ea = self.platform.elemental
        single = {}
        for ion in cfg.echem_single_ions:
            async with await self._station("electrochem"):
                t0 = tick()
                single[ion] = await self._call(
                    exp, ec.device_id, f"cycle_{ion}",
                    lambda ion=ion: ec.cycle(solid, {ion: 1.0}, cfg.echem_current_mA_g,
                                             cfg.echem_n_cycles), retryable=False)
                exp.raw_refs[f"echem_{ion}"] = self.store.save_trace(
                    exp.experiment_id, f"echem_{ion}", ec.device_id,
                    single[ion].as_arrays(), {"electrolyte_M": {ion: 1.0}})
                trace.add(f"echem_{ion}", True, t0)
        mixed = []
        for el in cfg.echem_mixed_electrolytes:
            tag = "+".join(f"{k}{v:g}" for k, v in sorted(el.items()))
            async with await self._station("electrochem"):
                t0 = tick()
                data = await self._call(
                    exp, ec.device_id, f"cycle_{tag}",
                    lambda el=el: ec.cycle(solid, dict(el), cfg.echem_current_mA_g,
                                           cfg.echem_n_cycles), retryable=False)
                exp.raw_refs[f"echem_{tag}"] = self.store.save_trace(
                    exp.experiment_id, f"echem_{tag}", ec.device_id, data.as_arrays(),
                    {"electrolyte_M": dict(el)})
                trace.add(f"echem_{tag}", True, t0)
            elements = tuple(sorted(set(el) | {"Fe", exp.parameters.metal}))
            async with await self._station("elemental"):
                electrode = await self._call(exp, ea.device_id, f"icp_electrode_{tag}",
                                             lambda: ea.measure(data.electrode, elements))
                electrolyte = await self._call(exp, ea.device_id, f"icp_electrolyte_{tag}",
                                               lambda: ea.measure(data.electrolyte, ("Fe",)))
            self.store.log_event("echem_digest", payload={
                "electrolyte_M": dict(el), "electrode": electrode.concentrations_mol_L,
                "spent_electrolyte": electrolyte.concentrations_mol_L},
                campaign_id=exp.campaign_id, experiment_id=exp.experiment_id)
            mixed.append((data, electrode, electrolyte))
        # The framework's own metal (Zn in a zinc framework) cannot be assayed
        # as inserted against the framework content; it is taken coulometrically.
        c = desc.composition
        if c is not None and not (math.isfinite(c.vacancy_fraction) and c.vacancy_fraction < 1):
            c = None
        fw = None
        if c is not None:
            from .schema import formula_weight
            fw = formula_weight(exp.parameters.metal, c.na_per_fu, c.vacancy_fraction,
                                c.water_per_fu, c.k_per_fu)
        return build_echem_descriptors(single, mixed,
                                       framework_metal=exp.parameters.metal,
                                       formula_weight=fw,
                                       fe_per_fu=(1.0 - c.vacancy_fraction) if c else None)

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    async def run(self, exp: Experiment) -> Experiment:
        """Execute one experiment.  Never raises for chemistry or hardware faults."""
        trace = WorkflowTrace()
        exp.started_at = timestamp()
        exp.status = ExperimentStatus.RUNNING
        self.store.log_event("experiment_started", campaign_id=exp.campaign_id,
                             experiment_id=exp.experiment_id)

        connected = await self.platform.ensure_online()
        if connected:
            self.store.log_event("devices_connected", payload={"devices": connected},
                                 campaign_id=exp.campaign_id,
                                 experiment_id=exp.experiment_id)

        problems = self.platform.synthesis_recipe_check(exp.parameters)
        if problems:
            exp.status = ExperimentStatus.FAILED
            exp.error = "infeasible recipe: " + "; ".join(problems)
            exp.finished_at = timestamp()
            self.store.finalize_experiment(exp, problems)
            return exp

        try:
            solid, liquid = await self._synthesize(exp, trace)
            exp.status = ExperimentStatus.CHARACTERIZING
            exp.descriptors = await self._characterize(exp, solid, liquid, trace)
        except ConsumableExhausted as err:
            exp.status = ExperimentStatus.FAILED
            exp.error = f"consumable: {err}"
        except HardwareFault as err:
            exp.status = ExperimentStatus.FAILED
            exp.error = f"hardware: {err}"
        except DeviceError as err:
            exp.status = ExperimentStatus.FAILED
            exp.error = f"device: {err}"
        except asyncio.CancelledError:
            exp.status = ExperimentStatus.FAILED
            exp.error = "cancelled"
            raise
        except Exception as err:  # unexpected: record, keep the campaign alive
            exp.status = ExperimentStatus.FAILED
            exp.error = f"{type(err).__name__}: {err}"

        qc_flags: list[str] = []
        if exp.status is ExperimentStatus.CHARACTERIZING:
            qc = quality_flags(exp.descriptors, exp.parameters)
            qc_flags = qc.flags
            if qc.passed:
                exp.objectives = compute_objectives(exp.descriptors, exp.parameters,
                                                    target_phase=self.config.target_phase,
                                                    objectives=self.config.objectives)
                exp.metadata["target_phase"] = self.config.target_phase
                exp.status = ExperimentStatus.COMPLETE
            else:
                exp.status = ExperimentStatus.QUARANTINED
                exp.error = qc.reason()

        exp.finished_at = timestamp()
        exp.metadata["stages"] = [
            {"name": s.name, "ok": s.ok, "duration_s": round(s.duration_s, 3),
             "detail": s.detail}
            for s in trace.stages
        ]
        self.store.finalize_experiment(exp, qc_flags)
        return exp
