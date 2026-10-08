# 5. Connecting real instruments

The orchestrator never talks to hardware directly. It calls six **protocols**, one per station,
defined in [`pba_autoworkflow/devices/base.py`](../../pba_autoworkflow/devices/base.py). The
simulated deck implements them; to go live, implement the same methods for your instruments,
one at a time. Planning, scheduling, analysis, quality control and provenance stay unchanged.

## The six protocols

All methods are `async`. A `VesselHandle` identifies a physical sample as it moves between
stations (`vessel_id`, `experiment_id`, `station`, `contents_mL`, `solid_present`).

| Station | Methods | Returns |
|---|---|---|
| `LiquidHandler` | `prepare_solution(vessel, components: dict[str, float], volume_mL)` — dispense a solution of the given molar composition | `VesselHandle` |
| | `meter_into(source, destination, volume_mL, rate_mL_min)` — controlled-rate addition | `VesselHandle` |
| | `adjust_ph(vessel, target_ph)` — titrate to the set-point | achieved pH (`float`) |
| `Reactor` | `load(vessel)`, `set_conditions(vessel, temperature_C, stir_rate_rpm)`, `hold(vessel, duration_s)` | `None` |
| | `unload(vessel)` | `VesselHandle` |
| `Workup` | `separate(vessel)` | `(solid, liquid)` handles |
| | `wash(solid, cycles=3, solvent="water")`, `dry(solid, temperature_C=70.0, duration_s=3600.0)` | `VesselHandle` |
| | `weigh(solid)` | dry mass in mg |
| `Diffractometer` | `measure(solid, two_theta_range, step_deg, exposure_s)` | `XRDPattern` |
| `ElementalAnalyzer` | `measure(solid, elements)` | `ICPResult` |

The data types a driver must return (`pba_autoworkflow.schema`):

| Type | Fields |
|---|---|
| `XRDPattern` | `two_theta_deg` (array), `intensity` (array), `wavelength_A`, `exposure_s` |
| `ICPResult` | `concentrations_mol_L` (dict, element → mol/L in the digest), `digest_mass_mg`, `digest_volume_mL`, `dry_mass_mg`, `carbon_wt_pct` |

## Writing a driver

Subclass `Device`, which provides the lifecycle (`connect`, `disconnect`, `status`, `reset`) and
the state machine (offline → idle → busy, or error), and implement the protocol methods:

```python
from pba_autoworkflow.devices.base import Device, HardwareFault, TransportError
from pba_autoworkflow.schema import XRDPattern

class MyDiffractometer(Device):
    capacity = 1                                   # samples held at once

    async def connect(self):
        ...                                        # open the connection; raise TransportError if unreachable
        await super().connect()

    async def measure(self, solid, two_theta_range, step_deg, exposure_s):
        self._require_ready()                      # refuse if offline or in error
        ...                                        # run the scan
        return XRDPattern(two_theta_deg=tt, intensity=counts,
                          wavelength_A=1.5406, exposure_s=exposure_s)
```

Then put it on the platform. `Platform` is a dataclass, so you can replace one station and keep
the others simulated while you bring the deck up:

```python
import dataclasses
from pba_autoworkflow.devices.simulated import build_simulated_platform

platform, _ = build_simulated_platform(seed=0, time_scale=0.0)
platform = dataclasses.replace(platform, diffractometer=MyDiffractometer("xrd-01"))
```

A real deck uses the same constructor with all five real drivers:
`Platform(liquid_handler=…, reactor=…, workup=…, diffractometer=…, elemental=…)`.

**Mixed decks need care.** Simulated downstream instruments generate data from the simulator's
record of each sample, so a real upstream instrument and a simulated downstream one will not
describe the same sample. Mixed decks are for testing a driver's communication and error
handling, not for producing results.

### Worked example: a watch-folder diffractometer

[`examples/custom_device.py`](../../examples/custom_device.py) is a complete driver for the
common case where the vendor software cannot be scripted but exports each scan as a two-column
`.xy` file. The driver waits for `<vessel_id>.xy` to appear in the export folder, waits until the
file has finished writing, and parses it. Its self-test feeds it a synthetic cubic pattern
(*a* = 10.20 Å):

```
$ python examples/custom_device.py
[xrd-01] scan requested: demo-b00-e00-S 10.0-60.0 deg, step 0.02, 1.0 s
parsed 2500 points -> a = 10.1999 A, 6 reflections indexed, phase cubic
```

## Errors: tell the orchestrator what went wrong

The exception class decides what happens next.

| Raise | When | The orchestrator |
|---|---|---|
| `TransportError` | Lost connection, timeout, bus error | Retries (2 retries, 1 s back-off by default), reconnecting offline devices before each experiment |
| `HardwareFault` | Clog, crash, interlock, unreadable output | Marks the experiment `failed`; never retried; the recipe is not blamed |
| `ConsumableExhausted` | Out of tips, plates or a stock solution | Marks the experiment `failed`; needs a person |

Two rules matter most:

- **Never return made-up data.** If a measurement did not happen, raise. Quality control catches implausible data, but it cannot detect plausible-looking defaults.
- **Use `TransportError` only for faults that a retry can fix.** Steps that change the sample irreversibly (ageing, workup) are never retried, whatever the exception.

## Capacity and scheduling

Each station has a capacity: one for most instruments, and `reactor_capacity` (default 4) for
the reactor. The scheduler never asks a device to hold more samples than its capacity, and at most
`max_in_flight` experiments (default 4) are in progress at once. Set these to match the deck.
If a fault happens while a vessel is in the reactor, the orchestrator unloads it so the position is
not lost.

## Recipe checks

Before a batch is committed, `Platform.synthesis_recipe_check()` rejects recipes the deck cannot
run. The built-in checks are: addition longer than 8 h, citrate above the stock-solubility budget,
and total volume above 50 mL. Override this method on your platform for your own limits; `dry-run`
reports the problems it finds.

## Bring-up checklist

1. Implement and test one driver at a time against the simulated deck (mixed deck, above).
2. Restrict the design space to what the deck can actually do (chapter 4).
3. `dry-run` the seed batch and check every recipe by eye.
4. Run a short campaign with `--time-scale 1.0` on the simulated deck to rehearse timing and station contention.
5. Run the seed batch on hardware with a small `--iterations 1`; inspect `status`, `report`, and the raw traces.
6. Check that replicate scatter on the real deck is comparable to the simulator's `--reproducibility` (default 5 %). If it is much larger, the optimiser will need more replicates.
