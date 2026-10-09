# 4. Python API

The command-line tool is a thin layer over four objects:

| Object | Module | Role |
|---|---|---|
| `Platform` | `pba_autoworkflow.devices.base` | The six instruments |
| `ProvenanceStore` | `pba_autoworkflow.provenance` | The campaign database and raw traces (a **directory**) |
| `CampaignConfig` | `pba_autoworkflow.campaign` | Budget, batch sizes, stopping rules |
| `Campaign` | `pba_autoworkflow.campaign` | The closed loop |

## A complete script

From [`examples/run_campaign.py`](../../examples/run_campaign.py):

```python
import asyncio

from pba_autoworkflow.campaign import Campaign, CampaignConfig
from pba_autoworkflow.devices.simulated import build_simulated_platform
from pba_autoworkflow.provenance import ProvenanceStore
from pba_autoworkflow.report import write_report
from pba_autoworkflow.schema import CategoricalSpec, default_design_space

platform, _backend = build_simulated_platform(seed=0, time_scale=0.0)

# Specs are immutable: build a modified copy (four metals, reactor limited to 80 °C).
base = default_design_space()
space = base.model_copy(update={
    "continuous": tuple(d.model_copy(update={"high": 80.0}) if d.name == "temperature_C" else d
                        for d in base.continuous),
    "categorical": (CategoricalSpec(name="metal", choices=("Mn", "Co", "Ni", "Cu"),
                                    description=base.categorical[0].description),),
})

cfg = CampaignConfig(campaign_id="example", n_seed=8, batch_size=6,
                     max_iterations=3, replicate_fraction=0.15, seed=0)
store = ProvenanceStore("runs/example")
campaign = Campaign(platform, store, cfg, space=space, progress=print)

records = asyncio.run(campaign.run())
write_report(campaign, "runs/example/report")
store.close()
```

Output:

```
iteration 0: HV=0.4280 (Δ+0.4280) best=0.2823 counts={'complete': 7, 'quarantined': 1} …
iteration 1: HV=0.5879 (Δ+0.1599) best=0.3394 counts={'failed': 1, 'complete': 5} …
iteration 2: HV=0.5886 (Δ+0.0007) best=0.3394 counts={'complete': 6} …

hypervolume per iteration: [0.428, 0.588, 0.589]
best: example-b01-e05 Mn Na1.27Mn[Fe(CN)6]0.88·1.2H2O
```

All 20 experiments used only the four allowed metals, and none exceeded 80 °C (maximum 77.1 °C).

`Campaign.run()` is a coroutine. Inside an existing event loop (for example Jupyter), use
`await campaign.run()` instead of `asyncio.run(...)`.

## `CampaignConfig`

| Field | Default | Meaning |
|---|---|---|
| `campaign_id` | random `pba-xxxxxxxx` | Identifier; an existing id in the store is resumed |
| `planner` | `"bayes"` | `"bayes"`, `"sobol"` or `"random"` after the seed batch |
| `seed_planner` | `"sobol"` | Planner for the first batch |
| `n_seed` | 12 | Size of the seed batch |
| `batch_size` | 6 | Experiments per later batch |
| `max_iterations` | 8 | Batches per `run()` call |
| `max_experiments` | 200 | Total experiment budget |
| `replicate_fraction` | 0.15 | Share of each batch spent on replicates |
| `max_in_flight` | 4 | Experiments executing concurrently |
| `reactor_capacity` | 4 | Reactor positions |
| `per_experiment_timeout_s` | `None` | Abort an experiment after this time |
| `hv_convergence_tol` | 5 × 10⁻⁴ | Hypervolume convergence tolerance |
| `hv_convergence_patience` | 3 | Batches below tolerance before stopping |
| `max_batch_failure_rate` | 0.5 | Halt if a larger share of a batch fails or is quarantined |
| `seed` | 0 | Random seed |
| `target_phase` | `None` | Framework phase to optimise for; `None` keeps the campaign's recorded target, else `pba_fm3m` |
| `objectives` | `None` | Objectives to maximise (chapter 3); `None` keeps the recorded ones, else `("target_phase_fraction", "crystallinity")` |

The CLI uses its own defaults for some of these (for example `--batch-size 8`).

## `WorkflowConfig`: how each experiment is run

Pass `workflow_config=WorkflowConfig(...)` to `Campaign` to change the measurement protocol.

| Field | Default |
|---|---|
| `xrd_range`, `xrd_step_deg`, `xrd_exposure_s` | (10, 60)° 2θ, 0.02°, 120 s |
| `wash_cycles` | 3 |
| `dry_duration_s` | 3600 s (drying temperature, pressure and gas are recipe parameters) |
| `icp_elements` | Na, K, Fe, Mn, Co, Ni, Cu, Zn |
| `run_ir` | `True` (when the platform has an IR spectrometer) |
| `echem_single_ions`, `echem_mixed_electrolytes` | filled from the campaign's objectives (chapter 3) |
| `echem_current_mA_g`, `echem_n_cycles` | 100 mA/g, 20 |
| `max_transport_retries`, `retry_backoff_s` | 2, 1.0 s |

## Reading results

```python
campaign.summary()              # dict: counts, hypervolume, Pareto size, stop reason
campaign.pareto_front()         # list[Experiment] on the feasible front
campaign.best_experiment()      # best by the scalarised score (feasible first)
campaign.replicate_statistics() # scatter of repeated recipes
campaign.dataframe()            # pandas DataFrame, one row per experiment

exp = campaign.best_experiment()
exp.parameters                  # SynthesisParameters (the recipe)
exp.descriptors.xrd             # phase_fractions, phase_lattice, crystallinity_index, lattice_a_A, …
exp.descriptors.composition     # na_per_fu, vacancy_fraction, formula, …
exp.objectives.values           # {'target_phase_fraction': …, 'crystallinity': …}
exp.objectives.feasible         # yield constraint satisfied?
```

To read a finished campaign without running anything:

```python
from pba_autoworkflow.provenance import ProvenanceStore

store = ProvenanceStore("runs/demo")
df = store.to_dataframe("demo")               # flat table of every experiment
events = store.event_log("demo")              # the full event log

# raw instrument data for one experiment
exp = next(e for e in store.load_campaign("demo") if e.experiment_id == "demo-b02-e05")
exp.raw_refs                                  # {'xrd': 'traces/….npz'}
xrd = store.load_trace(exp.raw_refs["xrd"])   # {'two_theta_deg': array, 'intensity': array}
```

## Using your own planner

Subclass `pba_autoworkflow.optimize.planner.Planner` and implement one method,
`suggest(self, batch_size: int, history: list[Experiment]) -> list[Suggestion]`, then pass it as
`Campaign(..., planner=my_planner)`. `make_planner(kind, space, seed)` builds the built-in planners.
