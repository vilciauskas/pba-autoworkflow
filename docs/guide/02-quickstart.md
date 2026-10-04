# 2. Quick start

This chapter runs a complete campaign on the **simulated deck**: six simulated instruments that
reproduce PBA co-precipitation chemistry, measurement noise and instrument faults. No hardware is
needed. All output below is real output from these commands (long lines shortened with `…`).

## Step 1: check a batch without running it

```bash
pba-autoworkflow dry-run --batch-size 6
```

```
       metal     c_metal_M       c_hcf_M      c_nacl_M   c_citrate_M            ph  temperature_C  …
          Ni        0.1805        0.2375         1.814         0.164          1.84         61.45  …
          Co       0.02696       0.01103         3.085        0.4021         4.161         53.18  …
          Mn       0.01251       0.08651        0.8373       0.07087         6.356          29.9  …
          Cu        0.0576       0.02721         4.608        0.2562         2.541          86.9  …
          Mn        0.1122       0.07497         3.486        0.3041         3.482         71.92  …
          Fe       0.01677       0.05374         2.252       0.03704         5.801         47.41  …
  ! citrate exceeds stock solubility budget

6 recipes, 1 feasibility problems
```

`dry-run` plans the seed batch and runs the platform's recipe checks, but touches no device.
Use it whenever you change the design space, before committing reagents.

## Step 2: run a campaign

```bash
pba-autoworkflow run --runs runs/demo --campaign-id demo --iterations 3 --batch-size 8 --time-scale 0
```

```
iteration 0: running 12 experiments (sobol)
  demo-b00-e01 failed — hardware: rx-01: over-temperature interlock tripped during ramp
  demo-b00-e06 quarantined — essentially no conversion detected
iteration 0: HV=0.4703 (Δ+0.4703) best=0.2988 counts={'complete': 10, 'failed': 1, 'quarantined': 1} …
iteration 1: running 8 experiments (qnehvi, replicate)
  demo-b01-e05 failed — hardware: wu-01: pellet lost during decant
iteration 1: HV=0.5903 (Δ+0.1200) best=0.3728 counts={'complete': 7, 'failed': 1} …
iteration 2: running 8 experiments (qnehvi, replicate)
iteration 2: HV=0.6749 (Δ+0.0846) best=0.4095 counts={'complete': 8} …
…
best recipe:
  metal                  Mn
  c_metal_M              0.0789072947048061
  …
  -> Na1.47Mn[Fe(CN)6]0.90·1.6H2O  capacity 68.5 mAh/g  yield 0.45
platform reproducibility (relative SD over replicate groups):
  rsd_na_inventory         0.1357
  rsd_framework_integrity  0.0312
  n_replicate_groups       1.0000
```

(Lines for experiments that completed normally are omitted.) The same command with the same
`--seed` gives identical results on any machine.

What happened:

- **Iteration 0** is the seed design: 12 points from a scrambled Sobol' sequence (`--n-seed`,
  default 12). The planner has no data yet, so it covers the space evenly.
- **Iterations 1 and 2** are chosen by the Bayesian planner (qNEHVI) from a Gaussian-process
  model of all results so far. A fraction of each batch (`--replicate-fraction`, default 0.15)
  repeats earlier recipes, to measure run-to-run scatter.
- **HV** is the dominated hypervolume of the feasible results, the campaign's measure of progress.
  It can only rise or stay level as data accumulate.
- Runs end in one of three states. **complete**: usable result. **failed**: an instrument fault,
  never counted against the recipe. **quarantined**: the data were produced but failed a physical
  sanity check, so they are kept out of the model. Chapter 3 lists the checks.

`--time-scale 0` skips all simulated instrument waiting. Use `2e-4` to see realistic contention
between stations within seconds, or `1.0` for a full-length timing rehearsal.

The default simulator injects faults on purpose, so the error-handling paths are exercised on
every run: 4 % of runs lose their product (`--failure-rate`), and each instrument operation has a
1 % chance of a terminal hardware fault (`--mechanical-fault-rate`) and a 2 % chance of a
communication error (`--transport-fault-rate`). Communication errors are retried automatically.
Set `--failure-rate 0 --mechanical-fault-rate 0 --transport-fault-rate 0` for a fault-free run.

## Step 3: resume

A campaign is resumed from its store; nothing is held in memory between commands. This works after
a crash, after a power cut, or when you want to add budget.

```bash
pba-autoworkflow resume --runs runs/demo --iterations 1 --time-scale 0
```

```
resumed demo with 28 experiments
iteration 3: running 8 experiments (qnehvi, replicate)
  demo-b03-e01 failed — hardware: rx-01: over-temperature interlock tripped during ramp
  demo-b03-e05 failed — infeasible recipe: citrate exceeds stock solubility budget
iteration 3: HV=0.7362 (Δ+0.0612) best=0.4260 counts={'complete': 6, 'failed': 2} …
```

The second failure shows the recipe check working inside the loop: the planner proposed a recipe
the deck cannot prepare, and it was rejected before any reagent was used.

## Step 4: inspect

```bash
pba-autoworkflow status --runs runs/demo
```

```
campaign      demo
experiments   36
status        {'complete': 31, 'failed': 4, 'quarantined': 1}
next batch    4

device        calls   busy_s   failed
  icp-01          32     16.3       0
  uv-01           32     16.1       0
  lh-01          139      9.9       1
  wu-01          129      3.9       1
  rx-01          136      3.2       2
  xrd-01          33      0.2       1
```

## Step 5: report

```bash
pba-autoworkflow report --runs runs/demo
```

```
experiments    runs/demo/report/experiments.csv
iterations     runs/demo/report/iterations.csv
summary        runs/demo/report/summary.json
overview       runs/demo/report/campaign_overview.png
best_pattern   runs/demo/report/best_pattern.png
```

| File | Contents |
|---|---|
| `experiments.csv` | One row per experiment: recipe, status, descriptors (lattice constant, domain size, composition), objectives |
| `iterations.csv` | One row per batch: counts, hypervolume, best score, wall time, bottleneck station |
| `summary.json` | Campaign summary, Pareto front, best recipe, stop reason |
| `campaign_overview.png` | The four-panel figure below |
| `best_pattern.png` | The XRD pattern of the best sample, with indexed reflections |

![Campaign overview](img/quickstart_overview.png)

*Top left: hypervolume against experiments run, rising from 0.470 to 0.736 over four batches.
Top right: the two objectives for every run, coloured by batch; open circles failed the yield
constraint, and red diamonds mark the Pareto front. Bottom left: run outcomes per batch. Bottom
right: station occupancy. With `--time-scale 0` no instrument time is simulated, so this panel is
empty here; run with `--time-scale 2e-4` or larger to see which station limits throughput.*

![Best sample XRD pattern](img/quickstart_best_pattern.png)

*The best sample after the resumed batch, demo-b03-e00, Na<sub>1.53</sub>Fe[Fe(CN)<sub>6</sub>]<sub>0.96</sub>·1.6H<sub>2</sub>O
(*a* = 10.146 Å):
measured pattern, fitted background, and the background-stripped pattern used for peak fitting.*

## Command reference

| Command | Purpose |
|---|---|
| `run` | Start a campaign, or continue one whose `--campaign-id` already exists in `--runs` |
| `resume` | Continue the campaign found in `--runs` |
| `status` | Experiment counts and per-device statistics |
| `report` | Write CSV, JSON and figures to `<runs>/report/` |
| `dry-run` | Plan and check one batch without touching a device |

Main options of `run` and `resume` (see `pba-autoworkflow run --help` for the full list):

| Option | Default | Meaning |
|---|---|---|
| `--runs` | `runs/demo` | Store directory (database and raw traces) |
| `--iterations` | 6 | Number of batches to run in this call |
| `--n-seed` | 12 | Size of the Sobol' seed batch |
| `--batch-size` | 8 | Experiments per Bayesian batch |
| `--planner` | `bayes` | `bayes` (qNEHVI), `sobol` or `random` |
| `--replicate-fraction` | 0.15 | Share of each batch spent on replicates |
| `--max-in-flight` | 4 | Experiments executing concurrently |
| `--reactor-capacity` | 4 | Reactor positions |
| `--seed` | 0 | Random seed for planner and simulator |
| `--time-scale` | 0.0 | Fraction of simulated instrument time actually waited |
| `--no-report` | off | Skip writing the report at the end |
