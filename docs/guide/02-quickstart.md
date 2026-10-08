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
          Ni        0.1805        0.2375          1.27         0.164          1.84         61.45  …
          Co       0.05442       0.02221         3.042        0.3068         6.478         44.96  …
          Zn       0.01164         0.113        0.6674       0.09726         5.064         38.73  …
          Cu           0.1       0.03782         2.441        0.3455         3.414         87.72  …
          Mn       0.05565       0.08154         3.084        0.4466         3.056         65.89  …
          Fe        0.0168       0.02878         1.364       0.02429           4.7          49.4  …

6 recipes, 0 feasibility problems
```

(The table is cut at the right; it continues with the remaining recipe columns.)

`dry-run` plans the seed batch and runs the platform's recipe checks, but touches no device.
Use it whenever you change the design space, before committing reagents.

## Step 2: run a campaign

```bash
pba-autoworkflow run --runs runs/demo --campaign-id demo --iterations 3 --batch-size 8 --time-scale 0
```

```
iteration 0: running 12 experiments (sobol)
  demo-b00-e01 failed — hardware: rx-01: over-temperature interlock tripped during ramp
  demo-b00-e02 failed — hardware: wu-01: pellet lost during decant
  demo-b00-e09 failed — infeasible recipe: citrate exceeds stock solubility budget
iteration 0: HV=0.6007 (Δ+0.6007) best=0.3404 counts={'complete': 9, 'failed': 3} …
iteration 1: running 8 experiments (qnehvi, replicate)
iteration 1: HV=0.7361 (Δ+0.1355) best=0.4115 counts={'complete': 8} …
iteration 2: running 8 experiments (qnehvi, replicate)
  demo-b02-e03 failed — hardware: rx-01: over-temperature interlock tripped during ramp
  demo-b02-e00 failed — hardware: wu-01: pellet lost during decant
iteration 2: HV=0.7361 (Δ+0.0000) best=0.4115 counts={'failed': 2, 'complete': 6} …
…
best recipe:
  metal                  Mn
  c_metal_M              0.041292938769153714
  …
  dry_temperature_C      27.938374050107715
  dry_atmosphere         air
  …
  -> Na0.89Mn[Fe(CN)6]0.90·1.6H2O  capacity 50.7 mAh/g  yield 0.63
platform reproducibility (relative SD over replicate groups):
  rsd_target_phase_fraction 0.0000
  rsd_crystallinity        0.0463
  n_replicate_groups       2.0000
  n_replicate_runs         4.0000
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
  sanity check, so they are kept out of the model. Chapter 3 lists the checks. The third
  failure here is the recipe check: a recipe whose citrate exceeds what the stock solution can
  supply is refused before any reagent is dispensed.
- With no `--target-phase`, the campaign optimises for the cubic framework (`pba_fm3m`). Step 6
  runs a campaign for a different polymorph.

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
iteration 3: HV=0.7361 (Δ+0.0000) best=0.4115 counts={'complete': 7, 'failed': 1} …
```

The resumed batch continues the numbering (iteration 3) and the model is rebuilt from the stored
results. It did not improve on the front this time; the hypervolume stays level rather than
falling.

## Step 4: inspect

```bash
pba-autoworkflow status --runs runs/demo
```

```
campaign      demo
experiments   36
status        {'complete': 30, 'failed': 6}
next batch    4

device        calls   busy_s   failed
  icp-01          30      1.4       0
  xrd-01          30      0.7       0
  wu-01          122      0.2       2
  rx-01          134      0.2       3
  lh-01          137      0.1       0
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
| `experiments.csv` | One row per experiment: recipe, status, descriptors (phase weight fractions and lattices, crystallinity, lattice constant, domain size, composition), objectives |
| `iterations.csv` | One row per batch: counts, hypervolume, best score, wall time, bottleneck station |
| `summary.json` | Campaign summary, Pareto front, best recipe, stop reason |
| `campaign_overview.png` | The four-panel figure below |
| `best_pattern.png` | The XRD pattern of the best sample with the whole-pattern phase fit |

![Campaign overview](img/quickstart_overview.png)

*Top left: hypervolume against experiments run, rising from 0.601 to 0.736 after the second batch
and level after that. Top right: target-phase fraction against crystallinity for every run,
coloured by batch; open circles failed the yield constraint, and the red diamond marks the Pareto
front. Most runs are single-phase cubic (fraction 1.0), so crystallinity is what separates them.
Of the runs below 0.9, three are Zn recipes that formed R-3c (with NaCl or ZnO) instead of the
cubic target and score 0, one Fe run contains 0.31 NaCl, and one poorly ordered Co run is
discounted for 0.62 unidentified intensity. Bottom left: run outcomes per batch. Bottom
right: station occupancy. With `--time-scale 0` no instrument time is simulated, so this panel is
empty here; run with `--time-scale 2e-4` or larger to see which station limits throughput.*

![Best sample XRD pattern](img/quickstart_best_pattern.png)

*The best sample, demo-b01-e04, Na<sub>0.89</sub>Mn[Fe(CN)<sub>6</sub>]<sub>0.90</sub>·1.6H<sub>2</sub>O
(*a* = 10.4973 Å, single-phase cubic): measured pattern, fitted background and the
whole-pattern phase fit (top); background-stripped pattern with the reflection ticks and weight
fraction of every identified phase (bottom). Peaks no library phase explains are marked with red
triangles.*

## Step 6: a polymorph campaign

Zinc hexacyanoferrate forms either the cubic framework or the rhombohedral R-3c phase
Na<sub>2</sub>Zn<sub>3</sub>[Fe(CN)<sub>6</sub>]<sub>2</sub>, depending on precipitation and drying. To
optimise for R-3c, name it as the target:

```bash
pba-autoworkflow run --runs runs/zn --campaign-id zn-r3c --target-phase znhcf_r3c --iterations 5 --batch-size 8 --time-scale 0
```

```
iteration 0: HV=0.0000 (Δ+0.0000) best=0.0161 counts={'complete': 10, 'failed': 2} …
iteration 1: HV=0.3674 (Δ+0.3674) best=0.2767 counts={'complete': 8} …
iteration 2: HV=0.4494 (Δ+0.0820) best=0.3101 counts={'complete': 7, 'failed': 1} …
iteration 3: HV=0.7059 (Δ+0.2565) best=0.4019 counts={'complete': 8} …
iteration 4: HV=0.7703 (Δ+0.0644) best=0.4350 counts={'complete': 8} …
…
best recipe:
  metal                  Zn
  …
  dry_temperature_C      48.6714157933453
  dry_atmosphere         air
  -> Na0.67Zn[Fe(CN)6]0.66·2.9H2O  capacity 19.0 mAh/g  yield 0.44
```

The search space still contains all six metals. The seed batch tried each of them twice. Only
zinc can form R-3c, so the first seed batch scores zero hypervolume, and the planner then moved
to zinc: 7 of 8 recipes in batch 1 and all 8 from batch 2 on. The low capacity is expected,
because Zn is not redox-active and only the Fe site stores charge.

![Polymorph campaign overview](img/polymorph_overview.png)

![Best R-3c sample](img/polymorph_best_pattern.png)

*The best sample, zn-r3c-b04-e01: 0.90 R-3c (blue ticks) and 0.10 cubic (orange ticks) by weight,
crystallinity 0.79. Chapter 3 describes how these fractions are computed and where they are
unreliable, for example in poorly ordered samples.*

A campaign's target phase is stored with it. `resume` keeps it, and naming a different
`--target-phase` for an existing campaign is refused; start a new campaign id instead.

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
| `--target-phase` | campaign's recorded target, else `pba_fm3m` | `pba_fm3m`, `pba_p21n` or `znhcf_r3c` |
| `--time-scale` | 0.0 | Fraction of simulated instrument time actually waited |
| `--no-report` | off | Skip writing the report at the end |
