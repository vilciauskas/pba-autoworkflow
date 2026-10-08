# 3. Concepts

## The closed loop

Each iteration of a campaign does five things:

1. **Plan.** Choose a batch of recipes (Sobol' seed design first, then Bayesian optimisation).
2. **Register.** Write every recipe to the store *before* it runs, so a crash never loses track of reagents already committed.
3. **Execute.** Run the batch through the stations concurrently, under station capacity limits.
4. **Analyse.** Reduce the raw data to descriptors, then to objectives, and apply quality control.
5. **Stop or continue.** Check the stopping rules.

Each experiment moves through the stations in this order:

```
liquid handler   prepare solution A (M²⁺ salt + citrate) and B (Na₄[Fe(CN)₆] + NaCl), adjust pH
reactor          load A, ramp to temperature, meter B into A, age, unload
workup           centrifuge, wash, dry, weigh
diffractometer   powder XRD of the solid
elemental        ICP digest of the solid (+ C analysis)
```

## Design space

The default space (`schema.default_design_space()`) has nine continuous parameters and one
categorical choice.

| Parameter | Range | Scale | Meaning |
|---|---|---|---|
| `metal` | Mn, Fe, Co, Ni, Cu | categorical | Divalent metal on the N-coordinated site |
| `c_metal_M` | 0.01–0.3 mol/L | log | M(II) sulfate/chloride in solution A |
| `c_hcf_M` | 0.01–0.3 mol/L | log | Na₄[Fe(CN)₆] in solution B |
| `c_nacl_M` | 0–3.5 mol/L | linear | Supporting NaCl; sets the Na⁺ activity |
| `c_citrate_M` | 0–0.45 mol/L | linear | Sodium citrate chelator |
| `ph` | 1–7 | linear | pH of solution A before addition |
| `temperature_C` | 25–90 °C | linear | Precipitation and ageing temperature |
| `addition_rate_mL_min` | 0.05–20 mL/min | log | Metering rate of B into A |
| `aging_time_h` | 0.5–24 h | log | Post-addition ageing |
| `stir_rate_rpm` | 200–1200 rpm | linear | Stirrer set-point |

Log-scaled parameters are searched uniformly in their logarithm. To change ranges or metals, see
chapter 4. Before a batch runs, the platform checks each recipe against physical limits (for
example, stock solubility); `dry-run` shows these warnings.

## From raw data to objectives

| Instrument | Raw data | Descriptors |
|---|---|---|
| XRD | 2θ pattern | Phase purity; number of impurity peaks; crystallinity index; cubic lattice constant *a*; coherent domain size and microstrain (Williamson–Hall, falling back to Scherrer); FWHM of (200); phase (cubic, rhombohedral, …); number of indexed reflections; indexing residual |
| ICP + C | Element concentrations, dry mass | Na per formula unit; Fe/M ratio; vacancy fraction *y*; water content; formula |
| Balance | Dry mass | Isolated yield |

The XRD pipeline is: SNIP background subtraction, Savitzky–Golay-smoothed peak search,
pseudo-Voigt fits, and cubic indexing. Indexing uses a truncated loss, so a peak from a secondary
phase costs a fixed penalty instead of pulling the lattice constant towards a wrong solution.

After indexing, each fitted peak is attributed either to the PBA (within 0.45° of an allowed
reflection at the refined *a*, or within 0.55° of an indexed one, which covers split components)
or to an unidentified secondary phase.

- **Phase purity** = PBA Bragg intensity ÷ all Bragg intensity. It is an *intensity* fraction,
  not a weight fraction; converting it would need reference intensity ratios for each impurity.
  It is an upper bound, because an impurity line that coincides with a PBA reflection is counted
  as PBA.
- **Crystallinity index** = PBA Bragg intensity ÷ (PBA Bragg intensity + amorphous halo). The
  halo is recovered by fitting the SNIP background with a constant, an exponential (air and
  low-angle scatter) and a Gaussian held near the (200) position. Impurity peaks are left out, so
  a crystalline impurity cannot raise it. On the simulated deck it ranks samples correctly
  (correlation 0.98 with the true ordered fraction) but reads about 0.15 low, because part of the
  halo is absorbed by the exponential term. Treat it as a relative measure.

Rhombohedral or monoclinic distortion is reported only when several of the reflections it should
split appear as doublets. A single impurity line next to a PBA reflection is not counted. On the
simulated deck the small rhombohedral split (0.16° 2θ) is not resolved by the peak search, so
distorted samples are currently reported as cubic. The distortion label is therefore not reliable
yet and is not used as an objective.

The campaign targets **phase formation** and maximises two objectives from the XRD pattern:

| Objective | Definition | Why |
|---|---|---|
| `phase_purity` | XRD phase purity (above) | Secondary phases (NaCl residue, M(OH)₂ or CuO at high pH, unreacted salts) mean the recipe did not form a single-phase PBA |
| `crystallinity` | XRD crystallinity index (above) | Separates well-ordered frameworks from poorly ordered or amorphous precipitates |

A run with no usable pattern scores 0 on both. The ICP composition is still measured and
recorded: it gives the formula used for the isolated yield, and feeds the charge-balance checks
below. It is not optimised.

and enforces one constraint: **isolated yield ≥ 0.35**. Runs below it are kept but marked infeasible.

The two objectives trade off against each other, so the campaign does not return a single
optimum. It maps the **Pareto front**: the set of recipes that cannot be improved in one objective
without losing in the other. Progress is measured by the **dominated hypervolume**, the area of
objective space dominated by the feasible results, with reference point (0, 0).

For ranking, for example to pick the "best recipe", runs are scored with an augmented Chebyshev
scalarisation. Every feasible run outranks every infeasible one.

## Quality control: complete, failed, quarantined

| Status | Meaning | Used by the optimiser? |
|---|---|---|
| `complete` | Data passed every check | Yes |
| `failed` | An instrument fault stopped the run | No; the recipe is not blamed |
| `quarantined` | Data were produced but are physically implausible | No; kept in the store for inspection |

A run is quarantined when any of these hold:

- no diffraction data; or, for a crystalline pattern, fewer than the required number of indexed
  reflections or no lattice constant. An **amorphous** product is not quarantined: it is a real
  outcome of the recipe and scores low on both objectives;
- lattice constant outside the window for that metal (for example 10.18–10.88 Å for Mn);
- a large indexing residual, or an implausible domain size;
- no elemental assay, or a vacancy fraction that the assay cannot determine;
- Na above the charge-balance ceiling, or Fe/M above the framework stoichiometry;
- isolated yield above theory.

These checks keep instrument artefacts and analysis failures from being modelled as chemistry,
which would otherwise steer the optimiser towards non-existent optima.

**Errors.** Communication errors (`TransportError`) are retried automatically, by default twice
with back-off. Hardware faults, exhausted consumables, and failures during irreversible steps
(ageing, workup) are not retried; the experiment is marked `failed`. If a fault happens while a
vessel is in the reactor, the vessel is ejected so its position is freed.

## Planners

| Planner | Used for | How it works |
|---|---|---|
| `sobol` | Seed batch | Scrambled Sobol' points in the continuous space, metals assigned by stratification |
| `bayes` | All later batches | Constrained q-Noisy Expected Hypervolume Improvement (qNEHVI) |
| `random` | Baseline | Uniform random sampling |

The Bayesian planner fits one Gaussian process per objective and one for the yield constraint.
Each uses a Matérn-5/2 kernel with automatic relevance determination over the continuous
parameters, multiplied by an exchangeable kernel over metals, so that information is shared
between metals. Hyperparameters, including the noise level, are fitted by maximising the
marginal likelihood. Candidates come from global Sobol' coverage plus local perturbations of the
current front. A batch is selected greedily with the *kriging believer* heuristic. Until at least
8 usable results exist, the planner continues with Sobol' points (logged as `sobol-warmup`).

**Replicates.** By default 15 % of each Bayesian batch repeats earlier recipes. Their scatter
shows whether a hypervolume gain is real or within measurement noise; see
`Campaign.replicate_statistics()`.

## Stopping rules

A campaign stops at the end of the `--iterations` requested, or earlier when one of these fires:

| Rule | Default | Setting |
|---|---|---|
| Experiment budget reached | 200 experiments | `max_experiments` |
| Platform health: too many failed or quarantined runs in a batch | > 50 % | `max_batch_failure_rate` |
| Hypervolume converged: \|ΔHV\| below tolerance for consecutive batches | 5 × 10⁻⁴ for 3 batches | `hv_convergence_tol`, `hv_convergence_patience` |

The platform-health rule halts the campaign for inspection rather than continuing to consume
reagents on a misbehaving deck. The stop reason is printed and recorded in `summary.json`.

## Provenance and resuming

Everything is stored under the `--runs` directory:

```
runs/demo/
├── campaign.sqlite      campaign config, experiments, results, device calls, event log
├── traces/              raw instrument data per experiment
└── report/              written by `report`
```

- Experiments are named `<campaign>-b<batch>-e<index>`, for example `demo-b02-e05`.
- Every device call is logged with its duration and outcome; `status` summarises them.
- Every step (campaign created, batch planned, experiment queued, finalised, iteration complete, campaign stopped) is an event in an append-only log.
- `run` on an existing campaign id, or `resume`, rebuilds the campaign entirely from the store:
  history, iteration records and the convergence test's memory. A campaign can therefore be
  stopped at any point and continued in a new process.
