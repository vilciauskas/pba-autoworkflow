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

The default space (`schema.default_design_space()`) has ten continuous parameters and two
categorical choices.

| Parameter | Range | Scale | Meaning |
|---|---|---|---|
| `metal` | Mn, Fe, Co, Ni, Cu, Zn | categorical | Divalent metal on the N-coordinated site |
| `c_metal_M` | 0.01–0.3 mol/L | log | M(II) sulfate/chloride in solution A |
| `c_hcf_M` | 0.01–0.3 mol/L | log | Na₄[Fe(CN)₆] in solution B |
| `c_nacl_M` | 0–3.5 mol/L | linear | Supporting NaCl; sets the Na⁺ activity |
| `c_citrate_M` | 0–0.45 mol/L | linear | Sodium citrate chelator |
| `ph` | 1–7 | linear | pH of solution A before addition |
| `temperature_C` | 25–90 °C | linear | Precipitation and ageing temperature |
| `addition_rate_mL_min` | 0.05–20 mL/min | log | Metering rate of B into A |
| `aging_time_h` | 0.5–24 h | log | Post-addition ageing |
| `stir_rate_rpm` | 200–1200 rpm | linear | Stirrer set-point |
| `dry_temperature_C` | 25–120 °C | linear | Drying temperature of the washed solid |
| `dry_atmosphere` | air, vacuum | categorical | Drying in ambient air or under dynamic vacuum |

Drying is part of the recipe because dehydration can change the polymorph: cubic zinc
hexacyanoferrate is reported to convert to the R-3c phase on drying at about 70 °C, which was the
platform's earlier fixed drying temperature. Lower water partial pressure (vacuum) lowers the
temperature of that conversion.

Log-scaled parameters are searched uniformly in their logarithm. To change ranges or metals, see
chapter 4. Before a batch runs, the platform checks each recipe against physical limits (for
example, stock solubility); `dry-run` shows these warnings.

## From raw data to objectives

| Instrument | Raw data | Descriptors |
|---|---|---|
| XRD | 2θ pattern | Weight fraction and refined cell of each crystalline phase; unidentified intensity share; dominant framework phase; crystallinity index; cubic lattice constant *a*; coherent domain size and microstrain (Williamson–Hall, falling back to Scherrer); FWHM of the strongest framework line |
| ICP + C | Element concentrations, dry mass | Na per formula unit; Fe/M ratio; vacancy fraction *y*; water content; formula |
| Balance | Dry mass | Isolated yield |

The XRD pipeline is: SNIP background subtraction, Savitzky–Golay-smoothed peak search and
pseudo-Voigt peak fits, followed by **whole-pattern phase quantification** against a library of
reference structures (`pba_autoworkflow/analysis/phases.py`).

**Reference phases.** The candidates depend on the recipe's metal:

| Metal | Framework phases | Secondary phases |
|---|---|---|
| Mn, Fe | cubic Fm-3m; monoclinic P2₁/n (Na-rich, "Prussian white" type) | NaCl; M(OH)₂ |
| Co, Ni | cubic Fm-3m | NaCl; M(OH)₂ |
| Cu | cubic Fm-3m | NaCl; CuO |
| Zn | cubic Fm-3m; rhombohedral R-3c Na₂Zn₃[Fe(CN)₆]₂ | NaCl; ZnO |

The library (`pba_autoworkflow/data/phase_library.json`) stores each phase's reflections as
multiplicity × |F|², computed from structures in the Crystallography Open Database by
`scripts/build_phase_library.py`; the COD numbers are in [REFERENCES.md](../../REFERENCES.md).
Two of the entries are approximations. The cubic Mn–Cu frameworks are an idealised model
(Na 1 per formula unit, *y* = 0.1) at the reference lattice constant. The cubic Zn entry is the
refined structure of zinc hexacyanoferrate(**III**), Zn₃[Fe(CN)₆]₂, so its line positions are right
but its intensities omit the Na⁺ of the Fe(II) compound the platform makes.

**Quantification**, for each candidate:

1. Refine the lattice within a few percent of the reference cell against the fitted peak positions:
   one scale for cubic and monoclinic cells, separate *a* and *c* for trigonal and hexagonal ones.
2. Synthesise its pattern from |F|² × Lorentz-polarisation with a pseudo-Voigt profile whose width
   follows its own domain size.
3. Fit the background-stripped pattern as a non-negative sum of all phase patterns, plus a free
   peak for every line no phase explains.
4. Convert the scale factors to **weight fractions** with the Hill–Howard relation,
   *w*ₚ ∝ *S*ₚ (*ZMV*)ₚ.

A phase is kept only if the pattern needs it. Either at least two of its strong lines are observed
where no other phase can explain them, or, for a framework phase, its own strongest line is (a minor
cubic Zn phase next to R-3c has only its (200) line clear of the dense R-3c pattern), or the
whole-pattern fit becomes clearly worse without it.
Without this test, a low-symmetry phase whose many lines lie close to cubic reflections absorbs
intensity mismatch and is reported although it isn't there.

Descriptors from this step:

- **`phase_fractions`**: weight fraction of each identified crystalline phase, summing to 1 over
  the crystalline material. The amorphous share is not included; it is reported as crystallinity.
- **`phase_lattice`**: refined cell of each identified phase.
- **`unidentified_fraction`**: share of the Bragg intensity in peaks no library phase explains.
- **`phase`**: the dominant framework phase, or `amorphous`.
- **Crystallinity index**: framework Bragg intensity ÷ (framework Bragg intensity + amorphous
  halo). The halo is recovered by fitting the SNIP background with a constant, an exponential and a
  Gaussian held near the strongest low-angle line of the dominant framework phase. On the
  simulated deck it ranks samples correctly (correlation 0.97 with the true ordered fraction) but
  reads about 0.15 low. Treat it as a relative measure.

**Accuracy on the simulated deck** (234 random recipes over all six metals and both drying
atmospheres; median and 90th-percentile absolute error of the weight fraction):

| Phase | Median | 90th percentile |
|---|---|---|
| cubic Fm-3m | 0.012 | 0.081 |
| rhombohedral R-3c (Zn) | 0.024 | 0.130 |
| NaCl | 0.015 | 0.062 |
| M(OH)₂, CuO, ZnO | 0.03–0.04 | 0.06–0.12 |
| monoclinic P2₁/n (Mn, Fe) | 0.095 | 0.447 |

The cubic lattice constant is recovered to 0.0002 Å (median). For R-3c-dominant zinc samples the
median error is 0.007 Å in *a* (12.47 Å) and 0.03 Å in *c* (32.9 Å).

**Known limitations.**

- **Cubic + monoclinic Mn/Fe mixtures are not quantified reliably** (median error about 0.17 in
  the monoclinic fraction for mixed samples). The monoclinic doublets are 0.1–0.2° apart and
  merge with the cubic line. Pure samples of either phase are identified correctly. Separating
  mixtures needs a full Rietveld refinement.
- **Poorly ordered Zn mixtures over-report R-3c.** At an ordered fraction of 0.7 the bias is
  +0.04; at 0.5 it is +0.06 to +0.13; at 0.35 or below it is +0.17 to +0.36. The many weak R-3c
  lines take up intensity the cubic phase should get. The crystallinity objective works against
  this in the Pareto front, but a campaign targeting R-3c should check its best low-crystallinity
  results by hand.
- **Unknown phases** appear only as `unidentified_fraction`. Their weight cannot be estimated
  without a structure, so the target-phase objective is discounted by their intensity share instead.
- Intensities come from fixed reference structures. Preferred orientation, microabsorption and
  Na/water occupancy are not refined.

Each campaign has one **target phase**: `pba_fm3m` (default), `pba_p21n` or `znhcf_r3c`. Set it
with `CampaignConfig(target_phase=…)` or `--target-phase`. It is stored with the campaign; resuming
without naming one keeps it, and naming a different one is refused. The campaign maximises two
objectives from the XRD pattern:

| Objective | Definition | Why |
|---|---|---|
| `target_phase_fraction` | Weight fraction of the target phase in the crystalline product × (1 − unidentified intensity share) | Polymorph and secondary-phase selectivity: other framework polymorphs, NaCl residue and hydroxides/oxides all count against it |
| `crystallinity` | XRD crystallinity index (above) | Separates well-ordered frameworks from poorly ordered or amorphous precipitates |

The target phase must be one some metal in the design space can form. For example, `znhcf_r3c`
needs Zn among the allowed metals, and a campaign is refused otherwise.

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
