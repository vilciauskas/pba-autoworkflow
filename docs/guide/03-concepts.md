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
liquid handler   prepare solution A (M²⁺ salt + citrate) and B (Na₄[Fe(CN)₆] or K₃[Fe(CN)₆] + NaCl), adjust pH
reactor          load A, ramp to temperature, meter B into A, age, unload
workup           centrifuge, wash, dry (temperature, pressure, gas), weigh
diffractometer   powder XRD of the solid
elemental        ICP digest of the solid (Na, K, metals, Fe; + C analysis)
IR (optional)    ATR-IR of the cyanide stretch: Fe(II) share
electrochem      cycling in single-ion and mixed electrolytes, then ICP of the electrode
  (only for electrochemical objectives)  and of the spent electrolyte
```

## Design space

The default space (`schema.default_design_space()`) has eleven continuous parameters and three
categorical choices.

| Parameter | Range | Scale | Meaning |
|---|---|---|---|
| `metal` | Mn, Fe, Co, Ni, Cu, Zn | categorical | Divalent metal on the N-coordinated site |
| `c_metal_M` | 0.01–0.3 mol/L | log | M(II) sulfate/chloride in solution A |
| `c_hcf_M` | 0.01–0.3 mol/L | log | Hexacyanoferrate in solution B |
| `hcf_precursor` | Na₄[Fe(CN)₆], K₃[Fe(CN)₆] | categorical | Fe(II) with Na⁺, or Fe(III) with K⁺ |
| `c_nacl_M` | 0–3.5 mol/L | linear | Supporting NaCl; sets the Na⁺ activity |
| `c_citrate_M` | 0–0.45 mol/L | linear | Sodium citrate chelator |
| `ph` | 1–7 | linear | pH of solution A before addition |
| `temperature_C` | 25–90 °C | linear | Precipitation and ageing temperature |
| `addition_rate_mL_min` | 0.05–20 mL/min | log | Metering rate of B into A |
| `aging_time_h` | 0.5–24 h | log | Post-addition ageing |
| `stir_rate_rpm` | 200–1200 rpm | linear | Stirrer set-point |
| `dry_temperature_C` | 25–120 °C | linear | Drying temperature of the washed solid |
| `dry_pressure_mbar` | 0.01–1013 mbar | log | Total pressure of the drying atmosphere |
| `dry_gas` | ambient, dry | categorical | Lab air (or a vacuum oven's residual water), or a desiccant / dry-gas purge |

**Drying** is part of the recipe because, for zinc hexacyanoferrate, the rate of water removal
selects the phase and the temperature sets the Fe reduction (Pilipavicius, Skarnulyte, Gece,
Vilciauskas). Slow dehydration gives anhydrous rhombohedral R-3c, even at room temperature; fast
dehydration into an undersaturated gas leaves a contracted, disordered cubic framework. Every run
records the **drying-rate index**

  Π = (*p*<sub>sat</sub>(*T*) − *p*<sub>w</sub>) / *P*<sub>total</sub>,

the gas's undersaturation divided by the total pressure, which scales the gas-phase transport
resistance. *p*<sub>w</sub> is 0 for `dry` gas; for `ambient` it is 13 mbar (lab air), capped at the
total pressure, because in an evacuated oven the residual gas is the water leaving the sample. The
paper's five routes have Π = 0.023, 1.9 and 1.9 (R-3c) and 246 and 1168 (disordered cubic). The
boundary lies somewhere in 2–250, which has not been measured. Π is a ranking heuristic, not a rate.

The **precursor** sets the Fe valence and the A cation. K₃[Fe(CN)₆] gives an Fe(III) framework,
part of which is reduced and compensated by K⁺; the paper's material is made this way.

Older stored recipes with `dry_atmosphere` still load: `air` becomes 1013 mbar and `vacuum` becomes
1 mbar, both with `ambient` gas, and the precursor defaults to Na₄[Fe(CN)₆].

Log-scaled parameters are searched uniformly in their logarithm. To change ranges or metals, see
chapter 4. Before a batch runs, the platform checks each recipe against physical limits (for
example, stock solubility); `dry-run` shows these warnings.

## From raw data to objectives

| Instrument | Raw data | Descriptors |
|---|---|---|
| XRD | 2θ pattern | Weight fraction and refined cell of each crystalline phase; unidentified intensity share; dominant framework phase; crystallinity index; cubic lattice constant *a*; coherent domain size and microstrain (Williamson–Hall, falling back to Scherrer); FWHM of the strongest framework line |
| ICP + C | Element concentrations, dry mass | A-site occupancy per formula unit for every assayed ion (`a_site`, e.g. Na and K); Fe/M ratio; vacancy fraction *y*; water content; formula; CN deficit 6·Fe − C (decyanation) |
| IR | ATR-IR, 1950–2300 cm⁻¹ | Fe(II) share of the hexacyanoferrate; band positions |
| ICP + IR | — | Charge-balance residual: measured Na + K minus the (1 − *y*)(3 + *f*<sub>Fe(II)</sub>) − 2 the framework charge needs |
| Electrochemistry | Galvanostatic curves; electrode and electrolyte digests | Per ion: formal potential, capacity, retention, plateau fraction; separation factors α; Fe dissolved |
| Balance | Dry mass | Isolated yield |
| Recipe | — | Drying-rate index Π |

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
  simulated deck it ranks samples correctly (correlation 0.96 with the true ordered fraction) but
  reads about 0.18 low. Treat it as a relative measure.

**Accuracy on the simulated deck** (236 random recipes over all six metals, both precursors and
both drying gases; median and 90th-percentile absolute error of the weight fraction):

| Phase | Median | 90th percentile |
|---|---|---|
| cubic Fm-3m | 0.008 | 0.052 |
| rhombohedral R-3c (Zn) | 0.061 | 0.152 |
| NaCl | 0.015 | 0.049 |
| M(OH)₂, CuO, ZnO | 0.02–0.03 | 0.05–0.10 |
| monoclinic P2₁/n (Mn, Fe) | 0.095 | 0.447 |

The P2₁/n row is from an earlier run of the same check with more Na-rich Mn/Fe samples. Only two
samples in this population contain it, because Na-rich recipes now share the A sites with K⁺. The
R-3c error is larger than in that earlier run (0.024 median): with drying now applied, most
random zinc recipes dry fast and give a poorly ordered disordered-cubic framework, where R-3c is
over-reported (see the limitations below).

The cubic lattice constant is recovered to 0.0001 Å (median). For R-3c-dominant zinc samples the
median error is 0.006 Å in *a* (12.47–12.61 Å, hydrated to anhydrous) and 0.06 Å in *c* (32.9 Å).

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
- **Disordered cubic ZnHCF reads as partly R-3c.** For the paper's fast-drying routes (true R-3c
  0.03–0.05) the analysis reports 0.21–0.24 R-3c, the same low-order bias as above.
- **Diffuse scattering is detected only partly.** A rapidly dehydrated framework scatters partly
  diffusely (reflections broadened to a ~2 nm coherence length). The analysis fits such a term to
  the raw pattern minus the Bragg model and counts it as ordered, not amorphous, scattering
  (`diffuse_fraction`). On simulated disordered-cubic Zn samples with a true diffuse share of 0.14,
  it found 0.04–0.08 (median) and nothing in about half the samples. It reported none for ordered
  frameworks. It is applied only to cubic frameworks: against the dense R-3c line list it gave
  false positives.
- **Unknown phases** appear only as `unidentified_fraction`. Their weight cannot be estimated
  without a structure, so the target-phase objective is discounted by their intensity share instead.
- Intensities come from fixed reference structures. Preferred orientation, microabsorption and
  Na/water occupancy are not refined.

Each campaign has one **target phase**: `pba_fm3m` (default), `pba_p21n` or `znhcf_r3c`. Set it
with `CampaignConfig(target_phase=…)` or `--target-phase`. It is stored with the campaign; resuming
without naming one keeps it, and naming a different one is refused. The target phase must be one
some metal in the design space can form; for example, `znhcf_r3c` needs Zn among the allowed metals.

Each campaign also has a set of **objectives**, all maximised and all scaled to [0, 1] with 0 the
worst value. The default is the XRD pair; choose others with `CampaignConfig(objectives=…)` or
`--objectives`. Like the target phase, they are stored and a different set is refused on resume.

| Objective | Definition | Measurement |
|---|---|---|
| `target_phase_fraction` | Weight fraction of the target phase × (1 − unidentified intensity share) | XRD |
| `crystallinity` | XRD crystallinity index (above) | XRD |
| `<a>_<b>_selectivity` | 0.5 + log₁₀ α<sub>a/b</sub> / 8 (α = 1 → 0.5; 10⁴ → 1; 10⁻⁴ → 0). `<b>_<a>_selectivity` is 1 minus it | electrochemistry |
| `<a>_retention` | Discharge capacity retention in the *a* electrolyte | electrochemistry |
| `<a>_capacity` | Second-cycle discharge capacity in *a* / 150 mAh g⁻¹ | electrochemistry |
| `framework_stability` | 1 − fraction of the electrode's Fe found in the mixed electrolyte(s) | electrochemistry |
| `zn_tolerance` | Alias of `zn_k_selectivity`: Zn²⁺ uptake despite K⁺ | electrochemistry |

*a* and *b* are any ions of the ion table, written in lower case: `li`, `na`, `k`, `nh4`, `mg`,
`ca`, `zn`, `al`. Examples: `k_zn_selectivity`, `li_na_selectivity`, `mg_retention`,
`nh4_capacity`. `analysis.objectives.objective_catalogue()` lists every valid name.

A missing measurement scores 0. The ICP composition, IR and charge balance are recorded for every
run but not optimised.

There is one constraint: **isolated yield ≥ 0.35**. Runs below it are kept but marked infeasible.

The objectives trade off against each other, so the campaign does not return a single optimum. It
maps the **Pareto front**: the recipes that cannot be improved in one objective without losing in
another. Progress is measured by the **dominated hypervolume** of the feasible results, with the
origin as the reference point. The report's objective-space panel shows the first two objectives.

## Ions

Everything ion-specific comes from one table, `pba_autoworkflow/ions.py`. Adding an ion means
adding one entry there; objective names, electrolytes, assays and pair keys follow from it.

| Ion | *z* | *r* (Å) | Δ<sub>hyd</sub>*G*° (kJ mol⁻¹) | Δ<sub>hyd</sub>*G*°/*z* | ICP | Notes |
|---|---|---|---|---|---|---|
| Li⁺ | 1 | 0.76 | −475 | −475 | yes | |
| Na⁺ | 1 | 1.02 | −365 | −365 | yes | |
| K⁺ | 1 | 1.38 | −295 | −295 | yes | K₂SO₄ ≤ 1.28 M |
| NH₄⁺ | 1 | 1.48 | −285 | −285 | **no** | taken by coulometry |
| Mg²⁺ | 2 | 0.72 | −1830 | −915 | yes | |
| Ca²⁺ | 2 | 1.00 | −1505 | −753 | yes | CaSO₄ insoluble: nitrate electrolyte |
| Zn²⁺ | 2 | 0.74 | −1955 | −978 | yes | pH < 6 |
| Al³⁺ | 3 | 0.535 | −4525 | −1508 | yes | pH < 3.5 |

Radii are Shannon (1976) six-coordinate values (NH₄⁺: Marcus 1991); hydration Gibbs energies are
the experimental values of Marcus (1991), Table 1.

## Electrochemistry and ion selectivity

Choosing any electrochemical objective adds a stage to the run. For each ion it needs, a fresh
electrode is cast from the powder and cycled (charge first, ending on a discharge) in a 1 mol/L
single-ion electrolyte. For each ion pair, another electrode is cycled in a **mixed electrolyte**,
then digested and assayed, and the spent electrolyte is assayed for Fe.

**XRD gate (optional).** Cycling takes days on real hardware. With
`WorkflowConfig(echem_gate={"target_phase_fraction": 0.5, "crystallinity": 0.25})`, only samples
whose XRD passes the thresholds are cycled. A gated sample keeps its XRD objectives, scores 0 on
the electrochemical ones, and records the reason in `metadata["echem_skipped"]`.

The gate is off by default because a score of 0 hides borderline samples from the optimiser. With
these thresholds, the step-7 campaign of chapter 2 skipped its only two zinc samples (cubic
fractions 0.30 and 0.49), the only runs with any Zn²⁺ tolerance. The optimiser then never
returned to zinc (2 of 29 completed runs), and the hypervolume stayed at 0. Use the gate when
cycling time dominates and the objectives do not depend on borderline samples. Set the thresholds
below the phase purity of the samples you expect to matter.

**Mixed electrolytes.** The ion expected to be taken up preferentially is made dilute against
1 mol/L of the other. Without that, it carries almost all the charge (K⁺ at 0.1 M against 1 M
Zn²⁺ carries more than 99 %), and the other ion's uptake disappears into the measurement error.
The prior guess of which ion is preferred is the less negative Δ<sub>hyd</sub>*G*°/*z* (easier
desolvation per unit charge, the ordering behind the reported PBA sequence K⁺ > Na⁺ > divalent
ions). The minor ion is 10 mM against an ion of the same charge and 5 mM otherwise. Three pairs
are tuned on the simulator and keep their earlier values:

| Pair | Mixed electrolyte |
|---|---|
| K⁺/Zn²⁺ | 1 M Zn²⁺ + 5 mM K⁺ |
| Na⁺/Zn²⁺ | 1 M Zn²⁺ + 20 mM Na⁺ |
| K⁺/Na⁺ | 1 M Na⁺ + 10 mM K⁺ |

`ions.electrolyte_recipe()` turns cation concentrations into salts: sulfates, unless one cation
would exceed its sulfate solubility (then nitrates for all). It also records the lowest hydrolysis
pH limit of the cations (Zn²⁺ < 6, Al³⁺ < 3.5). Each cell's recipe is written to the event log as
an `electrolyte` event. Override the electrolytes with `WorkflowConfig(echem_single_ions=…,
echem_mixed_electrolytes=…)`.

**Separation-factor keys** are written lower charge first, then the more weakly hydrated ion:
`K/Zn`, `Na/Zn`, `K/Na`, `Na/Li`, `Li/Mg`.

From the curves (`analysis/echem.py`):

- **Formal potential** E°′ per ion: midpoint of the charge- and discharge-averaged potentials of the
  second cycle. The first charge also extracts the as-made A cations, so it is not used.
- **Plateau fraction**: share of the discharge capacity delivered where |d*E*/d*q*| < 1 mV per
  mAh g⁻¹. On the simulated deck it is 0.4–0.55 for two-phase (conversion) insertion in R-3c and
  0 for solid-solution insertion in the disordered cubic phase.

From the mixed-electrolyte electrode:

- **Separation factor** α<sub>A/B</sub> = (*x*<sub>A</sub>/*x*<sub>B</sub>)<sub>solid</sub> /
  (*c*<sub>A</sub>/*c*<sub>B</sub>)<sub>electrolyte</sub>, from the inserted amounts per Fe. Concentrations
  stand in for activities, so use matched ionic strength.
- **Ions that cannot be assayed directly** are obtained by coulometry: charge passed in the final
  discharge minus the charge carried by the assayed ions, divided by the ion's charge. This needs
  the active mass and the formula weight from the as-made composition. Two cases need it:
  - The framework's own metal. In zinc hexacyanoferrate from the K/Zn mixed electrolyte, the
    inserted Zn²⁺ is about 0.01 (R-3c) to 0.05 (disordered cubic) per formula unit, against one
    framework Zn: about the size of the assay error.
  - NH₄⁺, which ICP does not see.

  Only one such ion per electrolyte can be recovered. An NH₄⁺/Zn²⁺ mixture on a zinc framework
  gives NaN for both, so measure that pair on another framework or assay NH₄⁺ separately (CHN
  nitrogen or ion chromatography).
- **Predicted α** from the single-ion formal potentials. Per electron, the exchange
  A⁺ + ½ Zn(host) ⇌ A(host) + ½ Zn²⁺ gives

  α<sub>A/B</sub> = (*z*<sub>B</sub>/*z*<sub>A</sub>) exp[*F*(E°′<sub>A</sub> − E°′<sub>B</sub>)/*RT*]
  *c*<sub>A</sub><sup>1/*z*<sub>A</sub> − 1</sup> *c*<sub>B</sub><sup>1 − 1/*z*<sub>B</sub></sup>,

  which for two monovalent ions is the familiar 59 mV per decade. A large gap between measured and
  predicted α would point to kinetic selectivity or to sites that only one ion uses.

**Accuracy on the simulated deck** (the paper's recipe and five drying routes, 6 seeds each): the
measured α<sub>K/Zn</sub> is 0.86 × the simulated true value (median; 10th–90th percentile 0.52–1.77);
the IR Fe(II) share is within 0.07 (median bias −0.01); K per formula unit from ICP is within 0.014.

The simulator's electrochemistry is **illustrative**. Its formal potentials put K⁺ above Na⁺ above
Zn²⁺, as reported for PBAs. The other ions are placed by rule, not fitted:
- other monovalent ions lie on the line through Na⁺ and K⁺ in ionic radius, giving NH₄⁺ > K⁺ >
  Na⁺ > Li⁺;
- other multivalent ions are offset from Zn²⁺ by 0.4 mV per kJ mol⁻¹ of Δ<sub>hyd</sub>*G*°/*z*,
  giving Ca²⁺ > Mg²⁺ > Zn²⁺ > Al³⁺.

Selectivities for those ions in the simulator are therefore consequences of these two rules. Its K–Zn gap is smaller in disordered cubic ZnHCF than in R-3c
(α<sub>K/Zn</sub> ≈ 1.4 × 10³ against 8–9 × 10³). That encodes a **hypothesis**: that the vacancy-related
sites of the disordered framework make it more Zn²⁺-tolerant. It has not been measured; testing it
is what the selectivity objectives are for.

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
parameters, multiplied by a kernel over each categorical choice (metal, precursor). By default
that kernel is exchangeable: every pair of different metals shares one learned correlation, so
information is shared between metals without assuming which are alike. Hyperparameters, including the noise level, are fitted by maximising the
marginal likelihood. Candidates come from global Sobol' coverage plus local perturbations of the
current front. A batch is selected greedily with the *kriging believer* heuristic. Until at least
8 usable results exist, the planner continues with Sobol' points (logged as `sobol-warmup`).

**Physical descriptors for metals (optional).** With `CampaignConfig(use_descriptors=True)` or
`--descriptors`, the metal and precursor are no longer unrelated labels. The correlation between
two metals becomes ρ + (1 − ρ) exp(−|*d*<sub>i</sub> − *d*<sub>j</sub>|² / 2*l*²), where *d* is
the standardised descriptor vector of each metal and *l* is fitted:

- metal: Shannon radius of M²⁺, Pauling electronegativity, Δ<sub>hyd</sub>*G*° of M²⁺, and whether
  M³⁺/M²⁺ is accessible;
- precursor: A-cation radius and Fe oxidation state.

Similar metals are then predicted to behave similarly, which is what allows a prediction for a
metal the campaign has not tried yet. On a synthetic test with one metal held out entirely, it
halved the prediction error for that metal (RMSE 2.07 → 1.08).

In closed-loop campaigns on the simulated deck (8 paired seeds, 12 seed runs plus 5 batches of
8, default XRD objectives), the mean hypervolume after the last batch was 0.804 with descriptors
against 0.792 without. It was higher after every batch, but the difference is within
seed-to-seed scatter (paired t-test p = 0.10–0.23, except p = 0.03 after batch 3, one of five comparisons). The
simulator's metal effects come from hand-made tables, not from the descriptors, so this test can
neither confirm nor rule out a real benefit. For that reason the option is off by default. It
should matter most when the campaign adds metals or ions it has little data on.

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
