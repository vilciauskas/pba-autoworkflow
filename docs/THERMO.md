# `pba_autoworkflow.thermo` — MLFF energies and the Na/vacancy convex hull

An optional subpackage: MACE-MP force-field energies for PBA composition series,
SQS decorations via icet, and a pseudo-binary convex hull in sodium content and
hexacyanoferrate vacancy fraction. The campaign runs without it.

## The headline finding: do not trust MACE-MP for this chemistry

Two independent validations both fail.

**Lattice trend across metals.** Relaxing the fully-loaded framework for each
analogue and comparing the cubic edge against the measured constants in
`pba_autoworkflow/schema.py`:

| model | MAE | rank correlation with measurement |
|---|---|---|
| mace-mp-small | 0.254 Å | **−0.70** |
| mace-mp-medium | 0.236 Å | **+0.10** |
| pet-mad-latest | 0.458 Å | **−0.70** |
| uma-s-1p1 (`omat` head) | 0.258 Å | **−0.70** |
| uma-s-1p1 (`odac` head) | 0.485 Å | **−0.70** |

A negative rank correlation means the model orders the metal series backwards.
Cross-metal stability ranking is therefore **not supported** by any of them.
Run it yourself:

    python -m pba_autoworkflow.thermo validate --model mace
    python -m pba_autoworkflow.thermo validate --model petmad
    python -m pba_autoworkflow.thermo validate --model uma --uma-task omat   # or odac

Each of these was a deliberate attempt to escape the previous failure, not a
sweep of whatever was installable:

- **PET-MAD** because the MAD training set is weighted toward distorted and
  off-equilibrium configurations rather than the relaxed ground-state crystals
  that dominate Materials Project.
- **UMA `omat`** because UMA is trained jointly across five domains rather than
  on inorganic crystals alone — and because it is the base potential behind TIP
  (arXiv:2608.14502), the free-energy method this module would most want.
- **UMA `odac`** because that head is trained on MOFs with adsorbed CO₂ and
  water, which is structurally the closest published analogue to a hydrated
  coordination-polymer framework.

**The failure is one shared systematic error, not four coincidences.** The
identical −0.70 is partly an artefact of the discrete Spearman metric at n = 5:
PET-MAD and UMA in fact predict *different* orderings. What they share is the
error's structure. Model error correlates with the measured lattice constant at
r = −1.00 (PET-MAD), −0.90 (UMA `omat`) and −1.00 (UMA `odac`): the larger the
true cell, the more the model under-predicts it. All three place **Cu above Mn**,
which is the exact inversion of the measured trend — Mn has the largest measured
cell of the series and Cu the smallest.

The predicted *spread* is not badly wrong (0.74–0.85 of the measured range), so
this is not insensitivity to the metal. The models resolve a trend of roughly
the right magnitude and point it the wrong way.

**Hypothesis, not a measurement.** The measured ordering
Mn > Co > Fe > Ni > Cu follows the high-spin M²⁺ ionic-radius sequence in
octahedral N-coordination. A PBA puts the divalent metal in a weak field (the N
end of cyanide, high-spin) and Fe in a strong field (the C end, low-spin). A
model that gets that spin-state split wrong would scramble the metal-size trend
in about this way. This is worth testing with DFT — it is *not* established
here, and no calculation in this module measures spin state.

**Environment conflict.** `mace-torch` 0.3.16 pins `e3nn==0.4.4`, while
`pet-mad` and `fairchem-core` require e3nn ≥ 0.5. Installing them together
leaves MACE unable to unpickle its own checkpoints
(`ValueError: too many values to unpack` from `e3nn.util.codegen`). Keep one
environment per backend; the `ASEForceFieldModel` base class exists so the
relaxation protocol is identical across them regardless. The MACE numbers above
were measured before that conflict and have not been re-measured per-metal.

UMA additionally requires accepting Meta's licence on HuggingFace and an
`HF_TOKEN`; without it the download fails with `GatedRepoError`.

**Environment conflict.** `mace-torch` 0.3.16 pins `e3nn==0.4.4`, while
`pet-mad` and `fairchem-core` require e3nn ≥ 0.5. Installing them together
leaves MACE unable to unpickle its own checkpoints
(`ValueError: too many values to unpack` from `e3nn.util.codegen`). Keep one
environment per backend; the `ASEForceFieldModel` base class exists so the
relaxation protocol is identical across them regardless.

**Mixing-energy magnitude.** The Mn and Ni series return features of
−996 and −860 meV per formula unit. Vacancy–vacancy interactions in a framework
roughly half empty by volume are weak — tens of meV per formula unit — so a
feature near an electron-volt is a model failure, not a miscibility gap. The
module says so rather than plotting it as a discovery
(`IMPLAUSIBLE_FEATURE_EV = 0.20`).

**Cross-model test: the thermodynamic error is measured, not just inferred.**
The magnitude argument above is a plausibility argument — it compares against an
expectation, not against a reference calculation. The following compares two
models against *each other* on the identical Mn series (y = 0, 0.25, 0.5; same
SQS decorations, 1×1×1 cell, fmax 0.08, 150 steps):

| model | E_mix(y = 0.25) | lattice spread over the series |
|---|---|---|
| mace-mp-small | −996 meV/f.u. | 0.350 Å |
| uma-s-1p1-omat | ≥ +12 meV/f.u. | 0.060 Å |

A 1.0 eV disagreement, about 39 × kT, that **flips sign**. At least one is badly
wrong, and nothing in either model says which.

UMA's y = 0.5 endpoint did not converge, so its +12 meV is a *bound*, not a
value — but the bound holds in the useful direction. Relaxation minimises energy,
so an unconverged run reports an energy above the true minimum; converging
y = 0.5 further makes E(0.5) more negative, which makes E_mix(0.25) *more*
positive. Reproducing MACE's −996 meV would require E(y = 0.5) to be 2.02 eV
**higher** than measured, which finishing the relaxation cannot do. The sign
disagreement is therefore robust.

**The structural and thermodynamic failures are the same failure.** MACE
contracts the cell by 0.350 Å (≈10 % of volume) at exactly y = 0.25 — the
composition carrying the −996 meV anomaly — and then relaxes back to −0.121 Å at
y = 0.5. A non-monotonic 10 % volume collapse on removing framework units whose
cavity is filled by water is not physical. UMA moves the lattice by 0.060 Å over
the whole series and returns a mixing energy near thermal. The anomalous energy
travels with an anomalous structure; it is not an independent energetic defect.

## UMA is the default backend — and what that is and is not licensed to mean

`DEFAULT_MODEL = "uma"` (task head `omat`). Every entry point now routes through
one registry, `mlff.build_model`. Previously `cli` defaulted to UMA while
`scripts/run_hull.py` and `scripts/mlff_frozen_scan.py` constructed MACE
unconditionally, so the same nominal calculation used different physics
depending on how it was invoked.

**Validated:** the *sign* of the vacancy mixing energy, on a matched
frozen-coordinate protocol against DFT — DFT +271, UMA +509 meV/f.u., while
mace-mp-small gives −996. Sign agreement is the specific claim.

**Not validated, and not claimed:**

- *Magnitude.* UMA is 1.9× DFT on that same quantity. Do not read an absolute
  mixing energy from it.
- *Cross-metal ranking.* UMA still fails the lattice-trend test with rank
  correlation −0.70, the same as MACE and PET-MAD. Being right about vacancies
  within one metal says nothing about ordering Mn against Ni.
- *Absolute lattice constants.* MAE 0.258 Å on the metal series.

So "always UMA" means UMA is the backend when a backend is needed — not that its
numbers are now trustworthy in general. The advisor's stability check still
reports `uninformative`, because the relaxed-coordinate mixing energy (+12 meV)
is below kT and `resolvable` is False. That remains correct.

Three defects surfaced while making the switch:

- **`water_reference_eV(model=None)` silently built a MACE water molecule** and
  cached it under a `mace-mp-*` key, so a UMA hull could be referenced against
  MACE water with nothing in the result disclosing it. The function's own
  docstring said a cross-model reference introduces an error larger than the
  features being resolved — it now raises instead.
- **An unrecognised model name fell through to MACE.** A typo like
  `--model umaa` silently ran the one backend known to invert the sign, and
  reported results under that name. Unknown names now raise.
- **`run_hull.py` hardcoded `"model": f"mace-mp-{args.model}"` in its output**,
  so once `--model` took a backend name, a UMA run would have been written to
  disk labelled `mace-mp-uma`. Those energies are the input to every hull and
  every cross-model comparison.

A missing licence token now raises an error naming the licence page, the
`HF_TOKEN` variable, and `--model petmad` as an ungated alternative, rather than
a bare `GatedRepoError` from inside `huggingface_hub`.

## Wiring into the campaign

UMA is now the default backend (`--model uma`, task head `--uma-task omat`).
Connecting it to the closed loop turned up a structural constraint worth stating
plainly, because it rules out the obvious design.

**A hull cannot score a proposed recipe.** `HullPrior` is indexed by
`(metal, vacancy_fraction)`. `metal` is a synthesis parameter; `vacancy_fraction`
is an *outcome* — derived from a measured Fe/M ratio plus a CHN carbon assay, and
living in `CompositionDescriptors`, not `SynthesisParameters`. The planner sets
concentrations, pH, temperature, addition rate, aging time and stir rate; none of
those tells you the vacancy fraction that will result. Scoring a proposal with
the hull would need a recipe → composition model, and that model is the
surrogate, fitted to the very data the prior is supposed to be independent of.

So the hull enters after measurement, as a falsifiable claim, via
`pba_autoworkflow.thermo.advisor.ThermoAdvisor`. Attach it with
`Campaign(..., thermo_advisor=advisor)`; each iteration records a
`thermo_advice` payload and changes nothing about what gets proposed. Two checks:

- **stability** — `HullPrior.disagreement` over accumulated `(metal, y, objective)`.
  With the current UMA hull this returns `uninformative`, and that is correct: the
  feature is +12 meV, below kT, so `resolvable` is False and there is nothing to
  falsify.
- **lattice** — the sharper test, and the one that pays off today. The models make
  *different* structural predictions along a fixed-metal series: MACE contracts
  the Mn cell 0.350 Å at y = 0.25, UMA moves it 0.060 Å across the whole range.
  XRD measures the cell edge on every sample anyway, so the campaign adjudicates
  between them at no extra experimental cost. It compares *changes* relative to
  the least-vacancy sample, never absolute edges — every model tested is 0.2–0.5 Å
  off in absolute terms, which would swamp the composition dependence being
  tested.

**A defect found while wiring it.** `HullPrior.penalty` returned
`weight × (1 − 0.5) = 0.075` whenever the prior had nothing to say — for a metal
with no computed hull, and for a hull whose feature is sub-thermal — while a
metal that *was* computed and found on its own hull paid `0.000`. Inside the
scalarizer that silently ranked an uncomputed metal below one the force field
happened to like: absence of evidence converted into evidence of absence, in the
one place where nothing reports it. `penalty` now returns exactly `0.0` when
uninformative, and `is_informative(metal)` makes the distinction explicit so
callers stop reading `stability_score`'s 0.5 as a mid-range opinion. A test in
`tests/test_thermo_advisor.py` locks this down; the assertion in
`tests/test_thermo.py` that encoded the old behaviour has been corrected.

Separately, `Planner.suggest(0)` crashed — reachable whenever replicates consume
a whole batch (`n_rep == n`) — because `random_base2` rejects n = 0 and the
categorical round-robin builds a float64 `np.array([])`, which is not a legal
index. All three planners now return `[]` for an empty request.

**Correction to an earlier claim.** The `validate` verdict previously said that
fixed-metal composition series "remain usable — systematic error cancels along a
homologous series." That was never measured, and this test shows it is false:
error does not cancel along the composition axis. The message has been corrected
to report the 1.0 eV disagreement instead.

The machinery is correct and tested. The numbers it currently produces are not
usable. Replace or fine-tune the force field on PBA reference data before acting
on a hull.

## Why the pseudo-binary hull, and not a formation-energy hull

A conventional Materials-Project-style hull against competing binaries would
report nearly every PBA as unstable — which is thermodynamically true and
experimentally useless, since PBAs are hydrated, defective, solution-precipitated
metastable phases. The pseudo-binary hull instead varies composition at *fixed*
metal, where systematic force-field error largely cancels along a homologous
series. That restriction is what makes the result defensible, and it is also why
the failed cross-metal validation above does not invalidate the approach — only
cross-metal conclusions.

## What the module refuses to do

Each of these was a defect found and fixed during development; each now has a
regression test named for the symptom.

- **Report a hull from a bent composition path.** On the charge-balanced line
  Na = 2 − 4y, so every species count is linear in y — until Na clamps at zero at
  y = 0.5. Past that the path bends and the tie-line subtraction leaves an
  uncancelled sodium chemical potential, which showed up as a spurious
  +2.2 eV/f.u. "miscibility gap". `check_path_linearity` catches it and marks the
  hull unresolvable.
- **Report a tendency from one interior point.** Three compositions give the hull
  a single degree of freedom; its shape carries no information.
- **Claim the water reference rescues anything.** A vacancy brings six aqua
  ligands, so n(H₂O) = 6y — linear in y, and therefore cancelling *identically* in
  the mixing-energy subtraction. Verified bit-for-bit. The grand potential is
  recorded because it is the physically correct object, not because it changes the
  hull. (An earlier version of this file claimed the correction brought a −1 eV
  feature down to the thermal scale. It does not; that claim was wrong.)
- **Attribute an energy to a composition the cell cannot hold.** 4 formula units
  cannot represent y = 0.125. `min_supercell_for` picks a cell with no rounding,
  and `build_pba` records `realized_*` alongside the request.
- **Present an unconverged or structurally broken relaxation as a result.**
  Unconverged points are drawn in red rather than dropped; bond-window checks flag
  a model that has torn the framework.

## Structure and physics

Rock-salt framework: M and Fe on interpenetrating FCC sublattices, bridged along
⟨100⟩, four formula units per conventional cubic cell. Bond lengths are held at
literature values (Fe–C 1.90 Å, C–N 1.15 Å) so the lattice mismatch from Mn to Cu
is absorbed by the M–N bond that actually flexes — verified within 0.015 Å.
Vacancies are hydrated by default: six aqua ligands complete the coordination
shells the missing hexacyanoferrate opens, and every metal ends up six-coordinate.
Hydration is on by default on the physical argument that an anhydrous cavity
leaves six metals under-coordinated, which should overestimate the vacancy
formation energy. That expectation is *not* measured here — the anhydrous
comparison run did not complete — so it stands as reasoning, not as a result from
this module. `hydrate_vacancies=False` is exposed if you want to test it.

## Computational cost, measured

On 22 CPU cores, no GPU:

| cell | atoms | single point | full relaxation |
|---|---|---|---|
| 1×1×1 | ~65 | 0.9 s | ~4 min |
| 2×2×2 | ~520 | 7.2 s | ~40 min |

Two things make this tractable. Internal coordinates converge in ~6 LBFGS steps,
but a general cell filter ran 200 steps without converging (max force 1.96 eV/Å);
the cubic lattice parameter is scanned and fitted instead — one degree of freedom
rather than six, better posed and faster, and it yields the curvature for free.
Second, water is placed geometrically with no hydrogen bonding and carries almost
all the initial force, so it is pre-relaxed with the framework fixed. Without that
every vacancy-bearing composition stalled at max force 0.6–0.9 eV/Å while the
vacancy-free endpoint converged in 6 steps.

A GPU or a remote host would change the accessible cell size substantially; none
was configured for these runs.

## Usage

    python -m pba_autoworkflow.thermo validate                      # check the force field FIRST
    python -m pba_autoworkflow.thermo hull --metal Mn --supercell 2
    python -m pba_autoworkflow.thermo hull --metal Mn --model analytic   # machinery, no force field
    python scripts/run_hull.py --metal Mn --workers 5      # one composition per core

`AnalyticModel` is a deliberately-named stand-in so the hull, plotting and
campaign code can be exercised without loading a network; nothing carrying
`model="analytic"` can be mistaken for a computed energy.

## How this enters the campaign

Through exactly one door, `HullPrior`, and only as a prior over *where to look*.
The orchestrator's rule is that predictions must never reach the measurement path
— a force-field artefact becoming an experimental result is the same failure as
the Fe/carbon bug. So the prior scores recipes before they run, never writes a
descriptor or an objective, defaults to a low weight, and returns 0.5
(uninformative) for a metal with no hull or a hull below the thermal scale.

`HullPrior.disagreement()` reports where the prior anti-correlates with
accumulated measurements — the mechanism by which the campaign can tell you the
computation was misleading. Given the validation results above, that is the method
to watch.
