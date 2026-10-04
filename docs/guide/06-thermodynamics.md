# 6. Thermodynamics module

`pba_autoworkflow.thermo` computes the energetics of Na/vacancy disorder in a PBA framework with
machine-learned interatomic potentials, and can report alongside a campaign how those predictions
compare with what the deck measures. It is optional and needs the `thermo` extra plus one
force-field backend (chapter 1).

> **Read this first.** The force fields tested here are **not validated** for quantitative PBA
> thermodynamics. Use the module as a diagnostic. It does not tell the optimiser what to do; see
> [Validation status](#validation-status).

## What it computes

For one metal M, the module builds supercells of Na<sub>x</sub>M[Fe(CN)<sub>6</sub>]<sub>1−y</sub>
along the charge-balanced path, with water in the vacancies. It distributes vacancies as
special quasirandom structures (icet), relaxes each cell, and builds the **mixing-energy hull**:
the energy of each composition relative to the straight line between the end members, with water
referenced out through a water chemical potential.

```bash
# hull for Mn at three vacancy fractions with UMA (needs HF_TOKEN)
python -m pba_autoworkflow.thermo hull --metal Mn --model uma --vacancies 0,0.25,0.5 --out runs/hull

# check a force field against measured lattice constants of five PBAs
python -m pba_autoworkflow.thermo validate --model uma --uma-task omat --out runs/validate
```

| Option | Meaning |
|---|---|
| `--model` | `uma` (default), `petmad`, `mace`, or `analytic` (a fast stand-in for testing) |
| `--uma-task` | UMA head: `omat` (inorganic crystals, default) or `odac` (frameworks with adsorbed water); `omc`, `omol`, `oc20` also exist |
| `--model-size` | `small`, `medium` or `large` (MACE) |
| `--supercell` | Supercell repetitions |
| `--vacancies` | Comma-separated vacancy fractions *y* |
| `--config-samples` | Extra random configurations per composition |
| `--no-sqs` | Use random instead of quasirandom vacancy placement |
| `--fmax`, `--steps` | Relaxation convergence (eV/Å) and step limit |

Relaxations of these cells are expensive on a CPU: a 520-atom cell took about 40 minutes per
composition with MACE in this project. Use a GPU or start with `--supercell 1`.

The module flags results it cannot support rather than reporting them silently. Unconverged
relaxations and implausible structures (for example broken metal coordination) are flagged; a
lattice-constant minimum at the edge of the scanned range is marked unreliable; and a hull whose
composition path is not linear in every species is marked as not interpretable, so the advisor
ignores it.

## The campaign advisor

`ThermoAdvisor` compares the force-field hull with each batch's measurements and records a verdict
in the iteration record. It is **read-only**: the hull is indexed by the vacancy fraction, which is
a measured outcome and not a recipe parameter, so the planner cannot use it to choose recipes.
From [`examples/thermo_advisor.py`](../../examples/thermo_advisor.py):

```python
from pba_autoworkflow.thermo.advisor import ThermoAdvisor
from pba_autoworkflow.thermo.hull import compute_hull
from pba_autoworkflow.thermo.mlff import build_model
from pba_autoworkflow.thermo.prior import HullPrior

model = build_model("uma")            # or "analytic" to try the mechanics in seconds
hulls = [compute_hull(m, model, vacancy_values=(0.0, 0.25, 0.5)) for m in ("Mn", "Ni")]
advisor = ThermoAdvisor(HullPrior.from_hulls(*hulls))

campaign = Campaign(platform, store, cfg, thermo_advisor=advisor)
```

After each batch, `record.thermo_advice` holds:

| Key | Meaning |
|---|---|
| `stability_verdict` | `agrees` or `disagrees` (measured trend against the hull); `insufficient-data`; or `uninformative` when the hull has no resolvable feature for any measured metal |
| `stability_correlation`, `n_stability` | The correlation behind the verdict, and how many results it used |
| `lattice_verdict`, `lattice_residual_A`, `n_lattice` | The same comparison for lattice constants, if predicted lattice constants were supplied |
| `notes` | Plain-language explanation |

At least 4 usable results on metals with an informative hull are needed before a verdict is given.
With an unfine-tuned foundation model, `uninformative` is the expected verdict. If the advisor itself fails,
the error is recorded and the batch is unaffected.

## Validation status

All numbers below are in [`docs/key_results.json`](../key_results.json); the methods are in
[`docs/THERMO.md`](../THERMO.md) and [`docs/DFT.md`](../DFT.md).

**Cross-metal lattice trend** (cubic lattice constant against measurement for Mn, Fe, Co, Ni, Cu):

| Model | MAE (Å) | Spearman ρ |
|---|---|---|
| MACE-MP-0 small | 0.254 | −0.70 |
| MACE-MP-0 medium | 0.236 | +0.10 |
| PET-MAD | 0.458 | −0.70 |
| UMA `omat` | 0.258 | −0.70 |
| UMA `odac` | 0.485 | −0.70 |

None of the five reproduces the ordering of lattice constants across the metal series, so none
should be used to rank metals.

**Mixing energy of Mn at *y* = 0.25** (meV per formula unit; *kT* at 25 °C = 25.7 meV):

| Method | E<sub>mix</sub> |
|---|---|
| DFT (PBE, frozen coordinates) | +271.2 |
| UMA `omat`, matched frozen coordinates | +509.2 |
| UMA `omat`, relaxed | +12.0 |
| MACE-MP-0 small, relaxed | −995.9 |

DFT and UMA agree on the **sign** (positive: no ordered vacancy phase at *y* = 0.25), and MACE
gives the opposite sign. That is the only result that has been validated, and only for Mn.
Magnitudes differ by about a factor of two, and the DFT itself lacks dispersion, Hubbard U and
ionic relaxation.

What would make the module quantitative: fine-tuning the force field on 20–50 DFT calculations of
your own compositions, at a level of theory that includes dispersion and Hubbard U.
