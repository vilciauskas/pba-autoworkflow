# 7. Troubleshooting

**The campaign stopped with "platform health: N/M of the batch failed or was quarantined".**
More than half of a batch (`max_batch_failure_rate`, default 0.5) failed or was quarantined. This
is deliberate: a deck that is misbehaving should not keep consuming reagents. Run `status` to see
which device is failing, and read the per-experiment reasons in `report/experiments.csv`. Fix the
cause, then `resume`. On the simulated deck with small batches this can also fire by chance; use
larger batches or lower fault rates.

**`unknown objective(s) [...]`.**
Electrochemical objective names are built from ion symbols in lower case: `<a>_<b>_selectivity`,
`<a>_retention`, `<a>_capacity`, with ions `li`, `na`, `k`, `nh4`, `mg`, `ca`, `zn`, `al`. The two
ions of a selectivity must differ. `analysis.objectives.objective_catalogue()` lists every valid
name.

**A separation factor is NaN.**
Two ions in the same mixed electrolyte could not be assayed directly: for example NH₄⁺ (invisible
to ICP) together with the framework's own metal (Zn²⁺ in zinc hexacyanoferrate). Only one such
ion per electrolyte can be recovered from the charge passed. Measure the pair on another framework
or assay NH₄⁺ separately. A NaN α scores 0.

**Electrochemical objectives are 0 for many samples.**
If `WorkflowConfig.echem_gate` is set, samples below its XRD thresholds are not cycled and score
0; the reason is in `metadata["echem_skipped"]` and in an `echem_gated` event. Lower the
thresholds or set `echem_gate=None` (chapter 3 explains why a gate can stall a campaign).

**`run` continued an old campaign instead of starting a new one.**
`run` resumes when the `--campaign-id` already exists in `--runs`. Use a new id or a new `--runs`
directory. A store is never overwritten.

**`ProvenanceStore(":memory:")` creates a folder called `:memory:`.**
`ProvenanceStore` takes a directory, not a SQLite path. In tests, use pytest's `tmp_path`.

**The planner keeps proposing Sobol' points (`sobol-warmup`) after the seed batch.**
The Gaussian process needs at least 8 usable results. Failed and quarantined runs do not count.
Increase `--n-seed` or reduce the fault rates.

**`dry-run` reports "citrate exceeds stock solubility budget" (or another recipe problem).**
The recipe is outside what the deck can prepare. Such recipes are reported before any hardware is
used. If the warning appears often, narrow the design space (chapter 4) or override
`Platform.synthesis_recipe_check()` with your deck's actual limits (chapter 5).

**The station-occupancy panel is empty, or shows only the analysis stations.**
With `--time-scale 0` no instrument time is simulated, so the panel shows only analysis compute and
says so in its title. Use `--time-scale 2e-4` or larger to see contention between stations.

**`GatedRepoError` or HTTP 401 when loading UMA.**
Request access to `facebook/UMA` on HuggingFace and export `HF_TOKEN` (chapter 1). Behind a proxy,
also set `HF_HUB_DISABLE_XET=1`.

**`ImportError` involving e3nn after installing a second force field.**
MACE and UMA/PET-MAD need incompatible e3nn versions. Use one environment per backend.

**GPAW fails with `PMIx_Init failed` or a hwloc error.**
MPI cannot start in that container. Build GPAW serially from source (see [`docs/DFT.md`](../DFT.md)).

**Force-field relaxations are very slow.**
Large PBA supercells are expensive on a CPU. Start with `--supercell 1`, set `OMP_NUM_THREADS` to
the number of physical cores, or run on a GPU.

## Running the tests

```bash
pip install -e ".[test,thermo]"
pytest -q -k "not one_degree and not stays_cubic and not overlap"   # skips three slow force-field tests
```

Tests that need a force-field backend are skipped automatically when the backend is not installed.
