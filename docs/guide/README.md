# PBA Autonomous Workflow: user guide

PBA Autonomous Workflow is a closed-loop orchestrator for the synthesis of Prussian blue
analogues (PBAs), Na<sub>x</sub>M[Fe(CN)<sub>6</sub>]<sub>1−y</sub>·nH<sub>2</sub>O. It proposes
co-precipitation recipes, runs them on an automated deck, reduces the powder XRD pattern to phase
purity, crystallinity and lattice descriptors (with ICP for composition), and uses multi-objective
Bayesian optimisation to choose the next batch. The current focus is phase formation.

Read the chapters in order the first time; after that, each one stands alone.

| Chapter | What it covers |
|---|---|
| [1. Installation](01-installation.md) | Python environments, optional extras, model weights |
| [2. Quick start](02-quickstart.md) | A complete campaign on the simulated deck, command by command |
| [3. Concepts](03-concepts.md) | Design space, objectives, quality control, planners, stopping rules, provenance |
| [4. Python API](04-python-api.md) | Running and customising campaigns from code; reading results |
| [5. Connecting real instruments](05-real-hardware.md) | The device protocols, error classes, and an example driver |
| [6. Thermodynamics module](06-thermodynamics.md) | Force-field hulls, the campaign advisor, and what has been validated |
| [7. Troubleshooting](07-troubleshooting.md) | Common problems and their fixes |

Runnable examples are in [`examples/`](../../examples):

- [`run_campaign.py`](../../examples/run_campaign.py): a campaign with a restricted design space
- [`custom_device.py`](../../examples/custom_device.py): a real-instrument driver template (watch-folder diffractometer), with a self-test
- [`thermo_advisor.py`](../../examples/thermo_advisor.py): a campaign with the force-field advisor attached

Background documents: [`THERMO.md`](../THERMO.md) (force-field validation),
[`DFT.md`](../DFT.md) (GPAW build and DFT results),
[`PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md) (technical summary of the project).
Software and method references: [`REFERENCES.md`](../../REFERENCES.md).

> **Status.** The orchestrator, analysis and optimiser are complete and tested. All campaign
> results in this repository come from the **simulated deck**; the package has not yet been run on
> laboratory hardware. Connecting it is described in chapter 5.
