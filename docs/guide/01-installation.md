# 1. Installation

Requires Python ≥ 3.11. Tested on Linux with Python 3.11 and 3.12.

## Orchestrator

```bash
git clone https://github.com/<your-account>/pba-autoworkflow.git
cd pba-autoworkflow
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest -q                       # optional check
```

This installs the closed loop, the simulated deck, the analysis routines and the optimiser.
The dependencies are NumPy, SciPy, pandas, Matplotlib and pydantic. The Bayesian optimiser is
implemented directly on NumPy/SciPy, so there is no PyTorch or BoTorch dependency.

The command-line tool is installed as `pba-autoworkflow`; `python -m pba_autoworkflow` is equivalent.

## Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `thermo` | ASE, icet | Building PBA supercells, special quasirandom structures, hull bookkeeping |
| `uma` | fairchem-core | The UMA force field (default backend of the thermodynamics module) |
| `petmad` | pet-mad | The PET-MAD force field |
| `mace` | mace-torch | The MACE-MP-0 force field |
| `test` | pytest, pytest-asyncio | The test suite |

```bash
pip install -e ".[test,thermo,uma]"
```

**Install one force-field backend per environment.** `mace-torch` pins `e3nn==0.4.4`, which
conflicts with the versions UMA and PET-MAD need. If you want to compare backends, make one
virtual environment per backend.

## UMA model weights

UMA is licence-gated on HuggingFace:

1. Request access at <https://huggingface.co/facebook/UMA> (usually approved within a day).
2. Create a read token under *Settings → Access Tokens*.
3. Export it before running anything that loads UMA:

```bash
export HF_TOKEN=hf_...          # never commit this
export HF_HUB_DISABLE_XET=1     # helps behind restrictive proxies
```

The weights (`uma-s-1p1`) are downloaded on first use and cached under `~/.cache/huggingface`.
MACE-MP-0 and PET-MAD weights are not gated and download automatically.

## DFT (optional)

The DFT scripts in `scripts/` use GPAW. On a normal workstation or cluster:

```bash
conda install -c conda-forge gpaw
gpaw install-data --gpaw --no-register setups/
export GPAW_SETUP_PATH=$PWD/setups/gpaw-setups-24.11.0
```

Where MPI cannot start (some containers), build GPAW serially from source as described in
[`docs/DFT.md`](../DFT.md).
