# SPDX-License-Identifier: GPL-3.0-or-later
"""Compute a Na/vacancy hull, one composition per worker process.

Each composition is an independent relaxation, so the series parallelizes across
cores with no communication.  Torch is pinned to one thread per worker: MACE on a
520-atom cell does not scale well across threads, and letting each worker grab all
cores makes the pool slower than running serially.

Usage:
    python scripts/run_hull.py --metal Mn --supercell 2 --workers 10
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

# Must be set before torch is imported anywhere in a worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _one(args: tuple) -> dict:
    """Relax one composition in a fresh process and return plain data."""
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import torch

    torch.set_num_threads(1)

    (metal, y, supercell, use_sqs, seed, fmax, steps, model_size,
     n_config_samples) = args

    from pba_autoworkflow.thermo.hull import _decoration
    from pba_autoworkflow.thermo.mlff import build_model
    from pba_autoworkflow.thermo.structures import PBAComposition, charge_balanced_na

    # model_size carries 'name:size' so the worker stays picklable
    _name, _, _size = model_size.partition(':')
    model = build_model(_name or 'uma', size=_size or 'small')
    comp = PBAComposition(metal, charge_balanced_na(y), y)
    sc = (supercell, supercell, supercell)

    t0 = time.time()
    atoms = _decoration(comp, sc, use_sqs, seed)
    res = model.relax(atoms, fmax=fmax, steps=steps)

    std = None
    if n_config_samples > 0:
        from pba_autoworkflow.thermo.sqs import sample_decorations

        energies = [model.relax(a, fmax=fmax, steps=steps).energy_per_fu_eV
                    for a in sample_decorations(comp, n_samples=n_config_samples,
                                                supercell=sc, seed=seed + 1000)]
        std = float(np.std(energies, ddof=1)) if len(energies) > 1 else None

    return {
        "metal": metal, "vacancy_fraction": y,
        "na_per_fu": float(atoms.info.get("realized_na_per_fu", comp.na_per_fu)),
        "energy_per_fu_eV": res.energy_per_fu_eV,
        "lattice_a_A": res.lattice_a_A, "converged": res.converged,
        "max_force_eV_A": res.max_force_eV_A, "n_steps": res.n_steps,
        "n_atoms": res.n_atoms, "warnings": res.extrapolation_warnings,
        "config_std_eV": std, "formula": comp.formula,
        "sqs_method": atoms.info.get("sqs_method", "sqs" if use_sqs else "random"),
        "wall_time_s": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metal", default="Mn")
    ap.add_argument("--vacancies", default="0.0,0.125,0.25,0.375,0.5")
    ap.add_argument("--supercell", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="uma",
                    choices=("uma", "petmad", "mace", "analytic"),
                    help="backend name (default uma)")
    ap.add_argument("--model-size", default="small",
                    help="only meaningful for --model mace")
    ap.add_argument("--fmax", type=float, default=0.08)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--config-samples", type=int, default=0)
    ap.add_argument("--no-sqs", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/hull")
    args = ap.parse_args()

    ys = [float(v) for v in args.vacancies.split(",")]
    jobs = [(args.metal, y, args.supercell, not args.no_sqs, args.seed + i,
             args.fmax, args.steps, f"{args.model}:{args.model_size}", args.config_samples)
            for i, y in enumerate(ys)]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import multiprocessing as mp

    t0 = time.time()
    rows: list[dict] = []
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(_one, jobs):
            rows.append(r)
            print(f"  y={r['vacancy_fraction']:.3f} E/fu={r['energy_per_fu_eV']:.4f} "
                  f"a={r['lattice_a_A']:.3f} conv={r['converged']} "
                  f"F={r['max_force_eV_A']:.3f} steps={r['n_steps']} "
                  f"({r['wall_time_s']:.0f}s)", flush=True)

    rows.sort(key=lambda r: r["vacancy_fraction"])
    payload = {
        # Record what actually ran.  This field used to be hardcoded to
        # f"mace-mp-{args.model}", so once --model became a backend name a UMA
        # run would have been written to disk labelled "mace-mp-uma" -- and the
        # stored energies are the input to every hull and every comparison.
        "metal": args.metal,
        "model": (f"mace-mp-{args.model_size}" if args.model == "mace"
                  else args.model),
        "supercell": [args.supercell] * 3, "fmax": args.fmax,
        "wall_time_s": time.time() - t0, "rows": rows,
    }
    path = out / f"energies_{args.metal}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {path} in {payload['wall_time_s']:.0f}s")


if __name__ == "__main__":
    main()
