# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line for the thermodynamics module.

    python -m pba_autoworkflow.thermo hull --metal Mn --supercell 2
    python -m pba_autoworkflow.thermo hull --metal Mn --model analytic     # no force field
    python -m pba_autoworkflow.thermo validate                            # lattice check

The ``validate`` subcommand is the one to run first on any new force field: it
relaxes the fully-loaded framework for each metal and compares the cubic edge
against the measured values.  A model that fails this has no business ranking
compositions, and finding that out costs minutes rather than a campaign.
"""

from __future__ import annotations

import argparse
import json
import pathlib


UMA_TASK = "omat"


#: The project default.  UMA is the only backend tested here that agrees with
#: DFT on the SIGN of the vacancy mixing energy (DFT +271, UMA +509 meV/f.u. on
#: a matched frozen-coordinate protocol); mace-mp-small inverts it (-996).  See
#: DFT.md for the calculation and for what remains unvalidated.
DEFAULT_MODEL = "uma"


def _model(name: str, size: str):
    """Delegate to the one backend registry in ``mlff.build_model``."""
    from .mlff import build_model

    return build_model(name, size=size, task=UMA_TASK)


def cmd_hull(args) -> int:
    from .hull import compute_hull, hull_report

    ys = tuple(float(v) for v in args.vacancies.split(","))
    model = _model(args.model, args.model_size)
    res = compute_hull(
        args.metal, model, vacancy_values=ys,
        supercell=(args.supercell,) * 3, temperature_C=args.temperature,
        use_sqs=not args.no_sqs, n_config_samples=args.config_samples,
        fmax=args.fmax, steps=args.steps, seed=args.seed, progress=True,
    )
    print()
    print(hull_report(res))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_dataframe().to_csv(out / f"hull_{args.metal}.csv", index=False)

    from .plot import plot_hull

    fig = plot_hull(res)
    fig.savefig(out / f"hull_{args.metal}.png", dpi=170, bbox_inches="tight")
    print(f"\nwrote {out}/hull_{args.metal}.csv and .png")
    return 0


def cmd_validate(args) -> int:
    """Relax each analogue's loaded framework and score against measurement."""
    import numpy as np
    from scipy.stats import spearmanr

    from .mlff import check_structure
    from .structures import LATTICE_A0_A, PBAComposition, build_pba

    model = _model(args.model, args.model_size)
    print(f"model: {getattr(model, 'name', args.model)}")
    print(f"{'metal':6s} {'measured':>9s} {'relaxed':>9s} {'error':>8s}  flags")
    exp, got = [], []
    for metal in ("Mn", "Fe", "Co", "Ni", "Cu"):
        atoms = build_pba(PBAComposition(metal, 2.0, 0.0),
                          supercell=(args.supercell,) * 3)
        res = model.relax(atoms, fmax=args.fmax, steps=args.steps)
        exp.append(LATTICE_A0_A[metal])
        got.append(res.lattice_a_A)
        print(f"{metal:6s} {LATTICE_A0_A[metal]:9.3f} {res.lattice_a_A:9.3f} "
              f"{res.lattice_a_A - LATTICE_A0_A[metal]:+8.3f}  "
              f"{'; '.join(res.extrapolation_warnings) or 'ok'}")

    exp_a, got_a = np.array(exp), np.array(got)
    rho = float(spearmanr(exp_a, got_a).statistic)
    mae = float(np.abs(got_a - exp_a).mean())
    print(f"\nMAE = {mae:.3f} A    rank correlation = {rho:+.3f}")
    if rho < 0.5:
        print(
            "  VERDICT: this force field does not reproduce the lattice trend across\n"
            "  metals.  Cross-metal stability ranking is NOT supported.\n"
            "\n"
            "  Fixed-metal composition series are NOT rescued by error cancellation.\n"
            "  An earlier version of this message claimed they were; that claim was\n"
            "  never measured and is now known to be false.  On the identical Mn\n"
            "  series (y = 0, 0.25, 0.5; same SQS decorations, same settings),\n"
            "  mace-mp-small gives E_mix(0.25) = -996 meV/f.u. while uma-s-1p1-omat\n"
            "  gives >= +12 meV/f.u. -- a 1.0 eV disagreement that flips sign, about\n"
            "  39x kT.  Systematic error does not cancel along the composition axis."
        )
    else:
        print("  VERDICT: lattice trend reproduced; cross-metal comparison is "
              "defensible to within the stated MAE.")

    if args.out:
        pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out, "lattice_validation.json").write_text(json.dumps(
            {"model": getattr(model, "name", args.model), "mae_A": mae,
             "spearman": rho,
             "rows": [{"metal": m, "measured_A": e, "relaxed_A": g}
                      for m, e, g in zip(("Mn", "Fe", "Co", "Ni", "Cu"), exp, got)]},
            indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pba_autoworkflow.thermo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="uma",
                        choices=("mace", "petmad", "uma", "analytic"))
    common.add_argument("--uma-task", default="omat",
                        choices=("omat", "odac", "omc", "omol", "oc20"),
                        help="UMA task head; omat for crystals, odac for "
                             "hydrated frameworks")
    common.add_argument("--model-size", default="small",
                        choices=("small", "medium", "large"))
    common.add_argument("--supercell", type=int, default=2)
    common.add_argument("--fmax", type=float, default=0.06)
    common.add_argument("--steps", type=int, default=60)
    common.add_argument("--out", default="runs/hull")

    h = sub.add_parser("hull", parents=[common], help="compute a Na/vacancy hull")
    h.add_argument("--metal", default="Mn")
    h.add_argument("--vacancies", default="0.0,0.125,0.25,0.375,0.5")
    h.add_argument("--temperature", type=float, default=25.0)
    h.add_argument("--config-samples", type=int, default=0)
    h.add_argument("--no-sqs", action="store_true")
    h.add_argument("--seed", type=int, default=1)
    h.set_defaults(func=cmd_hull)

    v = sub.add_parser("validate", parents=[common],
                       help="check the force field against measured lattice constants")
    v.set_defaults(func=cmd_validate)

    args = ap.parse_args(argv)
    if getattr(args, "uma_task", None):
        global UMA_TASK
        UMA_TASK = args.uma_task
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
