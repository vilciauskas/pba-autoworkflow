# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line entry points.

    python -m pba_autoworkflow run     --runs runs/demo --iterations 6 --batch-size 8
    python -m pba_autoworkflow resume  --runs runs/demo --iterations 3
    python -m pba_autoworkflow report  --runs runs/demo
    python -m pba_autoworkflow status  --runs runs/demo
    python -m pba_autoworkflow dry-run --batch-size 8

``run`` and ``resume`` are the same code path -- a campaign whose id already
exists in the store continues; ``resume`` merely defaults to reusing the id it
finds.  ``dry-run`` plans a batch and prints it without touching a device, which
is what you use to sanity-check a design space before committing reagent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .campaign import Campaign, CampaignConfig
from .devices.simulated import build_simulated_platform
from .optimize.planner import make_planner
from .provenance import ProvenanceStore
from .schema import default_design_space


def _add_campaign_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--runs", default="runs/demo",
                    help="provenance directory (database + raw traces)")
    ap.add_argument("--campaign-id", default=None)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--n-seed", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--planner", default="bayes", choices=["bayes", "sobol", "random"])
    ap.add_argument("--replicate-fraction", type=float, default=0.15)
    ap.add_argument("--max-in-flight", type=int, default=4)
    ap.add_argument("--reactor-capacity", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-phase", default=None,
                    choices=["pba_fm3m", "pba_p21n", "znhcf_r3c"],
                    help="framework phase to optimise for (default: the campaign's "
                         "recorded target, else pba_fm3m)")
    ap.add_argument("--objectives", default=None,
                    help="comma-separated objectives (default: the campaign's recorded "
                         "ones, else target_phase_fraction,crystallinity); e.g. "
                         "target_phase_fraction,k_zn_selectivity,zn_retention")
    ap.add_argument("--time-scale", type=float, default=0.0,
                    help="wall-clock compression for the simulated deck "
                         "(1.0 = real durations, 0.0 = instant)")
    ap.add_argument("--failure-rate", type=float, default=0.04,
                    help="chemistry-driven loss rate (product fails to pellet)")
    ap.add_argument("--mechanical-fault-rate", type=float, default=0.01,
                    help="per-operation terminal hardware fault rate")
    ap.add_argument("--transport-fault-rate", type=float, default=0.02,
                    help="per-operation retryable communication fault rate")
    ap.add_argument("--reproducibility", type=float, default=0.05,
                    help="relative run-to-run scatter of the simulated platform")
    ap.add_argument("--no-report", action="store_true")


def _build(args, campaign_id: str | None) -> tuple[Campaign, ProvenanceStore]:
    store = ProvenanceStore(args.runs)
    cid = campaign_id or args.campaign_id
    if cid is None:
        cid = CampaignConfig().campaign_id
    cfg = CampaignConfig(
        campaign_id=cid,
        planner=args.planner,
        n_seed=args.n_seed,
        batch_size=args.batch_size,
        max_iterations=args.iterations,
        replicate_fraction=args.replicate_fraction,
        max_in_flight=args.max_in_flight,
        reactor_capacity=args.reactor_capacity,
        seed=args.seed,
        target_phase=getattr(args, "target_phase", None),
        objectives=(tuple(o.strip() for o in args.objectives.split(",") if o.strip())
                    if getattr(args, "objectives", None) else None),
    )
    platform, _ = build_simulated_platform(
        seed=args.seed, time_scale=args.time_scale,
        failure_rate=args.failure_rate, reproducibility=args.reproducibility,
        mechanical_fault_rate=args.mechanical_fault_rate,
        transport_fault_rate=args.transport_fault_rate,
        reactor_capacity=args.reactor_capacity,
    )
    campaign = Campaign(platform, store, cfg,
                        progress=lambda m: print(m, flush=True))
    return campaign, store


def _existing_campaign_id(runs: str) -> str | None:
    store = ProvenanceStore(runs)
    rows = store.conn.execute(
        "SELECT campaign_id FROM campaigns ORDER BY created_at DESC LIMIT 1"
    ).fetchall()
    store.close()
    return rows[0]["campaign_id"] if rows else None


def cmd_run(args, resume: bool = False) -> int:
    cid = _existing_campaign_id(args.runs) if resume else args.campaign_id
    if resume and cid is None:
        print(f"no campaign found in {args.runs}", file=sys.stderr)
        return 2
    campaign, store = _build(args, cid)
    asyncio.run(campaign.run())
    summary = campaign.summary()
    print(json.dumps({k: summary[k] for k in
                      ("campaign_id", "n_experiments", "status_counts",
                       "hypervolume", "pareto_size", "bottleneck", "stop_reason")},
                     indent=2, default=str))
    best = summary["best"]
    if best:
        print("\nbest recipe:")
        for k, v in best["recipe"].items():
            print(f"  {k:22s} {v}")
        print(f"  -> {best['formula']}  "
              f"capacity {best['capacity_mAh_g']:.1f} mAh/g  "
              f"yield {best['isolated_yield']:.2f}")
    reps = campaign.replicate_statistics()
    if reps:
        print("\nplatform reproducibility (relative SD over replicate groups):")
        for k, v in reps.items():
            print(f"  {k:24s} {v:.4f}")
    if not args.no_report:
        from .report import write_report

        written = write_report(campaign, Path(args.runs) / "report")
        print("\nwrote:")
        for k, p in written.items():
            print(f"  {k:14s} {p}")
    store.close()
    return 0


def cmd_status(args) -> int:
    cid = args.campaign_id or _existing_campaign_id(args.runs)
    if cid is None:
        print(f"no campaign found in {args.runs}", file=sys.stderr)
        return 2
    store = ProvenanceStore(args.runs)
    history = store.load_campaign(cid)
    counts: dict[str, int] = {}
    for e in history:
        counts[e.status.value] = counts.get(e.status.value, 0) + 1
    print(f"campaign      {cid}")
    print(f"experiments   {len(history)}")
    print(f"status        {counts}")
    print(f"next batch    {store.next_batch_index(cid)}")
    util = store.device_utilization()
    if util:
        print("\ndevice        calls   busy_s   failed")
        for r in util:
            print(f"  {r['device_id']:<12s} {r['n_calls']:>5d} "
                  f"{(r['busy_s'] or 0.0):>8.1f} {r['n_failed']:>7d}")
    store.close()
    return 0


def cmd_report(args) -> int:
    cid = args.campaign_id or _existing_campaign_id(args.runs)
    if cid is None:
        print(f"no campaign found in {args.runs}", file=sys.stderr)
        return 2
    store = ProvenanceStore(args.runs)
    platform, _ = build_simulated_platform(seed=0, time_scale=0.0)
    cfg = CampaignConfig(campaign_id=cid)
    campaign = Campaign(platform, store, cfg)
    from .report import write_report

    written = write_report(campaign, Path(args.runs) / "report")
    for k, p in written.items():
        print(f"{k:14s} {p}")
    store.close()
    return 0


def cmd_dry_run(args) -> int:
    """Plan a batch and print it without touching hardware."""
    space = default_design_space()
    planner = make_planner(args.planner if args.planner != "bayes" else "sobol",
                           space, seed=args.seed)
    sugg = planner.suggest(args.batch_size, [])
    rows = [s.parameters.model_dump() for s in sugg]
    keys = ["metal", "c_metal_M", "c_hcf_M", "c_nacl_M", "c_citrate_M", "ph",
            "temperature_C", "addition_rate_mL_min", "aging_time_h"]
    print("  ".join(f"{k:>12s}" for k in keys))
    for r in rows:
        print("  ".join(
            f"{r[k]:>12.4g}" if isinstance(r[k], (int, float)) else f"{r[k]:>12s}"
            for k in keys
        ))
    problems = 0
    platform, _ = build_simulated_platform(seed=0, time_scale=0.0)
    for s in sugg:
        for msg in platform.synthesis_recipe_check(s.parameters):
            print(f"  ! {msg}")
            problems += 1
    print(f"\n{len(sugg)} recipes, {problems} feasibility problems")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pba_autoworkflow", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    for name in ("run", "resume"):
        p = sub.add_parser(name)
        _add_campaign_args(p)
    for name in ("report", "status"):
        p = sub.add_parser(name)
        p.add_argument("--runs", default="runs/demo")
        p.add_argument("--campaign-id", default=None)
    p = sub.add_parser("dry-run")
    _add_campaign_args(p)

    args = ap.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "resume":
        return cmd_run(args, resume=True)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "dry-run":
        return cmd_dry_run(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
