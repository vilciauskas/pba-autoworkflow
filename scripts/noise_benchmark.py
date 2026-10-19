# SPDX-License-Identifier: GPL-3.0-or-later
"""Closed-loop benchmark: XRD weight-fraction uncertainty as surrogate noise.

Paired R-3c-targeted campaigns on the simulated deck (same seed, chemistry,
faults and Sobol seed batch in both arms; only use_measurement_noise differs).
Scored on the simulator's TRUE R-3c weight fraction, which the optimiser never
sees:
  best_true  -- true R-3c fraction of the recipe the campaign reports as best
                (best_experiment(), by measured objectives)
  max_true   -- highest true R-3c fraction among all runs
  overclaim  -- measured minus true target fraction of the reported best
"""
import asyncio, json, sys, tempfile, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")


def run(args):
    seed, use = args
    import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
    from pba_autoworkflow.campaign import Campaign, CampaignConfig
    from pba_autoworkflow.devices.simulated import build_simulated_platform
    from pba_autoworkflow.provenance import ProvenanceStore
    plat, be = build_simulated_platform(seed=seed, time_scale=0.0)
    cfg = CampaignConfig(campaign_id=f"n{seed}", seed=seed, n_seed=12, batch_size=8, max_iterations=6,
                         hv_convergence_patience=99, max_batch_failure_rate=1.0,
                         target_phase="znhcf_r3c", use_measurement_noise=use)
    c = Campaign(plat, ProvenanceStore(tempfile.mkdtemp()), cfg)
    asyncio.run(c.run())
    def true_r3c(e):
        rec = be.vessels.get(f"{e.experiment_id}-A")
        return float(rec.latent.xrd_weight_fractions().get("znhcf_r3c", 0.0)) if rec and rec.latent else float("nan")
    usable = [e for e in c.usable]
    best = c.best_experiment()
    return dict(seed=seed, noise=use, n=len(c.history),
                best_true=true_r3c(best) if best else 0.0,
                best_meas=best.objectives.values["target_phase_fraction"] if best else 0.0,
                best_metal=best.parameters.metal if best else None,
                max_true=max((true_r3c(e) for e in usable), default=0.0),
                n_zn=sum(e.parameters.metal == "Zn" for e in usable))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobs = [(s, u) for s in range(n) for u in (False, True)]
    with Pool(min(20, len(jobs))) as pool:
        rows = pool.map(run, jobs)
    json.dump(rows, open("noise_benchmark.json", "w"), indent=1)
    import numpy as np
    from scipy import stats
    for u in (False, True):
        r = [x for x in rows if x["noise"] == u]
        print(("with noise   " if u else "without noise"), "best true R-3c mean %.3f | max true mean %.3f | overclaim (meas-true) mean %+.3f | Zn runs %.1f"
              % (np.mean([x["best_true"] for x in r]), np.mean([x["max_true"] for x in r]),
                 np.mean([x["best_meas"] - x["best_true"] for x in r]), np.mean([x["n_zn"] for x in r])))
    for k in ("best_true", "max_true"):
        d = np.array([[x[k] for x in rows if x["seed"] == s and x["noise"]][0] - [x[k] for x in rows if x["seed"] == s and not x["noise"]][0] for s in range(n)])
        print(f"paired {k} difference (noise - none): mean {d.mean():+.3f}, wins {(d > 1e-9).sum()}/{n}, ties {(np.abs(d) <= 1e-9).sum()}, t-test p={stats.ttest_1samp(d, 0).pvalue:.2f}")
