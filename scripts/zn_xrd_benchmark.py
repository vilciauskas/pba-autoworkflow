# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark of R-3c / cubic quantification on simulated zinc hexacyanoferrate.

A. Controlled grid: true R-3c share s, ordered fraction (crystallinity) c,
   disordered-cubic share of the cubic phase dc (contracted cell + diffuse
   component, as the simulator models fast drying), coherent domain d.
   Secondary phases are removed so only the framework polymorphs compete.
B. The paper's five drying routes (K3[Fe(CN)6] recipe), full simulator.

Usage: python zn_xrd_benchmark.py LABEL [n_noise_seeds]
Writes zn_bench_<LABEL>.csv and prints the summary tables.
"""
import copy, math, sys, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

S_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
C_GRID = (0.25, 0.35, 0.5, 0.7, 0.9)
DC_GRID = (0.0, 1.0)
D_GRID = tuple(float(v) for v in __import__("os").environ.get("D_GRID", "8,25").split(","))
ROUTES = [("RT, P2O5", 20, 1013, "dry", "R-3c"), ("RT, 8 mbar", 20, 8, "ambient", "R-3c"),
          ("120 C, air", 120, 1013, "ambient", "R-3c"), ("RT, 0.02 mbar", 20, 0.02, "ambient", "DC"),
          ("120 C, 8 mbar", 120, 8, "ambient", "DC")]
PAPER = dict(metal="Zn", hcf_precursor="K3FeCN6", c_metal_M=0.1, c_hcf_M=0.05, c_nacl_M=0.0,
             c_citrate_M=0.0, ph=5.5, temperature_C=25.0, addition_rate_mL_min=1.0,
             aging_time_h=1.0, stir_rate_rpm=600)


def _analyse(lat, seed):
    import os
    if os.environ.get("MIN_LINE_EVIDENCE"):
        import pba_autoworkflow.analysis.phases as PH
        PH.MIN_LINE_EVIDENCE = float(os.environ["MIN_LINE_EVIDENCE"])
    from pba_autoworkflow.sim.instruments import simulate_xrd
    from pba_autoworkflow.analysis.xrd import analyze_pattern
    t0 = time.time()
    x = analyze_pattern(simulate_xrd(lat, np.random.default_rng(seed)), "Zn")
    unc = getattr(x, "phase_fraction_sd", None) or {}
    return x, time.time() - t0, unc


def grid_job(args):
    s, c, dc, d, k = args
    from pba_autoworkflow.schema import SynthesisParameters
    from pba_autoworkflow.sim.ground_truth import GroundTruth, DC_DIFFUSE_SHARE
    gt = GroundTruth(seed=1, reproducibility=0.0, failure_rate=0.0)
    p = SynthesisParameters(**PAPER, dry_temperature_C=20.0, dry_pressure_mbar=1013.0, dry_gas="ambient")
    base = gt.precipitate(p, np.random.default_rng(5))
    lat = copy.deepcopy(base)
    lat.dried = True
    a_hyd = lat.lattice_a_A
    lat.polymorph_shares = {k_: v for k_, v in {"pba_fm3m/Zn": 1 - s, "znhcf_r3c/Zn": s}.items() if v > 1e-6}
    lat.crystallinity = c
    lat.domain_size_nm = d
    lat.nacl_fraction = 0.0
    lat.hydroxide_fraction = 0.0
    lat.diffuse_share = {}
    lat.disordered_cubic_share = dc if s < 1 else 0.0
    if lat.disordered_cubic_share > 0:
        lat.diffuse_share["pba_fm3m/Zn"] = DC_DIFFUSE_SHARE * lat.disordered_cubic_share
        lat.lattice_a_A = a_hyd * (1.0 - lat.disordered_cubic_share * (1.0 - 0.984 ** (1 / 3)))
    truth = lat.xrd_weight_fractions()
    x, dt, unc = _analyse(lat, 1000 + k + int(__import__("os").environ.get("SEED_OFFSET", "0")))
    return dict(set="grid", s=s, order=c, dc=dc, d_nm=d, seed=k,
                r3c_true=truth.get("znhcf_r3c", 0.0), r3c_meas=x.phase_fractions.get("znhcf_r3c", 0.0),
                cub_true=truth.get("pba_fm3m", 0.0), cub_meas=x.phase_fractions.get("pba_fm3m", 0.0),
                dcph_meas=x.phase_fractions.get("pba_fm3m_dc", 0.0),
                r3c_sd=unc.get("znhcf_r3c", float("nan")), unid=x.unidentified_fraction,
                cryst_meas=x.crystallinity_index, diffuse_true=lat.framework_diffuse_fraction(),
                diffuse_meas=x.diffuse_fraction, t_s=dt)


def route_job(args):
    name, T, P, g, meas, k = args
    from pba_autoworkflow.schema import SynthesisParameters
    from pba_autoworkflow.sim.ground_truth import GroundTruth
    gt = GroundTruth(seed=k, failure_rate=0.0)
    p = SynthesisParameters(**PAPER, dry_temperature_C=T, dry_pressure_mbar=P, dry_gas=g)
    off = int(__import__("os").environ.get("SEED_OFFSET", "0"))
    lat = gt.latent(p, np.random.default_rng(100 + k + off))
    truth = lat.xrd_weight_fractions()
    x, dt, unc = _analyse(lat, 200 + k + off)
    return dict(set="route", route=name, paper_phase=meas, seed=k, dc=lat.disordered_cubic_share,
                order=lat.crystallinity, d_nm=lat.domain_size_nm,
                r3c_true=truth.get("znhcf_r3c", 0.0), r3c_meas=x.phase_fractions.get("znhcf_r3c", 0.0),
                cub_true=truth.get("pba_fm3m", 0.0), cub_meas=x.phase_fractions.get("pba_fm3m", 0.0),
                dcph_meas=x.phase_fractions.get("pba_fm3m_dc", 0.0),
                r3c_sd=unc.get("znhcf_r3c", float("nan")), unid=x.unidentified_fraction,
                cryst_meas=x.crystallinity_index, diffuse_true=lat.framework_diffuse_fraction(),
                diffuse_meas=x.diffuse_fraction, t_s=dt)


def summarise(D: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    G = D[D.set == "grid"].copy()
    G["err"] = G.r3c_meas - G.r3c_true
    print("A. R-3c error (meas - true), median over seeds and domain sizes; rows: order, cols: true share s")
    for dc in DC_GRID:
        print(f"   disordered-cubic share of cubic phase = {dc:g}")
        t = G[G.dc == dc].pivot_table(index="order", columns="s", values="err", aggfunc="median")
        print(t.round(3).to_string())
    z = G[G.s == 0.0]
    print("   false R-3c (> 0.05) in pure cubic: %d / %d" % ((z.r3c_meas > 0.05).sum(), len(z)))
    o = G[G.s == 1.0]
    print("   cubic reported (> 0.05) in pure R-3c: %d / %d" % ((o.cub_meas + o.dcph_meas > 0.05).sum(), len(o)))
    e = G.err.abs()
    print("   |err| median %.3f  p90 %.3f  max %.3f | at order <= 0.35: median %.3f p90 %.3f"
          % (e.median(), e.quantile(.9), e.max(), e[G.order <= 0.35].median(), e[G.order <= 0.35].quantile(.9)))
    if G.r3c_sd.notna().any():
        cov = (e <= 2 * G.r3c_sd).mean()
        print("   reported sd: median %.3f; |err| <= 2 sd in %.0f %% of samples" % (G.r3c_sd.median(), 100 * cov))
    R = D[D.set == "route"]
    for dc in DC_GRID:
        g = G[(G.dc == dc) & (G.s < 1)]
        print("   diffuse share, dc=%g: true median %.3f, measured median %.3f, measured > 0 in %d / %d"
              % (dc, g.diffuse_true.median(), g.diffuse_meas.median(), (g.diffuse_meas > 0).sum(), len(g)))
    print("B. Paper routes (median over seeds)")
    print(R.groupby("route", sort=False)[["dc", "order", "r3c_true", "r3c_meas", "cub_meas", "diffuse_true", "diffuse_meas", "r3c_sd"]]
          .median().round(3).to_string())
    print("runtime: median %.2f s, max %.2f s per pattern" % (D.t_s.median(), D.t_s.max()))


if __name__ == "__main__":
    label = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    grid = [(s, c, dc, d, k) for s in S_GRID for c in C_GRID for dc in DC_GRID for d in D_GRID for k in range(n)]
    routes = [r + (k,) for r in ROUTES for k in range(2 * n + 2)]
    with Pool(20) as pool:
        rows = pool.map(grid_job, grid, chunksize=4) + pool.map(route_job, routes)
    D = pd.DataFrame(rows); D.to_csv(f"zn_bench_{label}.csv", index=False)
    summarise(D)
