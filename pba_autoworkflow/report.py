# SPDX-License-Identifier: GPL-3.0-or-later
"""Campaign reporting: the figures and tables you look at after a run.

Kept separate from :mod:`pba_autoworkflow.campaign` so that reporting can be re-run against
a stored campaign without touching the loop, and so the loop carries no plotting
dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis.objectives import OBJECTIVE_NAMES, scalarize
from .campaign import Campaign
from .optimize.planner import pareto_mask
from .schema import ExperimentStatus


def _simulated_time_scale(campaign: Campaign) -> float | None:
    """Wall-clock compression factor of a simulated deck, or ``None`` if real.

    Real drivers have no such attribute, so ``None`` means "these durations are
    physical" and the occupancy figures can be read at face value.
    """
    for device in campaign.platform.all_devices():
        backend = getattr(device, "backend", None)
        ts = getattr(backend, "time_scale", None)
        if ts is not None:
            return float(ts)
    return None


def campaign_table(campaign: Campaign):
    """Per-experiment table, ordered as run."""
    df = campaign.dataframe()
    if len(df) == 0:
        return df
    scalars = {}
    for e in campaign.history:
        scalars[e.experiment_id] = (
            scalarize(e.objectives) if e.objectives is not None else np.nan
        )
    df["scalar"] = df["experiment_id"].map(scalars)
    return df


def iteration_table(campaign: Campaign):
    import pandas as pd

    return pd.DataFrame([
        {
            "iteration": r.iteration,
            "n": r.n_suggested,
            "complete": r.counts.get("complete", 0),
            "quarantined": r.counts.get("quarantined", 0),
            "failed": r.counts.get("failed", 0),
            "hypervolume": r.hypervolume,
            "hv_delta": r.hv_delta,
            "best_scalar": r.best_scalar,
            "wall_time_s": r.wall_time_s,
            "bottleneck": r.bottleneck,
            "planner_note": r.planner_note,
        }
        for r in campaign.iterations
    ])


def plot_campaign(campaign: Campaign, path: str | Path):
    """Four-panel campaign overview.

    Panels, in the order a reviewer asks for them: is the loop learning
    (hypervolume against experiment count, with the seed boundary marked), where
    the samples landed in objective space and which are on the front, what the
    platform actually did (status counts per batch), and where the time went
    (station occupancy).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    ax_hv, ax_obj, ax_status, ax_util = axes.ravel()

    # -- 1. learning curve -------------------------------------------------- #
    usable = campaign.usable
    if campaign.iterations:
        n_per = np.cumsum([r.n_suggested for r in campaign.iterations])
        hv = [r.hypervolume for r in campaign.iterations]
        ax_hv.plot(n_per, hv, marker="o", color="#1f4e79", lw=1.8)
        n_seed = campaign.iterations[0].n_suggested
        ax_hv.axvline(n_seed, color="0.55", ls="--", lw=1.0)
        ax_hv.annotate("end of seed design", xy=(n_seed, ax_hv.get_ylim()[0]),
                       xytext=(4, 8), textcoords="offset points",
                       fontsize=8, color="0.35", rotation=90, va="bottom")
    ax_hv.set_xlabel("experiments run")
    ax_hv.set_ylabel("dominated hypervolume")
    ax_hv.set_title("Closed-loop progress", loc="left", fontsize=11)

    # -- 2. objective space ------------------------------------------------- #
    if usable:
        Y = np.array([[e.objectives.values[n] for n in OBJECTIVE_NAMES]
                      for e in usable])
        feas = np.array([e.objectives.feasible for e in usable])
        batch = np.array([e.batch_index for e in usable])
        ax_obj.scatter(Y[~feas, 0], Y[~feas, 1], s=26, facecolors="none",
                       edgecolors="0.6", linewidths=0.9,
                       label="yield constraint violated")
        sc = ax_obj.scatter(Y[feas, 0], Y[feas, 1], c=batch[feas], s=42,
                            cmap="viridis", edgecolors="white", linewidths=0.5,
                            label="feasible")
        if feas.any():
            Yf = Y[feas]
            mask = pareto_mask(Yf)
            order = np.argsort(Yf[mask, 0])
            ax_obj.plot(Yf[mask, 0][order], Yf[mask, 1][order], color="#c1272d",
                        lw=1.4, marker="D", ms=5, label="Pareto front", zorder=5)
            fig.colorbar(sc, ax=ax_obj, label="batch", pad=0.02)
        ax_obj.legend(fontsize=8, frameon=False, loc="lower left")
    ax_obj.set_xlabel(f"{OBJECTIVE_NAMES[0].replace('_', ' ')} →")
    ax_obj.set_ylabel(f"{OBJECTIVE_NAMES[1].replace('_', ' ')} →")
    ax_obj.set_title("Objective space", loc="left", fontsize=11)

    # -- 3. platform outcomes ----------------------------------------------- #
    batches = sorted({e.batch_index for e in campaign.history})
    statuses = [ExperimentStatus.COMPLETE, ExperimentStatus.QUARANTINED,
                ExperimentStatus.FAILED]
    colors = {"complete": "#2a6f4e", "quarantined": "#d9a441", "failed": "#8c2f39"}
    bottom = np.zeros(len(batches))
    for st in statuses:
        counts = np.array([
            sum(1 for e in campaign.history
                if e.batch_index == b and e.status is st)
            for b in batches
        ], dtype=float)
        ax_status.bar(batches, counts, bottom=bottom, label=st.value,
                      color=colors[st.value], width=0.72)
        bottom += counts
    ax_status.set_xlabel("batch")
    ax_status.set_ylabel("experiments")
    ax_status.set_xticks(batches)
    ax_status.legend(fontsize=8, frameon=False)
    ax_status.set_title("Platform outcomes per batch", loc="left", fontsize=11)

    # -- 4. station occupancy ----------------------------------------------- #
    # Occupancy only describes the *platform* when instrument dwell times were
    # actually simulated.  With ``time_scale = 0`` every dwell collapses to zero
    # and the measured busy time is analysis CPU instead, which ranks the stations
    # in a completely different order -- the elemental analyzer looks like the
    # bottleneck because its digest fit is the slowest computation, whereas with
    # realistic dwells the reactor dominates by an order of magnitude.  Both
    # numbers are real; only one answers "where is the platform's bottleneck".
    # So the panel reports which one it is showing rather than presenting the
    # convenient interpretation.
    rep = campaign.pool.report()
    if rep:
        ts = _simulated_time_scale(campaign)
        dwell_simulated = ts is None or ts > 0.0
        names = [str(r["station"]) for r in rep][::-1]
        vals = [float(r["occupancy"]) for r in rep][::-1]
        bars = ax_util.barh(names, vals, color="#3b6ea5", height=0.62)
        if vals:
            ax_util.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax_util.set_xlim(0, max(1.0, max(vals) * 1.25) if vals else 1.0)
        bn = campaign.pool.bottleneck()
        if dwell_simulated:
            ax_util.set_xlabel("fraction of campaign wall time occupied")
            title = ("Station occupancy — bottleneck: " + bn) if bn else \
                    "Station occupancy — no single bottleneck"
        else:
            ax_util.set_xlabel("fraction of wall time occupied "
                               "(analysis compute only)")
            # two lines: on one line this title ran past the figure edge
            title = ("Station occupancy\ninstrument dwell not simulated "
                     "(--time-scale 0)")
        ax_util.set_title(title, loc="left", fontsize=11)

    out = Path(path)
    fig.savefig(out, dpi=180)
    return out


def plot_best_pattern(campaign: Campaign, path: str | Path):
    """Diffractogram of the best sample with the indexed reflections marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = campaign.best_experiment()
    if best is None or "xrd" not in best.raw_refs:
        return None
    # Reflection labels come from the indexer's own table, never a local copy --
    # a second hand-maintained mapping is how a figure ends up labelled
    # inconsistently with the reflection table in the same report.
    from .analysis.xrd import (
        _HKL_LABELS as labels,
        find_and_fit_peaks,
        index_cubic,
        snip_background,
    )
    from .schema import XRDPattern

    arr = campaign.store.load_trace(best.raw_refs["xrd"])
    pattern = XRDPattern(arr["two_theta_deg"], arr["intensity"])
    peaks, tt, stripped = find_and_fit_peaks(pattern)
    a, assign, _ = index_cubic(peaks, pattern.wavelength_A)

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(9.0, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]}, constrained_layout=True,
    )
    ax.plot(pattern.two_theta_deg, pattern.intensity, color="0.25", lw=0.8,
            label="measured")
    ax.plot(tt, snip_background(np.asarray(pattern.intensity, float)),
            color="#c1272d", lw=1.0, ls="--", label="fitted background")
    for m, p in sorted(assign.items()):
        ax.axvline(p.two_theta_deg, color="#3b6ea5", lw=0.7, alpha=0.5)
        ax.annotate(labels.get(m, str(m)), xy=(p.two_theta_deg, ax.get_ylim()[1]),
                    xytext=(0, -12), textcoords="offset points", fontsize=7,
                    ha="center", color="#1f4e79")
    comp = best.descriptors.composition
    d = best.descriptors.xrd
    subtitle = (f"{comp.formula if comp else '?'} — "
                f"a = {a:.4f} Å, D = {d.domain_size_nm:.0f} nm, "
                f"{d.phase}" if d else "")
    ax.set_ylabel("intensity (counts)")
    ax.set_title(f"Best sample: {best.experiment_id}\n{subtitle}",
                 loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    ax_r.plot(tt, stripped, color="#2a6f4e", lw=0.8)
    ax_r.axhline(0.0, color="0.6", lw=0.7)
    ax_r.set_xlabel("2θ (degrees, Cu Kα)")
    ax_r.set_ylabel("background-\nstripped")

    out = Path(path)
    fig.savefig(out, dpi=180)
    return out


def write_report(campaign: Campaign, directory: str | Path) -> dict[str, Path]:
    """Write the full set of campaign deliverables into ``directory``."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    table = campaign_table(campaign)
    p = d / "experiments.csv"
    table.to_csv(p, index=False)
    written["experiments"] = p

    it = iteration_table(campaign)
    p = d / "iterations.csv"
    it.to_csv(p, index=False)
    written["iterations"] = p

    written["summary"] = campaign.write_summary(d / "summary.json")

    fig = plot_campaign(campaign, d / "campaign_overview.png")
    written["overview"] = fig
    pat = plot_best_pattern(campaign, d / "best_pattern.png")
    if pat is not None:
        written["best_pattern"] = pat
    return written
