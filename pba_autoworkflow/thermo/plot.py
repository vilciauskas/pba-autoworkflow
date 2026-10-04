# SPDX-License-Identifier: GPL-3.0-or-later
"""Figures for the Na/vacancy hull.

The plotting rule this module follows: a panel must not present a computed number
with more authority than the computation has.  Concretely -- unconverged points
are drawn differently from converged ones rather than dropped, the entropy scale
is drawn on the energy axis so a shallow feature cannot be read as a deep one, and
the force-field identity is written on the figure rather than left in the caller's
notes.
"""

from __future__ import annotations

import numpy as np

from .hull import HullResult


def plot_hull(result: HullResult, measured: list[tuple[float, float]] | None = None):
    """Two panels: the hull itself, and the lattice parameter against composition.

    ``measured`` is optional ``(vacancy_fraction, lattice_a_A)`` pairs from indexed
    diffraction patterns.  The lattice panel is the only *independent* check
    available on the force field here -- composition-resolved cell edges are
    measurable, mixing energies are not -- so it is worth plotting whenever
    experimental points exist.
    """
    import matplotlib.pyplot as plt

    pts = sorted(result.points, key=lambda p: p.vacancy_fraction)
    y = np.array([p.vacancy_fraction for p in pts])
    e = np.array([p.mixing_energy_eV * 1000.0 for p in pts])
    a = np.array([p.lattice_a_A for p in pts])
    conv = np.array([p.converged and not p.warnings for p in pts])
    on_hull = np.array([p.on_hull for p in pts])
    std = np.array([p.config_std_eV * 1000.0 if p.config_std_eV is not None else np.nan
                    for p in pts])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    # -- hull ------------------------------------------------------------------ #
    hull_y = y[on_hull]
    hull_e = e[on_hull]
    order = np.argsort(hull_y)
    ax.plot(hull_y[order], hull_e[order], "-", color="#1f77b4", lw=1.8, zorder=2,
            label="lower convex hull")

    if np.isfinite(std).any():
        ax.errorbar(y, e, yerr=std, fmt="none", ecolor="#888888", elinewidth=1,
                    capsize=3, zorder=3,
                    label="configurational spread (1 s.d.)")

    ax.scatter(y[conv & on_hull], e[conv & on_hull], s=70, zorder=4,
               color="#1f77b4", edgecolor="white", linewidth=1.2,
               label="on hull (converged)")
    ax.scatter(y[conv & ~on_hull], e[conv & ~on_hull], s=55, zorder=4,
               facecolor="white", edgecolor="#1f77b4", linewidth=1.6,
               label="above hull (converged)")
    if (~conv).any():
        ax.scatter(y[~conv], e[~conv], marker="x", s=70, zorder=5, color="#d62728",
                   linewidth=2.0, label="not converged / flagged")

    # The entropy scale, so a feature smaller than thermal disorder reads as such.
    scale = result.entropy_scale_eV * 1000.0
    ax.axhspan(-scale, scale, color="#ffcc99", alpha=0.35, zorder=0)
    ax.axhline(0.0, color="#555555", lw=0.8, ls=":", zorder=1)
    ax.text(0.98, scale, f"  $\\pm k_BT\\ln 2$ at {result.temperature_C:.0f} °C",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color="#8a5a00")

    ax.set_xlabel("hexacyanoferrate vacancy fraction $y$")
    ax.set_ylabel("mixing energy (meV / f.u.)")
    ax.set_title(f"Na$_x$ {result.metal}[Fe(CN)$_6$]$_{{1-y}}$ pseudo-binary hull",
                 loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="best")

    # -- lattice parameter ----------------------------------------------------- #
    ax2.plot(y, a, "o-", color="#2ca02c", lw=1.5, markersize=6,
             markeredgecolor="white", label=f"{result.model_name} (relaxed)")
    if measured:
        my = np.array([m[0] for m in measured])
        ma = np.array([m[1] for m in measured])
        ax2.scatter(my, ma, marker="s", s=55, color="#333333", zorder=4,
                    label="measured (indexed XRD)")
    ax2.set_xlabel("hexacyanoferrate vacancy fraction $y$")
    ax2.set_ylabel("cubic lattice parameter $a$ (Å)")
    ax2.set_title("Independent check: composition-resolved cell edge",
                  loc="left", fontsize=11)
    ax2.legend(fontsize=8, frameon=False)

    caveat = (f"{result.model_name}, "
              f"{result.supercell[0]}×{result.supercell[1]}×{result.supercell[2]} cell. "
              "Relative energies within one metal series only — "
              "not formation energies, and not comparable across metals.")
    fig.text(0.005, -0.04, caveat, fontsize=7.5, color="#555555", va="top")
    return fig


def plot_hull_grid(results: list[HullResult]):
    """One panel per metal, on a shared energy axis.

    The shared axis is the point: it shows how large the *between-metal* spread is
    relative to the within-metal features, which is the comparison that tells you
    whether cross-metal conclusions are supportable.  For MACE-MP on this
    chemistry they are not -- the model gets the lattice trend across metals
    backwards -- so this figure exists partly to make that visible.
    """
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, results):
        pts = sorted(res.points, key=lambda p: p.vacancy_fraction)
        y = np.array([p.vacancy_fraction for p in pts])
        e = np.array([p.mixing_energy_eV * 1000 for p in pts])
        oh = np.array([p.on_hull for p in pts])
        ax.plot(y[oh], e[oh], "-", color="#1f77b4", lw=1.6)
        ax.scatter(y[oh], e[oh], s=55, color="#1f77b4", edgecolor="white", zorder=3)
        ax.scatter(y[~oh], e[~oh], s=45, facecolor="white", edgecolor="#1f77b4",
                   linewidth=1.5, zorder=3)
        scale = res.entropy_scale_eV * 1000
        ax.axhspan(-scale, scale, color="#ffcc99", alpha=0.3, zorder=0)
        ax.axhline(0, color="#555", lw=0.8, ls=":", zorder=1)
        ax.set_title(f"{res.metal}", loc="left", fontsize=11)
        ax.set_xlabel("vacancy fraction $y$")
    axes[0].set_ylabel("mixing energy (meV / f.u.)")
    return fig
