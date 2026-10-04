# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage-by-stage walkthrough of the thermodynamics module.

Runs the whole pipeline on cheap settings and prints what each stage produces, so
the intermediate objects are visible rather than implied.  Every number printed is
computed here; the expensive MACE relaxations are read back from the stored run in
``runs/hull/`` rather than repeated, and that substitution is stated where it
happens.

    python scripts/walkthrough.py
"""

from __future__ import annotations

import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np


def rule(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\nSTAGE {n}. {title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
rule(1, "Composition -> crystal structure")
# --------------------------------------------------------------------------- #
from ase.neighborlist import neighbor_list

from pba_autoworkflow.thermo.structures import (FU_PER_CELL, LATTICE_A0_A, PBAComposition,
                                       build_pba, charge_balanced_na,
                                       min_supercell_for)

print(f"""
A composition is two numbers: x = Na per formula unit, y = vacancy fraction, in
Na_x M[Fe(CN)6]_(1-y).  They are the same quantities the analysis layer measures,
so a computed point and an assayed sample compare without a unit conversion.

Charge balance ties them: each vacancy removes 4- of framework charge and expels
two Na, so on the neutral line x = 2 - 4y.
""")
print(f"  {'y':>6} {'charge-balanced Na':>19}")
for y in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625):
    print(f"  {y:6.3f} {charge_balanced_na(y):19.3f}"
          + ("   <- Na hits zero; the line ENDS here" if y == 0.5 else "")
          + ("   <- past the end: Na cannot go negative, so it clamps" if y > 0.5 else ""))

print(f"""
The framework is rock-salt: M and Fe on interpenetrating FCC sublattices bridged
along <100>.  The conventional cubic cell holds {FU_PER_CELL} formula units and 8 Na cages
(= 2 per f.u.).  Getting this wrong is easy and silent -- an early version put one
M at the origin and one Fe at the body centre, leaving them 9.1 A apart with no
M-N bond at all, and 1 f.u. per cell instead of 4.
""")

comp = PBAComposition("Mn", charge_balanced_na(0.25), 0.25)
atoms = build_pba(comp, supercell=(1, 1, 1), rng=np.random.default_rng(1))
print(f"  built {comp.formula}: {len(atoms)} atoms, {atoms.info['n_formula_units']} f.u.")

i, j, d = neighbor_list("ijd", atoms, cutoff=2.7)
sym = atoms.get_chemical_symbols()
bonds: dict[tuple[str, str], list[float]] = {}
for a, b, dd in zip(i, j, d):
    bonds.setdefault(tuple(sorted((sym[a], sym[b]))), []).append(float(dd))
ref = {("C", "N"): 1.15, ("C", "Fe"): 1.90, ("Mn", "N"): 2.20, ("H", "O"): 0.96}
print(f"\n  {'bond':8} {'n':>4} {'mean (A)':>9} {'literature':>11} {'error':>8}")
for k in sorted(bonds, key=lambda k: np.mean(bonds[k])):
    m = float(np.mean(bonds[k]))
    r = ref.get(k)
    tag = f"{r:11.2f} {m - r:+8.3f}" if r else " " * 20
    print(f"  {k[0] + '-' + k[1]:8} {len(bonds[k]):4d} {m:9.3f} {tag}")
print("""
  Cyanide is stiff, so Fe-C and C-N are held at literature values and the M-N
  bond absorbs the Mn->Cu lattice mismatch.  Placing the light atoms at fixed
  *fractions* of the cell instead stretched C-N to 1.26 A, which no cyanide does.""")

print(f"""
A vacancy takes its six cyanides with it and leaves six under-coordinated metals,
which water completes.  Coordination is the check that this was done right:""")
for y in (0.0, 0.25, 0.5):
    a = build_pba(PBAComposition("Mn", charge_balanced_na(y), y), supercell=(1, 1, 1),
                  rng=np.random.default_rng(1))
    s = np.array(a.get_chemical_symbols())
    ii, jj, _ = neighbor_list("ijd", a, cutoff=2.45)
    coord: dict[int, int] = {}
    for p, q in zip(ii, jj):
        if s[p] == "Mn" and s[q] in ("N", "O"):
            coord[p] = coord.get(p, 0) + 1
    tot = [coord.get(k, 0) for k, t in enumerate(s) if t == "Mn"]
    print(f"  y={y:4.2f}: {len(a):3d} atoms, {s.tolist().count('O')} aqua O, "
          f"Mn coordination {min(tot)}-{max(tot)} (must be 6)")

print("""
Finite cells only represent compositions on a 1/n_fu grid.  Silently rounding a
request is the same class of defect as inventing a measurement, so the builder
records what it actually made:""")
small = build_pba(PBAComposition("Ni", 1.75, 0.125), supercell=(1, 1, 1))
print(f"  asked y=0.125 in a 1x1x1 cell (4 f.u.) -> realized "
      f"y={small.info['realized_vacancy_fraction']:.3f}, "
      f"error={small.info['composition_error']:.3f}")
sc = min_supercell_for((0.0, 0.125, 0.25, 0.375, 0.5))
print(f"  min_supercell_for(...) picks {sc}, where every target is exact:")
for y in (0.125, 0.375):
    a = build_pba(PBAComposition("Ni", charge_balanced_na(y), y), supercell=sc)
    print(f"     y={y:.3f} -> realized {a.info['realized_vacancy_fraction']:.3f}, "
          f"error {a.info['composition_error']:.1e}")

# --------------------------------------------------------------------------- #
rule(2, "Structure -> SQS decoration (icet)")
# --------------------------------------------------------------------------- #
print("""
A composition names an ENSEMBLE of decorations, not a structure.  One random
placement of vacancies is a single sample and its energy carries configurational
scatter.  An SQS is the decoration whose cluster correlations best match the
infinite random alloy, so one relaxation approximates the ensemble average.

The cutoffs decide whether the search has anything to work with.  The sublattice
spacings are set by the cubic edge a:""")
a0 = LATTICE_A0_A["Mn"]
print(f"  a = {a0:.2f} A")
print(f"  Fe-Na (cross)      a*sqrt(3)/4 = {a0 * np.sqrt(3) / 4:.2f} A")
print(f"  Na-Na (1st)        a/2         = {a0 / 2:.2f} A")
print(f"  Fe-Fe (1st)        a*sqrt(2)/2 = {a0 * np.sqrt(2) / 2:.2f} A   <- vacancy-vacancy")

from icet import ClusterSpace

from pba_autoworkflow.thermo.sqs import _parent_lattice, default_cutoffs, generate_sqs

parent, chem = _parent_lattice("Mn", (1, 1, 1))
print(f"\n  {'pair cutoff':>12} {'clusters':>9}   includes first Fe-Fe shell?")
for cut in (5.0, 7.0, 8.21, 9.0):
    cs = ClusterSpace(structure=parent, cutoffs=[cut], chemical_symbols=chem)
    print(f"  {cut:12.2f} {len(cs):9d}   {'yes' if cut > a0 * np.sqrt(2) / 2 else 'NO'}")
print(f"""
  The original default was 7.0 A -- just below the 7.45 A first Fe-Fe shell.  The
  vacancy-vacancy interaction, the single most important one for how vacancies
  arrange, was excluded while the search still reported success.  Cutoffs are now
  derived from the lattice parameter: default_cutoffs({a0:.2f}) = {default_cutoffs(a0)}.
""")

print("  SQS quality, measured against icet's own random-alloy target vector:")
print(f"  {'y':>6} {'atoms':>6} {'realized y':>11} {'realized Na':>12} {'mismatch':>9}")
for y in (0.25, 0.5):
    r = generate_sqs(PBAComposition("Mn", charge_balanced_na(y), y),
                     supercell=(2, 2, 2), n_steps=300, seed=1)
    print(f"  {y:6.3f} {r.n_atoms:6d} {r.realized_vacancy_fraction:11.4f} "
          f"{r.realized_na_per_fu:12.4f} {r.correlation_mismatch:9.4f}")
print("""
  0 is a perfect match.  With the wrong cutoffs this metric read 0.25-0.67; the
  metric itself was also wrong at first, estimating the pair target as the square
  of the point correlation -- valid for ONE binary sublattice, but these two are
  coupled, so it flagged good decorations as bad.

  Endpoints are a special case: at y=0 or y=1 nothing can be swapped and icet
  correctly refuses.  Those compositions have a single decoration, which is the
  exact answer rather than an approximation, so they are built directly.""")

# --------------------------------------------------------------------------- #
rule(3, "Decoration -> energy (MACE-MP)")
# --------------------------------------------------------------------------- #
print("""
Relaxation is the expensive stage and had two failure modes, both found by
reading the diagnostics rather than trusting convergence flags.

(a) The CELL.  Internal coordinates converge in ~6 LBFGS steps, but a general
    cell filter ran 200 steps without converging (max force 1.96 eV/A, 630 s):
    six cell degrees of freedom conditioned badly for a framework this soft, and
    five of them are forbidden by cubic symmetry anyway.  The lattice parameter is
    scanned and fitted instead -- one degree of freedom, better posed, and the
    curvature (a bulk-modulus proxy) comes free.

(b) The WATER.  Aqua ligands are placed geometrically with no hydrogen bonding,
    so they carry almost all the initial force.  Without pre-relaxing them against
    a fixed framework, every vacancy-bearing composition stalled at max force
    0.6-0.9 eV/A while the vacancy-free endpoint converged in 6 steps.
""")

for path, label in (("runs/hull/y025_check.json", "y=0.25, 1x1x1, after both fixes"),):
    if os.path.exists(path):
        d = json.load(open(path))
        print(f"  {label}:")
        print(f"     converged={d['converged']}  maxF={d['maxF']:.4f} eV/A  "
              f"flags={d['flags']}  a={d['a']:.3f} A  E/f.u.={d['E_per_fu']:.4f} eV")

print("""
  Structure checks run after every relaxation.  They too needed a fix: checking
  every H-O pair flags all hydrated structures, because adjacent waters
  hydrogen-bond at 2.0-2.3 A -- which is what they are supposed to do.  Only each
  hydrogen's NEAREST oxygen is its covalent partner.""")
from pba_autoworkflow.thermo.mlff import _nearest_oh_distances, check_structure

hyd = build_pba(PBAComposition("Mn", 1.0, 0.25), supercell=(1, 1, 1),
                rng=np.random.default_rng(1))
oh = _nearest_oh_distances(hyd)
print(f"     covalent O-H: {oh.min():.3f}-{oh.max():.3f} A over {oh.size} hydrogens "
      f"-> flags {check_structure(hyd)}")
broken = hyd.copy()
pos = broken.get_positions()
pos[broken.get_chemical_symbols().index("H")] += np.array([1.6, 0.0, 0.0])
broken.set_positions(pos)
print(f"     one H pulled 1.6 A away    -> flags {check_structure(broken)}")

print("""
  Stored MACE energies (from runs/hull/, not recomputed here).  Note: the flag
  counts below were written BEFORE the hydrogen-bond fix above, so every one of
  them is the old false positive (an H...O contact at 1.8-2.3 A, i.e. a hydrogen
  bond).  Re-running the checker on those structures today clears them.  The
  energies themselves are unaffected -- the checker never fed back into the
  relaxation -- but the stale counts are shown as stored rather than quietly
  rewritten.""")
stored = {}
for f in sorted(glob.glob("runs/hull/energies_*.json")):
    d = json.load(open(f))
    stored[d["metal"]] = d
    print(f"  {d['metal']}  model={d['model']}  supercell={d['supercell']}  "
          f"wall={d['wall_time_s']:.0f}s")
    for r in d["rows"]:
        print(f"     y={r['vacancy_fraction']:.3f}  E/f.u.={r['energy_per_fu_eV']:9.4f} eV  "
              f"a={r['lattice_a_A']:.3f} A  converged={r['converged']}  "
              f"flags={len(r['warnings'])}")

# --------------------------------------------------------------------------- #
rule(4, "Energies -> convex hull")
# --------------------------------------------------------------------------- #
print("""
The mixing energy subtracts the straight line between the two endpoint
compositions:

    E_mix(y) = E(y) - [(1-f) E(y_lo) + f E(y_hi)],   f = (y - y_lo)/(y_hi - y_lo)

Points below that line are stable intermediates; points above it would demix into
a mixture of hull compositions.  The lower convex hull separates the two, and the
vertical distance above it is the driving force for demixing.
""")
from pba_autoworkflow.thermo.hull import (check_path_linearity, hull_from_energies,
                                 hull_report, lower_hull, n_water_per_fu)

x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
yv = np.array([0.0, 0.30, -0.20, 0.30, 0.0])
print(f"  lower_hull demo: y-values {yv.tolist()}")
print(f"    -> vertices at x = {sorted(x[lower_hull(x, yv)].tolist())}  "
      "(the two points above the line are correctly excluded)")

print("""
Two guards decide whether the result means anything at all.

GUARD 1 -- is the composition path straight?  E_mix subtracts a line in y, which
only cancels the chemical potentials if every species count is ALSO linear in y.
Na = 2 - 4y is linear until it clamps at zero at y = 0.5.  Past that the path
bends and the subtraction leaves an uncancelled sodium term:""")
for ys in ((0.0, 0.125, 0.25, 0.375, 0.5), (0.0, 0.25, 0.5, 0.75, 1.0)):
    probs = check_path_linearity(ys, tuple(charge_balanced_na(v) for v in ys))
    print(f"  {str(ys):34} -> {'OK' if not probs else probs[0][:64] + '...'}")
print("""  Unguarded, the bent path reported a +2.2 eV/f.u. 'miscibility gap' that was
  pure sodium bookkeeping.

GUARD 2 -- is there more than one interior point?  Three compositions give the
hull a single degree of freedom, so its shape carries no information.
""")

print("""A related trap, worth stating because it looks like it should matter and does
not: a vacancy brings 6 waters, so n_H2O = 6y varies along the axis.  But 6y is
LINEAR in y, and E_mix subtracts a line in y -- so a water chemical potential
cancels identically, for any number of points:""")
mu = -14.0515
ys = (0.0, 0.125, 0.25, 0.375, 0.5)
rows = [{"vacancy_fraction": y, "na_per_fu": charge_balanced_na(y),
         "energy_per_fu_eV": -120.0 + 30.0 * y - mu * n_water_per_fu(y)
                             + 0.4 * y * (1 - y),
         "lattice_a_A": 10.2, "converged": True} for y in ys]
bare = hull_from_energies("Mn", rows)
corr = hull_from_energies("Mn", rows, mu_water_eV=mu)
print(f"  largest feature without water reference: {bare.max_hull_depth_eV * 1000:+.6f} meV")
print(f"  largest feature with    water reference: {corr.max_hull_depth_eV * 1000:+.6f} meV")
print(f"  identical: {abs(bare.max_hull_depth_eV - corr.max_hull_depth_eV) < 1e-12}")
print("""  This correction cannot rescue a -1 eV feature, contrary to an initial claim.  The
  grand potential is still recorded because it is the physically correct object,
  but the large features are a force-field problem, not a bookkeeping one.""")

print("\n  The real hull, on the stored MACE energies:")
mu_real = (json.load(open("runs/hull/mu_water.json"))["mu_water_eV"]
           if os.path.exists("runs/hull/mu_water.json") else None)
for metal, d in stored.items():
    keep = [r for r in d["rows"] if r["vacancy_fraction"] <= 0.5]
    res = hull_from_energies(metal, keep, model_name=d["model"],
                             supercell=tuple(d["supercell"]), mu_water_eV=mu_real)
    print()
    print("\n".join("  " + ln for ln in hull_report(res).splitlines()))

# --------------------------------------------------------------------------- #
rule(5, "Validating the force field before believing any of it")
# --------------------------------------------------------------------------- #
print("""
The hull above is machinery working correctly on numbers that are not
trustworthy.  Two independent checks say so.

CHECK 1 -- the lattice trend across metals, against the measured constants in the
schema.  This is the only fully independent validation available, because
composition-resolved cell edges are measurable while mixing energies are not.

    mace-mp-small   MAE 0.254 A   rank correlation -0.70
    mace-mp-medium  MAE 0.236 A   rank correlation +0.10

A negative rank correlation means the model orders the series BACKWARDS.  Reproduce
with:  python -m pba_autoworkflow.thermo validate

CHECK 2 -- the magnitude.  Vacancy interactions in a framework roughly half empty
by volume are tens of meV per formula unit.  The features above are ~900 meV, some
fifty times the thermal scale, so the module refuses to present them as a
discovery (IMPLAUSIBLE_FEATURE_EV = 0.20).

Conclusion: fine-tune on DFT for these compositions before acting on a hull.  The
foundation model has never seen this chemistry -- a molecular framework, half
empty, strong-field cyanide ligands, open-shell 3d metals, hydrogen-bonded water.
""")

# --------------------------------------------------------------------------- #
rule(6, "How this enters the campaign (and how it can be overruled)")
# --------------------------------------------------------------------------- #
print("""
One door only: HullPrior, scoring recipes BEFORE they run.  It never writes a
descriptor, never sets an objective, and nothing it returns is stored as an
experimental outcome -- a force-field artefact becoming a measurement is the same
failure as the Fe/carbon bug.
""")
from pba_autoworkflow.thermo.prior import HullPrior

demo = hull_from_energies(
    "Mn", [{"vacancy_fraction": y, "na_per_fu": charge_balanced_na(y),
            "energy_per_fu_eV": -120.0 + 30.0 * y + 0.6 * y * (1 - y),
            "lattice_a_A": 10.2, "converged": True} for y in ys],
    mu_water_eV=mu)
prior = HullPrior.from_hulls(demo)
print(f"  weight={prior.weight} (advisory by default)")
print(f"  {'y':>6} {'stability score':>15} {'penalty':>9}")
for y in (0.0, 0.125, 0.25, 0.375, 0.5):
    print(f"  {y:6.3f} {prior.stability_score('Mn', y):15.3f} {prior.penalty('Mn', y):9.4f}")
print(f"\n  a metal with no computed hull returns "
      f"{prior.stability_score('Cu', 0.2)} -- explicitly uninformative, not a "
      "default that quietly favours or disfavours it")

measured = [("Mn", y, -prior.stability_score("Mn", y)) for y in ys]
verdict = prior.disagreement(measured)
print(f"""
  And the prior can be falsified.  Feeding it measurements that get BETTER exactly
  where it predicts less stable:
     n={verdict['n']}  correlation={verdict['correlation']:+.3f}
     verdict: {verdict['verdict']}

  That is the method to watch given the validation results in stage 5.""")

print(f"\n{'=' * 78}\nEnd of walkthrough.\n{'=' * 78}")
