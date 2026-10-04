# GPAW for PBA structures — working local build

Serial GPAW 26.7.0, no MPI, running on this machine. Verified end to end.

## Why the conda build failed, and what fixed it

The conda-forge `gpaw` builds link OpenMPI, and OpenMPI could not initialise in the container used for this work:

    UCX  ERROR failed to get boot id
    hwloc/linux: failed to find sysfs cpu topology directory, aborting
    PMIX ERROR: PMIX_ERR_NOT_SUPPORTED in file hwloc/pmix_hwloc.c at line 441
    PMIx_Init failed ... Open MPI requires access to a local PMIx server

That is not a tuning problem. `hwloc` cannot read `/sys` CPU topology in that
container, PMIx needs it, and OpenMPI needs PMIx. Four sets of `OMPI_MCA_*`
workarounds (`ess=singleton`, `plm=isolated`, tmpdir redirection, posix shmem)
all failed identically. There is no environment variable that supplies a missing
`/sys`.

conda-forge does ship `nompi` builds, but only for gpaw **22.8.0** and Python
≤ 3.11 — three years behind, and predating the GPU backend. So instead:

**Build GPAW from source with no `siteconfig`.** MPI is opt-in in GPAW's build
(`siteconfig_example.py` sets `compiler = 'mpicc'` only when `mpi = True`), so a
plain source build is serial by default. One trap: GPAW 26.7 compiles its C
sources with `-DGPAW_CPP -std=c++17` and they `#include <algorithm>`, so the
build must be driven by a **C++** compiler. With plain `gcc` it dies at
`c/bmgs/zero.c:8: fatal error: algorithm: No such file or directory`.

    # env: python 3.12, libxc, openblas, fftw, cxx-compiler, ase, numpy,
    #      scipy, pybind11, setuptools, pydantic
    export CC=x86_64-conda-linux-gnu-c++ CXX=$CC
    export CPATH=$CONDA_PREFIX/include LIBRARY_PATH=$CONDA_PREFIX/lib
    pip wheel . --no-build-isolation --no-deps -w wheels/

Verified serial, not merely untested:

    ldd _gpaw.cpython-312-x86_64-linux-gnu.so | grep -c -E 'libmpi|pmix'   -> 0
    nm -D _gpaw...so | grep -c MPI_Init                                    -> 0
    python -c "from gpaw.mpi import world; print(world.size)"              -> 1

## PAW datasets

Now hosted on GitLab, not the DTU wiki. `gpaw install-data` with no flags also
reaches for `quantum-simulation.org` (SG15 pseudopotentials, unrelated); use
`--gpaw` to fetch only the official PAW setups:

    gpaw install-data --gpaw --no-register dft/setups
    export GPAW_SETUP_PATH=$PWD/dft/setups/gpaw-setups-24.11.0

101 PBE setups, including all ten elements this project needs
(Mn Fe Co Ni Cu C N O H Na).

## Validation

| check | result |
|---|---|
| H₂ total energy, PW(300) | −6.5948 eV |
| Mn atom, spin-polarized PW(400) | −5.7692 eV, moment **5.000 μB** |
| CN radical (the bridging ligand) | −11.1695 eV |

The Mn moment is the load-bearing one: free Mn is d⁵s², so 5 μB is the
high-spin value, and getting it exactly right means spin polarization is working
rather than silently collapsing to a low-spin solution.

## Measured cost — this corrects an earlier estimate

Real y = 0 cell, `scripts/dft_benchmark.py`:

    cell          : 10.530 A cubic, 64 atoms, Z_total = 604
    settings      : PW(340) eV, gamma-point only, spin-polarized
    per iteration : 14.1 s
    peak RSS      : 0.68 GB

**Local CPU-only DFT on these cells is practical.** An earlier, unmeasured
estimate had suggested otherwise. At 14 s per SCF iteration and 0.68 GB, the
numbers are:

| job | cost on this machine |
|---|---|
| single-point SCF (~60 iterations) | ~0.2 h |
| fixed-cell relaxation (~60 × 40) | ~9.4 h |
| lattice scan, 5 points, no ionic relaxation | ~1.2 h |

Only the per-iteration figure and the memory are measured; the rest are linear
extrapolations assuming iteration counts typical of magnetic 3d systems, and
they are labelled as such in the script's own output.

The consequence matters: **the decisive experiment is affordable here.** A
lattice scan at y = 0 and y = 0.25 costs about 2.3 h and settles whether the
0.350 Å contraction MACE predicts is real or an artefact — the question that
`THERMO.md` currently leaves open, and the one blocking any use of a
force-field hull as a campaign prior.

A production answer still wants a cluster: denser k-points, Hubbard U on the 3d
metals, larger supercells for configurational sampling, and the 20–50 structures
a fine-tune needs. But the yes/no discriminator does not.

## Lattice scan result — the test ran, and it does NOT decide

16 single-point SCFs, all converged: y = 0 and y = 0.25 on the Mn series, eight
lattice constants each from 9.65 to 11.75 Å, PW(340) eV, gamma-point,
spin-polarized, frozen fractional coordinates. `scripts/mlff_frozen_scan.py`
scanned UMA the *same frozen way* on the *same grid*, so the comparison is not
confounded by internal relaxation.

| | Δa (y = 0.25 − y = 0) | stable across fit windows? |
|---|---|---|
| **DFT (this work)** | **−0.126 Å**, window spread ±0.090 | no — range −0.221 to −0.042 |
| UMA, matched frozen scan | +0.047 Å, spread ±0.009 | yes |
| MACE, relaxed internals | −0.350 Å | (single value, not re-scannable) |

**Neither force field falls inside the DFT window range.** UMA sits 0.089 Å
outside the nearest edge, MACE 0.129 Å outside the far edge. MACE is further
out, but the DFT uncertainty is 71 % of the DFT value itself, so this does not
convict MACE and does not vindicate UMA. It is an inconclusive experiment,
reported as such.

Four reasons it came out that way, in order of how much they matter:

1. **The y = 0.25 series drifts in spin state across the grid** — total moment
   17.99 → 18.13 → 19.02 → 19.76 → 20.21 → 20.42 → 20.55 μB, and the a = 9.65 Å
   point collapsed to 9.12 μB (excluded from every fit as a different electronic
   solution). Those points are therefore not samples of one E(a) curve. Fixing
   the total moment (`occupations={'name': 'fixmagmom'}`) is the single largest
   improvement available.
2. **The curves are shallow.** Both minima are flat to within ~1 eV over 0.6 Å,
   so the fitted `a_min` moves with the choice of fit window — hence the spread
   quoted above rather than a single number with a false precision.
3. **Frozen internal coordinates.** No ionic relaxation, so water and Na cannot
   move into the vacancy. This was the affordability trade; it is also the
   respect in which the calculation is least like the real material.
4. **PBE overexpands this framework badly.** The DFT y = 0 minimum lands at
   10.85–10.98 Å against a measured 10.53 Å — an error of about +0.35 Å, which
   is *the same size as the MACE-versus-UMA disagreement being adjudicated*. No
   dispersion correction, and a hydrated open framework is precisely where
   dispersion matters. Absolute lattice constants from this setup are not
   trustworthy either.

**Note.** This scan was planned as a decisive test. It was affordable and it
ran, but at this level of theory it does not decide, for the reason given in
point 4: the method's own error on the absolute lattice constant is as large as
the effect being measured.

What a deciding calculation needs, and it is a cluster job rather than a local
afternoon: fixed magnetic moment across the scan, ionic relaxation at each
volume, a dispersion correction (D3 or similar), Hubbard U on the 3d metals, and
k-point sampling beyond gamma. The local build is the right place to develop and
debug that input, which is what it has now been used for.

## Mixing energies — the first DFT adjudication, and MACE fails it on sign

**Terminology first, because it decides what is computable.** A *formation*
energy is referenced to elemental standard states (bulk Mn, bulk Fe, graphite,
N₂ …). None of those were computed here, so no formation energy is reported and
none is claimed. What the hull — and therefore `HullPrior` — actually uses is
the *mixing* energy: the intermediate composition measured against the straight
line joining the endpoints of the charge-balanced path, y = 0 and y = 0.5.
Elemental references cancel in that difference, which is why it is the
well-posed quantity, and why the y = 0.5 endpoint is not optional.

Water is referenced out with a chemical potential from the same functional and
cutoff as the framework (PBE/PW(340) H₂O molecule, **−9.9031 eV**); a potential
taken at a different level of theory would not cancel. `n_water_per_fu` is
linear in y (0, 1.5, 3.0), so the water term cancels along the path by
construction.

23 DFT SCFs, all converged, all three compositions bracketed:

| composition | DFT a_min | DFT E per f.u. | points |
|---|---|---|---|
| y = 0 | 10.849 Å | −102.3947 eV | 8 |
| y = 0.25 | 10.792 Å | −92.7595 eV | 7 |
| y = 0.5 | 10.805 Å | −83.6667 eV | 7 |

| E_mix at y = 0.25 | value | protocol |
|---|---|---|
| **DFT** | **+271.2 meV/f.u.** | frozen coords |
| **UMA** | **+509.2 meV/f.u.** | frozen coords — *matched* |
| UMA | +12.0 meV/f.u. | relaxed coords |
| MACE-MP | −995.9 meV/f.u. | relaxed coords |

**MACE has the sign wrong.** DFT says +271 meV: the intermediate vacancy
fraction is *unstable* against the endpoints, so the system prefers to
demix. MACE says −996 meV: the intermediate is *specially stable*, an ordering
tendency. That is not a magnitude error to be scaled away — it is the opposite
physical prediction. A campaign prior built on it would have pushed the
optimizer toward y ≈ 0.25 precisely because DFT says that composition wants to
phase-separate.

**UMA gets the sign right.** On the matched frozen protocol UMA gives +509 meV
against DFT's +271 — the right sign and the right order of magnitude, too large
by a factor of 1.9. For a foundation model with no fine-tuning on this
chemistry, agreeing on sign and to within 2× on a quantity where the other
model inverts it is a meaningful difference in kind.

DFT's +271 meV is 10.6 × kT at 25 °C, so the demixing tendency is not thermal
noise at this level of theory.

**What this does not establish.** All of it is frozen-coordinate: water and Na
cannot relax into the vacancy, and the vacancy-bearing compositions have the
most to gain from relaxing, so ionic relaxation would lower y = 0.25 and y = 0.5
more than y = 0 and would *reduce* +271 meV by an unknown amount. The spin state
also drifts across each grid (total moment 17.7 → 20.8 μB at y = 0.5), so the
curves are not strictly one electronic state. Still PBE, gamma-point, no
dispersion, no Hubbard U, and the absolute lattice constants remain ~0.3 Å above
the measured 10.53 Å.

So the magnitude is not settled. The **sign** is the robust part, and the sign
is what disqualifies MACE: no amount of ionic relaxation turns −996 meV into a
positive number.

## GPU

`gpaw.gpu.cupy_is_fake` is `True` here, as expected with no GPU. The fake-CuPy
backend (`GPAW_CPUPY=1`) exercises the GPU code path on CPU, so the workflow can
be validated locally and submitted unchanged to a GPU node with
`GPAW(..., parallel={'gpu': True})`.

## Reproducing

On a normal workstation or cluster the conda-forge MPI build of GPAW usually works,
and the source build above is only needed where MPI cannot start. PAW datasets:

    gpaw install-data --gpaw --no-register setups/
    export GPAW_SETUP_PATH=$PWD/setups/gpaw-setups-24.11.0

A built wheel is platform-specific (cp312 / linux_x86_64) and is deliberately not
kept in this repository; rebuild from source with the recipe above.
