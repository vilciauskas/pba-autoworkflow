# SPDX-License-Identifier: GPL-3.0-or-later
"""Build ``pba_autoworkflow/data/phase_library.json`` from reference structures.

Run once after adding or changing a reference CIF (needs pymatgen, which the
package itself does not require at runtime)::

    python scripts/build_phase_library.py

For each phase the library stores, per group of reflections that stay
degenerate under the lattice strain allowed for that phase, the summed
``multiplicity * |F|^2`` per unit cell, together with the reference lattice,
the cell mass (amu, hydrogen included) and the cell volume.  That is everything
a whole-pattern phase quantification needs: line positions follow from the
(refined) lattice, line intensities from ``|F|^2`` times the Lorentz-
polarization factor, and weight fractions from the Hill-Howard relation
``w_p  ~  S_p * (Z M V)_p``.

Structure factors use the same atomic scattering-factor parameterisation as
pymatgen's ``XRDCalculator`` and a single isotropic displacement parameter
``B_ISO`` for all atoms (hexacyanometallate frameworks with disordered water
typically refine to B = 1-3 A^2).

Reference phases
----------------
Framework phases
  * ``pba_fm3m``  Na_xM[Fe(CN)6]_(1-y), Fm-3m.  For Mn, Fe, Co, Ni, Cu an
    idealised model (Na 1 per f.u. on 8c, y = 0.1, coordinated water on the
    vacant N sites, Fe-C 1.92 A, C-N 1.15 A) at the reference lattice constant
    of ``schema.LATTICE_A0_A``.  For Zn the refined cubic Zn3[Fe(CN)6]2.xH2O
    structure, COD 2020370.
  * ``pba_p21n``  Na2M[Fe(CN)6].2H2O, P2_1/n (Na-rich "Prussian white" type),
    COD 7047887 for Mn; the Fe variant substitutes Fe for Mn and scales the cell
    by a0(Fe)/a0(Mn).
  * ``znhcf_r3c`` Na2Zn3[Fe(CN)6]2.9H2O, R-3c, COD 2106959.
Secondary phases
  * ``nacl``      NaCl, COD 9003308
  * ``m_oh2``     brucite-type M(OH)2: Mn COD 9009111, Fe COD 9009104,
                  Co COD 1548810, Ni COD 1548811
  * ``cuo``       tenorite CuO, COD 7212242
  * ``zno``       zincite ZnO, COD 2300450
"""
from __future__ import annotations

import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS  # noqa: E402
from pymatgen.core import Element, Lattice, Structure  # noqa: E402
from pymatgen.io.cif import CifParser  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pba_autoworkflow.schema import LATTICE_A0_A  # noqa: E402

CIF_DIR = ROOT / "pba_autoworkflow" / "data" / "phases"
OUT = ROOT / "pba_autoworkflow" / "data" / "phase_library.json"

B_ISO = 2.0          # A^2, all atoms
D_MIN_A = 1.02       # covers 2theta <= 98 deg at Cu Ka1
REL_FLOOR = 1e-5     # drop reflection groups weaker than this fraction of the strongest
H_MASS = 1.00794


def _cif_meta(path: Path) -> dict:
    txt = path.read_text(errors="replace")

    def field(k):
        m = re.search(rf"^{k}\s+(.+?)\s*$", txt, re.M)
        return m.group(1).strip().strip("'\"") if m else ""
    authors = re.findall(r"^\s*'([^']+)'\s*$",
                         txt.split("_publ_author_name", 1)[1].split("_", 1)[0], re.M) \
        if "_publ_author_name" in txt else []
    formula = field("_chemical_formula_sum")
    z = field("_cell_formula_units_Z")
    return {
        "cod_id": path.stem.replace("COD_", ""),
        "journal": field("_journal_name_full"), "year": field("_journal_year"),
        "volume": field("_journal_volume"), "page": field("_journal_page_first"),
        "authors": authors[:4], "formula_sum": formula,
        "Z": float(z) if z else None,
    }


def _missing_h_mass(structure: Structure, meta: dict) -> float:
    """Mass of H listed in the CIF formula but absent from the atom sites."""
    m = re.search(r"\bH(\d*\.?\d*)\b", meta["formula_sum"] or "")
    if not m or not meta["Z"]:
        return 0.0
    h_formula = float(m.group(1) or 1.0) * meta["Z"]
    h_sites = float(structure.composition.get("H", 0.0))
    return max(h_formula - h_sites, 0.0) * H_MASS


def _f0(symbol: str, s2: np.ndarray) -> np.ndarray:
    """Atomic scattering factor, same parameterisation as pymatgen XRDCalculator."""
    coeffs = np.array(ATOMIC_SCATTERING_PARAMS[symbol])
    z = Element(symbol).Z
    return z - 41.78214 * s2 * np.sum(coeffs[:, 0][None, :] * np.exp(-coeffs[:, 1][None, :] * s2[:, None]), axis=1)


def structure_factor_table(structure: Structure, strain: str) -> list[dict]:
    """Reflection groups that remain degenerate under ``strain``.

    ``strain`` is ``"iso"`` (one scale for all lengths: cubic, monoclinic),
    or ``"ac"`` (independent a and c: hexagonal / trigonal).
    """
    lat = structure.lattice
    rec = lat.reciprocal_lattice_crystallographic
    hmax = [int(math.ceil(lat.abc[i] / D_MIN_A)) + 1 for i in range(3)]
    hs = np.array([(h, k, l) for h in range(-hmax[0], hmax[0] + 1)
                   for k in range(-hmax[1], hmax[1] + 1)
                   for l in range(-hmax[2], hmax[2] + 1) if (h, k, l) != (0, 0, 0)])
    g = hs @ rec.matrix
    inv_d = np.linalg.norm(g, axis=1)
    keep = inv_d <= 1.0 / D_MIN_A
    hs, inv_d = hs[keep], inv_d[keep]
    s2 = (inv_d / 2.0) ** 2

    F = np.zeros(len(hs), dtype=complex)
    fcache: dict[str, np.ndarray] = {}
    for site in structure:
        for sp, occ in site.species.items():
            sym = sp.symbol
            if sym not in fcache:
                fcache[sym] = _f0(sym, s2)
            phase = 2j * np.pi * (hs @ site.frac_coords)
            F += occ * fcache[sym] * np.exp(phase)
    F2 = (F * F.conjugate()).real * np.exp(-2.0 * B_ISO * s2)

    if strain == "ac":
        h, k, l = hs.T
        keys = list(zip(h * h + h * k + k * k, np.abs(l)))
    else:
        keys = list(np.round(1.0 / inv_d, 5))
    groups: dict = {}
    for key, hkl, f2, d in zip(keys, hs, F2, 1.0 / inv_d):
        gk = groups.setdefault(key, {"hkl": [int(x) for x in hkl], "f2m": 0.0, "d": float(d)})
        gk["f2m"] += float(f2)
    rows = [v for v in groups.values()]
    fmax = max(r["f2m"] for r in rows)
    rows = [r for r in rows if r["f2m"] > REL_FLOOR * fmax]
    rows.sort(key=lambda r: -r["d"])
    return rows


def idealised_pba(metal: str, na_per_fu: float = 1.0, vacancy: float = 0.1) -> Structure:
    a = LATTICE_A0_A[metal]
    x_c, x_n = 1.92 / a, (1.92 + 1.15) / a
    occ = 1.0 - vacancy
    species = [{"Fe": occ}, {metal: 1.0}, {"C": occ}, {"N": occ, "O": vacancy},
               {"Na": na_per_fu / 2.0}]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [x_c, 0, 0], [x_n, 0, 0], [0.25, 0.25, 0.25]]
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), species, coords)


def load_cif(cod_id: str) -> tuple[Structure, dict]:
    path = CIF_DIR / f"COD_{cod_id}.cif"
    s = CifParser(str(path), occupancy_tolerance=1.2).parse_structures(primitive=False)[0]
    return s, _cif_meta(path)


def cite(meta: dict) -> str:
    who = ", ".join(meta["authors"][:2]) + (" et al." if len(meta["authors"]) > 2 else "")
    return (f"COD {meta['cod_id']}; {who}, {meta['journal']} {meta['volume']} "
            f"({meta['year']}) {meta['page']}").replace("  ", " ").strip()


def entry(phase_id: str, variant: str, label: str, s: Structure, strain: str,
          framework: bool, window: float, source: str, extra_h_mass: float = 0.0) -> dict:
    lat = s.lattice
    return {
        "phase_id": phase_id, "variant": variant, "label": label,
        "framework": framework, "strain": strain, "window": window,
        "lattice": [round(x, 5) for x in (*lat.abc, *lat.angles)],
        "cell_mass_amu": round(float(s.composition.weight) + extra_h_mass, 3),
        "cell_volume_A3": round(float(lat.volume), 3),
        "source": source, "b_iso_A2": B_ISO,
        "reflections": structure_factor_table(s, strain),
    }


def build() -> dict:
    lib: dict[str, dict] = {}

    def add(e):
        lib[f"{e['phase_id']}/{e['variant']}" if e["variant"] else e["phase_id"]] = e

    for m in ("Mn", "Fe", "Co", "Ni", "Cu"):
        add(entry("pba_fm3m", m, f"Na{{x}}{m}[Fe(CN)6] (Fm-3m)", idealised_pba(m), "iso", True, 0.04,
                  "idealised model: Na 1/f.u. on 8c, y = 0.1, Fe-C 1.92 A, C-N 1.15 A, "
                  "a = schema.LATTICE_A0_A"))
    s, meta = load_cif("2020370")
    add(entry("pba_fm3m", "Zn", "Zn3[Fe(CN)6]2·xH2O (Fm-3m)", s, "iso", True, 0.03, cite(meta),
              _missing_h_mass(s, meta)))
    s, meta = load_cif("2106959")
    add(entry("znhcf_r3c", "Zn", "Na2Zn3[Fe(CN)6]2·9H2O (R-3c)", s, "ac", True, 0.03, cite(meta),
              _missing_h_mass(s, meta)))
    s, meta = load_cif("7047887")
    add(entry("pba_p21n", "Mn", "Na2Mn[Fe(CN)6]·2H2O (P2_1/n)", s, "iso", True, 0.03, cite(meta),
              _missing_h_mass(s, meta)))
    s_fe = s.copy()
    s_fe.replace_species({"Mn": "Fe"})
    s_fe.scale_lattice(s.volume * (LATTICE_A0_A["Fe"] / LATTICE_A0_A["Mn"]) ** 3)
    add(entry("pba_p21n", "Fe", "Na2Fe[Fe(CN)6]·2H2O (P2_1/n)", s_fe, "iso", True, 0.03,
              cite(meta) + "; Mn replaced by Fe, cell scaled by a0(Fe)/a0(Mn)",
              _missing_h_mass(s, meta)))

    s, meta = load_cif("9003308")
    add(entry("nacl", "", "NaCl", s, "iso", False, 0.015, cite(meta)))
    for m, cod in (("Mn", "9009111"), ("Fe", "9009104"), ("Co", "1548810"), ("Ni", "1548811")):
        s, meta = load_cif(cod)
        add(entry("m_oh2", m, f"{m}(OH)2", s, "ac", False, 0.02, cite(meta), _missing_h_mass(s, meta)))
    s, meta = load_cif("7212242")
    add(entry("cuo", "", "CuO (tenorite)", s, "iso", False, 0.015, cite(meta)))
    s, meta = load_cif("2300450")
    add(entry("zno", "", "ZnO (zincite)", s, "ac", False, 0.015, cite(meta)))
    return lib


if __name__ == "__main__":
    lib = build()
    OUT.write_text(json.dumps({"format": 1, "phases": lib}, indent=0))
    for k, e in lib.items():
        print(f"{k:16} {len(e['reflections']):4d} groups  V={e['cell_volume_A3']:8.1f}  "
              f"mass={e['cell_mass_amu']:8.1f}  {e['source'][:70]}")
