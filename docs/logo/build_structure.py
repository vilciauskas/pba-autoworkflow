"""Idealised cubic sodium iron hexacyanoferrate for the project logo.

Framework geometry is taken directly from the measured Prussian blue structure
COD 4343748 (Buser, Schwarzenbach, Petter, Ludi, Inorg. Chem. 16 (1977) 2704;
Pm-3m, a = 10.166 A): Fe-C 1.923 A, C-N 1.121 A, N-Fe 2.038 A, coordinated
water O at the O2 site (x = 0.21) around the [Fe(CN)6] vacancy, zeolitic water
O at the O4 site (x = 0.2608).

Idealisations (stated, not hidden):
  * the measured partial occupancies are replaced by one ordered vacancy at the
    body centre, i.e. Na1 Fe[Fe(CN)6]0.75 . 2.5 H2O per formula unit -- the same
    charge balance x = 2 - 4y used by pba_autoworkflow.thermo.structures;
  * Na+ is ADDED at four alternating sub-cube centres (Prussian blue itself has
    no Na) and zeolitic water occupies the other four;
  * H positions are not in the CIF; they are added with ideal water geometry
    (O-H 0.96 A, H-O-H 104.5 deg), coordinated-water H pointing away from Fe.
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np

A = 10.166                     # measured cubic edge, COD 4343748
X_C, X_N = 0.3108, 0.2005      # C1 and N1 fractional x (Fe at 0 and 0.5)
X_OW, X_OZ = 0.21, 0.2608      # O2 (coordinated water), O4 (zeolitic water)
R_OH, HOH = 0.96, np.radians(104.5)
EPS = 1e-6


def inside(f, lo=-EPS, hi=1 + EPS):
    return np.all(f >= lo) and np.all(f <= hi)


def water_h(o, bisector, perp):
    b = bisector / np.linalg.norm(bisector)
    p = perp - perp.dot(b) * b
    p /= np.linalg.norm(p)
    h = HOH / 2
    return [o + R_OH * (np.cos(h) * b + s * np.sin(h) * p) for s in (1, -1)]


def build(complete_octahedra: bool = True):
    atoms = []                                  # (role, element, cart xyz)

    def add(role, el, frac=None, cart=None):
        xyz = np.asarray(cart if cart is not None else np.asarray(frac) * A, float)
        for r, e, c in atoms:                   # de-duplicate shared positions
            if r == role and np.linalg.norm(c - xyz) < 1e-3:
                return
        atoms.append((role, el, xyz))

    ax = np.eye(3)
    # N-bonded Fe (fcc sites) on every corner and face centre of the cell
    for base in ([0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]):
        for t in itertools.product((0, 1), repeat=3):
            f = np.array(base) + t
            if inside(f):
                add("Fe_N", "Fe", f)
    # C-bonded Fe at edge centres (the body centre is the ordered vacancy)
    fe_c = []
    for base in ([.5, 0, 0], [0, .5, 0], [0, 0, .5]):
        for t in itertools.product((0, 1), repeat=3):
            f = np.array(base) + t
            if inside(f):
                fe_c.append(f); add("Fe_C", "Fe", f)
    dc, dn = 0.5 - X_C, 0.5 - X_N               # Fe_C -> C and Fe_C -> N, fractional
    for f in fe_c:
        for d in ax:
            for s in (1, -1):
                c, n = f + s * dc * d, f + s * dn * d
                if inside(c) or complete_octahedra:
                    add("C", "C", c)
                if inside(n):            # N only inside: no cyanide spikes out of the cell
                    add("N", "N", n)
    # coordinated water filling the body-centre vacancy, measured O2 site
    centre = np.array([.5, .5, .5])
    for i, d in enumerate(ax):
        for s in (1, -1):
            o_f = centre - s * (0.5 - X_OW) * d
            o = o_f * A
            add("O_coord", "O", cart=o)
            perp = ax[(i + 1 + (s > 0)) % 3]
            for h in water_h(o, (centre * A - o), perp):
                add("H", "H", cart=h)
    # sub-cube centres: Na+ on one tetrahedral set, zeolitic water on the other
    rng = np.random.default_rng(7)
    for t in itertools.product((0, 1), repeat=3):
        q = np.array(t)
        if q.sum() % 2 == 0:
            add("Na", "Na", 0.25 + 0.5 * q)
        else:
            o_f = np.where(q == 1, 1 - X_OZ, X_OZ)   # measured O4 site
            o = o_f * A
            add("O_zeo", "O", cart=o)
            for h in water_h(o, o - centre * A, rng.normal(size=3)):
                add("H", "H", cart=h)
    return atoms


def bonds(atoms):
    rules = {("Fe_C", "C"): 2.10, ("C", "N"): 1.30, ("N", "Fe_N"): 2.20,
             ("O_coord", "Fe_N"): 2.30, ("O_coord", "H"): 1.10, ("O_zeo", "H"): 1.10}
    out = []
    for i, j in itertools.combinations(range(len(atoms)), 2):
        ri, rj = atoms[i][0], atoms[j][0]
        cut = rules.get((ri, rj)) or rules.get((rj, ri))
        if cut and np.linalg.norm(atoms[i][2] - atoms[j][2]) < cut:
            kind = "coord" if "O_coord" in (ri, rj) and "Fe_N" in (ri, rj) else "covalent"
            out.append((i, j, kind))
    return out


def write_cif(atoms, path):
    """Ordered P1 model of the in-cell atoms (boundary images folded once)."""
    seen, lines = [], []
    for k, (role, el, c) in enumerate(atoms):
        f = np.mod(np.round(c / A, 6), 1.0)
        if any(np.allclose(f, g, atol=1e-4) for g in seen):
            continue
        seen.append(f)
        lines.append(f"{el}{len(seen)} {el} {f[0]:.5f} {f[1]:.5f} {f[2]:.5f} 1.0")
    with open(path, "w") as fh:
        fh.write("data_NaFeFeCN_idealised\n"
                 "_audit_creation_method 'idealised logo model built on COD 4343748 framework'\n"
                 f"_cell_length_a {A}\n_cell_length_b {A}\n_cell_length_c {A}\n"
                 "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
                 "_symmetry_space_group_name_H-M 'P 1'\n_symmetry_Int_Tables_number 1\n"
                 "loop_\n_symmetry_equiv_pos_as_xyz\n'x, y, z'\n"
                 "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
                 "_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n"
                 + "\n".join(lines) + "\n")
    return len(lines)


HERE = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    at = build(complete_octahedra=True)
    bd = bonds(at)
    centre = np.array([A / 2] * 3)
    json.dump({"a": A, "centre": centre.tolist(),
               "atoms": [{"role": r, "el": e, "xyz": c.tolist()} for r, e, c in at],
               "bonds": bd}, open(os.path.join(HERE, "structure.json"), "w"))
    n = write_cif(build(complete_octahedra=False), os.path.join(HERE, "NaFeFeCN_idealised.cif"))
    roles = {}
    for r, _, _ in at:
        roles[r] = roles.get(r, 0) + 1
    print("render atoms:", len(at), roles, "| bonds:", len(bd))
    print("CIF atoms in cell:", n)
