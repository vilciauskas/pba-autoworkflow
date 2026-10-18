# SPDX-License-Identifier: GPL-3.0-or-later
"""Properties of the inserting cations and of the framework metals.

One table drives everything ion-specific in the package: which selectivity,
retention and capacity objectives exist, which electrolytes the
electrochemistry stage prepares, how inserted amounts are assayed, and the
physical descriptors the optimizer uses to share information between metals
and between ions (see :mod:`pba_autoworkflow.optimize.surrogate`).  Adding an
ion means adding one :class:`Ion` entry; nothing else is keyed by ion name.

Sources
-------
* ``radius_A``: Shannon (1976) effective ionic radii, six-fold coordination
  (high-spin for Mn2+, Fe2+, Co2+), as tabulated in pymatgen.  NH4+ has no
  Shannon radius; the value is Marcus (1991).
* ``hydration_G_kJ``: experimental standard molar Gibbs energy of hydration,
  Delta_hyd G*, Marcus (1991) Table 1, J. Chem. Soc. Faraday Trans. 87, 2995.
* ``e0_V_SHE``: standard potential of M(z+)/M vs SHE (common tabulations, e.g.
  Bard, Parsons and Jordan, *Standard Potentials in Aqueous Solution*, 1985).
* ``electronegativity``: Pauling, as tabulated in pymatgen.
* ``sulfate_limit_M``: approximate cation concentration of the saturated
  sulfate near 20 degC, used only to decide whether a sulfate electrolyte is
  possible.  Nitrates of all listed cations are soluble well above 1 mol/L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Ion:
    symbol: str                       # "Li", "NH4", ...
    charge: int
    radius_A: float
    hydration_G_kJ: float
    e0_V_SHE: float = float("nan")
    electronegativity: float = float("nan")
    #: ICP can quantify it (NH4+ needs CHN nitrogen or ion chromatography)
    icp_assayable: bool = True
    #: element counts, for formula weights
    elements: dict[str, int] = field(default_factory=dict)
    sulfate_limit_M: float = 5.0
    #: above this pH the aquo ion hydrolyses (precipitates) at ~1 mol/L
    max_pH: float = 14.0

    @property
    def mass(self) -> float:
        from .schema import ATOMIC_WEIGHT
        return float(sum(ATOMIC_WEIGHT[e] * n for e, n in (self.elements or {self.symbol: 1}).items()))

    @property
    def hydration_per_charge_kJ(self) -> float:
        return self.hydration_G_kJ / self.charge


#: Inserting cations.
IONS: dict[str, Ion] = {i.symbol: i for i in (
    Ion("Li", 1, 0.76, -475.0, -3.04, 0.98, sulfate_limit_M=5.0),
    Ion("Na", 1, 1.02, -365.0, -2.71, 0.93, sulfate_limit_M=2.6),
    Ion("K", 1, 1.38, -295.0, -2.93, 0.82, sulfate_limit_M=1.28),
    Ion("NH4", 1, 1.48, -285.0, float("nan"), float("nan"), icp_assayable=False,
        elements={"N": 1, "H": 4}, sulfate_limit_M=10.0),
    Ion("Mg", 2, 0.72, -1830.0, -2.37, 1.31, sulfate_limit_M=2.9),
    Ion("Ca", 2, 1.00, -1505.0, -2.87, 1.00, sulfate_limit_M=0.015),
    Ion("Zn", 2, 0.74, -1955.0, -0.76, 1.65, sulfate_limit_M=3.6, max_pH=6.0),
    Ion("Al", 3, 0.535, -4525.0, -1.66, 1.61, sulfate_limit_M=2.1, max_pH=3.5),
)}

#: Framework (N-site) metals: Shannon radius of M2+ (VI, high spin), Pauling
#: electronegativity, Delta_hyd G* of M2+ (Marcus 1991; a measure of its
#: Lewis acidity), and whether M3+/M2+ is accessible in the framework (adds
#: a second redox couple to the hexacyanoferrate one).
FRAMEWORK_METALS: dict[str, dict[str, float]] = {
    "Mn": dict(radius_A=0.83, electronegativity=1.55, hydration_G_kJ=-1760.0, redox_active=1.0),
    "Fe": dict(radius_A=0.78, electronegativity=1.83, hydration_G_kJ=-1840.0, redox_active=1.0),
    "Co": dict(radius_A=0.745, electronegativity=1.88, hydration_G_kJ=-1915.0, redox_active=1.0),
    "Ni": dict(radius_A=0.69, electronegativity=1.91, hydration_G_kJ=-1980.0, redox_active=0.0),
    "Cu": dict(radius_A=0.73, electronegativity=1.90, hydration_G_kJ=-2010.0, redox_active=0.0),
    "Zn": dict(radius_A=0.74, electronegativity=1.65, hydration_G_kJ=-1955.0, redox_active=0.0),
}


def ion(symbol: str) -> Ion:
    try:
        return IONS[symbol]
    except KeyError:
        raise ValueError(f"unknown ion {symbol!r}; known: {sorted(IONS)}") from None


def charge(symbol: str) -> int:
    return ion(symbol).charge


def key(symbol: str) -> str:
    """Lower-case name used in objective names (``nh4_retention``)."""
    return symbol.lower()


def from_key(k: str) -> str:
    for s in IONS:
        if s.lower() == k:
            return s
    raise ValueError(f"unknown ion {k!r}; known: {sorted(s.lower() for s in IONS)}")


def pair_name(a: str, b: str) -> str:
    """Canonical key of a separation factor: lower charge first; for equal
    charge the more weakly hydrated ion first -- alpha_K/Zn, alpha_K/Na."""
    first, second = sorted((a, b), key=lambda s: (ion(s).charge, -ion(s).hydration_G_kJ))
    return f"{first}/{second}"


def expected_preferred(a: str, b: str) -> str:
    """Prior guess of which ion a PBA framework takes up preferentially.

    The more weakly hydrated ion per unit charge (less negative
    Delta_hyd G* / z) desolvates more easily; this is the ordering behind the
    reported PBA sequence Rb+ > K+ > Na+ > divalent ions.  It is used only to
    decide which ion is made dilute in a competitive-insertion electrolyte.
    """
    return a if ion(a).hydration_per_charge_kJ > ion(b).hydration_per_charge_kJ else b


#: Mixed electrolytes that have been tuned (major ion 1 mol/L): the minor ion's
#: concentration keeps both ions above ~5 % of the inserted charge for the
#: formal-potential gaps seen so far.
_TUNED_MIXTURES: dict[frozenset, dict[str, float]] = {
    frozenset({"Zn", "K"}): {"Zn": 1.0, "K": 0.005},
    frozenset({"Zn", "Na"}): {"Zn": 1.0, "Na": 0.02},
    frozenset({"Na", "K"}): {"Na": 1.0, "K": 0.01},
}


def competitive_electrolyte(a: str, b: str) -> dict[str, float]:
    """Default mixed electrolyte for measuring alpha between ``a`` and ``b``.

    The ion expected to be preferred is dilute (5 mmol/L against a 1 mol/L
    ion of different charge, 10 mmol/L against one of the same charge), so
    that the less-preferred ion still carries a measurable share of the
    charge.  Tuned pairs override this rule.  Override per campaign with
    ``WorkflowConfig.echem_mixed_electrolytes``.
    """
    tuned = _TUNED_MIXTURES.get(frozenset({a, b}))
    if tuned is not None:
        return dict(tuned)
    minor = expected_preferred(a, b)
    major = b if minor == a else a
    c_minor = 0.01 if charge(a) == charge(b) else 0.005
    return {major: 1.0, minor: c_minor}


@dataclass
class ElectrolyteRecipe:
    cations_M: dict[str, float]
    anion: str                               # "sulfate" or "nitrate"
    salts_M: dict[str, float]
    max_pH: float
    notes: list[str] = field(default_factory=list)


def _salt(symbol: str, z: int, anion: str) -> tuple[str, float]:
    """(formula, mol of salt per mol of cation)."""
    m = f"({symbol})" if symbol == "NH4" else symbol
    if anion == "nitrate":
        return (f"{m}NO3" if z == 1 else f"{symbol}(NO3){z}"), 1.0
    if z == 1:
        return f"{m}2SO4", 0.5
    if z == 2:
        return f"{symbol}SO4", 1.0
    return f"{symbol}2(SO4)3", 0.5


def electrolyte_recipe(cations_M: dict[str, float]) -> ElectrolyteRecipe:
    """Salts for an aqueous electrolyte with the given cation concentrations.

    Sulfate unless one of the cations would exceed its sulfate solubility
    (CaSO4 is the usual case), then nitrate for all salts so the mixture shares
    one anion.  ``max_pH`` is the lowest hydrolysis limit of the cations
    (Al3+ needs an acidic electrolyte).
    """
    cations = {s: float(c) for s, c in cations_M.items() if c}
    if not cations:
        raise ValueError("electrolyte needs at least one cation")
    for s, c in cations.items():
        ion(s)
        if c < 0:
            raise ValueError(f"negative concentration for {s}")
    notes = []
    over = [s for s, c in cations.items() if c > ion(s).sulfate_limit_M]
    anion = "nitrate" if over else "sulfate"
    if over:
        notes.append(f"{', '.join(over)} sulfate not soluble at this concentration: nitrate electrolyte")
    salts = {}
    for s, c in cations.items():
        name, per = _salt(s, charge(s), anion)
        salts[name] = c * per
    max_pH = min(ion(s).max_pH for s in cations)
    if max_pH < 7:
        limiting = [s for s in cations if ion(s).max_pH == max_pH]
        notes.append(f"keep pH below {max_pH:g} ({', '.join(limiting)} hydrolysis)")
    return ElectrolyteRecipe(cations, anion, salts, max_pH, notes)


def coulometric_ion(cations: list[str], framework_metal: str | None) -> list[str]:
    """Ions in an electrolyte whose inserted amount cannot be assayed directly.

    The framework's own metal (Zn in a zinc framework) is lost against the
    framework content; NH4+ is invisible to ICP.  At most one such ion can be
    recovered, by coulometric difference.
    """
    return [c for c in cations if c == framework_metal or not ion(c).icp_assayable]


def descriptor_matrix(names: list[str], table: dict[str, dict[str, float]] | None = None,
                      fields: tuple[str, ...] | None = None) -> np.ndarray:
    """Physical descriptor rows for categorical choices (raw, unstandardised)."""
    if table is None:
        table = {s: dict(radius_A=i.radius_A, charge=float(i.charge),
                         hydration_G_kJ=i.hydration_G_kJ) for s, i in IONS.items()}
    fields = fields or tuple(next(iter(table.values())).keys())
    return np.array([[float(table[n][f]) for f in fields] for n in names])


def _finite(x: float) -> bool:
    return x is not None and math.isfinite(x)
