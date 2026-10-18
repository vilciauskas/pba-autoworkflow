# SPDX-License-Identifier: GPL-3.0-or-later
"""Site template for hexacyanometallate frameworks.

A_x M [M'(CN)6]_(1-y) . z H2O, with

* **A site**: any mix of inserting cations from :mod:`pba_autoworkflow.ions`
  (Na+, K+, Li+, NH4+, Mg2+, Ca2+, Zn2+, Al3+), occupancies per formula unit;
* **M site**: the N-coordinated framework metal (charge ``m_charge``, 2 for
  the M(II) frameworks made here);
* **C site**: [M'(CN)6] with vacancy fraction y; M' is Fe by default, in its
  higher oxidation state ``c_high_ox`` (Fe(III)) or reduced by one
  (Fe(II)), with reduced share f;
* **water**: coordinated plus interstitial.

Everything that needs the formula -- formula weight, charge balance, the
formula string written to reports, the ICP reduction -- goes through one
template instance, so a different C-site metal or a new A-site ion changes
one place.  The module keeps the conventions of the earlier Na/K-only code:
``charge_balanced_a`` returns the A-site charge (equal to the A count for
monovalent ions), and :data:`PBA` reproduces the old formula weights exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ions import IONS, charge, ion


@dataclass(frozen=True)
class HexacyanometallateTemplate:
    c_metal: str = "Fe"
    c_high_ox: int = 3
    m_charge: int = 2
    #: vacancy fractions are clamped to this when computing masses (assay noise
    #: can give slightly negative values; > 0.9 is not a framework)
    max_vacancy: float = 0.9

    # ------------------------------------------------------------- charge
    def required_a_charge(self, vacancy_fraction: float, reduced_fraction: float = 1.0) -> float:
        """Positive A-site charge per formula unit for a neutral framework.

        [M'(CN)6] carries -(6 - ox); with ox = c_high_ox - f this is
        (1 - y)(6 - c_high_ox + f) - m_charge.  For Fe: (1 - y)(3 + f) - 2.
        """
        return (1.0 - vacancy_fraction) * (6.0 - self.c_high_ox + reduced_fraction) - self.m_charge

    @staticmethod
    def a_charge(a_site: dict[str, float]) -> float:
        return float(sum(charge(s) * x for s, x in a_site.items()))

    @staticmethod
    def a_count(a_site: dict[str, float]) -> float:
        return float(sum(a_site.values()))

    # -------------------------------------------------------------- mass
    def complex_mass(self) -> float:
        from .schema import ATOMIC_WEIGHT
        return ATOMIC_WEIGHT[self.c_metal] + 6 * (ATOMIC_WEIGHT["C"] + ATOMIC_WEIGHT["N"])

    def formula_weight(self, metal: str, a_site: dict[str, float], vacancy_fraction: float,
                       water_per_fu: float) -> float:
        from .schema import ATOMIC_WEIGHT
        y = min(max(vacancy_fraction, 0.0), self.max_vacancy)
        a_mass = sum(ion(s).mass * x for s, x in a_site.items())
        return (a_mass + ATOMIC_WEIGHT[metal] + (1.0 - y) * self.complex_mass()
                + water_per_fu * (2 * ATOMIC_WEIGHT["H"] + ATOMIC_WEIGHT["O"]))

    def anhydrous_weight(self, metal: str, a_site: dict[str, float], vacancy_fraction: float) -> float:
        return self.formula_weight(metal, a_site, vacancy_fraction, 0.0)

    # ----------------------------------------------------------- formula
    def formula(self, metal: str, a_site: dict[str, float], vacancy_fraction: float,
                water_per_fu: float) -> str:
        """``Na0.89K0.12Mn[Fe(CN)6]0.90·1.6H2O``.  Na is always written (as
        before); other A ions when above 0.005 per formula unit, in table order."""
        parts = [f"Na{a_site.get('Na', 0.0):.2f}"]
        for s in IONS:
            if s != "Na" and a_site.get(s, 0.0) > 0.005:
                parts.append(f"{s}{a_site[s]:.2f}")
        return (f"{''.join(parts)}{metal}[{self.c_metal}(CN)6]{1 - vacancy_fraction:.2f}"
                f"·{water_per_fu:.1f}H2O")


#: The template used throughout the package: M(II)[Fe(CN)6] frameworks.
PBA = HexacyanometallateTemplate()


def a_site_from(na: float = 0.0, k: float = 0.0, a_site: dict[str, float] | None = None
                ) -> dict[str, float]:
    """A-site dict from the legacy Na/K arguments, or a given dict."""
    if a_site is not None:
        return {s: float(x) for s, x in a_site.items()}
    out = {"Na": float(na)}
    if k:
        out["K"] = float(k)
    return out
