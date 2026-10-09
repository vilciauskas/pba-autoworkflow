# SPDX-License-Identifier: GPL-3.0-or-later
"""Core data model for the PBA self-driving-lab orchestrator.

Everything that crosses a module boundary is one of these types.  The design
space, the physical parameters of a single synthesis, the raw traces returned by
each instrument, the descriptors extracted from those traces, and the scalar
objectives handed to the optimizer are all declared here so that the provenance
store, the scheduler and the optimizer never have to agree informally on a dict
layout.

Units are part of the field name (``temperature_C``, ``aging_time_h``) because
unit confusion is the single most common source of silent error in an automated
platform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Chemistry constants
# --------------------------------------------------------------------------- #

FARADAY_C_PER_MOL = 96485.33212

#: Divalent transition metals that form Prussian-blue analogues of the type
#: Na_x M[Fe(CN)6]_(1-y) . zH2O by aqueous co-precipitation.
#: Zn is the exception that makes polymorphism a design question: with sodium
#: hexacyanoferrate(II) it forms either the cubic framework or the rhombohedral
#: Na2Zn3[Fe(CN)6]2 (R-3c, tetrahedral ZnN4), depending on synthesis and drying.
METALS: tuple[str, ...] = ("Mn", "Fe", "Co", "Ni", "Cu", "Zn")

#: Atomic weights (g/mol) needed for formula-weight bookkeeping.
ATOMIC_WEIGHT: dict[str, float] = {
    "Na": 22.9898,
    "K": 39.0983,
    "Mn": 54.9380,
    "Fe": 55.8450,
    "Co": 58.9332,
    "Ni": 58.6934,
    "Cu": 63.5460,
    "Zn": 65.38,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "H": 1.00794,
}

#: Shannon high-spin ionic radii (Angstrom, six-coordinate) for the N-coordinated
#: site.  Sets the baseline cubic lattice constant of the analogue.
IONIC_RADIUS_A: dict[str, float] = {
    "Mn": 0.830,
    "Fe": 0.780,
    "Co": 0.745,
    "Ni": 0.690,
    "Cu": 0.730,
    "Zn": 0.740,
}

#: Reference cubic (Fm-3m) lattice constant of the sodium-rich analogue, Angstrom.
#: These are reported values for Na_xM[Fe(CN)6] frameworks, consistent with the
#: structural sum a = 2*(d(Fe-C) + d(C=N) + d(N-M)) with d(Fe-C) ~ 1.92 A and
#: d(C=N) ~ 1.15 A: the cell edge spans two -N=C-Fe-C=N-M- linkages, so a is set
#: by the N-site metal-nitrogen distance.
#:
#: The series correlates with the divalent radius but is deliberately *not*
#: monotonic in it, because the radius is not the only term.  Cu(II) is d9 and its
#: Jahn-Teller distortion contracts the framework below what its radius implies,
#: and the Fe/Co pair is inverted relative to radius in the reported cubic phases.
#: A tidier table that interpolated these from ionic radii would be a fabrication;
#: the values here are the ones a Rietveld refinement of these materials actually
#: returns.
#:
#: Getting this baseline right matters beyond cosmetics: the QC window that
#: rejects mis-indexed patterns is derived from it, so a wrong intercept either
#: passes garbage or rejects good data across the whole metal series.
LATTICE_A0_A: dict[str, float] = {
    "Mn": 10.53,
    "Fe": 10.20,
    "Co": 10.30,
    "Ni": 10.24,
    "Cu": 10.11,
    # Cubic Zn3[Fe(CN)6]2.xH2O, COD 2020370 (the Zn reference is the cubic
    # polymorph; the rhombohedral one is described by its own R-3c cell).
    "Zn": 10.342,
}


def formula_weight(metal: str, na_per_fu: float, vacancy_fraction: float,
                   water_per_fu: float, k_per_fu: float = 0.0) -> float:
    """Formula weight (g/mol) of Na_n K_k M[Fe(CN)6]_(1-y) . zH2O."""
    hcf = ATOMIC_WEIGHT["Fe"] + 6 * (ATOMIC_WEIGHT["C"] + ATOMIC_WEIGHT["N"])
    # A measured vacancy may be marginally negative from assay error; an
    # occupancy above unity is not a physical structure, so clamp here.
    vacancy_fraction = min(max(vacancy_fraction, 0.0), 0.9)
    return (
        na_per_fu * ATOMIC_WEIGHT["Na"]
        + k_per_fu * ATOMIC_WEIGHT["K"]
        + ATOMIC_WEIGHT[metal]
        + (1.0 - vacancy_fraction) * hcf
        + water_per_fu * (2 * ATOMIC_WEIGHT["H"] + ATOMIC_WEIGHT["O"])
    )


def charge_balanced_a(vacancy_fraction: float, fe2_fraction: float = 1.0) -> float:
    """Monovalent A-cations per formula unit that make A_x M[Fe(CN)6]_(1-y) neutral.

    ``x = (1 - y)(3 + f) - 2`` for M(II), where ``f`` is the Fe(II) share of the
    hexacyanoferrate: [Fe(II)(CN)6] carries -4, [Fe(III)(CN)6] -3.  Fe(II) with
    no vacancies gives the familiar x = 2; Zn3[Fe(III)(CN)6]2 (y = 1/3, f = 0)
    gives x = 0.  Negative values mean the framework cannot be neutral without
    another anion (OH-, coordinated water deprotonation) and are returned as is.
    Same convention as :func:`pba_autoworkflow.thermo.structures.charge_balanced_na`.
    """
    return (1.0 - vacancy_fraction) * (3.0 + fe2_fraction) - 2.0


def theoretical_capacity_mAh_g(metal: str, na_per_fu: float,
                               vacancy_fraction: float, water_per_fu: float,
                               k_per_fu: float = 0.0) -> float:
    """Theoretical capacity for the extractable A-cation (Na + K) inventory.

    Zn(II) has no accessible redox couple, so in a Zn framework only the
    [Fe(CN)6] site can be oxidised and at most 1 - y A per formula unit is
    extractable (the rest stays in the formula weight).
    """
    fw = formula_weight(metal, na_per_fu, vacancy_fraction, water_per_fu, k_per_fu)
    a = na_per_fu + k_per_fu
    active = min(a, max(1.0 - vacancy_fraction, 0.0)) if metal == "Zn" else a
    return active * FARADAY_C_PER_MOL / (3.6 * fw)


#: Water partial pressure of ambient laboratory air, mbar (about 50 % relative
#: humidity at 20 degC).  Used when a recipe dries under ``dry_gas="ambient"``.
AMBIENT_WATER_PRESSURE_MBAR = 13.0


def water_saturation_pressure_mbar(temperature_C: float) -> float:
    """Saturation vapour pressure of liquid water (Buck 1996), mbar."""
    t = float(temperature_C)
    return 6.1121 * math.exp((18.678 - t / 234.5) * (t / (257.14 + t)))


def drying_water_pressure_mbar(pressure_mbar: float, gas: str) -> float:
    """Water partial pressure of the gas over the drying solid.

    ``dry`` gas (desiccant, dry purge): zero.  ``ambient``: lab-air humidity,
    capped at the total pressure -- in an evacuated oven the residual gas is
    taken to be the water leaving the sample, so p_w = P_total there.
    """
    if gas == "dry":
        return 0.0
    return min(float(pressure_mbar), AMBIENT_WATER_PRESSURE_MBAR)


def drying_rate_index(temperature_C: float, pressure_mbar: float, gas: str) -> float:
    """Dimensionless drying-rate proxy Pi = (p_sat(T) - p_w) / P_total.

    Undersaturation of the gas over the solid divided by the gas-phase transport
    resistance, which scales with total pressure in the diffusion regime.  A
    ranking heuristic, not a rate: free-molecular flow, heat supply and
    diffusion inside the powder cake are not in it.  For zinc hexacyanoferrate
    it separates slow drying (R-3c) from fast drying (disordered cubic) in
    Pilipavicius et al.; values of 2-250 were not tested there.  Clipped at
    1e-4 (a gas at or above saturation does not dry the solid).
    """
    pw = drying_water_pressure_mbar(pressure_mbar, gas)
    return max((water_saturation_pressure_mbar(temperature_C) - pw) / float(pressure_mbar), 1e-4)


#: Hexacyanoferrate precursors.  Na4[Fe(CN)6] delivers Fe(II) and Na+;
#: K3[Fe(CN)6] delivers Fe(III) and K+ (the route of the zinc paper).
HCF_PRECURSORS: tuple[str, ...] = ("Na4FeCN6", "K3FeCN6")
#: (A cation, A per [Fe(CN)6], Fe oxidation state) of each precursor
HCF_PRECURSOR_INFO: dict[str, tuple[str, int, int]] = {
    "Na4FeCN6": ("Na", 4, 2),
    "K3FeCN6": ("K", 3, 3),
}


# --------------------------------------------------------------------------- #
# Design space
# --------------------------------------------------------------------------- #


class ParameterSpec(BaseModel):
    """One controllable knob of the synthesis."""

    model_config = ConfigDict(frozen=True)

    name: str
    low: float
    high: float
    unit: str
    log_scale: bool = False
    description: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> "ParameterSpec":
        if not self.high > self.low:
            raise ValueError(f"{self.name}: high must exceed low")
        if self.log_scale and self.low <= 0:
            raise ValueError(f"{self.name}: log-scaled parameter needs low > 0")
        return self

    def to_unit(self, value: float) -> float:
        """Map a physical value onto [0, 1]."""
        if self.log_scale:
            return (math.log(value) - math.log(self.low)) / (
                math.log(self.high) - math.log(self.low)
            )
        return (value - self.low) / (self.high - self.low)

    def from_unit(self, u: float) -> float:
        """Map [0, 1] back onto the physical range."""
        u = min(max(float(u), 0.0), 1.0)
        if self.log_scale:
            return math.exp(
                math.log(self.low) + u * (math.log(self.high) - math.log(self.low))
            )
        return self.low + u * (self.high - self.low)


class CategoricalSpec(BaseModel):
    """A discrete choice, e.g. which divalent metal salt to precipitate."""

    model_config = ConfigDict(frozen=True)

    name: str
    choices: tuple[str, ...]
    description: str = ""


class DesignSpace(BaseModel):
    """The searchable space, plus the one-hot encoding used by the optimizer.

    The encoding is ``[continuous params in unit cube] + [one-hot blocks]``.  The
    optimizer only ever sees this vector; the physical meaning lives here.
    """

    model_config = ConfigDict(frozen=True)

    continuous: tuple[ParameterSpec, ...]
    categorical: tuple[CategoricalSpec, ...] = ()

    @property
    def continuous_names(self) -> list[str]:
        return [p.name for p in self.continuous]

    @property
    def dim(self) -> int:
        return len(self.continuous) + sum(len(c.choices) for c in self.categorical)

    @property
    def encoded_names(self) -> list[str]:
        names = list(self.continuous_names)
        for cat in self.categorical:
            names += [f"{cat.name}={c}" for c in cat.choices]
        return names

    def categorical_blocks(self) -> list[tuple[str, list[int]]]:
        """Column indices of each one-hot block, in encoding order."""
        blocks: list[tuple[str, list[int]]] = []
        offset = len(self.continuous)
        for cat in self.categorical:
            blocks.append((cat.name, list(range(offset, offset + len(cat.choices)))))
            offset += len(cat.choices)
        return blocks

    def encode(self, params: "SynthesisParameters") -> np.ndarray:
        raw = params.model_dump()
        x = [spec.to_unit(float(raw[spec.name])) for spec in self.continuous]
        for cat in self.categorical:
            value = raw[cat.name]
            if value not in cat.choices:
                raise ValueError(f"{cat.name}={value!r} outside {cat.choices}")
            x += [1.0 if value == c else 0.0 for c in cat.choices]
        return np.asarray(x, dtype=float)

    def decode(self, x: Sequence[float]) -> "SynthesisParameters":
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.dim:
            raise ValueError(f"expected {self.dim} encoded dims, got {x.size}")
        kwargs: dict[str, Any] = {
            spec.name: spec.from_unit(x[i]) for i, spec in enumerate(self.continuous)
        }
        for cat, (_, cols) in zip(self.categorical, self.categorical_blocks()):
            kwargs[cat.name] = cat.choices[int(np.argmax(x[cols]))]
        return SynthesisParameters(**kwargs)


class SynthesisParameters(BaseModel):
    """Physical recipe for one aqueous co-precipitation of a PBA.

    Two precursor solutions are prepared and combined: solution A carries the
    divalent metal salt plus (optionally) a citrate chelator; solution B carries
    the sodium hexacyanoferrate(II) plus supporting NaCl.  B is metered into A at
    a controlled rate, then the slurry is aged.
    """

    model_config = ConfigDict(extra="forbid")

    metal: Literal["Mn", "Fe", "Co", "Ni", "Cu", "Zn"] = "Mn"
    c_metal_M: float = Field(..., gt=0, description="M(II) salt in solution A")
    c_hcf_M: float = Field(..., gt=0, description="hexacyanoferrate in solution B")
    #: Na4[Fe(CN)6] (Fe(II), Na+) or K3[Fe(CN)6] (Fe(III), K+)
    hcf_precursor: Literal["Na4FeCN6", "K3FeCN6"] = "Na4FeCN6"
    c_nacl_M: float = Field(..., ge=0, description="supporting NaCl in solution B")
    c_citrate_M: float = Field(..., ge=0, description="sodium citrate chelator in A")
    ph: float = Field(..., ge=0.5, le=9.0)
    temperature_C: float = Field(..., ge=15.0, le=95.0)
    addition_rate_mL_min: float = Field(..., gt=0)
    aging_time_h: float = Field(..., gt=0)
    stir_rate_rpm: float = Field(..., ge=0, le=1500)
    #: Drying of the washed solid.  The rate of water removal selects the
    #: polymorph of zinc hexacyanoferrate (slow: R-3c, fast: disordered cubic)
    #: and temperature sets Fe(III) reduction, so drying is part of the recipe.
    #: Defaults reproduce the earlier fixed protocol (70 degC, ambient air).
    dry_temperature_C: float = Field(70.0, ge=20.0, le=150.0)
    #: total pressure of the drying atmosphere (1013 = ambient; < 100 = vacuum oven)
    dry_pressure_mbar: float = Field(1013.0, ge=0.005, le=1100.0)
    #: water activity of the drying gas: ``ambient`` lab air, or ``dry``
    #: (desiccant such as P2O5, or a dry-gas purge)
    dry_gas: Literal["ambient", "dry"] = "ambient"

    @model_validator(mode="before")
    @classmethod
    def _migrate_dry_atmosphere(cls, data):
        # Records written before drying pressure existed carry
        # ``dry_atmosphere`` (air | vacuum).  "vacuum" meant a dynamic-vacuum
        # oven; 1 mbar is taken as its pressure.
        if isinstance(data, dict) and "dry_atmosphere" in data:
            data = dict(data)
            atm = data.pop("dry_atmosphere")
            data.setdefault("dry_pressure_mbar", 1.0 if atm == "vacuum" else 1013.0)
        return data

    # Fixed platform geometry, not optimized but recorded for provenance.
    volume_A_mL: float = 10.0
    volume_B_mL: float = 10.0

    @property
    def total_volume_mL(self) -> float:
        return self.volume_A_mL + self.volume_B_mL

    @property
    def hcf_to_metal_ratio(self) -> float:
        n_metal = self.c_metal_M * self.volume_A_mL
        n_hcf = self.c_hcf_M * self.volume_B_mL
        return n_hcf / n_metal

    @property
    def citrate_to_metal_ratio(self) -> float:
        return self.c_citrate_M / self.c_metal_M

    @property
    def addition_time_min(self) -> float:
        return self.volume_B_mL / self.addition_rate_mL_min

    @property
    def drying_index(self) -> float:
        """Drying-rate proxy Pi of this recipe (see :func:`drying_rate_index`)."""
        return drying_rate_index(self.dry_temperature_C, self.dry_pressure_mbar, self.dry_gas)

    def summary(self) -> str:
        return (
            f"{self.metal} | [M]={self.c_metal_M:.3f} [HCF]={self.c_hcf_M:.3f} "
            f"[NaCl]={self.c_nacl_M:.2f} [cit]={self.c_citrate_M:.3f} "
            f"pH={self.ph:.1f} T={self.temperature_C:.0f}C "
            f"rate={self.addition_rate_mL_min:.2f}mL/min t={self.aging_time_h:.1f}h"
        )


def default_design_space() -> DesignSpace:
    """The design space used by the reference PBA campaign.

    Bounds are chosen to bracket the co-precipitation window reported for
    sodium-rich manganese/iron hexacyanoferrates: dilute-to-moderate precursor
    concentrations, a large supporting-electrolyte range (the Na+ activity that
    drives Na incorporation), citrate as the nucleation brake, mildly acidic pH
    to suppress hexacyanoferrate hydrolysis, and addition rates spanning three
    orders of supersaturation.
    """
    return DesignSpace(
        continuous=(
            ParameterSpec(name="c_metal_M", low=0.01, high=0.30, unit="mol/L",
                          log_scale=True,
                          description="M(II) sulfate/chloride in solution A"),
            ParameterSpec(name="c_hcf_M", low=0.01, high=0.30, unit="mol/L",
                          log_scale=True,
                          description="hexacyanoferrate in solution B"),
            ParameterSpec(name="c_nacl_M", low=0.0, high=3.5, unit="mol/L",
                          description="supporting NaCl, sets Na+ activity"),
            ParameterSpec(name="c_citrate_M", low=0.0, high=0.45, unit="mol/L",
                          description="sodium citrate chelator"),
            ParameterSpec(name="ph", low=1.0, high=7.0, unit="pH",
                          description="pH of solution A before addition"),
            ParameterSpec(name="temperature_C", low=25.0, high=90.0, unit="degC",
                          description="isothermal precipitation/ageing setpoint"),
            ParameterSpec(name="addition_rate_mL_min", low=0.05, high=20.0,
                          unit="mL/min", log_scale=True,
                          description="metering rate of B into A"),
            ParameterSpec(name="aging_time_h", low=0.5, high=24.0, unit="h",
                          log_scale=True, description="post-addition ageing"),
            ParameterSpec(name="stir_rate_rpm", low=200.0, high=1200.0, unit="rpm",
                          description="overhead stirrer setpoint"),
            ParameterSpec(name="dry_temperature_C", low=25.0, high=120.0, unit="degC",
                          description="drying temperature of the washed solid"),
            ParameterSpec(name="dry_pressure_mbar", low=0.01, high=1013.0, unit="mbar",
                          log_scale=True,
                          description="total pressure of the drying atmosphere"),
        ),
        categorical=(
            CategoricalSpec(name="metal", choices=METALS,
                            description="divalent metal on the N-coordinated site"),
            CategoricalSpec(name="hcf_precursor", choices=HCF_PRECURSORS,
                            description="Na4[Fe(CN)6] (Fe(II), Na+) or K3[Fe(CN)6] (Fe(III), K+)"),
            CategoricalSpec(name="dry_gas", choices=("ambient", "dry"),
                            description="drying gas: ambient lab air or dry (desiccant/purge)"),
        ),
    )


# --------------------------------------------------------------------------- #
# Raw instrument traces
# --------------------------------------------------------------------------- #


@dataclass
class XRDPattern:
    """Powder diffractogram as returned by the diffractometer driver."""

    two_theta_deg: np.ndarray
    intensity: np.ndarray
    wavelength_A: float = 1.5406  # Cu K-alpha1
    exposure_s: float = 0.0

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {"two_theta_deg": self.two_theta_deg, "intensity": self.intensity}


@dataclass
class ICPResult:
    """Elemental assay of the digested, washed, dried powder.

    ``concentrations_mol_L`` are the ICP-OES metal concentrations in the digest.
    ``carbon_wt_pct`` is the CHN combustion result on the same aliquot, and it is
    not optional garnish: for the iron analogue (Prussian blue itself) ICP reports
    only *total* Fe, because N-site and C-site iron are chemically identical in the
    digest.  Without an independent measure of the hexacyanoferrate content the
    vacancy fraction of Prussian blue is simply not determinable, and any split of
    total Fe between the two sites is an assumption rather than a measurement.
    Carbon gives that measure directly: C/6 is the hexacyanoferrate content per
    aliquot regardless of which metal occupies the N site.
    """

    concentrations_mol_L: dict[str, float]
    digest_mass_mg: float
    digest_volume_mL: float
    dry_mass_mg: float
    carbon_wt_pct: float = float("nan")


@dataclass
class IRSpectrum:
    """ATR-IR absorbance in the cyanide-stretch region."""

    wavenumber_cm1: np.ndarray
    absorbance: np.ndarray

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {"wavenumber_cm1": self.wavenumber_cm1, "absorbance": self.absorbance}


@dataclass
class EchemCycleData:
    """Galvanostatic cycling of one electrode in one electrolyte.

    ``electrolyte_M`` maps the inserting cation to its molar concentration
    (e.g. ``{"Zn": 1.0, "K": 0.2}``).  Curves are per cycle, as arrays of
    specific charge (mAh/g, cumulative within the half-cycle) and potential
    (V vs the reference named in ``reference``).  ``electrode`` and
    ``electrolyte`` are handles that can be sent to the elemental analyzer
    after the run (electrode digest; spent electrolyte).
    """

    electrolyte_M: dict[str, float]
    current_mA_g: float
    reference: str
    charge_q: list[np.ndarray]
    charge_E: list[np.ndarray]
    discharge_q: list[np.ndarray]
    discharge_E: list[np.ndarray]
    electrode: Any = None
    electrolyte: Any = None
    #: active material in the electrode and volume of electrolyte in the cell
    active_mass_mg: float = float("nan")
    electrolyte_volume_mL: float = float("nan")

    def as_arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for i, (q, e) in enumerate(zip(self.discharge_q, self.discharge_E)):
            out[f"dis{i}_q"], out[f"dis{i}_E"] = q, e
        for i, (q, e) in enumerate(zip(self.charge_q, self.charge_E)):
            out[f"chg{i}_q"], out[f"chg{i}_E"] = q, e
        return out


# --------------------------------------------------------------------------- #
# Extracted descriptors
# --------------------------------------------------------------------------- #


class XRDDescriptors(BaseModel):
    """Structural descriptors of one powder pattern.

    Undetermined numbers are NaN.  JSON has no NaN, so they are stored as null;
    the validator below turns them back into NaN on reload (a zinc sample that
    formed only the R-3c phase has no cubic lattice constant, for example).
    """

    lattice_a_A: float
    domain_size_nm: float
    crystallinity_index: float = Field(..., ge=0.0, le=1.0)
    fwhm_200_deg: float
    #: dominant crystalline framework phase (a phase id from the reference
    #: library, e.g. ``pba_fm3m``, ``pba_p21n``, ``znhcf_r3c``) or ``amorphous``
    phase: str
    n_peaks_indexed: int
    fit_residual: float
    #: weight fraction of each identified crystalline phase (Hill-Howard, from
    #: whole-pattern scale factors); sums to 1 over the crystalline material
    phase_fractions: dict[str, float] = Field(default_factory=dict)
    #: refined lattice (a, b, c, alpha, beta, gamma) of each identified phase
    phase_lattice: dict[str, list[float]] = Field(default_factory=dict)
    #: share of the Bragg intensity in peaks no library phase explains
    unidentified_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    n_unidentified_peaks: int = 0
    #: diffuse scattering around the framework reflections (short-range-ordered
    #: framework, e.g. disordered cubic ZnHCF) as a share of all framework
    #: scattering (Bragg + diffuse + amorphous halo); not counted as amorphous
    diffuse_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("lattice_a_A", "domain_size_nm", "fwhm_200_deg", "fit_residual",
                     mode="before")
    @classmethod
    def _null_is_nan(cls, v):
        return float("nan") if v is None else v


class CompositionDescriptors(BaseModel):
    """Composition as *measured*, not as idealized.

    ``vacancy_fraction`` is permitted to go slightly negative.  It is derived from
    a measured Fe/M ratio, and a nearly vacancy-free sample assays at Fe/M = 1
    plus or minus the assay error, so half of those measurements land above unity.
    Clipping the derived vacancy at zero would map every such sample onto exactly
    0.000, which destroys the ranking information in precisely the region the
    optimizer is trying to resolve and makes chemically distinct samples
    numerically identical.  The lower bound here is therefore a plausibility limit
    on assay error, not a physical one; :func:`quality_flags` decides whether a
    given deviation is tolerable, and the physical models clamp at their own point
    of use where a negative occupancy would be meaningless.

    ``vacancy_fraction`` may also be NaN, which means *not determinable from the
    measurements taken* -- not zero, and not a default.  The Fe analogue without a
    carbon assay is the case in point: returning a plausible-looking number from an
    unresolvable measurement is worse than returning nothing, because QC would pass
    it and the surrogate would train on it.  :func:`quality_flags` rejects NaN.
    """

    model_config = ConfigDict(allow_inf_nan=True)

    na_per_fu: float
    fe_per_metal: float
    vacancy_fraction: float
    water_per_fu: float
    formula: str
    k_per_fu: float = 0.0
    #: CN missing per formula unit, 6 Fe(C-site) - C, when both Fe and carbon
    #: were assayed (not for the Fe analogue): > 0 indicates decyanation
    cn_deficit_per_fu: float = float("nan")

    @field_validator("cn_deficit_per_fu", mode="before")
    @classmethod
    def _cn_null(cls, v):
        return float("nan") if v is None else v

    @property
    def a_per_fu(self) -> float:
        """Total monovalent A-cations (Na + K) per formula unit."""
        return self.na_per_fu + self.k_per_fu

    @field_validator("vacancy_fraction")
    @classmethod
    def _plausible_or_undetermined(cls, v: float) -> float:
        if math.isnan(v):
            return v  # explicitly undetermined
        if not -0.15 <= v <= 0.9:
            raise ValueError(
                f"vacancy_fraction {v} outside the plausible assay range "
                "[-0.15, 0.9]; use NaN for an undetermined composition"
            )
        return v


def _nan_for_null(data):
    """JSON stores NaN as null; turn nulls back into NaN on reload."""
    if not isinstance(data, dict):
        return data
    out = {}
    for k, v in data.items():
        if v is None:
            v = float("nan")
        elif isinstance(v, dict):
            v = {kk: (float("nan") if vv is None else vv) for kk, vv in v.items()}
        out[k] = v
    return out


class IRDescriptors(BaseModel):
    """Fe oxidation state from the cyanide-stretch bands.

    ``fe2_fraction`` = Fe(II) share of the hexacyanoferrate from the band areas
    and an assumed ratio of integrated absorptivities.  That ratio carries a
    calibration uncertainty, so treat the value as *relative* (ranking samples)
    unless the absorptivities were calibrated on reference compounds.
    """

    model_config = ConfigDict(allow_inf_nan=True)

    _nulls = model_validator(mode="before")(classmethod(lambda cls, d: _nan_for_null(d)))

    fe2_fraction: float
    nu_fe2_cm1: float
    nu_fe3_cm1: float
    fit_residual: float


class EchemDescriptors(BaseModel):
    """Electrochemistry and ion selectivity of one material.

    ``formal_potential_V`` is per inserting cation from its single-ion
    electrolyte (midpoint of charge and discharge average potentials).
    ``separation_factor`` maps ``"A/B"`` to the measured
    alpha = (x_A / x_B)_solid / (c_A / c_B)_electrolyte from the electrode
    digest after insertion from the mixed electrolyte (concentrations stand in
    for activities); ``predicted_separation_factor`` is the same quantity
    expected from the single-ion formal potentials.
    """

    model_config = ConfigDict(allow_inf_nan=True)

    _nulls = model_validator(mode="before")(classmethod(lambda cls, d: _nan_for_null(d)))

    formal_potential_V: dict[str, float] = Field(default_factory=dict)
    capacity_mAh_g: dict[str, float] = Field(default_factory=dict)
    retention: dict[str, float] = Field(default_factory=dict)
    plateau_fraction: dict[str, float] = Field(default_factory=dict)
    separation_factor: dict[str, float] = Field(default_factory=dict)
    predicted_separation_factor: dict[str, float] = Field(default_factory=dict)
    #: Fe found in the spent mixed electrolyte, as a fraction of electrode Fe
    dissolved_fe_fraction: float = float("nan")
    n_cycles: int = 0


class SampleDescriptors(BaseModel):
    """Everything the analysis layer extracted for one sample."""

    model_config = ConfigDict(allow_inf_nan=True)

    xrd: XRDDescriptors | None = None
    composition: CompositionDescriptors | None = None
    ir: IRDescriptors | None = None
    echem: EchemDescriptors | None = None
    isolated_yield: float | None = Field(default=None, ge=0.0, le=1.2)
    capacity_mAh_g: float | None = None
    #: recipe-derived drying-rate proxy Pi (not a measurement)
    drying_index: float | None = None
    #: A_measured - A_charge-balanced, from ICP (A, y) and IR (Fe(II) share):
    #: > 0 excess cations; < 0 a deficit (decyanation, or cations ICP does not
    #: see such as NH4+ or H3O+)
    charge_balance_residual: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _scalar_nulls(cls, data):
        # JSON stores NaN as null; these optional scalars are NaN when not computable.
        if isinstance(data, dict):
            data = dict(data)
            for k in ("capacity_mAh_g", "drying_index", "charge_balance_residual"):
                if k in data and data[k] is None:
                    data.pop(k)
        return data

    def flat(self) -> dict[str, float | str | None]:
        out: dict[str, float | str | None] = {
            "isolated_yield": self.isolated_yield,
            "capacity_mAh_g": self.capacity_mAh_g,
            "drying_index": self.drying_index,
            "charge_balance_residual": self.charge_balance_residual,
        }
        for prefix, block in (("xrd", self.xrd), ("comp", self.composition),
                              ("ir", self.ir), ("ec", self.echem)):
            if block is None:
                continue
            for k, v in block.model_dump().items():
                if isinstance(v, dict):          # e.g. phase_fractions -> xrd_w_<phase>
                    tag = {"phase_fractions": "w", "phase_lattice": "lat"}.get(k, k)
                    for kk, vv in v.items():
                        kk = str(kk).replace("/", "_")
                        out[f"{prefix}_{tag}_{kk}"] = (",".join(f"{x:.4f}" for x in vv)
                                                       if isinstance(vv, list) else vv)
                else:
                    out[f"{prefix}_{k}"] = v
        return out


# --------------------------------------------------------------------------- #
# Experiment lifecycle
# --------------------------------------------------------------------------- #


class ExperimentStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CHARACTERIZING = "characterizing"
    COMPLETE = "complete"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class Objectives(BaseModel):
    """Scalar targets handed to the optimizer, sign-normalized to *maximize*."""

    values: dict[str, float]
    constraints: dict[str, float] = Field(default_factory=dict)
    feasible: bool = True

    def vector(self, names: Sequence[str]) -> np.ndarray:
        return np.asarray([self.values[n] for n in names], dtype=float)


@dataclass
class Experiment:
    """One sample, from suggestion through characterization."""

    experiment_id: str
    campaign_id: str
    batch_index: int
    parameters: SynthesisParameters
    origin: str = "unknown"  # "doe" | "bo" | "manual" | "replicate"
    status: ExperimentStatus = ExperimentStatus.QUEUED
    descriptors: SampleDescriptors = field(default_factory=SampleDescriptors)
    objectives: Objectives | None = None
    raw_refs: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    def row(self) -> dict[str, Any]:
        """Flat record for dataframes and reports."""
        row: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "batch_index": self.batch_index,
            "origin": self.origin,
            "status": self.status.value,
        }
        row.update(self.parameters.model_dump())
        row.update(self.descriptors.flat())
        if self.objectives is not None:
            row.update({f"obj_{k}": v for k, v in self.objectives.values.items()})
            row.update({f"con_{k}": v for k, v in self.objectives.constraints.items()})
            row["feasible"] = self.objectives.feasible
        row["duration_s"] = self.duration_s
        row["error"] = self.error
        return row
