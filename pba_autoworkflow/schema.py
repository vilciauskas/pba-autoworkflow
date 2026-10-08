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
METALS: tuple[str, ...] = ("Mn", "Fe", "Co", "Ni", "Cu")

#: Atomic weights (g/mol) needed for formula-weight bookkeeping.
ATOMIC_WEIGHT: dict[str, float] = {
    "Na": 22.9898,
    "Mn": 54.9380,
    "Fe": 55.8450,
    "Co": 58.9332,
    "Ni": 58.6934,
    "Cu": 63.5460,
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
}


def formula_weight(metal: str, na_per_fu: float, vacancy_fraction: float,
                   water_per_fu: float) -> float:
    """Formula weight (g/mol) of Na_n M[Fe(CN)6]_(1-x) . zH2O."""
    hcf = ATOMIC_WEIGHT["Fe"] + 6 * (ATOMIC_WEIGHT["C"] + ATOMIC_WEIGHT["N"])
    # A measured vacancy may be marginally negative from assay error; an
    # occupancy above unity is not a physical structure, so clamp here.
    vacancy_fraction = min(max(vacancy_fraction, 0.0), 0.9)
    return (
        na_per_fu * ATOMIC_WEIGHT["Na"]
        + ATOMIC_WEIGHT[metal]
        + (1.0 - vacancy_fraction) * hcf
        + water_per_fu * (2 * ATOMIC_WEIGHT["H"] + ATOMIC_WEIGHT["O"])
    )


def theoretical_capacity_mAh_g(metal: str, na_per_fu: float,
                               vacancy_fraction: float, water_per_fu: float) -> float:
    """Two-electron theoretical capacity for the extractable Na inventory."""
    fw = formula_weight(metal, na_per_fu, vacancy_fraction, water_per_fu)
    return na_per_fu * FARADAY_C_PER_MOL / (3.6 * fw)


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

    metal: Literal["Mn", "Fe", "Co", "Ni", "Cu"] = "Mn"
    c_metal_M: float = Field(..., gt=0, description="M(II) salt in solution A")
    c_hcf_M: float = Field(..., gt=0, description="Na4[Fe(CN)6] in solution B")
    c_nacl_M: float = Field(..., ge=0, description="supporting NaCl in solution B")
    c_citrate_M: float = Field(..., ge=0, description="sodium citrate chelator in A")
    ph: float = Field(..., ge=0.5, le=9.0)
    temperature_C: float = Field(..., ge=15.0, le=95.0)
    addition_rate_mL_min: float = Field(..., gt=0)
    aging_time_h: float = Field(..., gt=0)
    stir_rate_rpm: float = Field(..., ge=0, le=1500)

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
                          description="Na4[Fe(CN)6] in solution B"),
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
        ),
        categorical=(
            CategoricalSpec(name="metal", choices=METALS,
                            description="divalent metal on the N-coordinated site"),
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


# --------------------------------------------------------------------------- #
# Extracted descriptors
# --------------------------------------------------------------------------- #


class XRDDescriptors(BaseModel):
    lattice_a_A: float
    domain_size_nm: float
    crystallinity_index: float = Field(..., ge=0.0, le=1.0)
    fwhm_200_deg: float
    phase: Literal["cubic", "rhombohedral", "monoclinic", "amorphous"]
    n_peaks_indexed: int
    fit_residual: float
    #: PBA share of the Bragg intensity, 0-1.  An *intensity* fraction, not a
    #: weight fraction (that would need reference intensity ratios).  None for
    #: results recorded before secondary phases were analysed.
    phase_purity: float | None = Field(default=None, ge=0.0, le=1.0)
    #: fitted peaks that index to no PBA reflection (candidate impurity lines)
    n_impurity_peaks: int = 0


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


class SampleDescriptors(BaseModel):
    """Everything the analysis layer extracted for one sample."""

    xrd: XRDDescriptors | None = None
    composition: CompositionDescriptors | None = None
    isolated_yield: float | None = Field(default=None, ge=0.0, le=1.2)
    capacity_mAh_g: float | None = None

    def flat(self) -> dict[str, float | str | None]:
        out: dict[str, float | str | None] = {
            "isolated_yield": self.isolated_yield,
            "capacity_mAh_g": self.capacity_mAh_g,
        }
        for prefix, block in (("xrd", self.xrd), ("comp", self.composition)):
            if block is None:
                continue
            for k, v in block.model_dump().items():
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
