# SPDX-License-Identifier: GPL-3.0-or-later
"""Latent ground truth for the simulated PBA platform.

This module is the stand-in for the chemistry.  It maps a recipe onto the *true*
state of the resulting solid -- Na content, [Fe(CN)6] vacancy fraction, hydration,
coherent domain size, phase, yield -- and nothing else in the package is allowed
to import it except the instrument simulators.  The analysis and optimization
layers must reach these quantities only by measuring simulated traces, exactly as
they would on the bench.  That discipline is what makes the loop a real test of
the orchestrator rather than a test of the simulator.

The functional forms are phenomenological but follow the established
co-precipitation picture for sodium-rich hexacyanoferrates:

* Vacancies are a *kinetic* defect.  Fast release of M(2+) into a supersaturated
  hexacyanoferrate solution buries [Fe(CN)6] vacancies; slowing the crystal
  growth (chelation by citrate, slow metering, low supersaturation) lets the
  framework complete.  Elevated temperature accelerates both growth and
  reorganization, so its effect on vacancies is non-monotonic.
* Sodium occupancy of the large A-site is set by Na+ activity in the mother
  liquor and capped by charge balance against the vacancy content.  Strongly
  acidic conditions substitute hydronium for Na.
* Interstitial/coordinated water fills vacancy sites, so hydration tracks the
  vacancy fraction.
* Coherent domain size grows with temperature, ageing time and chelation, and
  shrinks at high supersaturation and fast addition.
* Yield rises with driving force, temperature and time, and falls when citrate
  holds M(2+) in solution or acid dissolves the product.

Every campaign gets its own :class:`GroundTruth` instance seeded from the
campaign id, so an independent random offset makes the optimum sit in a slightly
different place each run.  That prevents a benchmark from being solved by
memorizing coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..schema import (
    IONIC_RADIUS_A,
    LATTICE_A0_A,
    SynthesisParameters,
    theoretical_capacity_mAh_g,
)


#: Solubility products of M(OH)2 at 25 C (textbook values, order of magnitude
#: is what matters here); Cu(OH)2 is the precursor of the CuO seen by XRD.
KSP_HYDROXIDE: dict[str, float] = {
    "Mn": 1.9e-13, "Fe": 4.9e-17, "Co": 5.9e-15, "Ni": 5.5e-16, "Cu": 2.2e-20,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


@dataclass
class LatentState:
    """True physical state of one synthesized sample."""

    na_per_fu: float
    vacancy_fraction: float
    water_per_fu: float
    domain_size_nm: float
    lattice_a_A: float
    phase: str
    crystallinity: float
    isolated_yield: float
    conversion: float
    capacity_mAh_g: float
    solid_mass_mg: float
    failed: bool = False
    metal: str = ""
    #: crystalline secondary phases, as integrated Bragg intensity relative to
    #: a fully ordered PBA pattern (see ``simulate_xrd``)
    nacl_fraction: float = 0.0
    hydroxide_fraction: float = 0.0
    failure_mode: str | None = None

    @property
    def formula(self) -> str:
        return (
            f"Na{self.na_per_fu:.2f}M[Fe(CN)6]{1 - self.vacancy_fraction:.2f}"
            f"·{self.water_per_fu:.1f}H2O"
        )


class GroundTruth:
    """Hidden response surface plus stochastic platform failures.

    Parameters
    ----------
    seed
        Controls both the campaign-specific offsets of the response surface and
        the run-to-run irreproducibility.
    reproducibility
        Standard deviation of the multiplicative lot-to-lot noise applied to the
        latent state (0.02 is a well-behaved platform, 0.10 is a bad day).
    failure_rate
        Baseline probability that a run fails for a mundane mechanical reason
        (clogged tip, lost pellet).  Chemistry-driven failures -- a run that
        simply produces no solid -- are computed, not sampled.
    """

    def __init__(self, seed: int = 0, reproducibility: float = 0.03,
                 failure_rate: float = 0.02) -> None:
        self.seed = int(seed)
        self.reproducibility = float(reproducibility)
        self.failure_rate = float(failure_rate)
        rng = np.random.default_rng(self.seed)

        # Campaign-specific shifts: where the sweet spot sits this time.
        self._t_opt_C = float(rng.uniform(55.0, 78.0))
        self._ph_opt = float(rng.uniform(2.4, 4.2))
        self._cit_opt_ratio = float(rng.uniform(0.55, 1.15))
        self._vacancy_floor = float(rng.uniform(0.015, 0.045))
        self._metal_bias = {
            m: float(b)
            for m, b in zip(
                IONIC_RADIUS_A, rng.normal(0.0, 0.035, size=len(IONIC_RADIUS_A))
            )
        }
        # Baseline cubic lattice constant per analogue, with a small per-campaign
        # offset so the benchmark cannot be solved by memorizing absolute values.
        self._a0 = {
            m: a + float(rng.normal(0.0, 0.012)) for m, a in LATTICE_A0_A.items()
        }

    # ------------------------------------------------------------------ #
    # Latent physics
    # ------------------------------------------------------------------ #

    def _supersaturation(self, p: SynthesisParameters) -> float:
        """Dimensionless driving force at the mixing point.

        Ion product relative to a reference solubility, damped by the fraction of
        M(2+) that citrate holds as a soluble complex.
        """
        free_metal = p.c_metal_M / (1.0 + 6.0 * p.c_citrate_M)
        ion_product = free_metal * p.c_hcf_M
        return float(np.log10(max(ion_product, 1e-9) / 1e-6))

    def latent(self, p: SynthesisParameters, rng: np.random.Generator) -> LatentState:
        S = self._supersaturation(p)                       # ~0 to 4
        s_norm = _clip(S / 4.0, 0.0, 1.4)
        cit_ratio = p.citrate_to_metal_ratio                # 0 to ~45
        cit_eff = _clip(cit_ratio / 1.5, 0.0, 1.6)          # saturating brake
        t_norm = (p.temperature_C - 25.0) / 65.0
        rate_norm = _clip(math.log10(p.addition_rate_mL_min / 0.05)
                          / math.log10(20.0 / 0.05), 0.0, 1.0)
        age_norm = _clip(math.log(p.aging_time_h / 0.5) / math.log(24.0 / 0.5), 0.0, 1.0)
        shear = _clip((p.stir_rate_rpm - 200.0) / 1000.0, 0.0, 1.0)
        excess_hcf = _clip(p.hcf_to_metal_ratio - 1.0, -0.9, 3.0)

        # -- vacancy fraction: kinetic defect, minimized by slow ordered growth
        z_vac = (
            -0.55
            + 1.55 * s_norm
            + 1.05 * rate_norm
            - 1.85 * cit_eff * (1.0 - 0.35 * cit_eff)      # diminishing returns
            - 1.25 * age_norm
            - 0.85 * _clip(excess_hcf, 0.0, 3.0) / 3.0     # HCF excess fills sites
            + 0.55 * shear
            - 1.10 * math.exp(-((p.temperature_C - self._t_opt_C) ** 2) / (2 * 18.0 ** 2))
            + 0.45 * max(0.0, (p.temperature_C - self._t_opt_C - 12.0) / 25.0)
            + 0.60 * max(0.0, (2.0 - p.ph))                # acid hydrolysis of HCF
            + 0.35 * max(0.0, (p.ph - 5.5))                # hydroxide co-precipitation
            + self._metal_bias[p.metal] * 6.0
        )
        vacancy = self._vacancy_floor + (0.62 - self._vacancy_floor) * _sigmoid(z_vac)

        # -- sodium occupancy: Na+ activity limited, charge balanced
        na_activity = p.c_nacl_M + 4.0 * p.c_hcf_M * (p.volume_B_mL / p.total_volume_mL)
        occ = 0.94 * na_activity / (0.55 + na_activity)     # Langmuir-like
        proton_penalty = _sigmoid((p.ph - 1.9) * 2.4)       # H3O+ competes below pH ~2
        na_max = 2.0 * (1.0 - vacancy)                      # charge balance ceiling
        na = na_max * occ * proton_penalty
        na = _clip(na, 0.02, 2.0)

        # -- hydration fills the vacancy cavities
        water = _clip(0.7 + 6.2 * vacancy + 0.9 * (1.0 - occ), 0.4, 6.0)

        # -- coherent domain size
        d = (
            14.0
            * (1.0 + 1.45 * age_norm)
            * (1.0 + 1.25 * _clip(t_norm, 0.0, 1.0))
            * (1.0 + 0.85 * min(cit_eff, 1.2))
            / (1.0 + 1.35 * rate_norm + 0.95 * max(0.0, s_norm - 0.35))
        )
        d = _clip(d, 3.5, 220.0)

        crystallinity = _clip(
            (1.0 - 1.35 * vacancy) * (1.0 - math.exp(-d / 12.0))
            * (1.0 - 0.30 * max(0.0, p.ph - 5.5)),
            0.02, 0.995,
        )

        # -- phase assignment follows Na content (rhombohedral distortion) and
        #    disorder (loss of long-range order at very small domains)
        if crystallinity < 0.18 or d < 5.0:
            phase = "amorphous"
        elif na > 1.72 and p.metal in ("Mn", "Fe"):
            phase = "monoclinic"
        elif na > 1.45:
            phase = "rhombohedral"
        else:
            phase = "cubic"

        lattice_a = (
            self._a0[p.metal]
            + 0.135 * vacancy
            - 0.048 * na
            + 0.010 * (p.temperature_C - 25.0) / 65.0
        )

        # -- conversion of hexacyanoferrate and isolated yield
        z_conv = (
            -1.05
            + 2.35 * s_norm
            + 1.35 * _clip(t_norm, 0.0, 1.0)
            + 1.15 * age_norm
            - 1.55 * cit_eff
            - 0.85 * max(0.0, (2.2 - p.ph))
            - 0.45 * _clip(excess_hcf, 0.0, 3.0) / 3.0
        )
        conversion = _clip(0.985 * _sigmoid(z_conv), 0.004, 0.985)
        # Handling losses: fine particles are lost in the wash steps.
        recovery = _clip(0.985 - 0.42 * math.exp(-d / 9.0), 0.45, 0.985)
        isolated = _clip(conversion * recovery, 0.0, 1.0)

        limiting_mol = min(
            p.c_metal_M * p.volume_A_mL, p.c_hcf_M * p.volume_B_mL
        ) * 1e-3
        from ..schema import formula_weight

        fw = formula_weight(p.metal, na, vacancy, water)
        solid_mass_mg = limiting_mol * isolated * fw * 1e3

        # -- electrochemical capacity proxy: Na inventory times utilization
        q_theo = theoretical_capacity_mAh_g(p.metal, na, vacancy, water)
        kinetic = _clip(1.06 / (1.0 + (d / 95.0) ** 1.7), 0.30, 1.0)
        defect = _clip(1.0 - 1.15 * vacancy, 0.15, 1.0)
        hydration = _clip(1.0 - 0.055 * water, 0.55, 1.0)
        capacity = q_theo * kinetic * defect * hydration * crystallinity ** 0.35

        # -- crystalline secondary phases (seen by XRD only: their mass and Na
        #    are deliberately not added to the gravimetric or ICP results)
        # NaCl left in the powder when the supporting electrolyte is concentrated.
        nacl_frac = 0.30 * _sigmoid((p.c_nacl_M - 2.6) / 0.30)
        # M(OH)2 once pH passes the solubility-product onset for the *free* metal;
        # citrate is taken to bind M2+ 1:1, which delays the onset.
        free_m = max(p.c_metal_M - p.c_citrate_M, 0.03 * p.c_metal_M)
        ph_onset = 14.0 + 0.5 * math.log10(KSP_HYDROXIDE[p.metal] / free_m)
        hydroxide_frac = 0.40 * _sigmoid((p.ph - ph_onset) / 0.25)

        state = LatentState(
            na_per_fu=na,
            vacancy_fraction=vacancy,
            water_per_fu=water,
            domain_size_nm=d,
            lattice_a_A=lattice_a,
            phase=phase,
            crystallinity=crystallinity,
            isolated_yield=isolated,
            conversion=conversion,
            capacity_mAh_g=capacity,
            solid_mass_mg=solid_mass_mg,
            metal=p.metal,
            nacl_fraction=nacl_frac,
            hydroxide_fraction=hydroxide_frac,
        )
        return self._apply_noise_and_failures(state, rng)

    # ------------------------------------------------------------------ #
    # Platform realism
    # ------------------------------------------------------------------ #

    def _apply_noise_and_failures(
        self, s: LatentState, rng: np.random.Generator
    ) -> LatentState:
        sd = self.reproducibility
        jitter = lambda v, k=1.0: float(v * max(0.0, 1.0 + rng.normal(0.0, sd * k)))  # noqa: E731

        s.vacancy_fraction = _clip(jitter(s.vacancy_fraction, 1.4), 0.005, 0.72)
        s.na_per_fu = _clip(jitter(s.na_per_fu), 0.01, 2.0)
        s.water_per_fu = _clip(jitter(s.water_per_fu), 0.1, 6.5)
        s.domain_size_nm = _clip(jitter(s.domain_size_nm, 1.8), 2.0, 260.0)
        s.crystallinity = _clip(jitter(s.crystallinity, 0.8), 0.01, 0.999)
        s.isolated_yield = _clip(jitter(s.isolated_yield, 1.2), 0.0, 1.0)
        s.conversion = _clip(jitter(s.conversion), 0.0, 0.999)
        s.capacity_mAh_g = jitter(s.capacity_mAh_g, 0.9)
        s.solid_mass_mg = jitter(s.solid_mass_mg, 1.5)
        s.lattice_a_A = float(s.lattice_a_A + rng.normal(0.0, 0.004))

        # Chemistry-driven failures only.  Mechanical failures (clogged tip,
        # thermal abort) are properties of the deck, not of the product, and are
        # sampled by the device that would experience them -- see
        # ``SimulatedBackend.mechanical_fault``.  Mixing the two here previously
        # allowed a latent state to be marked failed with a mode that no device
        # ever raised, so the run continued and was scored as a success.
        if s.solid_mass_mg < 1.5:
            s.failed = True
            s.failure_mode = "insufficient_solid"
        elif rng.random() < self.failure_rate:
            # Product lost in the decant: chemistry-adjacent (a poorly flocculated,
            # colloidal precipitate does not pellet), so it belongs here.
            s.failed = True
            s.failure_mode = "pellet_lost_in_decant"
        return s
