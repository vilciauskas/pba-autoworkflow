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
* The hexacyanoferrate precursor sets the Fe valence and the A cation:
  Na4[Fe(CN)6] gives Fe(II) frameworks filled with Na+ (oxidised to Fe(III)
  where the A sites stay empty), K3[Fe(CN)6] gives Fe(III) frameworks, part of
  which is reduced during precipitation and charge-compensated by K+.  K+ is
  taken up in preference to Na+ (``K_OVER_NA``).  The A content always obeys
  the charge balance x = (1 - y)(3 + f_Fe(II)) - 2.

The solid is formed in two stages that the simulated deck resolves where they
physically happen: :meth:`GroundTruth.precipitate` when the slurry is aged, and
:meth:`GroundTruth.dry` at the workup's drying step, with the temperature,
pressure and gas that step was actually commanded.  Drying follows the zinc
hexacyanoferrate study of Pilipavicius, Skarnulyte, Gece and Vilciauskas:

* Dehydration happens when the gas is drier than the hydrate's equilibrium
  water pressure (taken as 0.45 p_sat(T)): ambient air at room temperature
  leaves the hydrated cubic phase in place, a desiccator or any heating does not.
* For dehydrating cubic Zn3[Fe(CN)6]2 the *rate* of water removal selects the
  product, through the drying-rate proxy Pi = (p_sat - p_w)/P_total: slow
  (low Pi) gives anhydrous R-3c, fast (high Pi) freezes a contracted,
  disordered cubic framework that scatters partly diffusely.  The measured
  routes bracket the switch between Pi ~ 2 and ~ 250; the simulator puts it
  at ``ZN_PI_CRITICAL`` (shifted per campaign), which is a guess inside that range.
* Fe(III) is reduced on heating, more so under reduced pressure (XPS: 58-62 %
  Fe(II) after room-temperature routes, 66 % at 120 degC in air, 77 % at
  120 degC and 8 mbar).  Reduction in the dry solid has no cation source, so it
  is modelled as reductive decyanation (loss of a neutral CN per reduced Fe),
  which lowers the carbon assay slightly.  Whether the real material instead
  takes up residual K+ is open; the simulated K content is a prediction to test.

Every campaign gets its own :class:`GroundTruth` instance seeded from the
campaign id, so an independent random offset makes the optimum sit in a slightly
different place each run.  That prevents a benchmark from being solved by
memorizing coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..schema import (
    HCF_PRECURSOR_INFO,
    IONIC_RADIUS_A,
    LATTICE_A0_A,
    SynthesisParameters,
    charge_balanced_a,
    drying_rate_index,
    drying_water_pressure_mbar,
    formula_weight,
    theoretical_capacity_mAh_g,
    water_saturation_pressure_mbar,
)

#: A-site affinity of K+ relative to Na+ (Langmuir constants).  PBAs take up
#: K+ in preference to Na+; the factor is illustrative.
K_OVER_NA = 10.0
#: Fe(II) share of a K3[Fe(CN)6]-route framework straight after precipitation
#: (partial reduction in solution, compensated by K+).  An assumption: the
#: paper's XPS gives ~0.6 after room-temperature drying, but XPS samples the
#: surface and ferricyanides reduce under X-rays.
FE3_ROUTE_FE2_FRACTION = 0.35
#: Drying-rate proxy at which dehydrating cubic ZnHCF switches from R-3c to
#: disordered cubic (log-midpoint of the untested 2-250 range).
ZN_PI_CRITICAL = 20.0
#: Hydrate dehydrates when the gas water pressure is below this share of p_sat(T).
HYDRATE_WATER_ACTIVITY = 0.45
#: Share of a disordered-cubic framework's ordered scattering that is diffuse.
DC_DIFFUSE_SHARE = 0.35
#: Cell of the anhydrous R-3c phase relative to the hydrated library entry
#: (a 12.61 vs 12.47 A, c 32.96 vs 32.92 A).
R3C_ANHYDROUS_SCALE = (1.0113, 1.0012)


#: Solubility products of M(OH)2 at 25 C (textbook values, order of magnitude
#: is what matters here); Cu(OH)2 is the precursor of the CuO seen by XRD.
KSP_HYDROXIDE: dict[str, float] = {
    "Mn": 1.9e-13, "Fe": 4.9e-17, "Co": 5.9e-15, "Ni": 5.5e-16, "Cu": 2.2e-20,
    "Zn": 3.0e-17,   # epsilon-Zn(OH)2, precursor of the ZnO seen by XRD
}

#: Crystalline product of the high-pH side reaction, as a phase-library key.
HIGH_PH_PHASE: dict[str, str] = {
    "Mn": "m_oh2/Mn", "Fe": "m_oh2/Fe", "Co": "m_oh2/Co", "Ni": "m_oh2/Ni",
    "Cu": "cuo", "Zn": "zno",
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
    #: crystalline secondary phases, as mass fractions of the dried solid
    nacl_fraction: float = 0.0
    hydroxide_fraction: float = 0.0
    failure_mode: str | None = None
    #: framework polymorphs, as shares of the framework mass (phase-library keys)
    polymorph_shares: dict = field(default_factory=dict)
    #: lattice scale relative to the library reference, per non-cubic phase key
    lattice_scale: dict = field(default_factory=dict)
    k_per_fu: float = 0.0
    #: Fe(II) share of the hexacyanoferrate
    fe2_fraction: float = 1.0
    #: CN lost per formula unit by reductive decyanation during drying
    cn_loss_per_fu: float = 0.0
    #: share of each framework phase's ordered scattering that is diffuse
    diffuse_share: dict = field(default_factory=dict)
    #: framework share (of the Zn cubic phase) that is the disordered, dehydrated form
    disordered_cubic_share: float = 0.0
    #: extent of dehydration during drying, 0-1
    dehydration: float = 0.0
    dried: bool = False
    hcf_precursor: str = "Na4FeCN6"

    @property
    def a_per_fu(self) -> float:
        return self.na_per_fu + self.k_per_fu

    def crystalline_masses(self) -> dict[str, float]:
        """Mass of each crystalline phase per unit mass of solid (library keys).

        The disordered share of the framework (``1 - crystallinity``) scatters
        into the amorphous halo and is not a crystalline phase.
        """
        framework = max(1.0 - self.nacl_fraction - self.hydroxide_fraction, 0.0)
        # Bragg-visible mass: the diffuse share of a disordered framework is
        # reported separately (XRDDescriptors.diffuse_fraction), not as a phase.
        out = {k: framework * self.crystallinity * s * (1.0 - self.diffuse_share.get(k, 0.0))
               for k, s in self.polymorph_shares.items()}
        if self.nacl_fraction > 0:
            out["nacl"] = self.nacl_fraction
        if self.hydroxide_fraction > 0 and self.metal in HIGH_PH_PHASE:
            out[HIGH_PH_PHASE[self.metal]] = self.hydroxide_fraction
        return out

    def xrd_weight_fractions(self) -> dict[str, float]:
        """What a perfect XRD quantification would report, by phase id."""
        m = self.crystalline_masses()
        total = sum(m.values())
        out: dict[str, float] = {}
        for k, v in m.items():
            pid = k.split("/")[0]
            out[pid] = out.get(pid, 0.0) + (v / total if total > 0 else 0.0)
        return out

    def framework_diffuse_fraction(self) -> float:
        """Diffuse share of all framework scattering (Bragg + diffuse + halo)."""
        return self.crystallinity * sum(s * self.diffuse_share.get(k, 0.0)
                                        for k, s in self.polymorph_shares.items())

    @property
    def formula(self) -> str:
        k = f"K{self.k_per_fu:.2f}" if self.k_per_fu > 0.005 else ""
        return (
            f"Na{self.na_per_fu:.2f}{k}M[Fe(CN)6]{1 - self.vacancy_fraction:.2f}"
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
        # Campaign-specific offset of the Zn polymorph balance.
        self._zn_bias = float(rng.normal(0.0, 0.3))
        # Drawn last so that adding them left the earlier draws (and the
        # response surface of existing seeds) unchanged.
        self._zn_pi_critical = ZN_PI_CRITICAL * 10 ** float(rng.normal(0.0, 0.25))
        #: per-ion formal-potential offsets for the electrochemistry simulator
        self.echem_offsets = {ion: float(v) for ion, v in
                              zip(("Zn", "Na", "K"), rng.normal(0.0, 0.015, size=3))}
        #: true integrated-absorptivity ratio eps(Fe(III)-CN)/eps(Fe(II)-CN):
        #: the IR analysis assumes 0.25, so this is its calibration error
        self.ir_absorptivity_ratio = 0.25 * float(np.exp(rng.normal(0.0, 0.15)))

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
        """Precipitate and dry with the recipe's own drying conditions."""
        s = self.precipitate(p, rng)
        return self.dry(s, p.dry_temperature_C, p.dry_pressure_mbar, p.dry_gas, rng)

    def precipitate(self, p: SynthesisParameters, rng: np.random.Generator) -> LatentState:
        """The washed, still-wet solid (drying not yet applied)."""
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
        # Capped at 0.5: beyond that M[Fe(CN)6]_(1-y) cannot be charge-neutral.
        vacancy = self._vacancy_floor + (0.50 - self._vacancy_floor) * _sigmoid(z_vac)

        # -- A cations: Na+ and K+ compete for the A sites; charge balanced
        a_cation, a_per_hcf, fe_ox = HCF_PRECURSOR_INFO[p.hcf_precursor]
        from_hcf = a_per_hcf * p.c_hcf_M * (p.volume_B_mL / p.total_volume_mL)
        na_activity = p.c_nacl_M + (from_hcf if a_cation == "Na" else 0.0)
        k_activity = from_hcf if a_cation == "K" else 0.0
        eff = na_activity + K_OVER_NA * k_activity
        k_share = K_OVER_NA * k_activity / eff if eff > 0 else 0.0
        occ = 0.94 * eff / (0.55 + eff)                     # Langmuir-like
        proton_penalty = _sigmoid((p.ph - 1.9) * 2.4)       # H3O+ competes below pH ~2
        if fe_ox == 2:
            # Fe(II) framework: A sites fill toward the charge-balance ceiling;
            # unfilled sites leave Fe(III) behind (air oxidation).
            a_tot = max(charge_balanced_a(vacancy, 1.0), 0.0) * occ * proton_penalty
            fe2 = 1.0
        else:
            # Fe(III) framework: y <= 1/3 for neutrality; partial reduction in
            # solution is compensated by A+ from the mother liquor.
            vacancy = min(vacancy, 1.0 / 3.0)
            fe2 = _clip(FE3_ROUTE_FE2_FRACTION + rng.normal(0.0, 0.03), 0.0, 1.0)
            a_tot = max(charge_balanced_a(vacancy, fe2), 0.0)
        a_tot = _clip(a_tot, 0.02, 2.0)
        if fe_ox == 2:
            fe2 = _clip((a_tot + 2.0) / max(1.0 - vacancy, 1e-6) - 3.0, 0.0, 1.0)
        na, k_a = a_tot * (1.0 - k_share), a_tot * k_share

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

        # Only vacancies beyond the stoichiometric content disorder the
        # framework: y = 1/3 is the ordered vacancy topology of M3[Fe(III)(CN)6]2
        # and of zinc hexacyanoferrate.
        y_stoich = 1.0 / 3.0 if (fe_ox == 3 or p.metal == "Zn") else 0.0
        crystallinity = _clip(
            (1.0 - 1.35 * max(vacancy - y_stoich, 0.0)) * (1.0 - math.exp(-d / 12.0))
            * (1.0 - 0.30 * max(0.0, p.ph - 5.5)),
            0.02, 0.995,
        )

        # -- framework polymorphs
        lattice_scale: dict[str, tuple[float, ...]] = {}
        if p.metal == "Zn":
            # Zn: cubic Fm-3m (octahedral Zn, Zn3[Fe(CN)6]2 vacancy topology)
            # versus rhombohedral R-3c A2Zn3[Fe(CN)6]2 (tetrahedral ZnN4).
            # Precipitation only; drying converts the cubic phase (see dry()).
            # With an alkali ferrocyanide the hydrated R-3c phase precipitates
            # directly unless fast addition / high supersaturation trap the
            # cubic one; with K3[Fe(CN)6] there is little A+ to template it and
            # the hydrated cubic Zn3[Fe(CN)6]2 forms (as in the zinc paper).
            # Illustrative, not fitted.
            zn_excess = _clip(1.0 / p.hcf_to_metal_ratio - 1.5, 0.0, 3.0)
            f_a = eff / (0.3 + eff)
            z_r3c = (
                -0.4 + self._zn_bias
                + 2.0 * f_a
                - 1.6 * rate_norm
                - 1.0 * max(0.0, s_norm - 0.3)
                + 1.2 * _clip(t_norm, 0.0, 1.0)
                + 1.0 * age_norm
                - 0.6 * zn_excess
                - (3.5 if fe_ox == 3 else 0.0)
            )
            r3c = _sigmoid(z_r3c)
            # Zn3[Fe(CN)6]2 stoichiometry (y = 1/3), then charge balance.
            vacancy = 1.0 / 3.0 + (0.10 * (1.0 - r3c) * zn_excess / 3.0 if fe_ox == 2 else 0.0)
            if fe_ox == 2:
                fe2 = 1.0   # restart from the precursor's Fe(II), not the generic branch's value
            a_tot = _clip(max(charge_balanced_a(vacancy, fe2), 0.0)
                          * (proton_penalty if fe_ox == 2 else 1.0), 0.0, 2.0)
            if fe_ox == 2:
                fe2 = _clip((a_tot + 2.0) / (1.0 - vacancy) - 3.0, 0.0, 1.0)
            na, k_a = a_tot * (1.0 - k_share), a_tot * k_share
            water = 3.0 + 0.6 * (1.0 - r3c)
            shares = {"pba_fm3m/Zn": 1.0 - r3c, "znhcf_r3c/Zn": r3c}
            lattice_scale["znhcf_r3c/Zn"] = (1.0 + float(rng.normal(0.0, 0.004)),
                                             1.0 + float(rng.normal(0.0, 0.004)))
        elif p.metal in ("Mn", "Fe"):
            # A-rich hydrated Mn/Fe frameworks adopt the monoclinic P2_1/n cell.
            # The onset is placed where this simulator's A model can reach it
            # (A <= ~1.6 per f.u.); real materials transform closer to A ~ 1.7.
            p21n = _sigmoid((na + k_a - 1.42) / 0.05)
            shares = {f"pba_fm3m/{p.metal}": 1.0 - p21n, f"pba_p21n/{p.metal}": p21n}
            lattice_scale[f"pba_p21n/{p.metal}"] = (1.0 + float(rng.normal(0.0, 0.003)),)
        else:
            shares = {f"pba_fm3m/{p.metal}": 1.0}
        shares = {k: v for k, v in shares.items() if v > 1e-4}

        # -- dominant phase; loss of long-range order at very small domains
        if crystallinity < 0.18 or d < 5.0:
            phase = "amorphous"
        else:
            phase = max(shares, key=shares.get).split("/")[0]

        lattice_a = (
            self._a0[p.metal]
            + (0.0 if p.metal == "Zn" else 0.135 * vacancy - 0.048 * (na + k_a)
               - 0.06 * (1.0 - fe2))
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
        fw = formula_weight(p.metal, na, vacancy, water, k_a)
        solid_mass_mg = limiting_mol * isolated * fw * 1e3

        # -- electrochemical capacity proxy: A inventory times utilization
        q_theo = theoretical_capacity_mAh_g(p.metal, na, vacancy, water, k_a)
        kinetic = _clip(1.06 / (1.0 + (d / 95.0) ** 1.7), 0.30, 1.0)
        defect = _clip(1.0 - 1.15 * vacancy, 0.15, 1.0)
        hydration = _clip(1.0 - 0.055 * water, 0.55, 1.0)
        capacity = q_theo * kinetic * defect * hydration * crystallinity ** 0.35

        # -- crystalline secondary phases (seen by XRD only: their mass and Na
        #    are deliberately not added to the gravimetric or ICP results)
        # NaCl left in the powder when the supporting electrolyte is concentrated.
        nacl_frac = 0.25 * _sigmoid((p.c_nacl_M - 2.6) / 0.30)
        # M(OH)2 once pH passes the solubility-product onset for the *free* metal;
        # citrate is taken to bind M2+ 1:1, which delays the onset.
        free_m = max(p.c_metal_M - p.c_citrate_M, 0.03 * p.c_metal_M)
        ph_onset = 14.0 + 0.5 * math.log10(KSP_HYDROXIDE[p.metal] / free_m)
        hydroxide_frac = 0.35 * _sigmoid((p.ph - ph_onset) / 0.25)

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
            polymorph_shares=shares,
            lattice_scale=lattice_scale,
            k_per_fu=k_a,
            fe2_fraction=fe2,
            hcf_precursor=p.hcf_precursor,
        )
        return self._apply_noise_and_failures(state, rng)

    # ------------------------------------------------------------------ #
    # Drying
    # ------------------------------------------------------------------ #

    def dry(self, s: LatentState, temperature_C: float, pressure_mbar: float,
            gas: str, rng: np.random.Generator) -> LatentState:
        """Apply the drying step to a precipitated solid (in place; idempotent)."""
        if s.dried:
            return s
        s.dried = True
        T = float(temperature_C)
        psat = water_saturation_pressure_mbar(T)
        pw = drying_water_pressure_mbar(pressure_mbar, gas)
        p_eq = HYDRATE_WATER_ACTIVITY * psat
        dehyd = _sigmoid((p_eq - pw) / (0.08 * p_eq + 0.2))
        s.dehydration = dehyd
        pi = drying_rate_index(T, pressure_mbar, gas)
        water_before = s.water_per_fu

        if s.metal == "Zn":
            cub = s.polymorph_shares.get("pba_fm3m/Zn", 0.0)
            r3c = s.polymorph_shares.get("znhcf_r3c/Zn", 0.0)
            to_r3c = _sigmoid((math.log10(self._zn_pi_critical) - math.log10(pi)) / 0.3)
            conv = cub * dehyd * to_r3c                    # cubic -> anhydrous R-3c
            dc = cub * dehyd * (1.0 - to_r3c)              # cubic -> disordered cubic
            shares = {"pba_fm3m/Zn": cub - conv, "znhcf_r3c/Zn": r3c + conv}
            s.polymorph_shares = {k: v for k, v in shares.items() if v > 1e-4}
            cubic_left = cub - conv
            s.disordered_cubic_share = dc / cubic_left if cubic_left > 1e-6 else 0.0
            if s.disordered_cubic_share > 0.0 and "pba_fm3m/Zn" in s.polymorph_shares:
                s.diffuse_share["pba_fm3m/Zn"] = DC_DIFFUSE_SHARE * s.disordered_cubic_share
            # Disordered cubic is contracted: 98.4 % of the hydrated volume after
            # room-temperature vacuum, 93.3 % at 120 degC (paper, Table 1).
            vol = 1.0 - 0.016 - 0.052 * _sigmoid((T - 80.0) / 15.0)
            s.lattice_a_A *= 1.0 - s.disordered_cubic_share * (1.0 - vol ** (1.0 / 3.0))
            # Anhydrous R-3c cell (dehydrated share of the R-3c phase).
            if "znhcf_r3c/Zn" in s.polymorph_shares:
                a0, c0 = s.lattice_scale.get("znhcf_r3c/Zn", (1.0, 1.0))
                sa, sc = R3C_ANHYDROUS_SCALE
                s.lattice_scale["znhcf_r3c/Zn"] = (a0 * (1.0 + dehyd * (sa - 1.0)),
                                                   c0 * (1.0 + dehyd * (sc - 1.0)))
            s.water_per_fu *= 1.0 - 0.9 * dehyd
            # Loss of long-range order in the frozen framework
            s.crystallinity = _clip(s.crystallinity * (1.0 - 0.15 * dc), 0.01, 0.999)
            if s.crystallinity < 0.18:
                s.phase = "amorphous"
            elif s.phase != "amorphous":
                s.phase = max(s.polymorph_shares, key=s.polymorph_shares.get).split("/")[0]
        else:
            s.water_per_fu *= 1.0 - 0.35 * dehyd

        # Fe(III) reduction on heating, deeper under reduced pressure;
        # compensated by reductive decyanation (no cation source in a dry solid).
        g = _clip(math.log10(1013.0 / max(pressure_mbar, 1e-3)) / 2.0, 0.0, 1.0)
        # Calibrated on the XPS increments: +6 points at 120 degC in air,
        # +17 points at 120 degC and 8 mbar (relative to room-temperature routes).
        d_fe2 = min(1.0 - s.fe2_fraction,
                    _sigmoid((T - 90.0) / 12.0) * (0.066 + 0.119 * g))
        d_fe2 = max(0.0, d_fe2 * (1.0 + rng.normal(0.0, 0.05)))
        s.fe2_fraction = _clip(s.fe2_fraction + d_fe2, 0.0, 1.0)
        s.cn_loss_per_fu = d_fe2 * (1.0 - s.vacancy_fraction)

        # The weighed mass follows the water lost
        fw0 = formula_weight(s.metal, s.na_per_fu, s.vacancy_fraction, water_before, s.k_per_fu)
        fw1 = formula_weight(s.metal, s.na_per_fu, s.vacancy_fraction, s.water_per_fu, s.k_per_fu)
        s.solid_mass_mg *= fw1 / fw0
        return s

    # ------------------------------------------------------------------ #
    # Platform realism
    # ------------------------------------------------------------------ #

    def _apply_noise_and_failures(
        self, s: LatentState, rng: np.random.Generator
    ) -> LatentState:
        sd = self.reproducibility
        jitter = lambda v, k=1.0: float(v * max(0.0, 1.0 + rng.normal(0.0, sd * k)))  # noqa: E731

        s.vacancy_fraction = _clip(jitter(s.vacancy_fraction, 1.4), 0.005, 0.72)
        s.na_per_fu = _clip(jitter(s.na_per_fu), 0.0, 2.0)
        s.k_per_fu = _clip(jitter(s.k_per_fu), 0.0, 2.0)
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
