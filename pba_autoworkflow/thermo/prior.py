# SPDX-License-Identifier: GPL-3.0-or-later
"""Using a computed hull inside the experimental campaign.

The orchestrator's standing rule is that the analysis layer never imports the
simulator, so that a simulated campaign tests the orchestrator rather than the
simulator.  The same rule applies here for the same reason, in the opposite
direction: **the thermodynamics module must not feed the analysis path.**  A
computed energy is a prediction; a measured composition is an observation; and if
a prediction can reach the surrogate's training data then a force-field artefact
becomes an experimental result, which is precisely the failure the Fe/carbon bug
was.

So the hull enters the loop in exactly one place -- as a *prior over where to
look* -- and it enters with a knob that can turn it off:

:class:`HullPrior`
    Scores a proposed recipe by the predicted stability of the composition it is
    expected to produce.  The planner can use this to break ties among candidates
    the surrogate rates equally, and to avoid spending early batches in a region
    the force field says will phase-separate.

What it must never do, and does not:

*   Substitute for a measurement.  The prior scores *recipes before they run*; it
    never writes a descriptor, never sets an objective, and nothing it returns is
    stored as an experimental outcome.
*   Silently dominate.  ``weight`` defaults low, and
    :meth:`HullPrior.disagreement` reports where the prior and the accumulated
    measurements disagree -- which is the interesting scientific signal, and would
    be destroyed by a prior strong enough to prevent the campaign from ever
    sampling there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hull import HullResult


@dataclass
class HullPrior:
    """A stability prior over the Na/vacancy plane, per metal.

    ``weight`` scales the prior's influence; the default is deliberately small.
    The prior is advisory -- a force field that gets the cross-metal lattice trend
    backwards (as MACE-MP does for this chemistry) has no business overruling a
    measurement, and the campaign must remain able to falsify it.
    """

    hulls: dict[str, HullResult]
    weight: float = 0.15
    #: Hull distance beyond which a composition is treated as fully penalized,
    #: eV per formula unit.  Set from the entropy scale by default.
    saturation_eV: float = 0.10

    @classmethod
    def from_hulls(cls, *hulls: HullResult, weight: float = 0.15) -> "HullPrior":
        return cls(hulls={h.metal: h for h in hulls}, weight=weight)

    def stability_score(self, metal: str, vacancy_fraction: float) -> float:
        """Predicted stability in [0, 1]: 1 on the hull, decaying above it.

        Returns 0.5 -- explicitly uninformative -- for a metal with no computed
        hull, rather than a default that would silently favour or disfavour it.
        """
        h = self.hulls.get(metal)
        if h is None:
            return 0.5
        if not h.resolvable:
            # The hull's deepest feature is below the thermal scale: it carries no
            # usable information at synthesis temperature, so say so rather than
            # passing along noise.
            return 0.5

        ys = np.array([p.vacancy_fraction for p in h.points])
        es = np.array([p.e_above_hull_eV for p in h.points])
        order = np.argsort(ys)
        e_above = float(np.interp(vacancy_fraction, ys[order], es[order]))
        return float(math.exp(-e_above / max(1e-6, self.saturation_eV)))

    def is_informative(self, metal: str) -> bool:
        """Does this prior have anything to say about ``metal``?

        False when no hull was computed, and when the computed hull's deepest
        feature is below the thermal scale.  Callers must branch on this rather
        than reading :meth:`stability_score`'s 0.5 as a mid-range opinion.
        """
        h = self.hulls.get(metal)
        return h is not None and bool(h.resolvable)

    def penalty(self, metal: str, vacancy_fraction: float) -> float:
        """Additive penalty for a scalarized objective, >= 0.

        **Zero when the prior is uninformative**, not ``weight * (1 - 0.5)``.
        An earlier version returned the latter, which charged an uncomputed
        metal 0.075 while a computed-and-on-hull metal paid 0.000 -- so a metal
        nobody had run was silently ranked below one the force field liked.
        That converts absence of evidence into evidence of absence inside the
        scalarizer, where nothing reports it.  An uninformative prior must cost
        the same as no prior at all.
        """
        if not self.is_informative(metal):
            return 0.0
        return self.weight * (1.0 - self.stability_score(metal, vacancy_fraction))

    def disagreement(self, measured: "list[tuple[str, float, float]]") -> dict:
        """Where the prior and the measurements disagree.

        ``measured`` is ``(metal, vacancy_fraction, objective_value)`` triples from
        completed experiments.  A strong negative correlation between predicted
        stability and measured performance means the force field is wrong about
        this chemistry in a way worth reporting -- and worth lowering ``weight``
        over, or dropping the prior entirely.

        This is the method that makes the prior honest: it is the mechanism by
        which the campaign can tell you the computation was misleading.
        """
        rows = [(m, y, v) for m, y, v in measured
                if m in self.hulls and math.isfinite(v)]
        if len(rows) < 4:
            return {"n": len(rows), "correlation": None,
                    "verdict": "too few measurements to judge the prior"}

        pred = np.array([self.stability_score(m, y) for m, y, _ in rows])
        obs = np.array([v for _, _, v in rows])
        if pred.std() < 1e-9:
            return {"n": len(rows), "correlation": None,
                    "verdict": "prior is flat over the sampled region"}

        from scipy.stats import spearmanr

        rho = float(spearmanr(pred, obs).statistic)
        if rho < -0.3:
            verdict = ("prior anti-correlates with measurement: the force field is "
                       "misleading for this chemistry -- lower weight or drop it")
        elif rho > 0.3:
            verdict = "prior agrees with measurement"
        else:
            verdict = "prior carries little information about the measured objective"
        return {"n": len(rows), "correlation": rho, "verdict": verdict}
