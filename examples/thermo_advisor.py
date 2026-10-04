# SPDX-License-Identifier: GPL-3.0-or-later
"""Attach the force-field advisor to a campaign.

    python examples/thermo_advisor.py [output_dir]

The advisor is read-only: it never changes what the planner proposes.  After
each batch it compares the force-field hull with what was measured and records
a verdict in the iteration record.  ``"analytic"`` is a seconds-fast stand-in
model; use ``"uma"`` (needs the ``uma`` extra and an HF token) for real work.
"""

import asyncio
import sys

from pba_autoworkflow.campaign import Campaign, CampaignConfig
from pba_autoworkflow.devices.simulated import build_simulated_platform
from pba_autoworkflow.provenance import ProvenanceStore
from pba_autoworkflow.thermo.advisor import ThermoAdvisor
from pba_autoworkflow.thermo.hull import compute_hull
from pba_autoworkflow.thermo.mlff import build_model
from pba_autoworkflow.thermo.prior import HullPrior

out = sys.argv[1] if len(sys.argv) > 1 else "runs/thermo-demo"

model = build_model("analytic")
hulls = [compute_hull(m, model, vacancy_values=(0.0, 0.25, 0.5)) for m in ("Mn", "Ni")]
advisor = ThermoAdvisor(HullPrior.from_hulls(*hulls))

platform, _ = build_simulated_platform(seed=0, time_scale=0.0)
store = ProvenanceStore(out)
cfg = CampaignConfig(campaign_id="thermo-demo", n_seed=8, batch_size=6, max_iterations=2)
campaign = Campaign(platform, store, cfg, thermo_advisor=advisor)

for rec in asyncio.run(campaign.run()):
    print(rec.iteration, rec.thermo_advice["stability_verdict"], rec.thermo_advice["notes"])
store.close()
