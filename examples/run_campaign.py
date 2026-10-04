# SPDX-License-Identifier: GPL-3.0-or-later
"""Run a closed-loop campaign from Python against the simulated deck.

    python examples/run_campaign.py [output_dir]

This is the same code path as ``pba-autoworkflow run``; use it when you want to
change the design space, the configuration or the platform in code.
"""

import asyncio
import sys

from pba_autoworkflow.campaign import Campaign, CampaignConfig
from pba_autoworkflow.devices.simulated import build_simulated_platform
from pba_autoworkflow.provenance import ProvenanceStore
from pba_autoworkflow.report import write_report
from pba_autoworkflow.schema import CategoricalSpec, default_design_space

out = sys.argv[1] if len(sys.argv) > 1 else "runs/example"

# 1. The platform: six instruments behind async protocols.  Swap any of them for
#    a real driver (see examples/custom_device.py); nothing else changes.
platform, _backend = build_simulated_platform(seed=0, time_scale=0.0)

# 2. The design space.  Specs are immutable, so build a modified copy:
#    here, four metals and a reactor limited to 80 degC.
base = default_design_space()
space = base.model_copy(update={
    "continuous": tuple(d.model_copy(update={"high": 80.0}) if d.name == "temperature_C" else d
                        for d in base.continuous),
    "categorical": (CategoricalSpec(name="metal", choices=("Mn", "Co", "Ni", "Cu"),
                                    description=base.categorical[0].description),),
})

# 3. The campaign.  The store is a DIRECTORY holding the SQLite database and the
#    raw traces; an existing campaign_id in it is resumed, never overwritten.
cfg = CampaignConfig(campaign_id="example", n_seed=8, batch_size=6,
                     max_iterations=3, replicate_fraction=0.15, seed=0)
store = ProvenanceStore(out)
campaign = Campaign(platform, store, cfg, space=space, progress=print)

records = asyncio.run(campaign.run())

print("\nhypervolume per iteration:", [round(r.hypervolume, 3) for r in records])
best = campaign.best_experiment()
if best is not None:
    print("best:", best.experiment_id, best.parameters.metal,
          best.descriptors.composition.formula)
paths = write_report(campaign, out + "/report")
print("report files:", sorted(paths))
store.close()
