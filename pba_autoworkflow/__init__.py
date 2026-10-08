# SPDX-License-Identifier: GPL-3.0-or-later
"""pba_autoworkflow -- closed-loop orchestration for Prussian-blue-analogue synthesis.

Layers, bottom up:

``schema``       typed data model: design space, recipes, traces, descriptors
``devices``      abstract instrument protocols + simulated implementations
``sim``          hidden chemistry and forward instrument models (simulation only)
``analysis``     pattern indexing, spectral deconvolution, QC, objectives
``optimize``     mixed-variable GP surrogate and batch multi-objective planners
``provenance``   append-only SQLite record with raw traces on disk
``scheduler``    station pool and bounded-concurrency batch execution
``workflow``     one experiment, deck to descriptors
``campaign``     the closed loop, stopping rules, resumption, reporting

Typical use::

    from pba_autoworkflow import CampaignConfig, ProvenanceStore, build_simulated_platform
    from pba_autoworkflow.campaign import run_campaign_sync

    platform, backend = build_simulated_platform(seed=7)
    store = ProvenanceStore("runs/demo")
    campaign = run_campaign_sync(platform, store,
                                 CampaignConfig(n_seed=12, batch_size=6,
                                                max_iterations=6))
    print(campaign.summary()["best"])

To move to real hardware, implement the protocols in :mod:`pba_autoworkflow.devices.base`
against your instruments and pass a :class:`~pba_autoworkflow.devices.base.Platform` built
from those drivers instead of the simulated one.  Nothing above the device layer
changes.
"""

from .schema import (
    DesignSpace,
    Experiment,
    ExperimentStatus,
    Objectives,
    SampleDescriptors,
    SynthesisParameters,
    default_design_space,
)
from .provenance import ProvenanceStore
from .devices.base import Platform
from .devices.simulated import build_simulated_platform
from .campaign import Campaign, CampaignConfig, run_campaign, run_campaign_sync
from .workflow import ExperimentWorkflow, WorkflowConfig
from .scheduler import BatchScheduler, StationPool, default_capacities

from ._version import __version__

__all__ = [
    "DesignSpace", "Experiment", "ExperimentStatus", "Objectives",
    "SampleDescriptors", "SynthesisParameters", "default_design_space",
    "ProvenanceStore", "Platform", "build_simulated_platform",
    "Campaign", "CampaignConfig", "run_campaign", "run_campaign_sync",
    "ExperimentWorkflow", "WorkflowConfig",
    "BatchScheduler", "StationPool", "default_capacities",
    "__version__",
]
