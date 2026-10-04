# SPDX-License-Identifier: GPL-3.0-or-later
"""UMA is the default everywhere, and nothing silently substitutes for it."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from pba_autoworkflow.thermo import cli
from pba_autoworkflow.thermo.hull import water_reference_eV
from pba_autoworkflow.thermo.mlff import DEFAULT_MODEL, AnalyticModel, build_model


def test_default_backend_is_uma_in_every_entry_point():
    """cli, the parallel driver and the frozen scan must agree on the default.

    They did not: ``cli`` defaulted to UMA while ``run_hull.py`` and
    ``mlff_frozen_scan.py`` constructed MACE unconditionally, so the same
    nominal calculation used different physics depending on how it was invoked.
    """
    assert DEFAULT_MODEL == "uma"
    assert cli.DEFAULT_MODEL == "uma"
    cli_src = pathlib.Path(cli.__file__).read_text()
    assert 'default="uma"' in cli_src

    hull_src = pathlib.Path("scripts/run_hull.py").read_text()
    assert 'ap.add_argument("--model", default="uma"' in hull_src
    assert "MACEModel" not in hull_src, "the driver must go through build_model"

    scan_src = pathlib.Path("scripts/mlff_frozen_scan.py").read_text()
    assert "MACEModel" not in scan_src


def test_unknown_model_name_raises_instead_of_becoming_mace():
    """The old fallback made a typo run the one backend with the wrong sign."""
    for bad in ("umaa", "MACE-MP", "", "petmadd"):
        with pytest.raises(ValueError, match="unknown model"):
            build_model(bad)
    with pytest.raises(ValueError, match="unknown model"):
        cli._model("umaa", "small")


def test_analytic_backend_still_reachable_by_name():
    """Tests need a backend with no downloads; it must be explicit, not default."""
    assert isinstance(build_model("analytic"), AnalyticModel)


def test_water_reference_refuses_a_missing_model():
    """A cross-model water reference does not cancel in the grand potential.

    ``water_reference_eV`` used to accept ``model=None``, build a MACE water
    molecule, and cache it under a ``mace-mp-*`` key -- so a UMA or PET-MAD hull
    could be referenced against MACE water with nothing in the result saying so.
    The error that reference introduces is larger than the features the hull is
    trying to resolve.
    """
    with pytest.raises(ValueError, match="same model"):
        water_reference_eV(None)


def test_water_reference_is_cached_under_the_models_own_name():
    m = build_model("analytic")
    e1 = water_reference_eV(m)
    e2 = water_reference_eV(m)
    assert e1 == e2                      # cached, not recomputed
    assert np.isfinite(e1)


def test_run_hull_records_the_backend_that_actually_ran():
    """The output label was hardcoded to mace-mp-*, which would mislabel UMA runs.

    These energies are the input to every hull and every cross-model
    comparison, so a wrong ``model`` field silently poisons all of it.
    """
    # Strip comments: the fix carries a comment quoting the old expression, and
    # a naive substring search would match the explanation of the bug.
    code = "\n".join(line.split("#", 1)[0]
                     for line in pathlib.Path("scripts/run_hull.py").read_text().splitlines())
    assert 'f"mace-mp-{args.model}"' not in code
    assert "else args.model" in code
    assert 'args.model == "mace"' in code


def test_uma_gate_failure_names_the_fix(monkeypatch):
    """A licence-gated download must not surface as a raw GatedRepoError.

    UMA is the project default, so this is the first thing a fresh checkout
    hits, and huggingface_hub's own error says nothing about accepting a licence
    or setting a token.
    """
    import pba_autoworkflow.thermo.mlff as mlff

    class GatedRepoError(Exception):
        pass

    class FakePretrained:
        @staticmethod
        def get_predict_unit(version, device="cpu"):
            raise GatedRepoError("401 Client Error. Access to model facebook/UMA "
                                 "is restricted. You must have access to it.")

    import types
    fake = types.ModuleType("fairchem.core")
    fake.pretrained_mlip = FakePretrained
    fake.FAIRChemCalculator = lambda *a, **k: None
    monkeypatch.setitem(__import__("sys").modules, "fairchem.core", fake)

    with pytest.raises(RuntimeError) as exc:
        mlff.UMAModel()
    msg = str(exc.value)
    assert "huggingface.co/facebook/UMA" in msg      # where to accept the licence
    assert "HF_TOKEN" in msg                          # what to set
    assert "--model petmad" in msg                    # an ungated way to proceed
    assert "sign" in msg                              # why mace is not the answer


def test_dft_validation_numbers_are_recorded_with_the_default():
    """The reason UMA is the default must travel with the code, not just chat.

    The claim is specifically about SIGN agreement on a matched protocol; the
    magnitude is off by 1.9x and is not being claimed.
    """
    src = pathlib.Path(build_model("analytic").__class__.__module__
                       .replace(".", "/") + ".py").read_text()
    assert "+271" in src and "+509" in src and "-996" in src
    assert "DFT.md" in src
