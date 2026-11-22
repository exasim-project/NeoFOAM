# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for turbulence model selection under the merged family + ``fallback`` flag.

One family serves both solvers; the ``fallback`` flag picks the backend:

* ``fallback=False`` (incompressibleFluidNeoN) → a native
  :class:`~neofoam.turbulence.native.NeoNHandle` for a registered model.
* ``fallback=True`` (incompressibleFluid) → a
  :class:`~neofoam.turbulence.fallback.FallbackHandle`, built either from the
  model's co-located ``fallback=True`` op (registered) or straight from
  OpenFOAM's own run-time selection table (a name with no registered spec).

Parametrized over the discovered cases, whose ``expected.yaml`` declares how each
resolves: ``native`` is a registered dual model (both backends build);
``unregistered`` (Smagorinsky) has no spec, so only the pybFoam backend builds it;
``wrong_family`` (``LES { LESModel kEpsilon; }``) names a model registered in the
*other* family, which must be refused on both backends rather than silently built
as the RAS closure. This also exercises the pybFoam-bound ``select_from_case``
entry point.

The pybFoam turbulence factory is injected as a ``MagicMock`` throughout: what is
under test is which model selection picks, not what OpenFOAM then does with it —
so no mesh is needed.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from neofoam.tooling.casebuild import empty
from neofoam.turbulence import momentumTransportModel
from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.fallback import FallbackHandle
from neofoam.turbulence.native import NeoNHandle
from neofoam.turbulence.selection import (
    model_name,
    select_from_case,
    select_turbulence_model,
)
from turbulence._parity_case import turbulence_properties
from turbulence.conftest import CASES, Case

#: One case per resolution kind, taken from the discovered manifests.
UNREGISTERED = next(c for c in CASES if c.selection["resolves_to"] == "unregistered")
WRONG_FAMILY = next(c for c in CASES if c.selection["resolves_to"] == "wrong_family")

#: The turbulence dictionaries the parity tests run, keyed by the fallback ``correct``
#: op the model they select declares. The ``*Coeffs`` dicts select the same closure
#: through a full coefficient override, so they must resolve to the same op.
FALLBACK_OP_PER_DICT = {
    "laminar": "laminarCorrect",
    "kEpsilon": "kEpsilonCorrect",
    "kEpsilonCoeffs": "kEpsilonCorrect",
    "SpalartAllmaras": "spalartAllmarasCorrect",
    "SpalartAllmarasCoeffs": "spalartAllmarasCorrect",
    "kOmegaSST": "kOmegaSSTCorrect",
    "kOmegaSSTCoeffs": "kOmegaSSTCorrect",
    "realizableKE": "realizableKECorrect",
}

#: A *fallback-only* model: it registers a name (so selection and the MCP see it) but
#: declares **only** the pybFoam backend — no ``@build`` and no native ``@operation``,
#: just the one ``fallback=True`` correct pinned in :data:`FALLBACK_OP_PER_DICT`. It is
#: usable only when a solver selects ``fallback=True``.
FALLBACK_ONLY = "realizableKE"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_native_branch(case: Case) -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=case.path)
    if case.selection["resolves_to"] == "native":
        selected = select_turbulence_model(cfg, fallback=False, case_dir=case.path)
        assert isinstance(selected, NeoNHandle)
    else:  # unregistered → no native closure; wrong_family → the family guard
        with pytest.raises(ValueError):
            select_turbulence_model(cfg, fallback=False, case_dir=case.path)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_fallback_branch(case: Case) -> None:
    """Both fallback entry points agree: the config-taking one and ``select_from_case``."""
    cfg = TurbulencePropertiesConfig.load(case_dir=case.path)
    if case.selection["resolves_to"] == "wrong_family":
        with pytest.raises(ValueError):
            select_turbulence_model(cfg, fallback=True, case_dir=case.path, of_factory=MagicMock())
        with pytest.raises(ValueError):
            select_from_case(case.path, fallback=True, of_factory=MagicMock())
    else:  # a registered dual model, or an unregistered name OpenFOAM can build
        selected = select_turbulence_model(
            cfg, fallback=True, case_dir=case.path, of_factory=MagicMock()
        )
        assert isinstance(selected, FallbackHandle)
        from_case = select_from_case(case.path, fallback=True, of_factory=MagicMock())
        assert isinstance(from_case, FallbackHandle)


@pytest.mark.parametrize(("dict_name", "op_name"), sorted(FALLBACK_OP_PER_DICT.items()))
def test_registered_model_schedules_its_own_fallback_correct_op(
    dict_name: str, op_name: str, tmp_path: Path
) -> None:
    """Each registered model wires its own co-located ``fallback=True`` correct op.

    That the scheduled op then actually advances ``nut`` through real pybFoam is
    proven, once per wiring shape, by
    ``test_neon_turbulence_parity.test_fallback_nut_matches_pybfoam``; here the same
    claim is made for *every* dictionary the parity run uses, without a solver run.
    """
    case = (empty() | turbulence_properties(dict_name)).build_at(tmp_path).path
    cfg = TurbulencePropertiesConfig.load(case_dir=str(case))

    handle = select_turbulence_model(cfg, fallback=True, case_dir=str(case), of_factory=MagicMock())

    assert [op.metadata.op_name for op in handle.operations] == [op_name]


def test_fallback_only_model_is_registered_but_has_no_native_closure(tmp_path: Path) -> None:
    """A fallback-only model is selectable on the pybFoam path only; the native one raises."""
    assert FALLBACK_ONLY in momentumTransportModel.registered_names()

    case = (empty() | turbulence_properties(FALLBACK_ONLY)).build_at(tmp_path).path
    cfg = TurbulencePropertiesConfig.load(case_dir=case)

    with pytest.raises(ValueError, match="no native NeoN closure"):
        select_turbulence_model(cfg, fallback=False, case_dir=case)


def test_unregistered_model_resolves_to_the_openfoam_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A name with no spec is announced, gets the generic op, and builds via the factory."""
    cfg = TurbulencePropertiesConfig.load(case_dir=UNREGISTERED.path)
    factory = MagicMock()

    handle = select_turbulence_model(
        cfg, fallback=True, case_dir=UNREGISTERED.path, of_factory=factory
    )

    announcement = capsys.readouterr().out
    assert UNREGISTERED.selection["model_name"] in announcement
    assert "fallback" in announcement
    assert [op.metadata.op_name for op in handle.operations] == ["of_correct_turbulence"]

    handle.build()
    factory.assert_called_once()


def test_unregistered_model_has_no_native_closure() -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=UNREGISTERED.path)

    with pytest.raises(ValueError, match="no turbulence model registered"):
        select_turbulence_model(cfg, fallback=False, case_dir=UNREGISTERED.path)


def test_wrong_family_selection_names_the_registered_family() -> None:
    cfg = TurbulencePropertiesConfig.load(case_dir=WRONG_FAMILY.path)

    with pytest.raises(ValueError, match="registered as a RAS closure"):
        select_turbulence_model(
            cfg, fallback=True, case_dir=WRONG_FAMILY.path, of_factory=MagicMock()
        )


def test_undeterminable_model_raises() -> None:
    # Not expressible on disk: TurbulencePropertiesConfig pins simulationType to the
    # closed set, so an unknown one only reaches the selector duck-typed. The positive
    # paths (laminar / RASModel / LESModel resolution) are covered, manifest driven, by
    # test_config.py; here we only pin the unknown case.
    cfg = SimpleNamespace(simulationType="bogus")

    assert model_name(cfg) is None
    with pytest.raises(ValueError, match="cannot resolve the turbulence model"):
        select_turbulence_model(cfg, fallback=True, of_factory=MagicMock())
