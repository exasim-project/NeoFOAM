# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared case-discovery helpers for the turbulence and viscosity test suites.

Both suites ship self-contained OpenFOAM cases under ``<area>/cases/<name>/``,
each holding a real ``constant/<dict>`` plus an ``expected.yaml`` manifest, and
both build models exactly the way the incompressibleFluid solver does. The
discovery / selection / assertion machinery is identical between them; only the
domain hooks differ (config class, selector, native-wrap, fallback type). This
module factors out the common parts so each area's ``conftest.py`` is just a thin
binding — see ``test/turbulence/conftest.py`` and ``test/viscosity/conftest.py``.

Cases are *discovered* by globbing ``cases/*/expected.yaml``, so adding a case
directory extends coverage with zero test-module edits, and expected values come
from those manifests, never from literals baked into test bodies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from neofoam.framework.model import ModelSpec


@dataclass(frozen=True)
class Case:
    """A discovered case and its expectation manifest."""

    name: str  # directory name → parametrize id
    path: Path  # case dir (holds constant/<dict> + expected.yaml)
    config: dict[str, Any]  # manifest["config"]    — expected parsed dict
    selection: dict[str, Any]  # manifest["selection"] — {model_name, resolves_to}
    model: Optional[dict[str, Any]]  # manifest.get("model") — per-model runtime values


def discover_cases(cases_dir: Path) -> list[Case]:
    """Glob ``cases/*/expected.yaml`` under *cases_dir* and load each into a :class:`Case`."""
    cases: list[Case] = []
    for manifest in sorted(cases_dir.glob("*/expected.yaml")):
        data = yaml.safe_load(manifest.read_text())
        cases.append(
            Case(
                name=manifest.parent.name,
                path=manifest.parent,
                config=data["config"],
                selection=data["selection"],
                model=data.get("model"),
            )
        )
    return cases


def case_for(cases: list[Case], model_name: str) -> Case:
    """Return the native case in *cases* whose ``selection.model_name`` is *model_name*.

    Points a registered native model at the case the solver would feed it.
    """
    for case in cases:
        if (
            case.selection["resolves_to"] == "native"
            and case.selection["model_name"] == model_name
        ):
            return case
    raise LookupError(f"no native case for model {model_name!r}")


def make_build_as_solver(
    config_cls: Any,
    select_fn: Callable[[Any], Any],
    wrap_native: Optional[Callable[[Any], Any]] = None,
) -> Callable[[Case], Any]:
    """Build a ``build_as_solver(case)`` bound to a domain's hooks.

    Loads *config_cls* from the case dir, runs *select_fn*, and for a native
    :class:`ModelSpec` instantiates it at the case dir and passes the runtime
    through *wrap_native* when given (the viscosity suite uses the runtime as-is).
    Non-native selections are returned unbuilt. (The turbulence suite no longer
    uses this helper — it builds handles via the merged-family selector directly.)
    """

    def build_as_solver(case: Case) -> Any:
        cfg = config_cls.load(case_dir=case.path)
        selected = select_fn(cfg)
        if isinstance(selected, ModelSpec):
            runtime = selected.instantiate(case.path)
            return wrap_native(runtime) if wrap_native else runtime
        return selected

    return build_as_solver


def make_assert_selection(fallback_cls: type) -> Callable[[Any, Case], None]:
    """Build an ``assert_selection(selected, case)`` for a domain's *fallback_cls*."""

    def assert_selection(selected: Any, case: Case) -> None:
        resolves_to = case.selection["resolves_to"]
        if resolves_to == "native":
            assert isinstance(selected, ModelSpec)
            assert selected.name == case.selection["model_name"]
        elif resolves_to == "fallback":
            assert isinstance(selected, fallback_cls)
        else:
            raise AssertionError(f"unknown resolves_to: {resolves_to!r}")

    return assert_selection
