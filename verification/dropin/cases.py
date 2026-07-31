# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""A study is the solver-specific half of the suite: which tutorials, and how run.

The engine (staging, execution, comparison, report) is solver-agnostic. What
changes between ``incompressibleFluid`` and ``incompressibleVoF`` is the tutorial
group, the native solvers being replaced, and the fields diffed — all of which
live in a study's ``config.yaml`` + ``discover.py``, loaded here so both the
Snakefile and the per-case workers agree on the same case list.

``discover()`` is intentionally cheap (it only reads dictionaries), so re-running
it per worker instead of threading a manifest file through the DAG is the simpler
design and keeps the Snakefile's parse-time case list and the workers' case list
from ever drifting apart.
"""

from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from verification.dropin.foamdict import entry, read

__all__ = ["Case", "Study", "load_study", "subdomains"]


@dataclass
class Case:
    """One discovered tutorial, its static classification, and how to run it."""

    #: Filesystem-safe identifier (the DAG wildcard): ``simpleFoam__pitzDaily``.
    id: str
    #: Human-readable name, path-relative to the tutorial group.
    name: str
    path: Path
    native_solver: str
    #: The neofoam CLI command that replaces ``native_solver``. Empty when the
    #: candidates come from the study config's ``apps:`` rather than discovery.
    app: str
    #: Fields diffed after the run. Solver-specific.
    fields: tuple[str, ...]
    #: Static classification, kept optional so an enumerate-only ``discover`` (case
    #: selection lives in ``config.yaml``) need not compute them; a study that still
    #: tiers its tree fills them in.
    turbulence: str = "laminar"
    #: "A" is predicted-runnable; every other tier is a documented blocker.
    tier: str = ""
    reason: str = ""
    parallel: bool = False
    #: ``numberOfSubdomains`` when parallel, else 1 — the DAG's per-case thread count.
    subdomains: int = 1

    @property
    def native_label(self) -> str:
        """Names the native side by its solver, e.g. ``simpleFoam``."""
        return self.native_solver

    @property
    def neo_label(self) -> str:
        """Names the neofoam side by its solver token, e.g. ``incompressiblefluid``.

        The last word of the CLI command (``neofoam solver incompressiblefluid``) —
        so a case's two runs live in ``<native_solver>/`` and ``<neo_solver>/``
        rather than the anonymous ``native/`` and ``neo/``.
        """
        return self.app.split()[-1]


def case_id(name: str) -> str:
    """Turn a tutorial's path-name into a flat, wildcard-safe id."""
    return name.replace("/", "__")


def subdomains(case: Path) -> int:
    """``numberOfSubdomains`` from ``system/decomposeParDict*``, or 1 when serial."""
    for dict_path in sorted(case.glob("system/decomposeParDict*")):
        value = entry(read(dict_path), "numberOfSubdomains")
        if value.isdigit():
            return int(value)
    return 1


@dataclass
class Study:
    """A loaded study: its title, its cases, and the config that produced them."""

    title: str
    config_path: Path
    cases: list[Case]
    #: Raw config mapping, for study-specific flags (e.g. the VoF fallback pass).
    config: dict[str, Any]
    #: The neofoam commands compared against the native reference — one or more
    #: backends (``incompressiblefluid``, ``incompressiblefluidneon``). Native is
    #: run once per case as the shared reference; each of these is a candidate
    #: diffed against it.
    apps: tuple[str, ...] = ()
    #: Case name → ``{"reason": ..., "patch": {rel/dict/path: {dotted.key: value}}}``
    #: from the config's ``simplify:`` channel. Applied to *every* side at staging
    #: time, so both the native reference and the candidates solve the same case.
    simplifications: dict[str, dict[str, Any]] = field(default_factory=dict)

    def simplify(self, case_name: str) -> dict[str, Any]:
        """The simplification staged into both sides of *case_name*, or ``{}``."""
        return self.simplifications.get(case_name, {})

    @property
    def candidates(self) -> dict[str, str]:
        """Map a candidate's on-disk label to its full neofoam command."""
        return {app.split()[-1]: app for app in self.apps}

    @property
    def candidate_labels(self) -> list[str]:
        """The label of each candidate backend, in config order."""
        return [app.split()[-1] for app in self.apps]

    def by_id(self, case_id: str) -> Case:
        for case in self.cases:
            if case.id == case_id:
                return case
        raise KeyError(f"no case with id {case_id!r} in study {self.title!r}")


def _load_discover(path: Path) -> ModuleType:
    """Import a study's ``discover.py`` by file path."""
    spec = importlib.util.spec_from_file_location("_verify_discover", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load discover module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "discover", None)):
        raise AttributeError(f"{path} has no callable discover()")
    return module


def _drop_excluded(cases: list[Case], exclude: list[dict[str, str]] | None) -> list[Case]:
    """Drop config-excluded cases; every entry needs a case name and a reason."""
    if not exclude:
        return cases
    names: set[str] = set()
    for item in exclude:
        name = item.get("case") if isinstance(item, dict) else None
        if not name or not item.get("reason"):
            raise ValueError(f"exclude: each entry needs a `case` name and a `reason`: {item!r}")
        names.add(name)
    stale = names - {case.name for case in cases}
    if stale:
        raise ValueError(
            "exclude: names no selected case (a typo, or the case already left "
            f"the selection): {', '.join(sorted(stale))}"
        )
    return [case for case in cases if case.name not in names]


def _simplifications(
    cases: list[Case], simplify: list[dict[str, Any]] | None
) -> dict[str, dict[str, Any]]:
    """Index the config's ``simplify:`` by case name; each entry needs case/reason/patch."""
    if not simplify:
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for item in simplify:
        name = item.get("case") if isinstance(item, dict) else None
        if not name or not item.get("reason") or not item.get("patch"):
            raise ValueError(
                "simplify: each entry needs a `case` name, a `reason`, and a "
                f"non-empty `patch`: {item!r}"
            )
        entries[name] = {"reason": item["reason"], "patch": item["patch"]}
    stale = set(entries) - {case.name for case in cases}
    if stale:
        raise ValueError(
            "simplify: names no selected case (a typo, or the case already left "
            f"the selection): {', '.join(sorted(stale))}"
        )
    return entries


def load_study(config_path: Path) -> Study:
    """Load a study from its ``config.yaml``.

    The ``discover`` path in the config resolves relative to the config file's own
    directory, so a study is self-contained and can be run from anywhere.

    Two distinct case selectors, deliberately not merged: ``cases:`` is a selection
    handed *to* ``discover()`` (it resolves each named tutorial and never walks the
    tree), while ``only:`` post-filters a discover that *does* walk — the shape a
    study needs when it tiers its tutorial group. Both are applied here rather than
    in the Snakefile so the DAG, every worker, and the report's case table are
    derived from one list.

    ``exclude:`` is the third channel: it drops a selected case that can never run
    as a drop-in, each entry naming the case (same naming as ``cases:``/``only:``)
    with the reason it is out of scope::

        exclude:
          - case: interFoam/RAS/motorBike
            reason: "snappyHexMesh is not reproducible, so runs are not comparable"

    Applied last, so an excluded case is never staged and never reported. An entry
    naming no selected case is an error — a stale exclusion is either a typo (the
    case still runs) or dead documentation (the case is gone), and both should
    surface rather than sit silently in the config.

    ``simplify:`` is the fourth channel: it keeps a case in the sweep but substitutes
    the settings the neofoam solver does not support, each entry naming the case, why
    it is simplified, and the dictionary patch (dotted keys address sub-dicts)::

        simplify:
          - case: simpleFoam/pitzDaily
            reason: "SIMPLEC is not ported: run plain SIMPLE on both sides"
            patch:
              system/fvSolution:
                SIMPLE.consistent: "no"

    Unlike ``neo_patch`` (candidate-only), the patch is staged into *every* side, so
    the two runs still solve the same problem and a match stays meaningful. Same
    stale-entry rule as ``exclude:``: naming no selected case is an error.
    """
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text()) or {}
    discover_path = (config_path.parent / config["discover"]).resolve()
    module = _load_discover(discover_path)
    # A discover() that takes an argument reads its case selection from the config's
    # `cases:` list; a legacy no-arg discover() walks the tutorial tree itself. Both
    # signatures are supported so a study need only opt in to config-driven selection.
    if inspect.signature(module.discover).parameters:
        cases = module.discover(config.get("cases"))
    else:
        cases = module.discover()
    only = config.get("only")
    if only:
        cases = [case for case in cases if case.name in set(only)]
    cases = _drop_excluded(cases, config.get("exclude"))
    # The candidate backends: explicit `apps:` in the config, else the single app
    # each case was discovered with (back-compat — one backend, as before).
    apps = config.get("apps") or list(dict.fromkeys(case.app for case in cases))
    return Study(
        title=config.get("title", config_path.parent.name),
        config_path=config_path,
        cases=cases,
        config=config,
        apps=tuple(apps),
        simplifications=_simplifications(cases, config.get("simplify")),
    )
