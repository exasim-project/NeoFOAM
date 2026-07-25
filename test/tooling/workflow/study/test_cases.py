# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Loading a study wires config.yaml to its discover.py and tier titles.

This is the seam the Snakefile and every per-case worker share: both call
``load_study`` and must see the same case list. The discover module is loaded by
file path (a study is self-contained, not an installed package), so the test
writes a throwaway ``discover.py`` and config beside each other in ``tmp_path``
rather than depending on a sourced OpenFOAM.
"""

from pathlib import Path

import pytest

from neofoam.tooling.workflow.study.cases import Case, case_id, load_study

_DISCOVER_PY = """
from pathlib import Path
from neofoam.tooling.workflow.study.cases import Case, case_id

TIER_TITLES = {"A": "runnable"}

def discover():
    name = "simpleFoam/pitzDaily"
    return [Case(
        id=case_id(name), name=name, path=Path("/tut") / name,
        native_solver="simpleFoam", app="neofoam solver incompressiblefluid",
        fields=("U", "p"), turbulence="kEpsilon", tier="A",
    )]
"""


def test_case_id_is_wildcard_safe() -> None:
    """Slashes become the flat id Snakemake uses as a wildcard."""
    assert case_id("simpleFoam/pitzDaily") == "simpleFoam__pitzDaily"


def test_side_labels_name_each_run_by_its_solver() -> None:
    """A case's two runs are stored under solver-named dirs, not native/neo."""
    case = Case(
        id="simpleFoam__pitzDaily",
        name="simpleFoam/pitzDaily",
        path=Path("/tut/simpleFoam/pitzDaily"),
        native_solver="simpleFoam",
        app="neofoam solver incompressiblefluid",
        fields=("U",),
        turbulence="kEpsilon",
        tier="A",
    )
    assert case.native_label == "simpleFoam"
    assert case.neo_label == "incompressiblefluid"


def test_load_study_resolves_discover_relative_to_config(tmp_path: Path) -> None:
    (tmp_path / "discover.py").write_text(_DISCOVER_PY)
    (tmp_path / "config.yaml").write_text("title: demo\ndiscover: discover.py\n")

    study = load_study(tmp_path / "config.yaml")

    assert study.title == "demo"
    assert [c.id for c in study.cases] == ["simpleFoam__pitzDaily"]
    # tier titles fall back to the discover module's TIER_TITLES.
    assert study.tier_titles == {"A": "runnable"}
    assert study.by_id("simpleFoam__pitzDaily").native_solver == "simpleFoam"


def test_candidates_default_to_the_single_discovered_app(tmp_path: Path) -> None:
    """Without `apps:`, the one app each case carries is the sole candidate."""
    (tmp_path / "discover.py").write_text(_DISCOVER_PY)
    (tmp_path / "config.yaml").write_text("title: demo\ndiscover: discover.py\n")

    study = load_study(tmp_path / "config.yaml")

    assert study.candidate_labels == ["incompressiblefluid"]
    assert study.candidates == {"incompressiblefluid": "neofoam solver incompressiblefluid"}


def test_apps_config_declares_multiple_candidate_backends(tmp_path: Path) -> None:
    """`apps:` lists the backends diffed against the native reference, by label."""
    (tmp_path / "discover.py").write_text(_DISCOVER_PY)
    (tmp_path / "config.yaml").write_text(
        "title: demo\ndiscover: discover.py\n"
        "apps:\n"
        '  - "neofoam solver incompressiblefluid"\n'
        '  - "neofoam solver incompressiblefluidneon"\n'
    )

    study = load_study(tmp_path / "config.yaml")

    assert study.candidate_labels == ["incompressiblefluid", "incompressiblefluidneon"]
    assert study.candidates["incompressiblefluidneon"] == "neofoam solver incompressiblefluidneon"


def test_by_id_raises_on_unknown_case(tmp_path: Path) -> None:
    (tmp_path / "discover.py").write_text(_DISCOVER_PY)
    (tmp_path / "config.yaml").write_text("title: demo\ndiscover: discover.py\n")

    study = load_study(tmp_path / "config.yaml")

    with pytest.raises(KeyError):
        study.by_id("nope")


def test_config_only_key_is_available_for_subsetting(tmp_path: Path) -> None:
    """The slice-1 gate restricts the sweep via `only:`; it must reach the config."""
    (tmp_path / "discover.py").write_text(_DISCOVER_PY)
    (tmp_path / "config.yaml").write_text(
        "title: demo\ndiscover: discover.py\nonly:\n  - simpleFoam/pitzDaily\n"
    )

    study = load_study(tmp_path / "config.yaml")

    assert study.config["only"] == ["simpleFoam/pitzDaily"]
