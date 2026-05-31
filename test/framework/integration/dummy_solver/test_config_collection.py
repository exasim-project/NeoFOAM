# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Collect and save all of a solver's configs after LOAD/RESOLVE.

Drives two framework additions:

- ``LoadResult.config_classes`` — the declared config *schema* set across
  all models (including classes registered but not loaded), so a case can
  be scaffolded from schemas alone.
- ``neofoam.io.save_configs`` — write a set of config instances back to a
  case directory via their registered IO strategies (round-trips).

Pure-Python: the dummy solver and its YAML-bound configs need no pybFoam.
"""

from pathlib import Path
from typing import Any

from neofoam.framework.initialization import LoadResult
from neofoam.framework.model import Model
from neofoam.foam import fvSchemes, fvSolution
from neofoam.io import IOStrategy, YAML, collect_config_classes, save_configs

from .dummy_init import (
    MeshConfig,
    SolverConfig,
    create_init,
)

CONFIGS_DIR = Path(__file__).parent / "configs"

_CORE_CLASS_NAMES = {"SolverConfig", "MeshConfig", "CoreModel2", "DummyAlgorithm"}


def test_load_result_collects_all_core_config_instances() -> None:
    lr = create_init().run_load()
    names = {type(c).__name__ for c in lr.configs}
    assert _CORE_CLASS_NAMES <= names


def test_load_result_collects_optional_model_configs() -> None:
    lr = create_init().run_load()
    names = {type(c).__name__ for c in lr.configs}
    # Optional models (model1-4) load Model1Config/Model2Config/etc.
    assert any(n.startswith("Model") and n.endswith("Config") for n in names)


def test_config_classes_enumerates_declared_schema() -> None:
    lr = create_init().run_load()
    classes = lr.config_classes
    assert all(isinstance(c, type) for c in classes)
    assert len(classes) == len(set(classes))  # deduped
    by_name = {c.__name__ for c in classes}
    assert _CORE_CLASS_NAMES <= by_name


def test_config_classes_includes_declared_but_unloaded_classes() -> None:
    """A model whose @load short-circuits instantiate still contributes the
    config classes it registered via .config()."""
    probe = Model("LocalProbe")
    ProbeSlice = probe.config(fvSchemes)  # declared, never loaded

    @probe.load
    def _load(case_dir: Path, instance_id: str) -> SolverConfig:
        return SolverConfig.load(case_dir=case_dir, validate=False)

    rt = probe.instantiate(case_dir=CONFIGS_DIR, instance_id="LocalProbe")
    lr = LoadResult(core_models=[], optional_models=[rt])
    classes = lr.config_classes
    assert ProbeSlice in classes  # declared but not loaded
    assert SolverConfig in classes  # loaded via @load


def test_save_configs_round_trips(tmp_path: Path) -> None:
    lr = create_init().run_load()
    paths = save_configs(lr.configs, case_dir=tmp_path)
    assert paths and all(p.exists() for p in paths)
    for cfg in lr.configs:
        reloaded = type(cfg).load(case_dir=tmp_path, validate=False)
        assert reloaded.model_dump() == cfg.model_dump()


def test_save_configs_validates_clean(tmp_path: Path) -> None:
    lr = create_init().run_load()
    save_configs(lr.configs, case_dir=tmp_path)
    reloaded = [type(c).load(case_dir=tmp_path) for c in lr.configs]
    result = LoadResult(core_models=reloaded, optional_models=[])
    assert result.validate() == []


def test_scaffold_from_classes_only(tmp_path: Path) -> None:
    """Build configs from field values alone (no source case), save, reload."""
    solver_cfg = SolverConfig(
        param1=1e-5, param2=2.0, dt=0.01, endTime=10.0, parameters={}
    )
    mesh_cfg = MeshConfig(nPoints=100)

    save_configs([solver_cfg, mesh_cfg], case_dir=tmp_path)

    assert SolverConfig.load(case_dir=tmp_path).model_dump() == solver_cfg.model_dump()
    assert MeshConfig.load(case_dir=tmp_path).model_dump() == mesh_cfg.model_dump()


def test_per_spec_slice_is_collectible_and_savable(tmp_path: Path) -> None:
    """Per-spec fvSchemes/fvSolution YAML slices collect and round-trip."""
    m = Model("SliceProbe")
    Schemes = m.config(fvSchemes)
    Solution = m.config(fvSolution)
    IOStrategy(YAML("dummy_fvSchemes.yaml"))(Schemes)
    IOStrategy(YAML("dummy_fvSolution.yaml"))(Solution)
    Schemes.add(div="div(phi,U)", grad="grad(U)")
    Solution.add("U")

    rt = m.instantiate(case_dir=CONFIGS_DIR, instance_id="SliceProbe")
    cfgs = rt.configs
    assert {type(c).__name__ for c in cfgs} == {
        "SliceProbe_fvSchemes",
        "SliceProbe_fvSolution",
    }

    save_configs(cfgs, case_dir=tmp_path)
    for cfg in cfgs:
        reloaded = type(cfg).load(case_dir=tmp_path, validate=False)
        assert reloaded.model_dump() == cfg.model_dump()


def test_collect_config_classes_mixes_sources_and_dedups() -> None:
    """The collector accepts bare classes, specs, and instances, deduped."""
    probe = Model("Probe")
    ProbeSlice = probe.config(fvSchemes)

    classes = collect_config_classes(
        [
            SolverConfig,  # a bare class
            MeshConfig(nPoints=1),  # an instance
            probe,  # a spec carrying _config_classes
            SolverConfig,  # duplicate -> collapsed
        ]
    )

    assert classes.count(SolverConfig) == 1
    assert SolverConfig in classes
    assert MeshConfig in classes
    assert ProbeSlice in classes


def test_save_configs_skips_unbound_config(tmp_path: Path, recwarn: Any) -> None:
    """A config class with no IO strategy is skipped (warned), not raised."""
    from neofoam.io import BaseConfig

    class Unbound(BaseConfig):
        value: float = 1.0

    written = save_configs([Unbound()], case_dir=tmp_path)
    assert written == []
    assert any("Unbound" in str(w.message) for w in recwarn.list)
