# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the plugin families shared by every linear-solver backend."""

from typing import Any, ClassVar, Literal

import pytest

from linear_solvers._helpers import (
    _CASE,
    _CgOnlyPreconditioner,
    _DummyPreconditioner,
    _entry,
    _Fgmres,
    _OtherBackend,
    _OtherPreconditioner,
    _registered,
    _registered_preconditioner,
    _slice,
)
from neofoam.core.plugin_system import PluginSystem
from neofoam.foam import (
    GinkgoPreconditioner,
    GinkgoSolver,
    LinearSolver,
    Preconditioner,
    advanced,
)


def test_solver_schema_preconditioner_is_a_type_discriminated_union() -> None:
    field = _entry().model_json_schema()["$defs"]["Cg"]["properties"]["preconditioner"]
    arms = [arm["$ref"].rsplit("/", 1)[1] for arm in field["oneOf"]]
    assert (field["discriminator"]["propertyName"], arms, field["default"]) == (
        "type",
        ["NoPreconditioner", "Jacobi", "Ic", "Ilu", "Multigrid"],
        {"type": "none"},
    )


def test_late_incompatible_preconditioner_is_narrowed_per_solver() -> None:
    with _registered_preconditioner(_CgOnlyPreconditioner):
        defs = _entry().model_json_schema()["$defs"]
    listed = {
        name: "CgOnly" in defs[name]["properties"]["preconditioner"]["discriminator"]["mapping"]
        for name in ("Cg", "Bicg")
    }
    assert listed == {"Cg": True, "Bicg": False}


def test_late_preconditioner_reaches_existing_slice() -> None:
    slice_cls = _slice("LatePreconditioner")
    block = {"solver": "Cg", "preconditioner": "Dummy"}
    data = {"solvers": {"p": block, "pFinal": block}}
    data["solvers"] |= {"U": {"solver": "Ir"}, "UFinal": {"solver": "Ir"}}
    with _registered_preconditioner(_DummyPreconditioner):
        config = slice_cls.model_validate(data)
    assert type(config.solvers.p.root.preconditioner) is _DummyPreconditioner


def test_late_preconditioner_builds_its_mapper_word() -> None:
    with _registered_preconditioner(_DummyPreconditioner):
        entry = _entry().model_validate({"solver": "Bicg", "preconditioner": "Dummy"})
        settings = entry.root.runtime_settings(_CASE)
    assert settings["preconditioner"] == "diagonal"


def test_late_preconditioner_appears_in_every_solver_schema() -> None:
    with _registered_preconditioner(_DummyPreconditioner):
        defs = _entry().model_json_schema()["$defs"]
    refs = [str(defs[name]["properties"]["preconditioner"]) for name in ("Cg", "Gmres", "Ir")]
    assert all("_DummyPreconditioner" in ref for ref in refs)


def test_removed_preconditioner_leaves_existing_slice() -> None:
    slice_cls = _slice("LatePreconditionerRemoval")
    with _registered_preconditioner(_DummyPreconditioner):
        pass
    assert "_DummyPreconditioner" not in slice_cls.model_json_schema()["$defs"]


def test_other_backend_preconditioner_is_not_listed() -> None:
    with _registered_preconditioner(_OtherPreconditioner):
        defs = _entry().model_json_schema()["$defs"]
    assert "_OtherPreconditioner" not in defs


def test_duplicate_preconditioner_registration_fails() -> None:
    class DuplicateIc(GinkgoPreconditioner):
        mapper_word: ClassVar[str] = "DIC"
        type: Literal["Ic"] = "Ic"

    with pytest.raises(ValueError, match="'Ic' of DuplicateIc is already registered by Ic"):
        Preconditioner.register(DuplicateIc)


def test_schema_is_discriminated_on_solver() -> None:
    schema = _entry().model_json_schema()
    assert schema["discriminator"]["propertyName"] == "solver"


def test_schema_arms_are_the_backend_plugins() -> None:
    arms = [arm["$ref"].rsplit("/", 1)[1] for arm in _entry().model_json_schema()["oneOf"]]
    assert arms == ["Cg", "Bicg", "Bicgstab", "Gmres", "Ir", "GinkgoConfigFile"]


@pytest.mark.parametrize(
    ("arm", "tag"),
    [("Cg", "Cg"), ("Ir", "Ir"), ("GinkgoConfigFile", "Ginkgo")],
)
def test_schema_arm_has_const_solver(arm: str, tag: str) -> None:
    schema = _entry().model_json_schema()["$defs"][arm]
    assert schema["properties"]["solver"]["const"] == tag


@pytest.mark.parametrize("key", ["x-group", "x-group-title", "x-member-title", "backend"])
def test_schema_has_no_backend_annotation(key: str) -> None:
    assert key not in str(_entry().model_json_schema())


def test_family_base_declares_the_common_controls() -> None:
    assert set(LinearSolver.model_fields) == {"solver", "tolerance", "relTol", "maxIter", "minIter"}


def test_duplicate_solver_registration_fails() -> None:
    class DuplicateCg(GinkgoSolver):
        solver: Literal["Cg"] = "Cg"

    with pytest.raises(ValueError, match="'Cg' of DuplicateCg is already registered by Cg"):
        LinearSolver.register(DuplicateCg)


def test_late_plugin_reaches_existing_slice() -> None:
    slice_cls = _slice("LateRegistration")
    block = {"solver": "Fgmres", "preconditioner": {"type": "Ic"}}
    data = {"solvers": {"p": block, "pFinal": block}}
    data["solvers"] |= {"U": {"solver": "Ir"}, "UFinal": {"solver": "Ir"}}
    with _registered(_Fgmres):
        config = slice_cls.model_validate(data)
    assert type(config.solvers.p.root) is _Fgmres


def test_removed_plugin_leaves_existing_slice() -> None:
    slice_cls = _slice("LateRemoval")
    with _registered(_Fgmres):
        pass
    assert "_Fgmres" not in slice_cls.model_json_schema()["$defs"]


def test_other_backend_is_not_listed_on_a_ginkgo_slice() -> None:
    slice_cls = _slice("OtherBackend")
    with _registered(_OtherBackend):
        defs = slice_cls.model_json_schema()["$defs"]
    assert "_OtherBackend" not in defs


def test_config_file_is_the_last_registered_builtin() -> None:
    plugins = PluginSystem.list_plugins()["LinearSolver"]
    names = [plugin.__name__ for plugin in plugins if plugin.backend == "ginkgo"]
    assert names[-1] == "GinkgoConfigFile"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, {"nfLevel": "advanced"}),
        ({"group": "Hierarchy"}, {"nfLevel": "advanced", "nfGroup": "Hierarchy"}),
        ({"default_hint": 64}, {"nfLevel": "advanced", "nfDefaultHint": 64}),
        ({"default_hint": False}, {"nfLevel": "advanced", "nfDefaultHint": False}),
    ],
)
def test_advanced_emits_only_schema_metadata(
    kwargs: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert advanced(None, **kwargs).json_schema_extra == expected
