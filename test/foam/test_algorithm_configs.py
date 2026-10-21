# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for the typed ``system/fvSolution`` algorithm blocks
(:mod:`neofoam.foam.algorithm_configs`).

These classes replace the ``getOrDefault`` call sites the pressure-velocity
control factories used to carry, so the two things a test has to pin are the
*defaults* (what a case that omits a key gets) and the *reads* (that a key the
case does set actually arrives). Both run against two real ``system/fvSolution``
files under ``cases/``: ``algorithmDefaults`` has present-but-empty blocks, and
``algorithmBlocks`` sets every declared key to the opposite of its default, so
neither direction can pass by accident.

Every load uses ``validate=False`` — the call the framework's own auto-load and
the control factories make, so the coercion path under test is the production
one (OpenFOAM ``yes``/``no`` → ``bool``, and the deliberate ``-1`` corrector
sentinels staying untouched).

``correctPhi`` is the one key whose OpenFOAM default is not a literal but the
runtime value ``mesh.dynamic()``; :meth:`DynamicMeshControls.resolved` is what
substitutes it, so it is pinned in both directions (unset → the mesh's answer,
set → the case wins).
"""

from pathlib import Path

from neofoam.foam.algorithm_configs import (
    DynamicMeshControls,
    PimpleAlgorithmConfig,
    PisoAlgorithmConfig,
    PisoDynamicMeshControls,
    SimpleAlgorithmConfig,
)
from neofoam.solver.incompressibleVoF.models.pressure_velocity.control_factory import (
    VofPimpleAlgorithmConfig,
)

_CASES = Path(__file__).parent / "cases"
_DEFAULTS = _CASES / "algorithmDefaults"
_BLOCKS = _CASES / "algorithmBlocks"


# --- the blocks each class binds to ---------------------------------------


def test_each_config_binds_to_its_block_of_fvsolution() -> None:
    for cls, subdict in (
        (PimpleAlgorithmConfig, "PIMPLE"),
        (DynamicMeshControls, "PIMPLE"),
        (VofPimpleAlgorithmConfig, "PIMPLE"),
        (PisoAlgorithmConfig, "PISO"),
        (PisoDynamicMeshControls, "PISO"),
        (SimpleAlgorithmConfig, "SIMPLE"),
    ):
        assert cls.io_config is not None
        assert cls.io_config.file == "system/fvSolution"
        assert cls.io_config.subdict == subdict


# --- PIMPLE ----------------------------------------------------------------


def test_pimple_defaults_are_the_ones_pimple_control_read_applies() -> None:
    config = PimpleAlgorithmConfig.load(case_dir=_DEFAULTS, validate=False)
    assert config.nOuterCorrectors == 1
    assert config.nCorrectors == 2
    assert config.nNonOrthogonalCorrectors == 0
    assert config.momentumPredictor is True
    assert config.turbCorr is True
    assert config.turbOnFinalIterOnly is True
    assert config.finalOnLastPimpleIterOnly is False


def test_pimple_reads_every_declared_key_from_the_case() -> None:
    config = PimpleAlgorithmConfig.load(case_dir=_BLOCKS, validate=False)
    assert config.nOuterCorrectors == 3
    assert config.nCorrectors == 4
    assert config.nNonOrthogonalCorrectors == 2
    assert config.momentumPredictor is False
    assert config.turbCorr is False
    assert config.turbOnFinalIterOnly is False
    assert config.finalOnLastPimpleIterOnly is True


def test_the_vof_variant_adds_the_frozen_flow_switch() -> None:
    # interFoam's own key; the shared PIMPLE model does not carry it.
    assert VofPimpleAlgorithmConfig.load(case_dir=_DEFAULTS, validate=False).frozenFlow is False
    assert VofPimpleAlgorithmConfig.load(case_dir=_BLOCKS, validate=False).frozenFlow is True
    assert "frozenFlow" not in PimpleAlgorithmConfig.model_fields


def test_the_piso_class_reads_the_piso_block() -> None:
    # Same keys, other block: the PISO case's nCorrectors is 4, and the PIMPLE
    # block of the same file must not be what answered.
    assert PisoAlgorithmConfig.load(case_dir=_BLOCKS, validate=False).nCorrectors == 4
    assert PisoAlgorithmConfig.load(case_dir=_BLOCKS, validate=False).momentumPredictor is False


# --- SIMPLE ----------------------------------------------------------------


def test_simple_defaults_are_the_ones_simple_control_read_applies() -> None:
    config = SimpleAlgorithmConfig.load(case_dir=_DEFAULTS, validate=False)
    assert config.nNonOrthogonalCorrectors == 0
    assert config.momentumPredictor is True
    assert config.consistent is False


def test_simple_reads_every_declared_key_from_the_case() -> None:
    config = SimpleAlgorithmConfig.load(case_dir=_BLOCKS, validate=False)
    assert config.nNonOrthogonalCorrectors == 2
    assert config.momentumPredictor is False
    assert config.consistent is True


# --- createDyMControls.H ---------------------------------------------------


def test_dynamic_mesh_switches_default_to_off_and_correct_phi_to_unset() -> None:
    controls = DynamicMeshControls.load(case_dir=_DEFAULTS, validate=False)
    # ``None`` is "the case did not say" — OpenFOAM's default is mesh.dynamic().
    assert controls.correctPhi is None
    assert controls.checkMeshCourantNo is False
    assert controls.moveMeshOuterCorrectors is False


def test_dynamic_mesh_switches_are_read_from_the_case() -> None:
    controls = DynamicMeshControls.load(case_dir=_BLOCKS, validate=False)
    assert controls.correctPhi is False
    assert controls.checkMeshCourantNo is True
    assert controls.moveMeshOuterCorrectors is True


def test_the_piso_twin_reads_the_switches_from_the_piso_block() -> None:
    controls = PisoDynamicMeshControls.load(case_dir=_BLOCKS, validate=False)
    assert controls.correctPhi is False
    assert controls.checkMeshCourantNo is True
    assert controls.moveMeshOuterCorrectors is True


def test_an_unset_correct_phi_resolves_to_the_meshs_dynamic_flag() -> None:
    controls = DynamicMeshControls.load(case_dir=_DEFAULTS, validate=False)
    assert controls.resolved(mesh_dynamic=True).correctPhi is True
    assert controls.resolved(mesh_dynamic=False).correctPhi is False


def test_a_case_that_sets_correct_phi_wins_over_the_mesh_default() -> None:
    # The moving VoF tutorials ask for ``correctPhi no`` on a dynamic mesh.
    controls = DynamicMeshControls.load(case_dir=_BLOCKS, validate=False)
    assert controls.resolved(mesh_dynamic=True).correctPhi is False


def test_resolving_leaves_the_two_literal_switches_alone() -> None:
    controls = DynamicMeshControls.load(case_dir=_BLOCKS, validate=False)
    resolved = controls.resolved(mesh_dynamic=True)
    assert resolved.checkMeshCourantNo is True
    assert resolved.moveMeshOuterCorrectors is True
