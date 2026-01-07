# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for SimpleAlgorithm - SIMPLE algorithm for steady-state.

NOTE: Tests in this file are outdated - SIMPLE algorithm is not yet implemented
in the simplified architecture. These tests need to be rewritten once SIMPLE
is implemented following the same pattern as PimpleAlgorithm.
See test/solver/test_incompressible_fluid_pitzDaily.py for integration tests.
"""

import pytest
from unittest.mock import MagicMock, patch

from foamadapter.algorithms.pressure_velocity import (
    PressureVelocityAlgorithm,
    SimpleAlgorithm,
)
from foamadapter.framework.context import FieldUpdates

# Skip all tests since SIMPLE is not yet implemented
pytestmark = pytest.mark.skip(
    reason="SIMPLE algorithm not yet implemented in simplified architecture"
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_pybfoam():
    """Mock pybFoam module to avoid OpenFOAM dependencies."""
    with patch("foamadapter.algorithms.pressure_velocity.pyf") as mock_pyf:
        # Mock field classes
        mock_pyf.fvc = MagicMock()
        mock_pyf.fvm = MagicMock()
        mock_pyf.fvScalarMatrix = MagicMock()
        mock_pyf.fvVectorMatrix = MagicMock()
        mock_pyf.Word = MagicMock(return_value="mock_word")
        mock_pyf.solve = MagicMock()
        mock_pyf.adjustPhi = MagicMock()
        mock_pyf.constrainPressure = MagicMock()
        mock_pyf.constrainHbyA = MagicMock(return_value="mock_HbyA")
        mock_pyf.createPhi = MagicMock(return_value="mock_phi")
        mock_pyf.simpleControl = MagicMock(return_value="mock_simple_control")

        # Mock field types
        mock_pyf.volScalarField = MagicMock()
        mock_pyf.volVectorField = MagicMock()
        mock_pyf.surfaceScalarField = MagicMock()

        # Mock dictionary reading
        mock_fv_solution = MagicMock()
        mock_fv_solution.toc = MagicMock(
            return_value=["SIMPLE", "solvers", "relaxationFactors"]
        )

        # Mock SIMPLE subdictionary
        mock_simple_dict = MagicMock()

        # Setup get to work with type subscripting: simple_dict.get[int]("key")
        defaults = {
            "nNonOrthogonalCorrectors": 0,
            "consistent": False,
        }

        # Create a mock callable that returns defaults
        mock_typed_get = MagicMock(side_effect=lambda key: defaults.get(key, None))

        # Mock __getitem__ to return this callable (ignore the type argument)
        mock_simple_dict.get = MagicMock()
        mock_simple_dict.get.__getitem__ = MagicMock(return_value=mock_typed_get)

        # Mock residualControl subdictionary
        mock_residual_dict = MagicMock()
        mock_residual_dict.toc = MagicMock(return_value=["p", "U"])

        # Setup residual get with type subscripting
        residual_defaults = {"p": 1e-2, "U": 1e-3}
        mock_residual_typed_get = MagicMock(
            side_effect=lambda key: residual_defaults.get(key, None)
        )
        mock_residual_dict.get = MagicMock()
        mock_residual_dict.get.__getitem__ = MagicMock(
            return_value=mock_residual_typed_get
        )

        mock_simple_dict.subDict = MagicMock(return_value=mock_residual_dict)

        mock_fv_solution.subDict = MagicMock(return_value=mock_simple_dict)

        mock_pyf.dictionary = MagicMock()
        mock_pyf.dictionary.read = MagicMock(return_value=mock_fv_solution)

        yield mock_pyf


@pytest.fixture
def mock_fields():
    """Create mock field objects for testing."""
    U = MagicMock()
    U.assign = MagicMock()
    U.correctBoundaryConditions = MagicMock()
    U.relax = MagicMock()

    p = MagicMock()
    p.assign = MagicMock()
    p.select = MagicMock(return_value="mock_solver_dict")

    phi = MagicMock()
    phi.assign = MagicMock()

    UEqn = MagicMock()
    UEqn.A = MagicMock(return_value=1.0)
    UEqn.H = MagicMock(return_value="mock_H")
    UEqn.relax = MagicMock()

    return {"U": U, "p": p, "phi": phi, "UEqn": UEqn}


@pytest.fixture
def mock_turbulence():
    """Create mock turbulence model."""
    turbulence = MagicMock()
    turbulence.divDevReff = MagicMock(return_value="mock_divDevReff")
    turbulence.correct = MagicMock()
    return turbulence


@pytest.fixture
def mock_simple_control():
    """Create mock simple control object."""
    simple_control = MagicMock()
    simple_control.loop = MagicMock(return_value=False)  # Exit after one iteration
    simple_control.correctNonOrthogonal = MagicMock()
    simple_control.finalNonOrthogonalIter = MagicMock(return_value=True)

    # Mock iterator behavior
    simple_control.correctNonOrthogonal.__iter__ = lambda self: iter([True])
    simple_control.correctNonOrthogonal.__next__ = lambda self: True

    return simple_control


# ============================================================================
# Base Class Tests
# ============================================================================


def test_base_class_detects_simple(mock_pybfoam):
    """Test that base class correctly detects SIMPLE algorithm."""
    result = PressureVelocityAlgorithm.from_fv_solution()

    # PluginSystem returns wrapper, access .config for actual instance
    algo = result.config if hasattr(result, "config") else result

    assert isinstance(algo, SimpleAlgorithm)
    assert algo.algorithm_type == "SIMPLE"


def test_simple_method_instantiation(mock_pybfoam):
    """Test SimpleAlgorithm can be instantiated directly."""
    simple = SimpleAlgorithm()
    assert simple.algorithm_type == "SIMPLE"
    assert simple.nNonOrthogonalCorrectors == 0
    assert simple.consistent is False


def test_simple_method_load_settings(mock_pybfoam):
    """Test that SimpleAlgorithm loads settings from fvSolution."""
    simple = SimpleAlgorithm()
    simple.load_fv_solution()

    # Verify dictionary was read
    mock_pybfoam.dictionary.read.assert_called_once()


def test_simple_method_provides_fields(mock_pybfoam):
    """Test that SimpleAlgorithm provides correct fields."""
    simple = SimpleAlgorithm()
    provides = simple.provides

    assert "p" in provides
    assert "U" in provides
    assert "phi" in provides
    assert "simple_control" in provides


def test_simple_method_setup(mock_pybfoam):
    """Test that SimpleAlgorithm setup returns correct initializers."""
    simple = SimpleAlgorithm()
    initializers = simple.setup()

    # Should have 4 initializers: p, U, phi, simple_control
    assert len(initializers) == 4


def test_simple_method_name(mock_pybfoam):
    """Test SimpleAlgorithm name property."""
    simple = SimpleAlgorithm()
    assert simple.name() == "SIMPLE"


def test_simple_method_create_control(mock_pybfoam):
    """Test SimpleAlgorithm creates SimpleControl object."""
    from foamadapter.algorithms.control import SimpleControl

    simple = SimpleAlgorithm()
    mesh = MagicMock()

    control = simple.create_control(mesh)

    # Should return Python SimpleControl, not call pyf.simpleControl
    assert isinstance(control, SimpleControl)
    assert control.nNonOrthogonalCorrectors == simple.nNonOrthogonalCorrectors
    assert control.residualControl == simple.residualControl


# ============================================================================
# Operations Tests
# ============================================================================


def test_simple_method_has_operations(mock_pybfoam):
    """Test that SimpleAlgorithm has operation methods."""
    simple = SimpleAlgorithm()
    ops = simple.operations()

    assert ops is not None
    assert len(ops) == 2  # momentum and continuity

    # Check operation names
    op_names = [op.name for op in ops]
    assert "momentum" in op_names
    assert "continuity" in op_names


@pytest.mark.skip(
    reason="Requires real pybFoam types - can't mock fvm.ddt type checking"
)
def test_simple_momentum_operation(
    mock_pybfoam, mock_fields, mock_turbulence, mock_simple_control
):
    """Test SIMPLE momentum operation."""
    simple = SimpleAlgorithm()

    # Mock fvVectorMatrix constructor
    with patch(
        "foamadapter.algorithms.pressure_velocity.fvVectorMatrix"
    ) as mock_fvVectorMatrix:
        mock_UEqn = MagicMock()
        mock_UEqn.relax = MagicMock()
        mock_fvVectorMatrix.return_value = mock_UEqn

        result = simple.momentum(
            U=mock_fields["U"],
            phi=mock_fields["phi"],
            p=mock_fields["p"],
            turbulence=mock_turbulence,
            simple_control=mock_simple_control,
        )

        # Verify result
        assert isinstance(result, FieldUpdates)
        assert "UEqn" in result

        # Verify under-relaxation was applied
        mock_UEqn.relax.assert_called_once()

        # Verify solve was called
        mock_pybfoam.solve.assert_called_once()


@pytest.mark.skip(
    reason="Requires real pybFoam types - can't mock fvc.flux type checking"
)
def test_simple_continuity_operation(mock_pybfoam, mock_fields, mock_simple_control):
    """Test SIMPLE continuity operation."""
    simple = SimpleAlgorithm()
    simple.pRefCell = 0
    simple.pRefValue = 0.0

    with (
        patch(
            "foamadapter.algorithms.pressure_velocity.volScalarField"
        ) as mock_volScalarField,
        patch(
            "foamadapter.algorithms.pressure_velocity.volVectorField"
        ) as mock_volVectorField,
        patch(
            "foamadapter.algorithms.pressure_velocity.surfaceScalarField"
        ) as mock_surfaceScalarField,
        patch(
            "foamadapter.algorithms.pressure_velocity.fvScalarMatrix"
        ) as mock_fvScalarMatrix,
    ):
        # Setup mocks
        mock_rAU = MagicMock()
        mock_HbyA = MagicMock()
        mock_phiHbyA = MagicMock()
        mock_pEqn = MagicMock()
        mock_pEqn.setReference = MagicMock()
        mock_pEqn.solve = MagicMock()
        mock_pEqn.flux = MagicMock(return_value="mock_flux")

        mock_volScalarField.return_value = mock_rAU
        mock_volVectorField.return_value = mock_HbyA
        mock_surfaceScalarField.return_value = mock_phiHbyA
        mock_fvScalarMatrix.return_value = mock_pEqn

        # Mock simple_control iteration
        mock_simple_control.correctNonOrthogonal.return_value.__iter__ = (
            lambda self: iter([True])
        )
        mock_simple_control.finalNonOrthogonalIter.return_value = True

        result = simple.continuity(
            U=mock_fields["U"],
            p=mock_fields["p"],
            phi=mock_fields["phi"],
            UEqn=mock_fields["UEqn"],
            simple_control=mock_simple_control,
        )

        # Verify result
        assert isinstance(result, FieldUpdates)
        assert "U" in result
        assert "p" in result
        assert "phi" in result

        # Verify pressure equation was solved
        mock_pEqn.solve.assert_called_once()

        # Verify velocity was corrected and relaxed
        mock_fields["U"].assign.assert_called()
        mock_fields["U"].correctBoundaryConditions.assert_called_once()
        mock_fields["U"].relax.assert_called_once()


def test_simple_reference_cell(mock_pybfoam):
    """Test that SimpleAlgorithm stores reference cell and value."""
    simple = SimpleAlgorithm()
    simple.pRefCell = 42
    simple.pRefValue = 1.0

    assert simple.pRefCell == 42
    assert simple.pRefValue == 1.0


def test_simple_residual_control_loading(mock_pybfoam):
    """Test that residualControl is loaded correctly."""
    simple = SimpleAlgorithm()
    simple.load_fv_solution()

    # Should have loaded residual control
    assert isinstance(simple.residualControl, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
