# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for PimpleMethod - new discriminated union API.

Tests the new PressureVelocityAlgorithm.from_fv_solution() API
and PimpleMethod implementation.
"""

import pytest
from unittest.mock import MagicMock, patch

from foamadapter.algorithms.pressure_velocity import (
    PressureVelocityAlgorithm,
    PimpleMethod,
)
from foamadapter.framework.context import FieldUpdates


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
        mock_pyf.pimpleControl = MagicMock(return_value="mock_pimple_control")

        # Mock field types
        mock_pyf.volScalarField = MagicMock()
        mock_pyf.volVectorField = MagicMock()
        mock_pyf.surfaceScalarField = MagicMock()

        # Mock dictionary reading
        mock_fv_solution = MagicMock()
        mock_fv_solution.toc = MagicMock(
            return_value=["PIMPLE", "solvers", "relaxationFactors"]
        )

        mock_pimple_dict = MagicMock()
        # Setup get to work with type subscripting: pimple_dict.get[int]("key")
        defaults = {
            "nCorrectors": 2,
            "nNonOrthogonalCorrectors": 0,
            "momentumPredictor": True,
        }

        # Create a mock callable that returns defaults
        mock_typed_get = MagicMock(side_effect=lambda key: defaults.get(key, None))

        # Mock __getitem__ to return this callable (ignore the type argument)
        mock_pimple_dict.get = MagicMock()
        mock_pimple_dict.get.__getitem__ = MagicMock(return_value=mock_typed_get)
        mock_fv_solution.subDict = MagicMock(return_value=mock_pimple_dict)

        mock_pyf.dictionary.read = MagicMock(return_value=mock_fv_solution)

        yield mock_pyf


@pytest.fixture
def mock_fields():
    """Create mock field objects for testing."""
    U = MagicMock()
    U.name = "U"
    U.assign = MagicMock()
    U.correctBoundaryConditions = MagicMock()

    p = MagicMock()
    p.name = "p"
    p.assign = MagicMock()
    p.select = MagicMock(return_value="mock_solver_dict")

    phi = MagicMock()
    phi.name = "phi"
    phi.assign = MagicMock()

    return {"U": U, "p": p, "phi": phi}


@pytest.fixture
def mock_turbulence():
    """Create mock turbulence model."""
    turbulence = MagicMock()
    turbulence.divDevReff = MagicMock(return_value="mock_divDevReff")
    return turbulence


@pytest.fixture
def mock_pimple_control():
    """Create mock pimple control object."""
    control = MagicMock()
    control.loop = MagicMock(return_value=False)  # No loop by default
    control.correct = MagicMock(return_value=False)
    control.correctNonOrthogonal = MagicMock(return_value=False)
    control.momentumPredictor = MagicMock(return_value=True)
    control.finalInnerIter = MagicMock(return_value=True)
    control.finalNonOrthogonalIter = MagicMock(return_value=True)
    return control


# ============================================================================
# Base Class Tests
# ============================================================================


def test_base_class_detects_pimple(mock_pybfoam):
    """Test that base class detects PIMPLE from fvSolution."""
    base = PressureVelocityAlgorithm()
    base.load_fv_solution("system/fvSolution")

    assert base.algorithm_type == "PIMPLE"
    mock_pybfoam.dictionary.read.assert_called_once_with("system/fvSolution")


def test_base_class_from_fv_solution_returns_pimple_method(mock_pybfoam):
    """Test factory method returns PimpleMethod instance."""
    result = PressureVelocityAlgorithm.from_fv_solution()

    # PluginSystem returns wrapper, access .config for actual instance
    algorithm = result.config if hasattr(result, "config") else result

    # Should return PimpleMethod via discriminated union
    assert isinstance(algorithm, PimpleMethod)
    assert algorithm.algorithm_type == "PIMPLE"


def test_base_class_raises_on_no_algorithm(mock_pybfoam):
    """Test that base class raises error if no algorithm found."""
    mock_pybfoam.dictionary.read.return_value.toc = MagicMock(
        return_value=["solvers", "relaxationFactors"]
    )

    base = PressureVelocityAlgorithm()
    with pytest.raises(ValueError, match="No algorithm found in fvSolution"):
        base.load_fv_solution()


# ============================================================================
# PimpleMethod Tests
# ============================================================================


def test_pimple_method_creation():
    """Test PimpleMethod can be created directly."""
    method = PimpleMethod(pRefCell=0, pRefValue=0.0)

    assert method.algorithm_type == "PIMPLE"
    assert method.pRefCell == 0
    assert method.pRefValue == 0.0
    assert method.nCorrectors == 2  # default
    assert method.momentumPredictor is True  # default


def test_pimple_method_loads_settings(mock_pybfoam):
    """Test PimpleMethod loads settings from fvSolution."""
    method = PimpleMethod()
    method.load_fv_solution()

    # Should have loaded settings from mock
    assert method.nCorrectors == 2
    assert method.nNonOrthogonalCorrectors == 0
    assert method.momentumPredictor is True


def test_pimple_method_name():
    """Test PimpleMethod returns correct name."""
    method = PimpleMethod()
    assert method.name() == "PIMPLE"


def test_pimple_method_create_control(mock_pybfoam):
    """Test PimpleMethod creates PimpleControl."""
    from foamadapter.algorithms.control import PimpleControl

    method = PimpleMethod()
    mock_mesh = MagicMock()

    control = method.create_control(mock_mesh)

    # Should return Python PimpleControl, not call pyf.pimpleControl
    assert isinstance(control, PimpleControl)
    assert control.nCorrectors == method.nCorrectors
    assert control.nNonOrthogonalCorrectors == method.nNonOrthogonalCorrectors
    assert control.momentumPredictor_enabled == method.momentumPredictor
    assert control.turbCorr_enabled == method.turbCorr


def test_pimple_method_has_operations():
    """Test PimpleMethod has operations collection."""
    method = PimpleMethod()
    ops = method.operations()

    assert ops is not None
    assert len(ops) == 2  # momentum and continuity

    # Check operation names
    op_names = [op.name for op in ops]
    assert "momentum" in op_names
    assert "continuity" in op_names


@pytest.mark.skip(
    reason="Requires real pybFoam types - can't mock fvm.ddt type checking"
)
def test_pimple_method_momentum_operation(
    mock_pybfoam, mock_fields, mock_turbulence, mock_pimple_control
):
    """Test PimpleMethod momentum operation."""
    method = PimpleMethod(pRefCell=0, pRefValue=0.0)

    # Setup equation mock
    mock_UEqn = MagicMock()
    mock_UEqn.relax = MagicMock()
    mock_pybfoam.fvVectorMatrix.return_value = mock_UEqn

    # Execute momentum
    result = method.momentum(
        U=mock_fields["U"],
        phi=mock_fields["phi"],
        p=mock_fields["p"],
        turbulence=mock_turbulence,
        pimple_control=mock_pimple_control,
    )

    # Check result
    assert isinstance(result, FieldUpdates)
    assert "UEqn" in result

    # Check equation was assembled and relaxed
    mock_UEqn.relax.assert_called_once()

    # Check momentum predictor was solved
    mock_pimple_control.momentumPredictor.assert_called_once()
    mock_pybfoam.solve.assert_called_once()


def test_pimple_method_continuity_operation(
    mock_pybfoam, mock_fields, mock_pimple_control
):
    """Test PimpleMethod continuity operation."""
    method = PimpleMethod(pRefCell=0, pRefValue=0.0)

    # Setup mocks for equation
    mock_UEqn = MagicMock()
    mock_UEqn.A = MagicMock(return_value=1.0)
    mock_UEqn.H = MagicMock(return_value="mock_H")

    mock_pEqn = MagicMock()
    mock_pEqn.setReference = MagicMock()
    mock_pEqn.solve = MagicMock()
    mock_pEqn.flux = MagicMock(return_value="mock_flux")
    mock_pybfoam.fvScalarMatrix.return_value = mock_pEqn

    # No loops for simple test
    mock_pimple_control.loop.return_value = False

    # Execute continuity
    result = method.continuity(
        U=mock_fields["U"],
        p=mock_fields["p"],
        phi=mock_fields["phi"],
        UEqn=mock_UEqn,
        pimple_control=mock_pimple_control,
    )

    # Check result
    assert isinstance(result, FieldUpdates)
    assert "U" in result
    assert "p" in result
    assert "phi" in result


@pytest.mark.skip(
    reason="Requires real pybFoam types - can't mock volScalarField constructor"
)
def test_pimple_method_continuity_with_loops(
    mock_pybfoam, mock_fields, mock_pimple_control
):
    """Test PimpleMethod continuity with PIMPLE loops."""
    method = PimpleMethod(pRefCell=0, pRefValue=0.0)

    # Setup equation mocks
    mock_UEqn = MagicMock()
    mock_UEqn.A = MagicMock(return_value=1.0)
    mock_UEqn.H = MagicMock(return_value="mock_H")

    mock_pEqn = MagicMock()
    mock_pEqn.setReference = MagicMock()
    mock_pEqn.solve = MagicMock()
    mock_pEqn.flux = MagicMock(return_value="mock_flux")
    mock_pybfoam.fvScalarMatrix.return_value = mock_pEqn

    # Setup loops: one outer loop, one corrector, one non-orthogonal
    loop_count = [1]
    correct_count = [1]
    non_orth_count = [1]

    def mock_loop():
        if loop_count[0] > 0:
            loop_count[0] -= 1
            return True
        return False

    def mock_correct():
        if correct_count[0] > 0:
            correct_count[0] -= 1
            return True
        return False

    def mock_non_orth():
        if non_orth_count[0] > 0:
            non_orth_count[0] -= 1
            return True
        return False

    mock_pimple_control.loop.side_effect = mock_loop
    mock_pimple_control.correct.side_effect = mock_correct
    mock_pimple_control.correctNonOrthogonal.side_effect = mock_non_orth

    # Execute
    method.continuity(
        U=mock_fields["U"],
        p=mock_fields["p"],
        phi=mock_fields["phi"],
        UEqn=mock_UEqn,
        pimple_control=mock_pimple_control,
    )

    # Check pressure equation was solved
    mock_pEqn.solve.assert_called()
    mock_pEqn.setReference.assert_called_with(0, 0.0, False)

    # Check velocity was corrected
    mock_fields["U"].assign.assert_called()
    mock_fields["U"].correctBoundaryConditions.assert_called()


# ============================================================================
# Integration Tests
# ============================================================================


def test_from_fv_solution_full_workflow(mock_pybfoam):
    """Test full workflow: fvSolution → PimpleMethod with loaded settings."""
    # Factory creates and loads
    result = PressureVelocityAlgorithm.from_fv_solution()

    # PluginSystem returns wrapper, access .config
    algorithm = result.config if hasattr(result, "config") else result

    # Should be PimpleMethod
    assert isinstance(algorithm, PimpleMethod)
    assert algorithm.algorithm_type == "PIMPLE"

    # Should have operations
    ops = algorithm.operations()
    assert len(ops) == 2

    # Should have correct name
    assert algorithm.name() == "PIMPLE"


def test_pimple_method_provides_fields():
    """Test PimpleMethod provides required fields."""
    method = PimpleMethod()

    provides = method.provides
    assert "p" in provides
    assert "U" in provides
    assert "phi" in provides
    assert "pimple_control" in provides


def test_pimple_method_setup_initializers(mock_pybfoam):
    """Test PimpleMethod returns setup initializers."""
    method = PimpleMethod()

    initializers = method.setup()

    # Should have initializers for p, U, phi, pimple_control
    assert len(initializers) >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
