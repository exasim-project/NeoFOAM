# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Tests for Pydantic-based control condition classes.

Tests cover:
- Generic iteration conditions
- Residual convergence conditions
- Boolean flag conditions
- Control bundle classes (PimpleControl, SimpleControl)
- Pydantic validation and type safety
- Boolean composition with framework's Condition class
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock
from pydantic import ValidationError

# Add src to path to allow direct import of control module
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


# Import control module directly to avoid OpenFOAM dependencies from algorithms package
def _load_control_module():
    """Load control module directly without triggering algorithms/__init__.py"""
    control_path = src_path / "foamadapter" / "algorithms" / "control.py"
    spec = importlib.util.spec_from_file_location("control_module", control_path)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    return control


def _load_condition_class():
    """Load Condition class from framework"""
    cond_path = src_path / "foamadapter" / "framework" / "conditions.py"
    spec = importlib.util.spec_from_file_location("conditions_module", cond_path)
    cond_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cond_mod)
    return cond_mod.Condition


control = _load_control_module()
Condition = _load_condition_class()

IterationCountCondition = control.IterationCountCondition
ResidualConvergenceCondition = control.ResidualConvergenceCondition
SingleIterationCondition = control.SingleIterationCondition
BooleanFlagCondition = control.BooleanFlagCondition
PimpleControl = control.PimpleControl
SimpleControl = control.SimpleControl


class TestIterationCountCondition:
    """Test generic iteration count condition."""

    def test_single_iteration(self, mock_context):
        """Test condition for single iteration."""
        condition = IterationCountCondition(nIterations=1)

        # First call returns True
        assert condition(mock_context) is True
        assert condition.is_final() is True

        # Second call returns False
        assert condition(mock_context) is False

    def test_multiple_iterations(self, mock_context):
        """Test condition for multiple iterations."""

        condition = IterationCountCondition(nIterations=3)

        # Should iterate 3 times
        iteration_count = 0
        while condition(mock_context):
            iteration_count += 1

        assert iteration_count == 3

    def test_is_final_flag(self, mock_context):
        """Test is_final() flag on last iteration."""

        condition = IterationCountCondition(nIterations=2)

        assert condition(mock_context) is True
        assert condition.is_final() is False

        assert condition(mock_context) is True
        assert condition.is_final() is True

        assert condition(mock_context) is False

    def test_reset(self, mock_context):
        """Test reset functionality."""

        condition = IterationCountCondition(nIterations=2)

        # First iteration
        assert condition(mock_context) is True
        assert condition(mock_context) is True
        assert condition(mock_context) is False

        # Reset and iterate again
        condition.reset()
        assert condition(mock_context) is True
        assert condition(mock_context) is True
        assert condition(mock_context) is False

    def test_linked_condition_reset(self, mock_context):
        """Test automatic reset of linked condition."""

        outer = IterationCountCondition(nIterations=2)
        inner = IterationCountCondition(nIterations=3)

        # Link inner to outer
        outer.link_condition(inner)

        # First outer iteration - inner should reset automatically
        assert outer(mock_context) is True

        # Inner iterations
        assert inner(mock_context) is True
        assert inner(mock_context) is True
        assert inner(mock_context) is True
        assert inner(mock_context) is False

        # Second outer iteration - inner should reset automatically
        assert outer(mock_context) is True

        # Inner should be reset and iterate again
        assert inner(mock_context) is True
        assert inner(mock_context) is True
        assert inner(mock_context) is True
        assert inner(mock_context) is False

    def test_validation_positive_iterations(self):
        """Test Pydantic validation for positive iterations."""

        # Valid: positive iterations
        condition = IterationCountCondition(nIterations=1)
        assert condition.nIterations == 1

        # Invalid: zero iterations
        with pytest.raises(ValidationError):
            IterationCountCondition(nIterations=0)

        # Invalid: negative iterations
        with pytest.raises(ValidationError):
            IterationCountCondition(nIterations=-1)


class TestResidualConvergenceCondition:
    """Test residual-based convergence condition."""

    def test_no_residual_control(self, mock_context):
        """Test condition continues when no residualControl specified."""

        condition = ResidualConvergenceCondition()

        # Should continue iteration (return True)
        assert condition(mock_context) is True
        assert condition.converged() is False

    def test_convergence_all_residuals_low(self, mock_context):
        """Test convergence when all residuals below tolerance."""

        condition = ResidualConvergenceCondition(residualControl={"p": 1e-2, "U": 1e-3})

        # Mock residuals below tolerance
        def mock_get_residual(ctx, field_name):
            return {"p": 5e-3, "U": 5e-4}[field_name]

        condition._get_residual = mock_get_residual

        # Should stop iteration (return False) due to convergence
        assert condition(mock_context) is False
        assert condition.converged() is True

    def test_no_convergence_high_residuals(self, mock_context):
        """Test continuation when residuals above tolerance."""

        condition = ResidualConvergenceCondition(residualControl={"p": 1e-2, "U": 1e-3})

        # Mock residuals above tolerance
        def mock_get_residual(ctx, field_name):
            return {"p": 5e-2, "U": 5e-3}[field_name]

        condition._get_residual = mock_get_residual

        # Should continue iteration (return True)
        assert condition(mock_context) is True
        assert condition.converged() is False

    def test_partial_convergence(self, mock_context):
        """Test no convergence when only some residuals below tolerance."""

        condition = ResidualConvergenceCondition(residualControl={"p": 1e-2, "U": 1e-3})

        # Mock: p converged, U not converged
        def mock_get_residual(ctx, field_name):
            return {"p": 5e-3, "U": 5e-2}[field_name]

        condition._get_residual = mock_get_residual

        # Should continue iteration (return True)
        assert condition(mock_context) is True
        assert condition.converged() is False

    def test_reset(self, mock_context):
        """Test reset after convergence."""

        condition = ResidualConvergenceCondition(residualControl={"p": 1e-2})

        # Mock residuals below tolerance
        condition._get_residual = lambda ctx, field: 5e-3

        # Converge
        assert condition(mock_context) is False
        assert condition.converged() is True

        # Reset
        condition.reset()
        assert condition.converged() is False

    def test_validation_residual_control_dict(self):
        """Test Pydantic validation for residualControl field."""

        # Valid: dictionary
        condition = ResidualConvergenceCondition(residualControl={"p": 1e-2, "U": 1e-3})
        assert condition.residualControl["p"] == 1e-2

        # Valid: empty dict (default)
        condition2 = ResidualConvergenceCondition()
        assert condition2.residualControl == {}


class TestSingleIterationCondition:
    """Test single iteration (once per reset) condition."""

    def test_single_execution(self, mock_context):
        """Test condition executes once then returns False."""

        condition = SingleIterationCondition()

        # First call returns True
        assert condition(mock_context) is True

        # Subsequent calls return False
        assert condition(mock_context) is False
        assert condition(mock_context) is False

    def test_reset(self, mock_context):
        """Test reset allows another execution."""

        condition = SingleIterationCondition()

        # First execution
        assert condition(mock_context) is True
        assert condition(mock_context) is False

        # Reset and execute again
        condition.reset()
        assert condition(mock_context) is True
        assert condition(mock_context) is False


class TestBooleanFlagCondition:
    """Test simple boolean flag condition."""

    def test_enabled_flag(self, mock_context):
        """Test condition when enabled."""

        condition = BooleanFlagCondition(enabled=True)

        # Should always return True
        assert condition(mock_context) is True
        assert condition(mock_context) is True

    def test_disabled_flag(self, mock_context):
        """Test condition when disabled."""

        condition = BooleanFlagCondition(enabled=False)

        # Should always return False
        assert condition(mock_context) is False
        assert condition(mock_context) is False

    def test_default_enabled(self, mock_context):
        """Test default value is enabled."""

        condition = BooleanFlagCondition()

        assert condition.enabled is True
        assert condition(mock_context) is True


class TestPimpleControl:
    """Test PIMPLE control bundle."""

    def test_initialization(self, mock_mesh):
        """Test PimpleControl initializes with configuration."""

        control = PimpleControl(
            nCorrectors=3,
            nNonOrthogonalCorrectors=2,
            momentumPredictor=True,
            turbCorr=False,
        )

        assert control.nCorrectors == 3
        assert control.nNonOrthogonalCorrectors == 2
        assert control.momentumPredictor_enabled is True
        assert control.turbCorr_enabled is False

    def test_pimple_loop_single_iteration(self, mock_mesh, mock_context):
        """Test PIMPLE outer loop executes once per time step."""

        control = PimpleControl(nCorrectors=2)

        # First call returns True
        assert control.loop(mock_context) is True

        # Second call returns False (already executed)
        assert control.loop(mock_context) is False

        # Reset for next time step
        control.reset()
        assert control.loop(mock_context) is True

    def test_corrector_loop(self, mock_mesh, mock_context):
        """Test corrector loop iterations."""

        control = PimpleControl(nCorrectors=3)

        # Start outer loop
        assert control.loop(mock_context) is True

        # Corrector loop should iterate 3 times
        corrector_count = 0
        while control.correct(mock_context):
            corrector_count += 1

        assert corrector_count == 3

    def test_non_orthogonal_loop(self, mock_mesh, mock_context):
        """Test non-orthogonal correction loop."""

        control = PimpleControl(nCorrectors=1, nNonOrthogonalCorrectors=2)

        assert control.loop(mock_context) is True
        assert control.correct(mock_context) is True

        # Should iterate nNonOrthogonalCorrectors + 1 times (2 + 1 = 3)
        non_ortho_count = 0
        while control.correctNonOrthogonal(mock_context):
            non_ortho_count += 1

        assert non_ortho_count == 3

    def test_nested_loops(self, mock_mesh, mock_context):
        """Test complete nested loop structure."""

        control = PimpleControl(nCorrectors=2, nNonOrthogonalCorrectors=1)

        total_non_ortho = 0

        while control.loop(mock_context):
            while control.correct(mock_context):
                while control.correctNonOrthogonal(mock_context):
                    total_non_ortho += 1

        # Should execute: 1 outer * 2 correctors * 2 non-ortho = 4 times
        assert total_non_ortho == 4

    def test_final_inner_iter(self, mock_mesh, mock_context):
        """Test finalInnerIter flag."""

        control = PimpleControl(nCorrectors=2)

        assert control.loop(mock_context) is True

        assert control.correct(mock_context) is True
        assert control.finalInnerIter() is False

        assert control.correct(mock_context) is True
        assert control.finalInnerIter() is True

        assert control.correct(mock_context) is False

    def test_final_non_orthogonal_iter(self, mock_mesh, mock_context):
        """Test finalNonOrthogonalIter flag."""

        control = PimpleControl(nCorrectors=1, nNonOrthogonalCorrectors=1)

        assert control.loop(mock_context) is True
        assert control.correct(mock_context) is True

        assert control.correctNonOrthogonal(mock_context) is True
        assert control.finalNonOrthogonalIter() is False

        assert control.correctNonOrthogonal(mock_context) is True
        assert control.finalNonOrthogonalIter() is True

        assert control.correctNonOrthogonal(mock_context) is False

    def test_momentum_predictor_flag(self, mock_mesh):
        """Test momentumPredictor flag."""

        control_enabled = PimpleControl(momentumPredictor=True)
        control_disabled = PimpleControl(momentumPredictor=False)

        assert control_enabled.momentumPredictor() is True
        assert control_disabled.momentumPredictor() is False

    def test_turb_corr_flag(self, mock_mesh):
        """Test turbCorr flag."""

        control_enabled = PimpleControl(turbCorr=True)
        control_disabled = PimpleControl(turbCorr=False)

        assert control_enabled.turbCorr() is True
        assert control_disabled.turbCorr() is False

    def test_validation(self, mock_mesh):
        """Test Pydantic validation."""

        # Valid configuration
        control = PimpleControl(nCorrectors=2, nNonOrthogonalCorrectors=1)
        assert control.nCorrectors == 2

        # Invalid: negative correctors
        with pytest.raises(ValidationError):
            PimpleControl(nCorrectors=-1)

        # Invalid: zero correctors
        with pytest.raises(ValidationError):
            PimpleControl(nCorrectors=0)


class TestSimpleControl:
    """Test SIMPLE control bundle."""

    def test_initialization(self, mock_mesh):
        """Test SimpleControl initializes with configuration."""

        control = SimpleControl(
            nNonOrthogonalCorrectors=2, residualControl={"p": 1e-2, "U": 1e-3}
        )

        assert control.nNonOrthogonalCorrectors == 2
        assert control.residualControl["p"] == 1e-2
        assert control.residualControl["U"] == 1e-3

    def test_loop_with_convergence(self, mock_mesh, mock_context):
        """Test loop stops when converged."""

        control = SimpleControl(residualControl={"p": 1e-2})

        # Mock residuals below tolerance
        control._residual_check._get_residual = lambda ctx, field: 5e-3

        # First call should detect convergence and return False
        assert control.loop(mock_context) is False
        assert control.converged() is True

    def test_loop_without_convergence(self, mock_mesh, mock_context):
        """Test loop continues when not converged."""

        control = SimpleControl(residualControl={"p": 1e-2})

        # Mock residuals above tolerance
        control._residual_check._get_residual = lambda ctx, field: 5e-2
        mock_context.runTime.loop.return_value = True

        # Should continue iteration
        assert control.loop(mock_context) is True
        assert control.converged() is False

    def test_loop_runtime_limit(self, mock_mesh, mock_context):
        """Test loop stops when runtime limit reached."""

        control = SimpleControl()

        # Mock runtime loop returns False
        mock_context.runTime.loop.return_value = False

        # Should stop iteration
        assert control.loop(mock_context) is False

    def test_get_residual_condition(self, mock_mesh):
        """Test accessing residual condition for composition."""

        control = SimpleControl(residualControl={"p": 1e-2})

        # Should return ResidualConvergenceCondition instance
        residual_cond = control.get_residual_condition()
        assert isinstance(residual_cond, ResidualConvergenceCondition)
        assert residual_cond.residualControl == {"p": 1e-2}

    def test_validation(self, mock_mesh):
        """Test Pydantic validation."""

        # Valid configuration
        control = SimpleControl(nNonOrthogonalCorrectors=2)
        assert control.nNonOrthogonalCorrectors == 2

        # Invalid: negative non-ortho correctors
        with pytest.raises(ValidationError):
            SimpleControl(nNonOrthogonalCorrectors=-1)


class TestPydanticFeatures:
    """Test Pydantic-specific features."""

    def test_field_descriptions(self):
        """Test Field descriptions are present."""

        fields = IterationCountCondition.model_fields
        assert "nIterations" in fields
        assert fields["nIterations"].description is not None

    def test_model_dump(self, mock_mesh):
        """Test model serialization (Pydantic feature)."""

        control = PimpleControl(
            nCorrectors=2,
            momentumPredictor=True,
        )

        # Can export to dict (excluding private fields and mesh)
        data = control.model_dump(
            exclude={
                "mesh",
                "_loop",
                "_corrector",
                "_non_ortho",
                "_momentum_predictor",
                "_turb_corr",
            }
        )
        assert data["nCorrectors"] == 2
        assert data["momentumPredictor_enabled"] is True

    def test_arbitrary_types_allowed(self, mock_mesh):
        """Test arbitrary_types_allowed for Context and other arbitrary types."""

        # Should not raise ValidationError for control instantiation
        control = PimpleControl()
        assert control.nCorrectors >= 1


class TestConditionComposition:
    """Test composition with framework's Condition class."""

    def test_wrap_in_condition(self, mock_context):
        """Test wrapping condition in framework's Condition class."""

        residual_check = ResidualConvergenceCondition(residualControl={"p": 1e-2})

        # Mock residuals
        residual_check._get_residual = lambda ctx, field: 5e-3

        # Wrap in framework Condition
        wrapped = Condition(residual_check, "ResidualConvergence")

        # Should work with framework Condition
        assert wrapped.name == "ResidualConvergence"
        assert wrapped(mock_context) is False  # Converged

    def test_boolean_operators(self, mock_context):
        """Test combining conditions with boolean operators."""

        residual_check = ResidualConvergenceCondition(residualControl={"p": 1e-2})
        residual_check._get_residual = lambda ctx, field: 5e-2  # Not converged

        iter_check = IterationCountCondition(nIterations=2)

        # Wrap in framework Condition
        residual_cond = Condition(residual_check, "Residuals")
        iter_cond = Condition(iter_check, "Iterations")

        # Test AND operator
        combined = residual_cond & iter_cond
        assert combined.name == "(Residuals & Iterations)"

        # Test OR operator
        combined2 = residual_cond | iter_cond
        assert combined2.name == "(Residuals | Iterations)"

        # Test NOT operator
        inverted = ~residual_cond
        assert inverted.name == "~Residuals"


# Fixtures


@pytest.fixture
def mock_mesh():
    """Create mock mesh object."""
    return MagicMock()


@pytest.fixture
def mock_context():
    """Create mock Context object."""
    context = MagicMock()
    context.runTime.loop.return_value = True
    return context
