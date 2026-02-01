# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for stage decorators."""

import pytest

pytestmark = pytest.mark.skip(reason="Outdated - framework refactored (@Model decorators removed)")

# from dataclasses import dataclass, field
# from typing import Any
#
# from foamadapter.framework.model import Model
# from foamadapter.framework.initialization import SolverInitializer


def test_multiple_methods_same_stage():
    """Test that multiple methods can have same stage decorator."""

    @dataclass
    class MultiMethodModel:
        name: str = "multi"
        files_read: bool = False
        configured: bool = True
        setup_complete: bool = True
        data1: Any = None
        data2: Any = None

        @Model.load
        def load_data1(self):
            self.data1 = "loaded1"

        @Model.load
        def load_data2(self):
            self.data2 = "loaded2"
            self.files_read = True

    @dataclass
    class MultiSolver:
        files_read: bool = True
        configured: bool = True
        setup_complete: bool = True
        model: MultiMethodModel = field(default_factory=MultiMethodModel)

        def get_models(self):
            return [self.model]

    solver = MultiSolver()
    initializer = SolverInitializer(solver)
    initializer._run_load()

    assert solver.model.data1 == "loaded1"
    assert solver.model.data2 == "loaded2"


def test_method_without_decorator_not_called():
    """Test that methods without stage decorators are not called."""

    @dataclass
    class SelectiveModel:
        name: str = "selective"
        files_read: bool = True
        configured: bool = True
        setup_complete: bool = True
        decorated_called: bool = False
        undecorated_called: bool = False

        @Model.load
        def decorated_method(self):
            self.decorated_called = True

        def undecorated_method(self):
            self.undecorated_called = True

    @dataclass
    class SelectiveSolver:
        files_read: bool = True
        configured: bool = True
        setup_complete: bool = True
        model: SelectiveModel = field(default_factory=SelectiveModel)

        def get_models(self):
            return [self.model]

    solver = SelectiveSolver()
    initializer = SolverInitializer(solver)
    initializer._run_load()

    assert solver.model.decorated_called
    assert not solver.model.undecorated_called
