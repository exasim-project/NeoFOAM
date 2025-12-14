# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

"""Tests for error handling during initialization."""

import pytest
from dataclasses import dataclass, field

from foamadapter.framework.model import Model
from foamadapter.framework.initialization import SolverInitializer
from .test_fixtures import TransportConfig


def test_configure_fails_if_required_model_not_found():
    """Test behavior when model dependency is missing."""
    
    @dataclass
    class OrphanModel:
        name: str = "orphan"
        files_read: bool = True
        configured: bool = False
        setup_complete: bool = False
        
        @Model.configure
        def check_dependency(self, registry):
            missing = registry.get("nonexistent")
            if not missing:
                raise RuntimeError("Required model 'nonexistent' not found")
            self.configured = True
    
    @dataclass
    class OrphanSolver:
        files_read: bool = True
        configured: bool = True
        setup_complete: bool = True
        model: OrphanModel = field(default_factory=OrphanModel)
        
        def get_models(self):
            return [self.model]
    
    solver = OrphanSolver()
    initializer = SolverInitializer(solver)
    
    with pytest.raises(RuntimeError, match="Required model 'nonexistent' not found"):
        initializer._run_configure()


def test_validation_error_in_configure():
    """Test that validation errors are raised during CONFIGURE."""
    
    @dataclass
    class InvalidTransport:
        name: str = "transport"
        files_read: bool = True
        configured: bool = False
        setup_complete: bool = False
        config: TransportConfig = field(default_factory=lambda: TransportConfig(viscosity=-1.0))
        
        @Model.configure
        def validate(self, registry):
            if self.config.viscosity <= 0:
                raise ValueError("Viscosity must be positive")
            self.configured = True
    
    @dataclass
    class InvalidSolver:
        files_read: bool = True
        configured: bool = True
        setup_complete: bool = True
        model: InvalidTransport = field(default_factory=InvalidTransport)
        
        def get_models(self):
            return [self.model]
    
    # This should fail during model creation due to Pydantic validation
    with pytest.raises(Exception):
        solver = InvalidSolver()
        initializer = SolverInitializer(solver)
        initializer._run_configure()
