# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Honest case validation — CheckRegistry + isolated checks (§5)."""

from neofoam.framework.validation.checks import (
    SOLVER_COMPANION,
    check_boussinesq_gravity,
    check_constraint_patches,
    check_div_scheme,
    check_laminar_wall_functions,
    check_pimple_final,
    check_required_files,
    check_solver_companion,
    default_registry,
    is_boussinesq,
    mesh_patch_types,
    turbulence_type,
    validate,
)
from neofoam.framework.validation.registry import (
    CaseContext,
    Check,
    CheckRegistry,
    Finding,
    ValidationReport,
)

__all__ = [
    "Finding",
    "ValidationReport",
    "CaseContext",
    "Check",
    "CheckRegistry",
    "validate",
    "default_registry",
    "SOLVER_COMPANION",
    "mesh_patch_types",
    "is_boussinesq",
    "turbulence_type",
    "check_required_files",
    "check_constraint_patches",
    "check_solver_companion",
    "check_pimple_final",
    "check_boussinesq_gravity",
    "check_laminar_wall_functions",
    "check_div_scheme",
]
