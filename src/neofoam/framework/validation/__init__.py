# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Honest case validation — CheckRegistry + isolated checks (§5)."""

from neofoam.framework.validation.checks import (
    check_boussinesq_gravity,
    check_constraint_patches,
    check_div_scheme,
    check_gamg_smoother,
    check_laminar_wall_functions,
    check_pimple_final,
    check_required_files,
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
    "mesh_patch_types",
    "is_boussinesq",
    "turbulence_type",
    "check_required_files",
    "check_constraint_patches",
    "check_gamg_smoother",
    "check_pimple_final",
    "check_boussinesq_gravity",
    "check_laminar_wall_functions",
    "check_div_scheme",
]
