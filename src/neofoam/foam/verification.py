# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Verification logic for fvSchemes / fvSolution requirements.

Collects requirements from active operations and verifies them against
dict data (parsed from OpenFOAM files or provided directly).

Two layers:
  Layer 1 (structural): required entries exist in the dict
  Layer 2 (value):      entry values parse into valid typed scheme models
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, create_model
from pydantic import ValidationError as PydanticValidationError

from neofoam.foam.requirements import SchemeRequirement, SolverRequirement
from neofoam.foam.schemes import (
    DdtScheme,
    DivScheme,
    GradScheme,
    InterpolationScheme,
    LaplacianScheme,
    SnGradScheme,
)

SECTION_TO_TYPE: dict[str, Any] = {
    "ddtSchemes": DdtScheme,
    "divSchemes": DivScheme,
    "gradSchemes": GradScheme,
    "laplacianSchemes": LaplacianScheme,
    "snGradSchemes": SnGradScheme,
    "interpolationSchemes": InterpolationScheme,
}


@dataclass(frozen=True)
class VerificationError:
    """A single verification failure."""

    field: str
    error_type: str  # "missing_entry" or "invalid_scheme"
    message: str
    file_name: str
    subdict: str | None = None
    input_value: Any = None


def collect_requirements(
    operations: list[Any],
) -> tuple[list[SchemeRequirement], list[SolverRequirement]]:
    """Gather deduplicated requirements from active operations.

    Each operation is expected to have ``scheme_requirements`` and
    ``solver_requirements`` attributes (set by ``@fvSchemes.add`` /
    ``@fvSolution.add`` decorators or stored on ``OperationDef``).
    """
    schemes: list[SchemeRequirement] = []
    solvers: list[SolverRequirement] = []
    seen_schemes: set[tuple[str, str]] = set()
    seen_solvers: set[str] = set()

    for op in operations:
        for req in getattr(op, "scheme_requirements", []):
            key = (req.section, req.key)
            if key not in seen_schemes:
                seen_schemes.add(key)
                schemes.append(req)
        for req in getattr(op, "solver_requirements", []):
            if req.field not in seen_solvers:
                seen_solvers.add(req.field)
                solvers.append(req)

    return schemes, solvers


def verify_fvschemes(
    fv_schemes: dict[str, Any],
    scheme_reqs: list[SchemeRequirement],
    scheme_type_map: dict[str, Any] | None = None,
) -> list[VerificationError]:
    """Verify that required fvSchemes entries exist and values are valid.

    Args:
        fv_schemes: Parsed fvSchemes dict (section → {key → value}).
        scheme_reqs: Required entries from active operations.
        scheme_type_map: Optional mapping of section name → Pydantic type
            for Layer 2 value validation (e.g. ``{"ddtSchemes": DdtScheme}``).

    Returns:
        List of errors (empty if all valid).
    """
    errors: list[VerificationError] = []

    for req in scheme_reqs:
        section_dict = fv_schemes.get(req.section, {})

        # Layer 1: entry must exist (specific key, or "default" fallback
        # unless default is "none" which means no unlisted schemes allowed)
        has_key = req.key in section_dict
        has_valid_default = (
            "default" in section_dict and section_dict["default"] != "none"
        )
        if not has_key and not has_valid_default:
            errors.append(
                VerificationError(
                    field=f"{req.section}.{req.key}",
                    error_type="missing_entry",
                    message=f"fvSchemes.{req.section}.{req.key} is required but missing",
                    file_name="system/fvSchemes",
                    subdict=req.section,
                )
            )
            continue

        # Layer 2: validate value type if type map provided
        if scheme_type_map is not None:
            scheme_type = scheme_type_map.get(req.section)
            if scheme_type is not None:
                value = section_dict.get(req.key, section_dict.get("default"))
                try:
                    TypeAdapter(scheme_type).validate_python(value)
                except PydanticValidationError:
                    errors.append(
                        VerificationError(
                            field=f"{req.section}.{req.key}",
                            error_type="invalid_scheme",
                            message=f"invalid scheme value '{value}'",
                            file_name="system/fvSchemes",
                            subdict=req.section,
                            input_value=value,
                        )
                    )

    return errors


def verify_fvsolution(
    fv_solution: dict[str, Any],
    solver_reqs: list[SolverRequirement],
) -> list[VerificationError]:
    """Verify that required fvSolution solver entries exist.

    Args:
        fv_solution: Parsed fvSolution dict (must have "solvers" sub-dict).
        solver_reqs: Required solver entries from active operations.

    Returns:
        List of errors (empty if all valid).
    """
    errors: list[VerificationError] = []
    solvers = fv_solution.get("solvers", {})

    for req in solver_reqs:
        if not _solver_key_exists(solvers, req.field):
            errors.append(
                VerificationError(
                    field=f"solvers.{req.field}",
                    error_type="missing_entry",
                    message=f"fvSolution.solvers.{req.field} is required but missing",
                    file_name="system/fvSolution",
                    subdict="solvers",
                )
            )

    return errors


def _solver_key_exists(solvers: dict[str, Any], field: str) -> bool:
    """Check if a solver entry exists, handling OpenFOAM regex keys.

    OpenFOAM fvSolution uses keys like ``(U|nuTilda)`` to define a single
    solver entry for multiple fields. This function matches ``"U"`` against
    ``"(U|nuTilda)"`` by checking if the field appears inside parenthesised
    pipe-separated lists.
    """
    if field in solvers:
        return True
    for key in solvers:
        if key.startswith("(") and key.endswith(")"):
            alternatives = key[1:-1].split("|")
            if field in alternatives:
                return True
    return False


def collect_requirements_from_models(
    models: list[Any],
) -> tuple[list[SchemeRequirement], list[SolverRequirement]]:
    """Gather requirements from active operations on model specs.

    If a model has an ``operation_collection`` (which selects active operations
    based on runtime state like ``use_boussinesq``), uses that to get only
    the operations that will actually run. Otherwise falls back to all
    registered ``_operations``.

    Works with ``ModelSpec``, ``ModelRuntime``, or any object with
    ``_operations``.
    """
    all_ops: list[Any] = []
    for m in models:
        ops = _get_active_operations(m)
        all_ops.extend(ops)
    return collect_requirements(all_ops)


def _get_active_operations(model: Any) -> list[Any]:
    """Get the active operations for a model, respecting operation_collection."""
    # ModelSpec with operation_collection — call it to get active ops
    if (
        hasattr(model, "_operation_collection_func")
        and model._operation_collection_func is not None
    ):
        try:
            result = model._operation_collection_func(model)
            # result is an Operations object — extract OperationDef-like objects
            # The Operations contains Operation objects which have metadata
            # but we need scheme_requirements from the original functions.
            # The operation_collection wraps functions, losing decorator metadata.
            # Fall back to matching by name against _operations.
            active_names = set()
            for op in result:
                name = getattr(op, "operation_name", None) or getattr(
                    getattr(op, "metadata", None), "op_name", None
                )
                if name:
                    active_names.add(name)
            if active_names:
                return [
                    op
                    for op in getattr(model, "_operations", [])
                    if op.name in active_names
                ]
        except Exception:
            pass

    # Direct _operations list (ModelSpec without operation_collection)
    if hasattr(model, "_operations"):
        return list(model._operations)

    # ModelRuntime → delegate to spec
    if hasattr(model, "spec"):
        return _get_active_operations(model.spec)

    return []


def verify_solver_setup(
    fv_schemes: dict[str, Any],
    fv_solution: dict[str, Any],
    models: list[Any],
    model_configs: list[Any] | None = None,
    scheme_type_map: dict[str, Any] | None = None,
) -> list[VerificationError]:
    """Run full verification: collect from models, verify against dicts.

    Args:
        fv_schemes: Parsed fvSchemes dict.
        fv_solution: Parsed fvSolution dict.
        models: List of ModelSpec or ModelRuntime objects.
        model_configs: Optional list of BaseConfig instances for field constraint validation.
        scheme_type_map: Optional section -> Pydantic type map for Layer 2.

    Returns:
        List of all errors (empty if everything is valid).
    """
    scheme_reqs, solver_reqs = collect_requirements_from_models(models)

    errors: list[VerificationError] = []
    errors.extend(verify_fvschemes(fv_schemes, scheme_reqs, scheme_type_map))
    errors.extend(verify_fvsolution(fv_solution, solver_reqs))

    if model_configs:
        for cfg in model_configs:
            if hasattr(cfg, "check_validation"):
                for ve in cfg.check_validation():
                    errors.append(
                        VerificationError(
                            field=str(ve.field),
                            error_type=ve.error_type,
                            message=ve.message,
                            file_name=ve.file_name,
                            subdict=ve.subdict,
                            input_value=ve.input_value,
                        )
                    )

    return errors


# ============================================================================
# Pydantic model builder — typed validation from requirements
# ============================================================================


def _sanitize_field_name(section: str, key: str) -> str:
    """Convert section + key to a valid Python field name.

    ``"ddtSchemes"`` + ``"ddt(U)"`` → ``"ddtSchemes_ddt_U"``
    """
    raw = f"{section}_{key}"
    return raw.replace("(", "_").replace(")", "").replace(",", "_").replace(".", "_")


def build_scheme_model(
    scheme_reqs: list[SchemeRequirement],
) -> type[BaseModel]:
    """Build a typed Pydantic model from scheme requirements.

    Each requirement becomes a field typed with the correct scheme union
    (``DdtScheme``, ``DivScheme``, etc.). Sections without a known type
    mapping (e.g., ``wallDist``) get ``str``.

    The resulting model can:
    - Validate parsed fvSchemes values (structural + type)
    - Expose ``model_json_schema()`` for AI introspection
    - Collect all validation errors at once
    """
    fields: dict[str, Any] = {}
    for req in scheme_reqs:
        name = _sanitize_field_name(req.section, req.key)
        typ = SECTION_TO_TYPE.get(req.section, str)
        fields[name] = (typ, Field(description=f"{req.section}.{req.key}"))
    return create_model("FvSchemesRequirements", **fields)


def flatten_fvschemes(
    fv_data: dict[str, Any],
    scheme_reqs: list[SchemeRequirement],
) -> dict[str, Any]:
    """Extract values from a nested fvSchemes dict matching requirements.

    For each requirement, looks up the concrete key first, then falls back
    to ``"default"`` (unless default is ``"none"``).
    """
    result: dict[str, Any] = {}
    for req in scheme_reqs:
        section = fv_data.get(req.section, {})
        value = section.get(req.key)
        if value is None:
            default = section.get("default")
            if default is not None and default != "none":
                value = default
        if value is not None:
            name = _sanitize_field_name(req.section, req.key)
            result[name] = value
    return result
