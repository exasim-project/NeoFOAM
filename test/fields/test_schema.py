# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for :func:`neofoam.fields.schema.schema_for`.

Construction-only tests live here (no disk IO). The full OpenFOAM
round-trip lives in ``test_loader.py`` so the IO surface is exercised
once against the canonical case fixture rather than per-arm.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neofoam.fields.bc import FixedValueBC, GenericBC, NoSlipBC, ZeroGradientBC
from neofoam.fields.decl import FieldDecl
from neofoam.fields.schema import schema_for
from neofoam.fields.value_types import Scalar, Vector


def _U() -> FieldDecl:
    return FieldDecl(
        name="U",
        dimensions=[0, 1, -1, 0, 0, 0, 0],
        value_type=Vector,
        allowed_bcs=(NoSlipBC, FixedValueBC),
        write=True,
    )


def _T() -> FieldDecl:
    return FieldDecl(
        name="T",
        dimensions=[0, 0, 0, 1, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=(FixedValueBC, ZeroGradientBC, GenericBC),
    )


# -- shape ------------------------------------------------------------


def test_schema_binds_io_to_field_file() -> None:
    Cls = schema_for(_U())
    assert Cls.io_config is not None
    assert Cls.io_config.file == "0/U"


def test_schema_caches_per_decl_identity() -> None:
    decl = _U()
    assert schema_for(decl) is schema_for(decl)


def test_distinct_decls_get_distinct_classes() -> None:
    assert schema_for(_U()) is not schema_for(_U())  # two FieldDecls, two classes


def test_foam_file_header_pinned_from_value_type() -> None:
    UCls = schema_for(_U())
    TCls = schema_for(_T())
    u = UCls.model_validate({})
    t = TCls.model_validate({})
    assert u.FoamFile.field_class == "volVectorField"
    assert u.FoamFile.object == "U"
    assert t.FoamFile.field_class == "volScalarField"
    assert t.FoamFile.object == "T"


def test_dimensions_default_from_declaration() -> None:
    Cls = schema_for(_U())
    instance = Cls.model_validate({})
    assert instance.dimensions == [0, 1, -1, 0, 0, 0, 0]


def test_dimensions_accept_bracket_string() -> None:
    Cls = schema_for(_U())
    instance = Cls.model_validate({"dimensions": "[ 1 -2 3 0 0 0 0 ]"})
    assert instance.dimensions == [1, -2, 3, 0, 0, 0, 0]


def test_dimensions_serialise_to_bracket_string() -> None:
    Cls = schema_for(_U())
    instance = Cls.model_validate({"dimensions": [0, 1, -1, 0, 0, 0, 0]})
    dumped = instance.model_dump(by_alias=True)
    assert dumped["dimensions"] == "[0 1 -1 0 0 0 0]"


# -- allowed_bcs gating ----------------------------------------------


def test_allowed_bcs_accept_listed_arm() -> None:
    Cls = schema_for(_U())
    instance = Cls.model_validate({"boundaryField": {"floor": {"type": "noSlip"}}})
    assert isinstance(instance.boundaryField["floor"], NoSlipBC)


def test_allowed_bcs_reject_unlisted_arm() -> None:
    Cls = schema_for(_U())  # ``allowed_bcs=(NoSlipBC, FixedValueBC)`` only
    with pytest.raises(ValidationError):
        Cls.model_validate({"boundaryField": {"floor": {"type": "zeroGradient"}}})


def test_generic_bc_fallback_lets_unknown_arms_through() -> None:
    Cls = schema_for(_T())  # includes GenericBC
    instance = Cls.model_validate(
        {
            "boundaryField": {
                "floor": {
                    "type": "fixedFluxPressure",
                    "rho": "rhok",
                    "value": "uniform 0",
                }
            }
        }
    )
    bc = instance.boundaryField["floor"]
    assert isinstance(bc, GenericBC)
    assert bc.type == "fixedFluxPressure"


def test_empty_allowed_bcs_is_error() -> None:
    decl = FieldDecl(
        name="phi",
        dimensions=[0, 3, -1, 0, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=(),
    )
    with pytest.raises(ValueError, match="allowed_bcs is empty"):
        schema_for(decl)


def test_unsupported_value_type_is_error() -> None:
    class _Bogus:
        pass

    decl = FieldDecl(
        name="bogus",
        dimensions=[0, 0, 0, 0, 0, 0, 0],
        value_type=_Bogus,
        allowed_bcs=(NoSlipBC,),
    )
    with pytest.raises(TypeError, match="unsupported value_type"):
        schema_for(decl)


# -- internalField defaults ------------------------------------------


def test_internal_field_default_zero_for_scalar() -> None:
    Cls = schema_for(_T())
    instance = Cls.model_validate({})
    assert instance.internalField == "uniform 0"


def test_internal_field_default_zero_for_vector() -> None:
    Cls = schema_for(_U())
    instance = Cls.model_validate({})
    assert instance.internalField == "uniform (0 0 0)"


def test_initial_value_baked_into_default() -> None:
    decl = FieldDecl(
        name="T",
        dimensions=[0, 0, 0, 1, 0, 0, 0],
        value_type=Scalar,
        allowed_bcs=(FixedValueBC,),
        initial_value=300.0,
    )
    instance = schema_for(decl).model_validate({})
    assert instance.internalField == "uniform 300.0"
