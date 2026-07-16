# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for the BC discriminated union (``neofoam.fields.bc``).

Round-trip every typed arm through ``model_validate`` / ``model_dump``,
verify discriminator dispatch on a union, and assert the fallback /
restrictive behaviour of :class:`GenericBC` inclusion.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from neofoam.fields.bc import (
    AlphatWallFunctionBC,
    CalculatedBC,
    CyclicAMIBC,
    CyclicBC,
    EmptyBC,
    FixedFluxPressureBC,
    FixedValueBC,
    GenericBC,
    InletOutletBC,
    MovingWallVelocityBC,
    NoSlipBC,
    PressureInletOutletVelocityBC,
    SlipBC,
    SymmetryBC,
    SymmetryPlaneBC,
    TurbulentIntensityKineticEnergyInletBC,
    TurbulentMixingLengthDissipationRateInletBC,
    TurbulentMixingLengthFrequencyInletBC,
    WallFunctionBC,
    ZeroGradientBC,
    build_bc_union,
)


# -- arms round-trip --------------------------------------------------


def test_no_slip_round_trip() -> None:
    bc = NoSlipBC.model_validate({"type": "noSlip"})
    assert bc.model_dump() == {"type": "noSlip"}


def test_fixed_value_scalar_round_trip() -> None:
    bc = FixedValueBC.model_validate({"type": "fixedValue", "value": 0.0})
    assert bc.value == 0.0
    assert bc.model_dump() == {"type": "fixedValue", "value": 0.0}


def test_fixed_value_uniform_literal_round_trip() -> None:
    bc = FixedValueBC.model_validate({"type": "fixedValue", "value": "uniform 0"})
    assert bc.value == "uniform 0"


def test_fixed_value_vector_list_round_trip() -> None:
    bc = FixedValueBC.model_validate({"type": "fixedValue", "value": [0.0, 0.0, 0.0]})
    assert bc.value == [0.0, 0.0, 0.0]


def test_zero_gradient_round_trip() -> None:
    bc = ZeroGradientBC.model_validate({"type": "zeroGradient"})
    assert bc.model_dump() == {"type": "zeroGradient"}


def test_generic_bc_keeps_extras() -> None:
    """Unknown BC types parse via GenericBC; extras pass through."""
    payload = {"type": "fixedFluxPressure", "rho": "rhok", "value": "uniform 0"}
    bc = GenericBC.model_validate(payload)
    assert bc.type == "fixedFluxPressure"
    dumped = bc.model_dump()
    # extra="allow" → ``rho`` / ``value`` survive the round-trip.
    assert dumped == payload


# -- build_bc_union dispatch -----------------------------------------


def _wrap(union: object) -> type[BaseModel]:
    """Build a tiny pydantic model whose ``bc`` field uses ``union``."""

    class _Holder(BaseModel):
        bc: union  # type: ignore[valid-type]

    return _Holder


def test_build_bc_union_single_arm_is_arm() -> None:
    assert build_bc_union([NoSlipBC]) is NoSlipBC


def test_build_bc_union_dedupes() -> None:
    Holder = _wrap(build_bc_union([NoSlipBC, FixedValueBC, NoSlipBC]))
    instance = Holder.model_validate({"bc": {"type": "noSlip"}})
    assert isinstance(instance.bc, NoSlipBC)


def test_build_bc_union_discriminated_dispatch() -> None:
    """Literal-only arms dispatch via ``Field(discriminator="type")``."""
    Holder = _wrap(build_bc_union([NoSlipBC, FixedValueBC]))

    no_slip = Holder.model_validate({"bc": {"type": "noSlip"}})
    assert isinstance(no_slip.bc, NoSlipBC)

    fv = Holder.model_validate({"bc": {"type": "fixedValue", "value": 1.0}})
    assert isinstance(fv.bc, FixedValueBC)
    assert fv.bc.value == 1.0


def test_build_bc_union_rejects_unknown_without_generic() -> None:
    """No ``GenericBC`` → unknown ``type`` is a validation error."""
    Holder = _wrap(build_bc_union([NoSlipBC, FixedValueBC]))
    with pytest.raises(ValidationError):
        Holder.model_validate({"bc": {"type": "zeroGradient"}})


def test_build_bc_union_accepts_unknown_via_generic() -> None:
    """Including ``GenericBC`` switches to smart-union; unknowns parse."""
    Holder = _wrap(build_bc_union([NoSlipBC, FixedValueBC, GenericBC]))
    instance = Holder.model_validate(
        {"bc": {"type": "fixedFluxPressure", "rho": "rhok", "value": "uniform 0"}}
    )
    # Pydantic smart-union picks GenericBC when no literal arm matches.
    assert isinstance(instance.bc, GenericBC)
    assert instance.bc.type == "fixedFluxPressure"


def test_build_bc_union_known_arms_win_over_generic() -> None:
    """When a literal arm matches the discriminator, generic is bypassed."""
    Holder = _wrap(build_bc_union([NoSlipBC, FixedValueBC, GenericBC]))
    instance = Holder.model_validate({"bc": {"type": "noSlip"}})
    assert isinstance(instance.bc, NoSlipBC)


def test_build_bc_union_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one arm is required"):
        build_bc_union([])


# -- dict[str, PatchBC] shape that boundaryField will use ------------


def test_boundary_field_dict_shape() -> None:
    """Mirror the shape a per-field schema's ``boundaryField`` carries."""
    bc_type = build_bc_union([NoSlipBC, FixedValueBC, GenericBC])

    class FieldFile(BaseModel):
        boundaryField: dict[str, bc_type]  # type: ignore[valid-type]

    file = FieldFile.model_validate(
        {
            "boundaryField": {
                "floor": {"type": "noSlip"},
                "ceiling": {"type": "fixedValue", "value": 1.0},
                "fixedWalls": {
                    "type": "fixedFluxPressure",
                    "rho": "rhok",
                    "value": "uniform 0",
                },
            }
        }
    )
    assert isinstance(file.boundaryField["floor"], NoSlipBC)
    assert isinstance(file.boundaryField["ceiling"], FixedValueBC)
    assert isinstance(file.boundaryField["fixedWalls"], GenericBC)

    redumped = file.model_dump()
    assert redumped["boundaryField"]["fixedWalls"]["rho"] == "rhok"


# -- topology-only arms ----------------------------------------------


@pytest.mark.parametrize(
    "cls,name",
    [
        (EmptyBC, "empty"),
        (SlipBC, "slip"),
        (SymmetryBC, "symmetry"),
        (SymmetryPlaneBC, "symmetryPlane"),
        (CyclicBC, "cyclic"),
    ],
)
def test_topology_arms_round_trip(cls: type, name: str) -> None:
    bc = cls.model_validate({"type": name})
    assert bc.model_dump() == {"type": name}


# -- value-carrying arms ---------------------------------------------


def test_calculated_round_trip() -> None:
    bc = CalculatedBC.model_validate({"type": "calculated", "value": "uniform 0"})
    assert bc.value == "uniform 0"


def test_inlet_outlet_round_trip_scalar() -> None:
    payload = {
        "type": "inletOutlet",
        "inletValue": "uniform 200",
        "value": "uniform 200",
    }
    bc = InletOutletBC.model_validate(payload)
    assert bc.inletValue == "uniform 200"
    assert bc.value == "uniform 200"
    assert bc.model_dump() == payload


def test_inlet_outlet_round_trip_vector_list() -> None:
    bc = InletOutletBC.model_validate(
        {"type": "inletOutlet", "inletValue": [0.0, 0.0, 0.0], "value": [1.0, 0.0, 0.0]}
    )
    assert bc.value == [1.0, 0.0, 0.0]


def test_fixed_flux_pressure_with_rho() -> None:
    """The buoyancy-tutorial shape: type + rho + value."""
    payload = {"type": "fixedFluxPressure", "rho": "rhok", "value": "uniform 0"}
    bc = FixedFluxPressureBC.model_validate(payload)
    assert bc.rho == "rhok"
    assert bc.model_dump() == payload


def test_fixed_flux_pressure_without_rho() -> None:
    bc = FixedFluxPressureBC.model_validate(
        {"type": "fixedFluxPressure", "value": "uniform 0"}
    )
    assert bc.rho is None


def test_pressure_inlet_outlet_velocity_round_trip() -> None:
    bc = PressureInletOutletVelocityBC.model_validate(
        {"type": "pressureInletOutletVelocity", "value": [0.0, 0.0, 0.0]}
    )
    assert bc.value == [0.0, 0.0, 0.0]
    assert bc.inletValue is None


def test_moving_wall_velocity_round_trip() -> None:
    bc = MovingWallVelocityBC.model_validate(
        {"type": "movingWallVelocity", "value": "uniform (0 0 0)"}
    )
    assert bc.value == "uniform (0 0 0)"


def test_cyclic_ami_with_value() -> None:
    bc = CyclicAMIBC.model_validate({"type": "cyclicAMI", "value": "uniform (0 0 0)"})
    assert bc.value == "uniform (0 0 0)"


def test_cyclic_ami_without_value() -> None:
    bc = CyclicAMIBC.model_validate({"type": "cyclicAMI"})
    assert bc.value is None


# -- turbulence wall functions (single-class, multi-Literal type) ----


@pytest.mark.parametrize(
    "type_name",
    [
        "kqRWallFunction",
        "nutkWallFunction",
        "nutUSpaldingWallFunction",
        "nutLowReWallFunction",
        "epsilonWallFunction",
        "omegaWallFunction",
    ],
)
def test_wall_function_round_trip(type_name: str) -> None:
    bc = WallFunctionBC.model_validate({"type": type_name, "value": "uniform 0"})
    assert bc.type == type_name
    assert bc.value == "uniform 0"


def test_wall_function_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        WallFunctionBC.model_validate({"type": "myCustomWF", "value": "uniform 0"})


@pytest.mark.parametrize(
    "type_name",
    ["compressible::alphatWallFunction", "alphatJayatillekeWallFunction"],
)
def test_alphat_wall_function_round_trip(type_name: str) -> None:
    bc = AlphatWallFunctionBC.model_validate(
        {"type": type_name, "value": "uniform 0", "Prt": 0.85}
    )
    assert bc.type == type_name
    assert bc.Prt == 0.85


def test_alphat_wall_function_without_prt() -> None:
    bc = AlphatWallFunctionBC.model_validate(
        {"type": "compressible::alphatWallFunction", "value": "uniform 0"}
    )
    assert bc.Prt is None


# -- turbulence inlet specifiers -------------------------------------


def test_turbulent_intensity_k_inlet() -> None:
    bc = TurbulentIntensityKineticEnergyInletBC.model_validate(
        {
            "type": "turbulentIntensityKineticEnergyInlet",
            "intensity": 0.05,
            "value": "uniform 1",
        }
    )
    assert bc.intensity == 0.05
    assert bc.value == "uniform 1"


def test_turbulent_mixing_length_dissipation_inlet() -> None:
    bc = TurbulentMixingLengthDissipationRateInletBC.model_validate(
        {
            "type": "turbulentMixingLengthDissipationRateInlet",
            "mixingLength": 0.005,
            "value": "$internalField",
        }
    )
    assert bc.mixingLength == 0.005


def test_turbulent_mixing_length_frequency_inlet() -> None:
    bc = TurbulentMixingLengthFrequencyInletBC.model_validate(
        {
            "type": "turbulentMixingLengthFrequencyInlet",
            "mixingLength": 0.005,
            "value": "$internalField",
        }
    )
    assert bc.mixingLength == 0.005


# -- discriminated-union dispatch across the new arms -----------------


def test_build_bc_union_dispatches_to_wall_function() -> None:
    """A union including the multi-literal WallFunctionBC routes
    every covered type string to that arm."""
    Holder = _wrap(build_bc_union([FixedValueBC, WallFunctionBC]))
    for tn in ("kqRWallFunction", "nutkWallFunction", "epsilonWallFunction"):
        inst = Holder.model_validate({"bc": {"type": tn, "value": "uniform 0"}})
        assert isinstance(inst.bc, WallFunctionBC)
        assert inst.bc.type == tn


def test_build_bc_union_disambiguates_fixed_value_and_fixed_flux() -> None:
    Holder = _wrap(build_bc_union([FixedValueBC, FixedFluxPressureBC]))
    fv = Holder.model_validate({"bc": {"type": "fixedValue", "value": 0.0}})
    ff = Holder.model_validate(
        {"bc": {"type": "fixedFluxPressure", "rho": "rhok", "value": "uniform 0"}}
    )
    assert isinstance(fv.bc, FixedValueBC)
    assert isinstance(ff.bc, FixedFluxPressureBC)
