# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Typed boundary-condition arms used by per-field schemas.

Every BC arm is a Pydantic model carrying at minimum a ``type`` literal
that doubles as the discriminator. :func:`build_bc_union` returns the
``Annotated[Union[...], Field(discriminator="type")]`` annotation a
per-field schema's ``boundaryField`` uses.

Two typed arms ship with Phase 1:

* :class:`NoSlipBC` — ``noSlip`` wall BC, no extra parameters.
* :class:`FixedValueBC` — ``fixedValue`` BC carrying a scalar / vector
  / OpenFOAM-style uniform literal value.

:class:`GenericBC` is the catch-all fallback. It keeps any unknown BC
parsing — ``zeroGradient``, ``fixedFluxPressure``, wall-functions, … —
through ``extra="allow"`` so the LLM (or a hand-written case) never
hits a schema wall on unseen BC types. Adding a new typed arm only
means appending another ``BaseModel`` subclass and listing it in the
relevant model's ``allowed_bcs=[...]``.

A small :class:`ZeroGradientBC` is provided as a second concrete arm
because it appears in every tutorial; it's still optional from the
field's POV (each field's ``allowed_bcs`` decides what's permitted).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Sequence, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    model_serializer,
)

from neofoam.fields.value_types import (
    FieldValue,
    Scalar,
    Vector,
    to_uniform_literal,
)


class NoSlipBC(BaseModel):
    """OpenFOAM ``noSlip`` wall BC.

    Carries no parameters; serialises to ``{ type noSlip; }``.
    """

    type: Literal["noSlip"] = "noSlip"


class FixedValueBC(BaseModel):
    """OpenFOAM ``fixedValue`` BC.

    The ``value`` is stored as it appears on disk — an OpenFOAM uniform
    literal (``"uniform 0"`` / ``"uniform (0 0 0)"``) or a bare Python
    scalar / list for ergonomic Python use; the per-field schema's
    serializer in Step 4 normalises the wire form.
    """

    type: Literal["fixedValue"] = "fixedValue"
    value: FieldValue[Any]


class ZeroGradientBC(BaseModel):
    """OpenFOAM ``zeroGradient`` BC. No parameters."""

    type: Literal["zeroGradient"] = "zeroGradient"


# ---------------------------------------------------------------------------
# Topology-only arms (no parameters; just the ``type`` discriminator). These
# arms cover ~30% of all BC instances in the upstream OpenFOAM tutorials —
# ``empty`` alone is ~14% (every 1D/2D / wedge case carries it on every
# field), and ``slip`` / ``symmetry`` / ``symmetryPlane`` / ``cyclic`` round
# out the universal set used on both volScalarField and volVectorField.
# ---------------------------------------------------------------------------


class EmptyBC(BaseModel):
    """OpenFOAM ``empty``. Topology marker for the empty direction in
    1D / 2D meshes (wedges, slabs). Carries no parameters."""

    type: Literal["empty"] = "empty"


class SlipBC(BaseModel):
    """OpenFOAM ``slip``. Free-slip wall — normal velocity zero,
    tangential unconstrained. No parameters."""

    type: Literal["slip"] = "slip"


class SymmetryBC(BaseModel):
    """OpenFOAM ``symmetry``. Mirror BC for a generic-shape patch."""

    type: Literal["symmetry"] = "symmetry"


class SymmetryPlaneBC(BaseModel):
    """OpenFOAM ``symmetryPlane``. Mirror BC for a flat patch
    (cheaper than the generic ``symmetry`` arm)."""

    type: Literal["symmetryPlane"] = "symmetryPlane"


class CyclicBC(BaseModel):
    """OpenFOAM ``cyclic``. Periodic BC linked to a counterpart patch
    in the polyMesh. No payload beyond ``type``."""

    type: Literal["cyclic"] = "cyclic"


class CyclicAMIBC(BaseModel):
    """OpenFOAM ``cyclicAMI``. Periodic BC with Arbitrary Mesh
    Interface — used between rotating / sliding patches that do not
    share a node-conformal mesh. Carries an optional ``value``
    placeholder some tutorials use to seed the AMI's initial state."""

    type: Literal["cyclicAMI"] = "cyclicAMI"
    value: Optional[FieldValue[Any]] = None


# ---------------------------------------------------------------------------
# Value-carrying arms — the patch reads ``value`` (sometimes
# ``inletValue`` / ``rho``) every time it is written.
# ---------------------------------------------------------------------------


class CalculatedBC(BaseModel):
    """OpenFOAM ``calculated``. Derived BC — ``value`` is computed by
    another part of the solver (e.g. wall-function ``nut``); the
    on-disk placeholder keeps the patch read/writeable."""

    type: Literal["calculated"] = "calculated"
    value: FieldValue[Any]


class InletOutletBC(BaseModel):
    """OpenFOAM ``inletOutlet``. Switches between ``inletValue`` when
    flux is into the domain and ``zeroGradient`` when out. Both
    payload fields accept the bare Python form (``float`` / ``list``)
    or an OpenFOAM uniform / ``$internalField`` literal."""

    type: Literal["inletOutlet"] = "inletOutlet"
    inletValue: FieldValue[Any]
    value: FieldValue[Any]


class FixedFluxPressureBC(BaseModel):
    """OpenFOAM ``fixedFluxPressure``. Pressure BC that balances the
    flux through the patch given the density-weighted body force —
    standard for ``p_rgh`` in Boussinesq buoyancy cases. ``rho`` names
    the density field (e.g. ``"rhok"``) when needed."""

    type: Literal["fixedFluxPressure"] = "fixedFluxPressure"
    value: FieldValue[Scalar]
    rho: Optional[str] = None


class PressureInletOutletVelocityBC(BaseModel):
    """OpenFOAM ``pressureInletOutletVelocity``. Velocity counterpart
    to a pressure inlet/outlet — tangential velocity zero on inflow,
    zero-gradient on outflow. Vector-only."""

    type: Literal["pressureInletOutletVelocity"] = "pressureInletOutletVelocity"
    value: FieldValue[Vector]
    inletValue: Optional[FieldValue[Vector]] = None


class MovingWallVelocityBC(BaseModel):
    """OpenFOAM ``movingWallVelocity``. No-slip wall whose velocity
    follows the mesh motion (rotating shafts, oscillating bodies,
    sliding meshes). Vector-only. The ``value`` placeholder is what
    the field initial-condition reader sees; the BC overrides it at
    every time step from the mesh velocity."""

    type: Literal["movingWallVelocity"] = "movingWallVelocity"
    value: FieldValue[Vector]


# ---------------------------------------------------------------------------
# Turbulence wall functions — k / epsilon / omega / nut all share the same
# ``(type, value)`` payload across their wall-function variants, so a single
# typed arm with a multi-string ``Literal`` discriminator covers the family
# without one class per name. The PIMPLE-family tutorial survey shows these
# six dominate: ``epsilonWallFunction`` (45% of epsilon patches),
# ``kqRWallFunction`` (37% of k), ``nutkWallFunction`` (30% of nut),
# ``omegaWallFunction`` (24% of omega), plus the ``nutUSpaldingWallFunction``
# / ``nutLowReWallFunction`` variants for low-Re cases.
# ---------------------------------------------------------------------------


class WallFunctionBC(BaseModel):
    """OpenFOAM turbulence wall-function BCs sharing ``(type, value)``.

    Covers ``kqRWallFunction`` (k at walls), ``nutkWallFunction`` /
    ``nutUSpaldingWallFunction`` / ``nutLowReWallFunction`` (nut),
    ``epsilonWallFunction``, and ``omegaWallFunction``. The wall
    function picks which RAS variant the solver applies at the patch;
    the on-disk payload is identical across variants, so they share
    one typed arm.
    """

    type: Literal[
        "kqRWallFunction",
        "nutkWallFunction",
        "nutUSpaldingWallFunction",
        "nutLowReWallFunction",
        "epsilonWallFunction",
        "omegaWallFunction",
    ]
    value: FieldValue[Scalar]


class AlphatWallFunctionBC(BaseModel):
    """Thermal-diffusivity wall functions for ``alphat``.

    The buoyant-flow tutorials use two variants on ``alphat`` patches:
    ``compressible::alphatWallFunction`` (incl. the ``compressible::``
    namespace prefix from OpenFOAM) and ``alphatJayatillekeWallFunction``
    (the Jayatilleke thermal wall law). Both share the same
    ``(type, value, Prt?)`` shape; ``Prt`` is the optional turbulent
    Prandtl number some variants accept.
    """

    type: Literal[
        "compressible::alphatWallFunction",
        "alphatJayatillekeWallFunction",
    ]
    value: FieldValue[Scalar]
    Prt: Optional[float] = None


# ---------------------------------------------------------------------------
# Turbulence inlet specifiers — set k / epsilon / omega from an inlet
# intensity or mixing-length plus a placeholder ``value``. Each arm has a
# different parameter (``intensity`` vs ``mixingLength``) so they get
# their own typed classes.
# ---------------------------------------------------------------------------


class TurbulentIntensityKineticEnergyInletBC(BaseModel):
    """OpenFOAM ``turbulentIntensityKineticEnergyInlet``.

    Scalar inlet BC for ``k`` that computes its value from the local
    velocity magnitude and a prescribed turbulence ``intensity``
    (dimensionless fraction). The on-disk ``value`` is a placeholder
    used until the first update from the inlet ``U``.
    """

    type: Literal["turbulentIntensityKineticEnergyInlet"] = (
        "turbulentIntensityKineticEnergyInlet"
    )
    intensity: float
    value: FieldValue[Scalar]


class TurbulentMixingLengthDissipationRateInletBC(BaseModel):
    """OpenFOAM ``turbulentMixingLengthDissipationRateInlet``.

    Scalar inlet BC for ``epsilon`` that computes its value from the
    local ``k`` and a prescribed ``mixingLength`` (length scale in m).
    Pairs with :class:`TurbulentIntensityKineticEnergyInletBC` on the
    upstream k field.
    """

    type: Literal["turbulentMixingLengthDissipationRateInlet"] = (
        "turbulentMixingLengthDissipationRateInlet"
    )
    mixingLength: float
    value: FieldValue[Scalar]


class TurbulentMixingLengthFrequencyInletBC(BaseModel):
    """OpenFOAM ``turbulentMixingLengthFrequencyInlet``.

    Scalar inlet BC for ``omega``, k-omega's counterpart to the
    dissipation-rate inlet. Computes ``omega`` from the local ``k``
    and a prescribed ``mixingLength``.
    """

    type: Literal["turbulentMixingLengthFrequencyInlet"] = (
        "turbulentMixingLengthFrequencyInlet"
    )
    mixingLength: float
    value: FieldValue[Scalar]


class GenericBC(BaseModel):
    """Fallback BC arm: any ``type`` string, free-form extra fields.

    Allows the schema to consume BCs without a dedicated typed arm
    (``fixedFluxPressure``, ``kqRWallFunction``, …) without losing the
    payload. The OpenFOAM strategy's ``ConfigDict(extra="allow")`` path
    already round-trips dict-shaped extras unchanged.

    A non-fallback ``allowed_bcs=[...]`` that *omits* :class:`GenericBC`
    will reject any unknown ``type`` at validation; including it adds
    the fallback. See :func:`build_bc_union`.
    """

    model_config = ConfigDict(extra="allow")
    type: str

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any, info: SerializationInfo) -> dict[str, Any]:
        """Map open-set values to OpenFOAM literals when writing OpenFOAM.

        ``GenericBC``'s payload is untyped ``extra`` keys, so the per-field
        :data:`FieldValue` serializer can't reach them; mirror it here — render
        any list/number value as a ``uniform`` literal under the ``openfoam``
        format context, pass everything else (incl. ``type`` and string values)
        through. Plain ``model_dump`` is unchanged.
        """
        data: dict[str, Any] = handler(self)
        if (info.context or {}).get("format") == "openfoam":
            return {key: to_uniform_literal(val) for key, val in data.items()}
        return data


#: Public union type alias for typing call sites that want the open set.
#: ``build_bc_union`` returns the equivalent ``Annotated[...]`` annotation
#: parameterised by a caller-chosen arm list, so this alias is only used
#: where every concrete arm is acceptable (e.g. the agent's case-spec
#: aggregate; not for individual field schemas, which constrain the
#: arms). ``GenericBC`` is intentionally NOT in this alias — it would
#: collapse the discriminator dispatch into the smart-union fallback
#: (its ``type: str`` is not a literal); callers that want the open
#: set with the fallback pass ``[..., GenericBC]`` to
#: :func:`build_bc_union` directly.
PatchBC = Annotated[
    Union[
        NoSlipBC,
        FixedValueBC,
        ZeroGradientBC,
        EmptyBC,
        SlipBC,
        SymmetryBC,
        SymmetryPlaneBC,
        CyclicBC,
        CyclicAMIBC,
        CalculatedBC,
        InletOutletBC,
        FixedFluxPressureBC,
        PressureInletOutletVelocityBC,
        MovingWallVelocityBC,
        WallFunctionBC,
        AlphatWallFunctionBC,
        TurbulentIntensityKineticEnergyInletBC,
        TurbulentMixingLengthDissipationRateInletBC,
        TurbulentMixingLengthFrequencyInletBC,
    ],
    Field(discriminator="type"),
]


def build_bc_union(arms: Sequence[type[BaseModel]]) -> Any:
    """Return a discriminated-union annotation over the given BC ``arms``.

    The returned annotation is suitable for a Pydantic model field:

    .. code-block:: python

        bc = build_bc_union([NoSlipBC, FixedValueBC])

        class Schema(BaseModel):
            boundaryField: dict[str, bc]

    Discrimination is on the ``type`` field. The arms must therefore
    each declare ``type: Literal["…"]`` (the bundled :class:`NoSlipBC` /
    :class:`FixedValueBC` / :class:`ZeroGradientBC` arms already do; the
    fallback :class:`GenericBC` uses ``type: str``).

    Pydantic's discriminator dispatch is exact-match on the literal,
    so :class:`GenericBC` — whose ``type`` is ``str``, not a literal —
    cannot share a discriminated union with literal arms. When
    :class:`GenericBC` is included, this function builds a plain
    ``Union[...]`` instead: validation still dispatches on ``type``
    via Pydantic's smart-union mode, with :class:`GenericBC` acting as
    the "anything else" fallback.

    Args:
        arms: BC classes to include. Order is preserved; deduplicated
            on identity (a duplicate class is a no-op).

    Returns:
        An ``Annotated[Union[...]]`` for the homogeneous literal-only
        case, or a bare ``Union[...]`` when :class:`GenericBC` is
        included. Both are valid pydantic field annotations.

    Raises:
        ValueError: if ``arms`` is empty.
    """
    if not arms:
        raise ValueError("build_bc_union: at least one arm is required")

    # Dedupe while preserving order.
    seen: list[type[BaseModel]] = []
    for arm in arms:
        if arm not in seen:
            seen.append(arm)

    # Union with a single arm is just that arm (Union[X] == X in typing
    # semantics, but Pydantic still accepts a single-arm Union; keep it
    # uniform for callers that introspect).
    if len(seen) == 1:
        return seen[0]

    union = Union[tuple(seen)]  # type: ignore[valid-type]
    has_generic = any(arm is GenericBC for arm in seen)
    if has_generic:
        # Smart-union dispatch — GenericBC's ``type: str`` is incompatible
        # with ``Field(discriminator="type")``.
        return union
    return Annotated[union, Field(discriminator="type")]
