# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared operator matrix for the cross-backend (pybFoam vs neon) parity tests.

Each entry names one explicit finite-volume operator that both backends expose,
the value type of its result, the div-scheme variants it is compared under, and
its comparison tolerances. The runners (``run_openfoam_ops.py`` /
``run_neon_ops.py``) and ``test_operator_parity.py`` all import from here so the
matrix lives in exactly one place.

Implicit operators (``fvm.*`` vs ``nn.imp.*``) are deferred: pybFoam exposes
``fvMatrix`` opaquely, so a fair comparison needs either matrix-export bindings
on both sides or a one-step ``nfb.PDESolverScalar`` transport-solve comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

# Div-scheme variants shared by both backends. NeoN registers the ``linear``,
# ``upwind`` and ``linearUpwind`` interpolation kernels; ``{field}`` is the
# convected field name (linearUpwind needs its gradient scheme argument).
DIV_SCHEMES: dict[str, str] = {
    "linear": "Gauss linear",
    "upwind": "Gauss upwind",
    "linearUpwind": "Gauss linearUpwind grad({field})",
}


@dataclass(frozen=True)
class OpSpec:
    """One operator in the parity matrix.

    ``atol_scale`` is relative to the reference magnitude: the effective
    absolute tolerance is ``atol_scale * max(|reference|)`` — div/laplacian
    results scale with 1/V, so a fixed atol would be meaningless across meshes.
    ``z_atol`` optionally loosens the z component on 2D (empty-patch) meshes,
    matching the ``ApproxVector({1e-12, 1e-12, 1e-4})`` convention of
    ``test/operators.cpp``.
    """

    kind: str  # "vol_scalar" | "vol_vector" | "surf_scalar"
    div_schemes: tuple[str, ...] = ("linear",)
    rtol: float = 1e-9
    atol_scale: float = 1e-12
    z_atol: float | None = None


OPERATORS: dict[str, OpSpec] = {
    "interpolate_T": OpSpec(kind="surf_scalar", rtol=1e-12, atol_scale=1e-14),
    "flux_U": OpSpec(kind="surf_scalar", rtol=1e-12, atol_scale=1e-13),
    "grad_T": OpSpec(kind="vol_vector", rtol=1e-12, atol_scale=1e-12, z_atol=1e-4),
    "div_phi": OpSpec(kind="vol_scalar", rtol=1e-9, atol_scale=1e-12),
    "div_phi_T": OpSpec(
        kind="vol_scalar",
        div_schemes=("linear", "upwind", "linearUpwind"),
        rtol=1e-9,
        atol_scale=1e-12,
    ),
    "div_phi_U": OpSpec(
        kind="vol_vector",
        div_schemes=("linear", "upwind", "linearUpwind"),
        rtol=1e-9,
        atol_scale=1e-12,
        z_atol=1e-4,
    ),
    "laplacian_Gamma_T": OpSpec(kind="vol_scalar", rtol=1e-9, atol_scale=1e-10),
    "laplacian_Gamma_U": OpSpec(
        kind="vol_vector", rtol=1e-9, atol_scale=1e-10, z_atol=1e-4
    ),
}


def ops_for_scheme(scheme: str) -> list[str]:
    """Operator keys evaluated under a given div-scheme variant.

    The base ``linear`` variant runs the full matrix; the other variants only
    re-run the div operators whose scheme actually changed.
    """
    return [name for name, spec in OPERATORS.items() if scheme in spec.div_schemes]


_FOAMFILE_HEADER = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      {name};
}}
"""


def fv_schemes_text(div_scheme: str) -> str:
    """Generate the ``system/fvSchemes`` content for one div-scheme variant.

    Both backends read this same file (pybFoam via the mesh registry, neon via
    ``nfb.map_fv_schemes(rt.fv_schemes_dict)``), which is what guarantees they
    discretise with identical schemes. Laplacians use ``uncorrected`` because
    that is the only face-normal-gradient kernel NeoN registers.
    """
    div_t = DIV_SCHEMES[div_scheme].format(field="T")
    div_u = DIV_SCHEMES[div_scheme].format(field="U")
    return (
        _FOAMFILE_HEADER.format(name="fvSchemes")
        + f"""
ddtSchemes
{{
    default         Euler;
}}

gradSchemes
{{
    default         none;
    grad(T)         Gauss linear;
    grad(U)         Gauss linear;
}}

divSchemes
{{
    default         none;
    div(phi,T)      {div_t};
    div(phi,U)      {div_u};
}}

laplacianSchemes
{{
    default         none;
    laplacian(Gamma,T) Gauss linear uncorrected;
    laplacian(Gamma,U) Gauss linear uncorrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         uncorrected;
}}

// ************************************************************************* //
"""
    )
