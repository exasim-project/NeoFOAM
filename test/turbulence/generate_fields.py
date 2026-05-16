# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Analytical field generators and OpenFOAM field writers for turbulence tests.

Used by conftest.py to create non-uniform initial conditions at test time.
"""

from pathlib import Path
from typing import IO

import numpy as np


# ---------------------------------------------------------------------------
# Analytical field generators
# ---------------------------------------------------------------------------


def compute_velocity(cc: np.ndarray) -> np.ndarray:
    """Parabolic channel flow profile with cross-flow perturbation.

    Non-zero curl ensures vorticity magnitude Omega != 0, exercising
    the SA production term.

    Returns shape (N, 3).
    """
    x, y = cc[:, 0], cc[:, 1]
    n = len(x)
    h = 0.0254  # channel half-height

    y_norm = np.clip(y / h, -1.0, 1.0)
    U_x = 10.0 * np.maximum(1.0 - y_norm**2, 0.01)
    U_y = 0.5 * np.sin(2.0 * np.pi * x / 0.15) * y_norm
    U_z = np.zeros(n)
    return np.column_stack([U_x, U_y, U_z])


def compute_nuTilda(cc: np.ndarray, nu: float = 1e-5) -> np.ndarray:
    """nuTilda varying with position: higher away from center.

    Returns shape (N,).
    """
    y = cc[:, 1]
    h = 0.0254
    y_norm = np.clip(np.abs(y) / h, 0.01, 1.0)
    return nu * (3.0 + 50.0 * y_norm)


def compute_k(cc: np.ndarray, nu: float = 1e-5) -> np.ndarray:
    """Turbulent kinetic energy varying with position.

    Higher k away from walls, lower near walls.
    Returns shape (N,).
    """
    y = cc[:, 1]
    h = 0.0254
    y_norm = np.clip(np.abs(y) / h, 0.01, 1.0)
    # k ranges from ~0.001 near walls to ~0.5 in core
    return 0.001 + 0.5 * (1.0 - np.exp(-5.0 * y_norm))


def compute_epsilon(cc: np.ndarray, nu: float = 1e-5) -> np.ndarray:
    """Dissipation rate varying with position.

    Higher epsilon near walls, lower in core.
    Returns shape (N,).
    """
    y = cc[:, 1]
    h = 0.0254
    y_norm = np.clip(np.abs(y) / h, 0.01, 1.0)
    k_vals = compute_k(cc, nu)
    # epsilon ~ k^(3/2) / l, with l ~ y near wall
    # Simplified: higher near walls
    return 0.09 * k_vals**2 / (nu * (1.0 + 100.0 * y_norm))


def compute_nut_from_k_epsilon(
    k: np.ndarray, epsilon: np.ndarray, Cmu: float = 0.09,
) -> np.ndarray:
    """nut = Cmu * k² / epsilon"""
    return Cmu * k**2 / np.maximum(epsilon, 1e-20)


def compute_nut_from_nuTilda(
    nuTilda: np.ndarray, nu: float = 1e-5, Cv1: float = 7.1,
) -> np.ndarray:
    """nut = nuTilda * fv1, consistent with SA model.

    Returns shape (N,).
    """
    chi = nuTilda / nu
    fv1 = chi**3 / (chi**3 + Cv1**3)
    return nuTilda * fv1


# ---------------------------------------------------------------------------
# OpenFOAM field file writers
# ---------------------------------------------------------------------------


def _write_header(f: IO[str], field_class: str, name: str) -> None:
    f.write(
        f"FoamFile\n{{\n"
        f"    version     2.0;\n"
        f"    format      ascii;\n"
        f"    class       {field_class};\n"
        f"    object      {name};\n"
        f"}}\n"
    )


def _read_boundary_block(filepath: Path) -> str:
    """Extract boundaryField {...} block from an existing OpenFOAM field file."""
    text = filepath.read_text()
    idx = text.find("boundaryField")
    if idx < 0:
        msg = f"No boundaryField in {filepath}"
        raise ValueError(msg)
    return text[idx:]


def write_scalar_field(
    filepath: Path,
    name: str,
    dimensions: str,
    values: np.ndarray,
    boundary_text: str,
) -> None:
    """Write nonuniform volScalarField in OpenFOAM format."""
    with open(filepath, "w") as f:
        _write_header(f, "volScalarField", name)
        f.write(f"dimensions      {dimensions};\n\n")
        f.write(f"internalField   nonuniform List<scalar>\n{len(values)}\n(\n")
        for v in values:
            f.write(f"{v:.15e}\n")
        f.write(")\n;\n\n")
        f.write(boundary_text)


def write_vector_field(
    filepath: Path,
    name: str,
    dimensions: str,
    values: np.ndarray,
    boundary_text: str,
) -> None:
    """Write nonuniform volVectorField in OpenFOAM format."""
    with open(filepath, "w") as f:
        _write_header(f, "volVectorField", name)
        f.write(f"dimensions      {dimensions};\n\n")
        f.write(f"internalField   nonuniform List<vector>\n{len(values)}\n(\n")
        for vx, vy, vz in values:
            f.write(f"({vx:.15e} {vy:.15e} {vz:.15e})\n")
        f.write(")\n;\n\n")
        f.write(boundary_text)
