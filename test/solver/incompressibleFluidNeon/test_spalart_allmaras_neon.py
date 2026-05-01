# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tests for NeoN Spalart-Allmaras turbulence model."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from neofoam.solver.incompressibleFluid.models.spalartAllmaras import (
    SpalartAllmarasConfig,
)


def _check_openfoam_available() -> bool:
    try:
        result = subprocess.run(["blockMesh", "-help"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


requires_openfoam = pytest.mark.skipif(
    not _check_openfoam_available(),
    reason="OpenFOAM not available",
)

REPO_ROOT = Path(__file__).parent.parent.parent.parent


# --- Unit tests (no OpenFOAM required) ---


def test_sa_config_defaults() -> None:
    """SA config has correct default constants."""
    cfg = SpalartAllmarasConfig()
    assert cfg.Cb1 == pytest.approx(0.1355)
    assert cfg.Cb2 == pytest.approx(0.622)
    assert cfg.Cv1 == pytest.approx(7.1)
    assert cfg.sigma == pytest.approx(2.0 / 3.0)
    assert cfg.kappa == pytest.approx(0.41)


def test_sa_config_cw1_derived() -> None:
    """Cw1 is correctly computed from Cb1, kappa, Cb2, sigma."""
    cfg = SpalartAllmarasConfig()
    expected = cfg.Cb1 / cfg.kappa**2 + (1 + cfg.Cb2) / cfg.sigma
    assert cfg.Cw1 == pytest.approx(expected)


# --- Detection tests (require OpenFOAM for case setup) ---


def _convert_to_piso(test_case: Path) -> None:
    """Convert a PIMPLE case to PISO for the NeoN solver."""
    fv_solution = test_case / "system" / "fvSolution"
    content = fv_solution.read_text()

    # Replace PIMPLE section with PISO
    content = content.replace("PIMPLE", "PISO")

    # Add momentumPredictor if not present
    if "momentumPredictor" not in content:
        content = content.replace(
            "nCorrectors",
            "momentumPredictor yes;\n    nCorrectors",
        )

    # Add nNonOrthogonalCorrectors if missing
    if "nNonOrthogonalCorrectors" not in content:
        content = content.replace(
            "nCorrectors         2;",
            "nCorrectors         2;\n    nNonOrthogonalCorrectors 0;",
        )

    fv_solution.write_text(content)

    # Ensure nuTilda solver entry exists (not just "(U|nuTilda)")
    # The NeoN solver maps solvers by exact name
    fv_solution_content = fv_solution.read_text()
    if '"(U|nuTilda)"' in fv_solution_content or "'(U|nuTilda)'" in fv_solution_content:
        # Add explicit nuTilda entry matching the U solver
        fv_solution_content = fv_solution_content.replace(
            '"(U|nuTilda)"',
            "U",
        )
        # Add separate nuTilda entry
        nuTilda_entry = """
    nuTilda
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
"""
        # Insert after the U block
        fv_solution_content = fv_solution_content.replace(
            "    UFinal",
            nuTilda_entry + "    UFinal",
        )
        # Also rename UFinal references
        fv_solution_content = fv_solution_content.replace(
            '"(U|nuTilda)Final"', "UFinal"
        )
        fv_solution.write_text(fv_solution_content)

    # Disable adjustTimeStep (use fixed deltaT for PISO)
    control_dict = test_case / "system" / "controlDict"
    cd_content = control_dict.read_text()
    cd_content = cd_content.replace("adjustTimeStep  yes;", "adjustTimeStep  no;")
    control_dict.write_text(cd_content)


@requires_openfoam
def test_pitzDaily_SA_neon_runs() -> None:
    """End-to-end: NeoN solver with SA turbulence completes without error."""
    source_case = REPO_ROOT / "tutorials" / "pitzDaily_SA"
    test_case = REPO_ROOT / "test_cases" / "pitzDaily_SA_neon_e2e"

    try:
        if test_case.exists():
            shutil.rmtree(test_case)
        test_case.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_case, test_case)

        # Setup 0 from 0.orig
        orig_dir = test_case / "0.orig"
        zero_dir = test_case / "0"
        if orig_dir.exists():
            if zero_dir.exists():
                shutil.rmtree(zero_dir)
            shutil.copytree(orig_dir, zero_dir)

        subprocess.run(
            ["blockMesh", "-case", str(test_case)],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

        # Convert to PISO and set very short run
        _convert_to_piso(test_case)

        # Set very short endTime (just 2 time steps)
        control_dict = test_case / "system" / "controlDict"
        cd_content = control_dict.read_text()
        lines = cd_content.split("\n")
        new_lines = []
        for line in lines:
            if line.strip().startswith("endTime"):
                new_lines.append("endTime         0.0002;")
            elif line.strip().startswith("writeInterval"):
                new_lines.append("writeInterval   0.0002;")
            elif line.strip().startswith("writeControl"):
                new_lines.append("writeControl    timeStep;")
            else:
                new_lines.append(line)
        control_dict.write_text("\n".join(new_lines))

        # Run solver as subprocess to avoid OpenFOAM double-init segfault
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                "from neofoam.solver.incompressibleFluidNeon import run; run(['.'])",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(test_case),
            env={**os.environ, "PWD": str(test_case)},
        )
        assert result.returncode == 0, (
            f"NeoN SA solver failed:\nstdout: {result.stdout[-1000:]}\n"
            f"stderr: {result.stderr[-1000:]}"
        )

    finally:
        if test_case.exists():
            shutil.rmtree(test_case)


@requires_openfoam
def test_detect_sa_on_sa_case() -> None:
    """detect_sa() returns a model when turbulenceProperties has SA."""
    from neofoam.solver.incompressibleFluidNeon.models.turbulence.spalartAllmaras import (
        detect_sa,
    )

    source_case = REPO_ROOT / "tutorials" / "pitzDaily_SA"
    test_case = REPO_ROOT / "test_cases" / "pitzDaily_SA_detect_test"

    try:
        if test_case.exists():
            shutil.rmtree(test_case)
        test_case.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_case, test_case)

        # Setup 0 from 0.orig
        orig_dir = test_case / "0.orig"
        zero_dir = test_case / "0"
        if orig_dir.exists():
            if zero_dir.exists():
                shutil.rmtree(zero_dir)
            shutil.copytree(orig_dir, zero_dir)

        subprocess.run(
            ["blockMesh", "-case", str(test_case)],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

        original_dir = Path.cwd()
        os.chdir(test_case)
        try:
            result = detect_sa()
            assert result is not None, "detect_sa() should return a model for SA case"
        finally:
            os.chdir(original_dir)

    finally:
        if test_case.exists():
            shutil.rmtree(test_case)


def test_compute_wall_distance_binding_exists() -> None:
    """compute_wall_distance should be available in neofoam_bindings."""
    from neofoam import neofoam_bindings as nfb

    assert hasattr(nfb, "compute_wall_distance"), (
        "neofoam_bindings should expose compute_wall_distance"
    )


def test_sa_neon_model_has_build() -> None:
    """SA NeoN model should have a build step returning fields."""
    from neofoam.solver.incompressibleFluidNeon.models.turbulence.spalartAllmaras import (
        sa_neon,
    )

    assert hasattr(sa_neon, "_build_func"), "SA model should have a build function"
    assert sa_neon._build_func is not None


def test_sa_build_includes_div_dev_reff_correction() -> None:
    """SA build step should include div_dev_reff_correction field."""
    from neofoam.solver.incompressibleFluidNeon.models.turbulence.spalartAllmaras import (
        sa_neon,
    )

    # Call the build function to get the list of InitSteps
    steps = sa_neon._build_func(sa_neon)
    step_names = [s.name for s in steps]
    assert "fields.div_dev_reff_correction" in step_names, (
        f"SA build should include div_dev_reff_correction field, got: {step_names}"
    )


def test_sa_neon_model_has_operations() -> None:
    """SA NeoN model should have turbulence_correction operation via operation_collection."""
    from neofoam.solver.incompressibleFluidNeon.models.turbulence.spalartAllmaras import (
        sa_neon,
    )

    assert sa_neon._operation_collection_func is not None, (
        "SA model should have operation_collection"
    )


@requires_openfoam
def test_detect_sa_on_laminar_case() -> None:
    """detect_sa() returns None when no turbulenceProperties or laminar."""
    from neofoam.solver.incompressibleFluidNeon.models.turbulence.spalartAllmaras import (
        detect_sa,
    )

    source_case = REPO_ROOT / "tutorials" / "cavity"
    test_case = REPO_ROOT / "test_cases" / "cavity_detect_test"

    try:
        if test_case.exists():
            shutil.rmtree(test_case)
        test_case.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_case, test_case)

        orig_dir = test_case / "0.orig"
        zero_dir = test_case / "0"
        if orig_dir.exists():
            if zero_dir.exists():
                shutil.rmtree(zero_dir)
            shutil.copytree(orig_dir, zero_dir)

        subprocess.run(
            ["blockMesh", "-case", str(test_case)],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

        original_dir = Path.cwd()
        os.chdir(test_case)
        try:
            result = detect_sa()
            assert result is None, "detect_sa() should return None for laminar case"
        finally:
            os.chdir(original_dir)

    finally:
        if test_case.exists():
            shutil.rmtree(test_case)
