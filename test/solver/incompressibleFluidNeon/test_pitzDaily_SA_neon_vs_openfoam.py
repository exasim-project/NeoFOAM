# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
E2E comparison test: NeoN incompressibleFluidNeon (SA) vs OpenFOAM pimpleFoam.

Runs both solvers on pitzDaily_SA for a short duration and compares
U, p, nuTilda, nut fields at the final time step.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent


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


def _setup_base_case(source_case: Path, test_case: Path) -> None:
    """Copy tutorial case, restore 0 from 0.orig, run blockMesh."""
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

    result = subprocess.run(
        ["blockMesh", "-case", str(test_case)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"blockMesh failed: {result.stderr}"


def _set_fixed_timing(
    test_case: Path, end_time: float, delta_t: float, write_interval: float
) -> None:
    """Set fixed deltaT, endTime, writeInterval and disable adjustTimeStep."""
    control_dict = test_case / "system" / "controlDict"
    content = control_dict.read_text()
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("endTime"):
            new_lines.append(f"endTime         {end_time};")
        elif stripped.startswith("deltaT"):
            new_lines.append(f"deltaT          {delta_t};")
        elif stripped.startswith("writeControl"):
            new_lines.append("writeControl    timeStep;")
        elif stripped.startswith("writeInterval"):
            new_lines.append(f"writeInterval   {write_interval};")
        elif stripped.startswith("adjustTimeStep"):
            new_lines.append("adjustTimeStep  no;")
        else:
            new_lines.append(line)
    control_dict.write_text("\n".join(new_lines))


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
    fv_solution_content = fv_solution.read_text()
    if '"(U|nuTilda)"' in fv_solution_content:
        # First rename regex entries to plain names
        fv_solution_content = fv_solution_content.replace('"(U|nuTilda)"', "U")
        fv_solution_content = fv_solution_content.replace(
            '"(U|nuTilda)Final"', "UFinal"
        )
        # Now insert nuTilda entry before UFinal
        nuTilda_entry = """
    nuTilda
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
"""
        fv_solution_content = fv_solution_content.replace(
            "    UFinal", nuTilda_entry + "    UFinal"
        )
        fv_solution.write_text(fv_solution_content)

    # Disable adjustTimeStep
    control_dict = test_case / "system" / "controlDict"
    cd_content = control_dict.read_text()
    cd_content = cd_content.replace("adjustTimeStep  yes;", "adjustTimeStep  no;")
    control_dict.write_text(cd_content)


def _ensure_pimple_single_outer(test_case: Path) -> None:
    """Ensure PIMPLE has nOuterCorrectors=1 (equivalent to PISO)."""
    fv_solution = test_case / "system" / "fvSolution"
    content = fv_solution.read_text()
    if "nOuterCorrectors" not in content:
        content = content.replace(
            "nCorrectors",
            "nOuterCorrectors 1;\n    nCorrectors",
        )
        fv_solution.write_text(content)


def _simplify_solvers_for_neon(test_case: Path) -> None:
    """Replace GAMG with PCG for pressure solver (Ginkgo doesn't support GAMG/Multigrid preconditioner config)."""
    fv_solution = test_case / "system" / "fvSolution"
    content = fv_solution.read_text()
    # Replace GAMG solver block for p with PCG
    content = content.replace(
        "solver           GAMG;\n        tolerance        1e-7;\n"
        "        relTol           0.01;\n        smoother         DICGaussSeidel;",
        "solver          PCG;\n        preconditioner  DIC;\n"
        "        tolerance       1e-7;\n        relTol          0.01;",
    )
    fv_solution.write_text(content)


def _simplify_schemes_for_neon(test_case: Path) -> None:
    """Simplify fvSchemes to NeoN-supported subset (upwind/linear, uncorrected)."""
    fv_schemes = test_case / "system" / "fvSchemes"
    content = fv_schemes.read_text()
    # div(phi,U): linearUpwind -> upwind
    content = content.replace("Gauss linearUpwind grad(U)", "Gauss upwind")
    # laplacian: corrected -> uncorrected
    content = content.replace("Gauss linear corrected", "Gauss linear uncorrected")
    # snGrad: corrected -> uncorrected
    content = content.replace("default         corrected", "default         uncorrected")
    fv_schemes.write_text(content)


def _replace_wall_functions(test_case: Path) -> None:
    """Replace nutUSpaldingWallFunction with fixedValue for NeoN compatibility."""
    nut_file = test_case / "0" / "nut"
    if nut_file.exists():
        content = nut_file.read_text()
        content = content.replace("nutUSpaldingWallFunction", "fixedValue")
        nut_file.write_text(content)


def _add_executor_config(test_case: Path) -> None:
    """Add 'executor CPU;' to controlDict for NeoN runtime adapter."""
    control_dict = test_case / "system" / "controlDict"
    content = control_dict.read_text()
    if "executor" not in content:
        # Insert after FoamFile block
        content = content.replace(
            "// * * * * * * * * * * * *",
            "executor        CPU;\n\n// * * * * * * * * * * * *",
        )
        control_dict.write_text(content)


def _setup_neon_case(
    source_case: Path,
    test_case: Path,
    end_time: float,
    delta_t: float,
    write_interval: float,
) -> None:
    """Setup NeoN case: copy, blockMesh, convert PIMPLE->PISO, fix timing."""
    _setup_base_case(source_case, test_case)
    _convert_to_piso(test_case)
    _replace_wall_functions(test_case)
    _simplify_solvers_for_neon(test_case)
    _simplify_schemes_for_neon(test_case)
    _add_executor_config(test_case)
    _set_fixed_timing(test_case, end_time, delta_t, write_interval)


def _setup_openfoam_case(
    source_case: Path,
    test_case: Path,
    end_time: float,
    delta_t: float,
    write_interval: float,
) -> None:
    """Setup OpenFOAM case: copy, blockMesh, ensure single outer corrector."""
    _setup_base_case(source_case, test_case)
    _replace_wall_functions(test_case)
    _simplify_solvers_for_neon(test_case)
    _simplify_schemes_for_neon(test_case)
    _ensure_pimple_single_outer(test_case)
    _set_fixed_timing(test_case, end_time, delta_t, write_interval)


def _run_neon_solver(test_case: Path) -> None:
    """Run NeoN solver via subprocess (avoids OpenFOAM double-init segfault)."""
    # Write a helper script that disables FPE *after* OpenFOAM init
    # (OpenFOAM re-enables feenableexcept during its init)
    script = test_case / "_run_solver.py"
    script.write_text(
        "from neofoam.solver.incompressibleFluidNeon import run\n"
        "run(['.'], disable_fpe=True)\n"
    )
    env = {**os.environ, "PWD": str(test_case)}
    env.pop("FOAM_SIGFPE", None)  # Unset to prevent OpenFOAM FPE trapping
    result = subprocess.run(
        ["uv", "run", "python", str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(test_case),
        env=env,
    )
    # Accept exit even with Kokkos cleanup crashes (SIGSEGV=139, SIGABRT=134)
    # as long as solver completed successfully (printed "End")
    if result.returncode != 0 and "End" not in result.stdout:
        raise AssertionError(
            f"NeoN solver failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )


def _run_pimpleFoam(test_case: Path) -> None:
    """Run OpenFOAM pimpleFoam via subprocess."""
    result = subprocess.run(
        ["pimpleFoam", "-case", str(test_case)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"pimpleFoam failed: {result.stderr[-2000:]}"


def _read_field_values(time_dir: Path, field_name: str) -> Any:
    """Read field values directly from OpenFOAM ASCII file."""
    field_path = time_dir / field_name
    if not field_path.exists():
        raise FileNotFoundError(f"Field file not found: {field_path}")

    content = field_path.read_text()

    # Find the internalField section
    idx = content.find("internalField")
    if idx == -1:
        raise ValueError(f"No internalField found in {field_path}")

    # Find the data block (between parentheses after nonuniform)
    start = content.find("(", idx)
    end = content.find(")", start)
    if start == -1 or end == -1:
        raise ValueError(f"Could not parse field data in {field_path}")

    data_str = content[start + 1 : end].strip()
    lines = [line.strip() for line in data_str.split("\n") if line.strip()]

    values = []
    for line in lines:
        if line.startswith("(") and line.endswith(")"):
            components = line[1:-1].split()
            values.append([float(c) for c in components])
        else:
            try:
                values.append(float(line))
            except ValueError:
                continue

    return np.array(values)


def _get_final_time(case_dir: Path) -> Path:
    """Get the last output time directory."""
    time_dirs = sorted(
        [
            d
            for d in case_dir.iterdir()
            if d.is_dir()
            and d.name.replace(".", "").replace("-", "").isdigit()
            and float(d.name) > 0
        ],
        key=lambda x: float(x.name),
    )
    assert len(time_dirs) > 0, f"No output time directories in {case_dir}"
    return time_dirs[-1]


@requires_openfoam
def test_pitzDaily_SA_neon_vs_pimpleFoam() -> None:
    """Compare NeoN SA solver against pimpleFoam on pitzDaily case."""
    source_case = REPO_ROOT / "tutorials" / "pitzDaily_SA"

    test_case_neon = REPO_ROOT / "test_cases" / "pitzDaily_SA_neon_comparison"
    test_case_of = REPO_ROOT / "test_cases" / "pitzDaily_SA_of_comparison"

    end_time = 0.001  # 10 time steps at deltaT=0.0001
    delta_t = 0.0001
    write_interval = 10  # write at final time only (every 10 steps)

    try:
        # Setup cases
        print("\n=== Setting up pitzDaily_SA test cases ===")
        _setup_neon_case(source_case, test_case_neon, end_time, delta_t, write_interval)
        _setup_openfoam_case(
            source_case, test_case_of, end_time, delta_t, write_interval
        )

        # Run solvers
        print("\n=== Running NeoN solver ===")
        _run_neon_solver(test_case_neon)

        print("\n=== Running pimpleFoam ===")
        _run_pimpleFoam(test_case_of)

        # Debug: list time directories
        for label, tc in [("NeoN", test_case_neon), ("OF", test_case_of)]:
            dirs = sorted(
                [d.name for d in tc.iterdir() if d.is_dir()
                 and d.name.replace(".", "").replace("-", "").isdigit()
                 and float(d.name) > 0]
            )
            print(f"  {label} time dirs: {dirs}")
            if dirs:
                final_dir = tc / dirs[-1]
                print(f"  {label} files in {dirs[-1]}: {[f.name for f in final_dir.iterdir()]}")

        # Compare final output fields
        final_neon = _get_final_time(test_case_neon)
        final_of = _get_final_time(test_case_of)

        print(f"\nComparing: NeoN({final_neon.name}) vs OF({final_of.name})")

        fields_to_compare = ["U", "p", "nuTilda", "nut"]
        all_match = True
        rtol = 1e-3
        atol = 1e-8

        for field_name in fields_to_compare:
            vals_neon = _read_field_values(final_neon, field_name)
            vals_of = _read_field_values(final_of, field_name)

            assert vals_neon.shape == vals_of.shape, (
                f"{field_name}: shape mismatch {vals_neon.shape} vs {vals_of.shape}"
            )

            max_abs_diff = float(np.max(np.abs(vals_neon - vals_of)))
            max_val = float(np.max(np.abs(vals_of)))
            rel_diff = (
                max_abs_diff / (max_val + 1e-15) if max_val > 0 else max_abs_diff
            )

            match = bool(np.allclose(vals_neon, vals_of, rtol=rtol, atol=atol))
            status = "OK" if match else "MISMATCH"
            print(
                f"  {field_name}: {status} "
                f"(max_abs={max_abs_diff:.2e}, rel={rel_diff:.2e})"
            )

            if not match:
                all_match = False

        assert all_match, (
            f"Field values differ between NeoN and pimpleFoam (rtol={rtol}, atol={atol})"
        )
        print("\n=== Test PASSED: NeoN matches pimpleFoam ===")

    finally:
        pass  # temporarily skip cleanup for debugging
        # for tc in [test_case_neon, test_case_of]:
        #     if tc.exists():
        #         shutil.rmtree(tc)
        #         print(f"Cleaned up: {tc}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
