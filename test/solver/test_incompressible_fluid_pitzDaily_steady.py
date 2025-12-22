# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Integration test for the IncompressibleFluid solver using the pitzDaily_steady case.
This test requires OpenFOAM to be properly sourced in the environment.
Tests the SIMPLE algorithm for steady-state simulations.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Disable OpenFOAM floating point exception trapping BEFORE any imports
# This prevents FPE errors when pytest tries to print object representations
os.environ["FOAM_SIGFPE"] = ""


# Fix LD_LIBRARY_PATH to use system libstdc++ before conda's
# This is needed for pybFoam which is compiled against newer libstdc++
def fix_ld_library_path():
    """Prepend system lib paths to LD_LIBRARY_PATH to avoid GLIBCXX version conflicts."""
    system_lib_paths = ["/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"]
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    # Prepend system paths
    new_paths = [p for p in system_lib_paths if Path(p).exists()]
    if current_ld_path:
        new_paths.append(current_ld_path)

    if new_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths)


fix_ld_library_path()


def check_openfoam_available():
    """Check if OpenFOAM is available and properly configured."""
    try:
        result = subprocess.run(
            ["blockMesh", "-help"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_pybfoam_available():
    """Check if pybFoam can be imported without library errors."""
    try:
        import pybFoam  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


requires_openfoam = pytest.mark.skipif(
    not check_openfoam_available() or not check_pybfoam_available(),
    reason="OpenFOAM not available or pybFoam libraries not properly configured",
)


# skip
@pytest.mark.skip(reason="Test currently disabled, relax is not properly supported")
@requires_openfoam
def test_incompressible_fluid_pitzDaily_steady():
    """Test the IncompressibleFluid solver on the pitzDaily_steady tutorial case with SIMPLE algorithm."""
    # Import here to avoid import errors when OpenFOAM is not available
    from foamadapter.solver import IncompressibleFluid

    # Setup paths - go up from test/solver to repo root
    repo_root = Path(__file__).parent.parent.parent
    source_case = repo_root / "tutorials" / "pitzDaily_steady"
    test_case = repo_root / "test_cases" / "pitzDaily_steady_incompressible_test"

    # Clean up any existing test case
    if test_case.exists():
        shutil.rmtree(test_case)

    try:
        # Copy the case
        test_case.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_case, test_case)

        # Setup initial conditions
        orig_dir = test_case / "0.orig"
        zero_dir = test_case / "0"
        if orig_dir.exists():
            if zero_dir.exists():
                shutil.rmtree(zero_dir)
            shutil.copytree(orig_dir, zero_dir)

        # Run blockMesh
        result = subprocess.run(
            ["blockMesh", "-case", str(test_case)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"blockMesh failed: {result.stderr}"

        # Modify controlDict for a short test run
        control_dict = test_case / "system" / "controlDict"
        control_dict_content = control_dict.read_text()

        # Replace endTime with a smaller value for testing (50 iterations instead of 2000)
        control_dict_content = control_dict_content.replace(
            "endTime         2000;", "endTime         50;"
        )
        # Write output more frequently for testing
        control_dict_content = control_dict_content.replace(
            "writeInterval   100;", "writeInterval   25;"
        )
        control_dict.write_text(control_dict_content)

        # Change to the test case directory so OpenFOAM can find files
        original_dir = Path.cwd()
        os.chdir(test_case)

        try:
            # Create solver instance and run it
            # Use a separate function to avoid pytest trying to repr the solver on error
            def run_solver():
                solver = IncompressibleFluid(argv=["incompressibleFluid"])
                solver.run()

            run_solver()
        finally:
            # Change back to original directory
            os.chdir(original_dir)

        # Verify output was written
        time_dirs = sorted(
            [
                d
                for d in test_case.iterdir()
                if d.is_dir() and d.name.replace(".", "").replace("-", "").isdigit()
            ]
        )
        assert len(time_dirs) >= 2, "Expected at least 0 and one output time directory"

        # Check that result files exist
        final_time = time_dirs[-1]
        assert (final_time / "U").exists(), "Velocity field U not found in final time"
        assert (final_time / "p").exists(), "Pressure field p not found in final time"

        # Check for turbulence model fields (k-epsilon model)
        assert (final_time / "k").exists(), (
            "Turbulent kinetic energy field k not found in final time"
        )
        assert (final_time / "epsilon").exists(), (
            "Turbulent dissipation field epsilon not found in final time"
        )

        print(f"Test successful! Output written to: {time_dirs[-1]}")
        print(
            f"Steady-state SIMPLE algorithm completed {len(time_dirs) - 1} output iterations"
        )

    finally:
        # Clean up test case
        if test_case.exists():
            shutil.rmtree(test_case)
            print(f"Cleaned up test case: {test_case}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
