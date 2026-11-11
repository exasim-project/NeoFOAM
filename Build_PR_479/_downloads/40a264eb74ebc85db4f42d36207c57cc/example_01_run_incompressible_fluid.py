"""
Run your first NeoFOAM case
===========================

Run the bundled ``pitzDaily`` benchmark end-to-end with the
``incompressibleFluid`` solver and plot the final velocity field.
"""

# %%
# Imports
# -------

import contextlib
import subprocess

import pyvista as pv

from neofoam.solver import incompressibleFluid
from neofoam.tutorial import clone_case

# %%
# Clone the bundled case
# ----------------------
# ``clone_case`` copies the case to a fresh tempdir and restores
# ``0.orig/`` to ``0/`` (what ``Allrun`` normally does). The
# ``.foam`` marker satisfies pyvista's OpenFOAMReader at the end.

case = clone_case("pitzDaily")
(case / "pitzDaily.foam").touch()
print(f"working copy: {case}")

# %%
# Generate the mesh
# -----------------
# ``blockMesh`` reads ``system/blockMeshDict`` and writes
# ``constant/polyMesh/``.

subprocess.run(["blockMesh", "-case", str(case)], check=True)
assert (case / "constant" / "polyMesh" / "points").exists()

# %%
# Run the solver
# --------------
# Equivalent to running ``neofoam solver incompressiblefluid`` from
# the shell — that CLI command calls the same ``run`` underneath.
# We ``chdir`` into the case the way an ``Allrun`` script does.

with contextlib.chdir(case):
    incompressibleFluid.run(["."])

# %%
# Plot the final velocity field
# -----------------------------
# pyvista colours by the magnitude of ``U`` automatically. A
# flat-blue plot or zero-range colour bar would mean the solve
# didn't produce data — re-check the log in that case.

reader = pv.OpenFOAMReader(str(case / "pitzDaily.foam"))
reader.set_active_time_value(reader.time_values[-1])
mesh = reader.read()["internalMesh"]

pl = pv.Plotter(off_screen=True, window_size=(900, 360))
pl.add_mesh(
    mesh,
    scalars="U",
    cmap="viridis",
    scalar_bar_args={"title": "|U| [m/s]"},
)
pl.view_xy()
pl.camera.zoom(1.2)
pl.show()
