"""
In-situ post-processing: monitor a run as CSV
=============================================

A case can declare *tables* — a source, a pipeline of nodes and a write
cadence — and NeoFOAM checks them after every time step, evaluating each on
its own cadence and appending it to ``postProcessing/<name>.csv`` in the case
directory. Nothing is written unless the case asks for it, and no post-run
tooling is involved: the numbers land while the solver runs.

There are two front doors, and a case may use both. ``system/postProcess.yaml``
declares tables as data; ``system/postProcess.py`` builds them in Python with
``|`` and can register brand-new node types that the YAML then uses by name.

This tutorial runs the bundled ``postProcessing/cavity`` case — a laminar
lid-driven cavity — and plots its kinetic energy over time.

If you haven't run a NeoFOAM case yet, do
:doc:`example_01_run_incompressible_fluid` first.
"""

# %%
# Imports
# -------

import contextlib
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from neofoam.solver import incompressibleFluid
from neofoam.tutorial import clone_case

# %%
# Clone the bundled case
# ----------------------
# ``postProcessing/cavity`` is a 20x20 lid-driven cavity that runs in a
# couple of seconds. Beside the usual ``system/`` dictionaries it ships the
# two post-processing declaration files. The empty ``.foam`` marker is what
# pyvista's OpenFOAMReader opens further down.

case = clone_case("postProcessing/cavity")
(case / "cavity.foam").touch()
print(f"working copy: {case}")

# %%
# The declarative front door
# --------------------------
# ``source``, each ``pipeline`` entry and ``write_control`` are open
# mappings resolved by their ``type`` against a plugin registry, so a table
# needs no code (an unknown key in one of them is rejected, not ignored).
# Note the second table: ``square`` is not a NeoFOAM node — the script below
# registers it. The file lives at
# ``tutorials/postProcessing/cavity/system/postProcess.yaml``:
#
# .. literalinclude:: ../../tutorials/postProcessing/cavity/system/postProcess.yaml
#    :language: yaml

assert (case / "system" / "postProcess.yaml").exists()

# %%
# The script front door
# ---------------------
# The decorated functions run at load time and return a pipeline, so a
# scripted table *is* the same object a declared one resolves to. The
# ``@Node.register`` class at the top is what makes ``type: square`` valid
# in the YAML above — the script is executed first, on purpose.
#
# .. literalinclude:: ../../tutorials/postProcessing/cavity/system/postProcess.py
#    :language: python

assert (case / "system" / "postProcess.py").exists()

# %%
# Mesh and run
# ------------
# Exactly what ``./Allrun`` does. The ``[U_in_mm_per_s]`` lines in the log
# come from the ``print`` node in the declared vector pipeline.

subprocess.run(["blockMesh", "-case", str(case)], check=True)

with contextlib.chdir(case):
    incompressibleFluid.run(["."])

# %%
# The flow, before and after
# --------------------------
# The case is a lid-driven cavity: the top wall slides in ``+x`` and drags the
# fluid along with it. At the first written time only a thin shear layer under
# the lid has picked up any speed. By the last one that sweep has turned down
# the right wall and back along the middle — the primary vortex forming —
# while the bottom of the box is still close to rest. Both panels share one
# colour scale, and the arrows show the direction. The CSV tables below are
# the numbers behind this picture.

reader = pv.OpenFOAMReader(str(case / "cavity.foam"))
written_times = [t for t in reader.time_values if t > 0]
first_time, last_time = written_times[0], written_times[-1]


def velocity_mesh(time_value):
    reader.set_active_time_value(time_value)
    mesh = reader.read()["internalMesh"].cell_data_to_point_data()
    mesh["magU"] = np.linalg.norm(mesh["U"], axis=1)
    return mesh


panels = [(t, velocity_mesh(t)) for t in (first_time, last_time)]
clim = (0.0, max(float(mesh["magU"].max()) for _, mesh in panels))

pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1000, 450))
for column, (time_value, mesh) in enumerate(panels):
    pl.subplot(0, column)
    pl.add_mesh(
        mesh,
        scalars="magU",
        cmap="turbo",
        clim=clim,
        show_scalar_bar=column == 1,
        # Under the field rather than across it: the panels are square in a
        # wider-than-tall viewport, so the bottom strip is free.
        scalar_bar_args={
            "title": "|U| [m/s]",
            "position_x": 0.15,
            "position_y": 0.01,
            "width": 0.7,
            "height": 0.05,
        },
    )
    # The arrows are decoration; a pyvista build that dislikes the glyph
    # filter should still leave the coloured field standing.
    with contextlib.suppress(Exception):
        arrows = mesh.glyph(orient="U", scale="magU", factor=0.08, tolerance=0.02)
        pl.add_mesh(arrows, color="black")
    pl.add_text(f"t = {time_value:g} s", font_size=11)
    pl.view_xy()
pl.show()

# %%
# What was written
# ----------------
# One flat CSV per table, ``time`` first column, created on the first write.
# The two kinetic-energy tables — one declared, one scripted — hold the same
# numbers by two routes. The two ``volume_U`` tables carry an ``interval: 10``
# cadence, so they have a tenth of the rows; a vector aggregation becomes one
# column per component.

tables = sorted((case / "postProcessing").glob("*.csv"))
for path in tables:
    head = path.read_text().splitlines()
    print(f"{path.name}  ({len(head) - 1} rows)")
    for line in head[:3]:
        print(f"    {line}")

# %%
# Plot the kinetic energy over time
# ---------------------------------
# ``kinetic_energy.csv`` is :math:`\frac{1}{2} \int_\Omega |U|^2 \, dV` at every
# step — the monitor you would watch to see a transient settle. The lid drags
# the fluid into motion, so the curve rises steeply from rest and then flattens
# as the primary vortex reaches its steady strength; a flat line from the first
# step would mean the solve never advanced.

time, kinetic_energy = np.loadtxt(
    case / "postProcessing" / "kinetic_energy.csv",
    delimiter=",",
    skiprows=1,
    unpack=True,
)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(time, kinetic_energy, color="#1f77b4")
ax.set_xlabel("time [s]")
ax.set_ylabel(r"$\frac{1}{2}\int_\Omega |U|^2 \, dV$  [m$^5$/s$^2$]")
ax.set_title("kinetic_energy — written every time step while the solver ran")
ax.grid(alpha=0.3)
fig.tight_layout()

# %%
# Where to go from here
# ---------------------
# - :doc:`/reference/postprocessing` lists the available nodes, the
#   write-cadence options and the recipe for adding your own node.
# - Post-processing is serial-only for now: a ``-parallel`` run of a case
#   that declares tables is refused rather than writing per-rank partials.
