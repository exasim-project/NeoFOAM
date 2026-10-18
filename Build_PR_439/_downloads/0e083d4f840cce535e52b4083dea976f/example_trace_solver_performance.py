"""
Trace solver performance with OpenTelemetry
===========================================

Measure where a run spends its time — per operation, per init step, and
per MPI rank — by opting a case into NeoFOAM's OpenTelemetry tracing.

Tracing is **opt-in twice**: the ``neofoam[telemetry]`` extra must be
installed (``pip install neofoam[telemetry]``), and the case must carry a
``telemetry`` sub-dict in ``system/controlDict``. Without either, the
instrumentation is a no-op. When active, every framework operation
(``time_loop``, ``momentum``, ``continuity``, …) and every init step
becomes a span, and each rank writes:

- ``<case>/telemetry/rank<N>.spans.jsonl`` — one JSON span per line
  (nesting, timestamps, attributes), and
- ``<case>/telemetry/rank<N>.summary.json`` — per-operation
  count/total/mean/min/max wall-clock seconds.

This page runs the bundled ``hotRoom`` buoyancy case for a few steps and
reads the summary back.
"""

# %%
# Prepare the case
# ----------------
# Clone ``hotRoom``, mesh it, and initialize the hot-air blob — what its
# ``Allrun`` would do. We then shorten the run to 20 time steps.

import contextlib
import subprocess

import matplotlib.pyplot as plt
import pybFoam as pyf

from neofoam import telemetry
from neofoam.solver import incompressibleFluid
from neofoam.tutorial import clone_case

case = clone_case("hotRoom")
subprocess.run(["blockMesh", "-case", str(case)], check=True)
subprocess.run(["setFields", "-case", str(case)], check=True)

control_dict = case / "system" / "controlDict"
cd = pyf.dictionary.read(str(control_dict))
cd.set("endTime", 40)
cd.write(str(control_dict))

# %%
# Opt the case into tracing
# -------------------------
# The ``telemetry`` sub-dict is all it takes — every key shown here is
# optional (``enabled`` defaults to ``yes``, ``directory`` to
# ``telemetry``, ``summary`` to ``yes``). ``subDictOrAdd`` creates the
# sub-dict but returns a detached copy, so re-fetch it with ``subDict``
# to get a live handle before setting the keys.

cd = pyf.dictionary.read(str(control_dict))
cd.subDictOrAdd("telemetry")
telemetry_dict = cd.subDict("telemetry")
telemetry_dict.set("enabled", "yes")
telemetry_dict.set("directory", "telemetry")
telemetry_dict.set("summary", "yes")
cd.write(str(control_dict))

# %%
# Run the solver
# --------------
# ``run()`` reads the dict before initialization, so the init/build steps
# are traced too. In parallel each rank writes its own file: decompose the
# case and launch e.g. ``mpirun -np 4 python -m <driver>`` with
# ``run(["incompressibleFluid", "-parallel"])`` to get ``rank0`` …
# ``rank3`` span files, each tagged with ``mpi.rank`` / ``mpi.size``.

with contextlib.chdir(case):
    incompressibleFluid.run(["."])

# %%
# Where did the time go? — the summary bar chart
# ----------------------------------------------
# :func:`neofoam.telemetry.plot_summary` reads ``rank<N>.summary.json`` and
# draws the per-operation total wall-clock. ``time_loop`` is the whole loop;
# ``momentum`` / ``continuity`` are the PIMPLE stages inside it, and their
# ``.assemble`` / ``.solve`` children split matrix build from linear solve.
# Passing an axes lets the docs gallery (or a report) capture the figure;
# ``neofoam telemetry plot <case>`` does the same from the command line.

fig, ax = plt.subplots(figsize=(8, 5))
telemetry.plot_summary(case / "telemetry" / "rank0.summary.json", top=12, ax=ax)

# %%
# The interactive timeline — open in Perfetto
# -------------------------------------------
# The summary flattens the run; the *spans* keep the full hierarchy (each
# carries its parent span id and exact start/end). :func:`~neofoam.telemetry.
# write_chrome_trace` turns them into a ``trace.json`` you drop into
# https://ui.perfetto.dev or ``chrome://tracing`` for a zoomable flame
# timeline — one process row per MPI rank, so you can see exactly which
# inner-loop iteration was slow (``neofoam telemetry trace <case>`` from the
# CLI). Operation authors add finer detail with
# ``neofoam.telemetry.span("my.step")`` or ``neofoam.telemetry.instrument()``
# — those spans nest under the operation's span automatically.

trace_path = telemetry.write_chrome_trace(case)
print(f"wrote {trace_path} — open it in https://ui.perfetto.dev")
