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
import json
import subprocess

import pybFoam as pyf

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
# Where did the time go?
# ----------------------
# The summary aggregates every span by name. ``time_loop`` is the whole
# loop; ``momentum``/``continuity`` are the PIMPLE stages inside it, and
# ``init.*`` entries are the one-off build steps.

summary = json.loads((case / "telemetry" / "rank0.summary.json").read_text())
print(f"rank {summary['mpi']['rank']} of {summary['mpi']['size']}")

by_total = sorted(
    summary["spans"].items(), key=lambda item: item[1]["total_s"], reverse=True
)
print(f"{'operation':<28} {'count':>5} {'total [s]':>10} {'mean [s]':>10}")
for name, stats in by_total[:12]:
    print(
        f"{name:<28} {stats['count']:>5} {stats['total_s']:>10.4f} "
        f"{stats['mean_s']:>10.4f}"
    )

# %%
# The raw spans keep the full hierarchy — each line carries its parent
# span id, so a trace viewer (or a few lines of Python) can reconstruct
# exactly which inner-loop iteration was slow. Operation authors can add
# finer detail from inside an operation with
# ``neofoam.telemetry.span("my.step")`` or by decorating a helper with
# ``neofoam.telemetry.instrument()`` — those spans nest under the
# operation's span automatically.

spans = [
    json.loads(line)
    for line in (case / "telemetry" / "rank0.spans.jsonl").read_text().splitlines()
]
momentum = [s for s in spans if s["name"] == "momentum"]
print(f"{len(spans)} spans total, {len(momentum)} momentum solves")
