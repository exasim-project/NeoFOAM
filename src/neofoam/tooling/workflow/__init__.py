# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case-staging workflow: geometry → mesh, parameter sweeps, drop-in studies.

This package is a *directory of deep sub-modules*, not one flat namespace — import
from the sub-module that owns your concern, each of which states its own interface:

* :mod:`neofoam.tooling.workflow.geometry` — the ``manifest.json`` schema
  (:class:`~neofoam.tooling.workflow.geometry.PatchSet`) and the deterministic
  ``PatchSet`` → mesh-dict mappers (:func:`~neofoam.tooling.workflow.geometry.build_mesh_inputs`).
  The one entry point for going from staged geometry to mesh inputs.
* :mod:`neofoam.tooling.workflow.sweep` — parameter sweeps over a saved base case:
  the :class:`~neofoam.tooling.workflow.sweep.Sweep` object owns the round-trip
  (export a runnable Snakemake workflow dir, read it back), plus the wizard's canvas
  node/edge model.
* :mod:`neofoam.tooling.workflow.study` — OpenFOAM tutorial drop-in studies: take
  a tutorial verbatim, swap only the solver token in its ``Allrun``, run both sides
  and diff the final-time fields. A study is a directory of three files
  (``Snakefile`` + ``config.yaml`` + ``discover.py``) over the packaged pipeline.
* :mod:`neofoam.tooling.workflow.rules` — the packaged Snakemake rule library, in
  two named sets: the *mesh sweep* (per mesh variant ``setup_mesh`` → blockMesh →
  snappyHexMesh → checkMesh, then per case ``setup`` → ``solve``), composed by
  :meth:`~neofoam.tooling.workflow.rules.RuleRegistry.plan`; and the *drop-in study*
  (``study.smk`` → build_case → swap_solver → run → compare → report), included
  whole.
* :mod:`neofoam.tooling.workflow.dag` — render an exported sweep's
  ``snakemake --dag`` / ``--rulegraph`` as canvas nodes and edges.
* :mod:`neofoam.tooling.workflow.paramspace` — the swept-dimension csv/yaml layer
  the sweep persistence builds on.
* :mod:`neofoam.tooling.workflow.sweep_runner` — the ``python -m …sweep_runner`` CLI
  the generated Snakefile shells out to (mesh staging, per-tool runs, case setup).

Where the STLs come from is out of scope — any upstream tool that writes
``constant/triSurface/*.stl`` plus a ``manifest.json`` plugs in at
:mod:`~neofoam.tooling.workflow.geometry`.

This top-level ``__init__`` intentionally re-exports **nothing**: the concerns
above used to share one flat namespace; keeping the surface on the sub-modules is
what makes each a small, deep interface. Import
``from neofoam.tooling.workflow.geometry import PatchSet``, not
``from neofoam.tooling.workflow import PatchSet``.
"""

from __future__ import annotations

__all__: list[str] = []
