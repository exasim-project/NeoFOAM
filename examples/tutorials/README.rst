Tutorials
=========

Guided lessons for newcomers. Each tutorial is a runnable Python
script that executes at documentation build time against the cases
shipped under ``tutorials/`` in the repo — the printed log output and
figures on every page are exactly what the script produced. Download
the ``.py`` file or the rendered Jupyter notebook from any tutorial
to run it locally.

Read them in order:

1. **Run your first NeoFOAM case.** Drive the bundled ``pitzDaily``
   benchmark through the ``incompressibleFluid`` solver end-to-end
   and visualise the final pressure field. Establishes the case
   layout and the ``clone_case`` helper every later tutorial uses.
2. **Add a passive scalar transport model.** Extend
   ``incompressibleFluid`` with a new physics model — without
   modifying the solver. Walks through the four plugin decorators
   (``@detect`` / ``@load`` / ``@build`` / ``@operation``) and shows
   the dye spreading from the inlet at the final time step.
3. **Build a scalar-transport solver from scratch.** Author a
   complete solver — ``SolverSpec``, ``StagedInit``, ``DAGResolver``
   entry point — for unsteady advection-diffusion. Includes a worked
   example of an ``InitializationGraphError`` cycle so you've seen the
   DAG fail before you have to debug it for real.
4. **Typed configs across three file formats.** Define a
   ``BaseConfig`` once and bind it to YAML, JSON, or an OpenFOAM
   dictionary by changing one decorator. Covers load / mutate / save
   round-trips, the subdict pattern that lets several configs share a
   file, and how Pydantic constraints catch bad input on load.
5. **Monitor a run with in-situ post-processing.** Declare *tables* —
   a source, a pipeline of nodes, a write cadence — and have the
   solver append each one to ``postProcessing/<name>.csv`` while it
   runs. Covers both front doors (``system/postProcess.yaml`` and
   ``system/postProcess.py``), a case-defined ``@Node.register`` node
   used by name from the YAML, and the cavity's kinetic energy
   plotted over time.
