# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The engine behind ``verification/foam_tutorials`` — the OpenFOAM drop-in suite.

The claim under test is *drop-in*: take an OpenFOAM tutorial verbatim, swap only
the solver token in its ``Allrun``, run both, and diff the final-time fields. If a
case needs any other edit to run, that is a solver defect to report — not
something the harness works around.

Nothing here is solver-specific: *which* tutorials are swept, and how they are
tiered, is study configuration and lives with the study under
``verification/foam_tutorials/<solver>/`` as a ``config.yaml`` + ``discover.py``.

The machinery is split so each stage is a separate Snakemake rule with its result
on disk (so a run is parallel across cases and resumable):

* :mod:`~verification.dropin.cases` — load a study's ``config.yaml`` +
  ``discover.py`` into the :class:`~verification.dropin.cases.Study`
  the Snakefile and every worker re-derive identically.
* :mod:`~verification.dropin.foamdict` — static dictionary readers, for
  classifying a tutorial without running it.
* :mod:`~verification.dropin.stage` — build one side of a comparison
  with :mod:`neofoam.tooling.casebuild`, then swap the solver token.
* :mod:`~verification.dropin.execute` — run ``Allrun``, classify how it
  failed if it did.
* :mod:`~verification.dropin.compare` — diff the two runs field by field.
* :mod:`~verification.dropin.report` — render the HTML report.
* :mod:`~verification.dropin.runner` — the CLI the packaged
  ``study_*.smk`` rules shell out to.
"""

__all__: list[str] = []
