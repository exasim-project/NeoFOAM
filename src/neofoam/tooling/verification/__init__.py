# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The engine behind ``verification/foam_tutorials`` — the OpenFOAM drop-in suite.

The claim under test is *drop-in*: take an OpenFOAM tutorial verbatim, swap only
the solver token in its ``Allrun``, run both, and diff the final-time fields. If a
case needs any other edit to run, that is a solver defect to report — not
something the harness works around.

This package holds the machinery, split so each stage is a separate Snakemake
rule with its result on disk (so a run is parallel across cases and resumable):

* :mod:`~neofoam.tooling.verification.foamdict` — static dictionary readers, for
  classifying a tutorial without running it.
* :mod:`~neofoam.tooling.verification.stage` — build one side of a comparison
  with :mod:`neofoam.tooling.casebuild`, then swap the solver token.
* :mod:`~neofoam.tooling.verification.execute` — run ``Allrun``, classify how it
  failed if it did.
* :mod:`~neofoam.tooling.verification.compare` — diff the two runs field by field.
* :mod:`~neofoam.tooling.verification.report` — render the HTML report.
* :mod:`~neofoam.tooling.verification.runner` — the CLI the packaged
  ``verify_*.smk`` rules shell out to.

*Which* tutorials are swept, and how they are tiered, is study configuration and
lives with the study under ``verification/foam_tutorials/<solver>/``. Nothing
here is solver-specific.
"""

__all__: list[str] = []
