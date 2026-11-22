# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Standalone ``-parallel`` driver for the incompressibleVoF comparison tests.

Launched once per MPI rank via ``mpirun … python _parallel_driver.py`` with the
decomposed case as the working directory, mirroring how native ``interFoam
-parallel`` runs: each rank builds its own ``argList``/``Time`` under the shared
MPI session. Kept as a real module (not a string embedded in a test) so it is
linted and importable like any other source file — the same arrangement as
``test/solver/incompressibleFluid/_parallel_driver.py``.
"""

import os

from neofoam.solver.incompressibleVoF import run


def main() -> None:
    # OpenFOAM's SIGFPE trap can fire in some Python float paths; the case-file
    # tests disable it exactly as the single-phase parallel driver does.
    os.environ["FOAM_SIGFPE"] = ""
    run(["incompressibleVoF", "-parallel"])


if __name__ == "__main__":
    main()
