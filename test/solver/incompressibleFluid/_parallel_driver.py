# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Standalone ``-parallel`` driver for the incompressibleFluid comparison tests.

Launched once per MPI rank via ``mpirun … python _parallel_driver.py`` with the
decomposed case as the working directory. Each rank builds its own
``argList``/``Time`` under the shared MPI session, mirroring how native
``pimpleFoam -parallel`` runs. Kept as a real module (not a string embedded in a
test) so it is linted and importable like any other source file.
"""

import os

from neofoam.solver.incompressibleFluid import run


def main() -> None:
    # OpenFOAM's SIGFPE trap can fire in some Python float paths; the case-file
    # tests disable it exactly as the plain-port driver does.
    os.environ["FOAM_SIGFPE"] = ""
    run(["incompressibleFluid", "-parallel"])


if __name__ == "__main__":
    main()
