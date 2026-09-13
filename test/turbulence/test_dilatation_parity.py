# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Parity of the NeoN closures' **dilatation terms** against pybFoam.

OpenFOAM's ``kEpsilon``/``kOmegaSST`` carry ``fvm::SuSp(divU, ...)`` terms that
vanish only for a conservative flux, and every other parity case here seeds a
divergence-free velocity on purpose (:mod:`test_wall_function_parity`) — so a
closure that drops them still matches to round-off there, while drifting ~1e-3 of
peak per ``correct`` step on a real SIMPLE iterate.

Hence :func:`_parity_case.wall_function_case` with
``U = (0, 2x + 1.5z + (y - 0.5)^2, 0)``: no wall-normal component (the wall
functions stay exercised unchanged), but ``div U = 2(y - 0.5)`` changes sign
across the box, so both halves of the ``SuSp`` split bind within one run.

Bounds are the wall-function module's: round-off for ``kEpsilon``, 1e-6 for
``kOmegaSST``, whose ``F1``/``F2`` blending associates differently on the two
backends. One ``Foam::Time`` per process, so the roles run as subprocesses of
:mod:`_parity_worker` — once per model, not once per compared field.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neofoam.tooling.casebuild import Step, patch
from turbulence._parity_case import run_worker, wall_function_case

#: ``model -> (the fields it owns, agreement bound as a fraction of the field's peak)``.
MODELS: dict[str, tuple[tuple[str, ...], float]] = {
    "kEpsilon": (("nut", "k", "epsilon"), 1e-12),
    "kOmegaSST": (("nut", "k", "omega"), 1e-6),
}

#: One parameter per compared field, so a mismatch names the model *and* the field.
COMPARISONS = [(model, field) for model, (fields, _) in MODELS.items() for field in fields]

#: The unit cube of the wall-function case, 4 cells per side.
CELLS_PER_SIDE = 4
N_CELLS = CELLS_PER_SIDE**3


def dilatation_velocity() -> Step:
    """Rewrite ``0/U``'s internal field to ``(0, 2x + 1.5z + (y - 0.5)^2, 0)``.

    blockMesh numbers the single hex block x-fastest; the cell centres are recomputed
    here rather than read from a mesh because a parent process must not construct a
    ``Foam::Time``.
    """
    cell = np.arange(N_CELLS)
    x = (cell % CELLS_PER_SIDE + 0.5) / CELLS_PER_SIDE
    y = (cell // CELLS_PER_SIDE % CELLS_PER_SIDE + 0.5) / CELLS_PER_SIDE
    z = (cell // CELLS_PER_SIDE**2 + 0.5) / CELLS_PER_SIDE
    u_y = 2.0 * x + 1.5 * z + (y - 0.5) ** 2
    body = " ".join(f"(0 {v:.16e} 0)" for v in u_y)
    return patch("0/U", internalField=f"nonuniform List<vector> {N_CELLS} ( {body} )")


@pytest.fixture(scope="module")
def stepped_cases(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Take the one ``correct`` step per model, once, on both backends."""
    cases = {}
    for model in MODELS:
        pipeline = wall_function_case(model) | dilatation_velocity()
        case = pipeline.build_at(tmp_path_factory.mktemp(model) / "case").path
        run_worker("mesh", case)  # the pipeline wrote the 0/ fields; do not reseed
        run_worker("reference", case)  # pybFoam
        run_worker("subject", case)  # the NeoN closure
        cases[model] = case
    return cases


@pytest.mark.parametrize(("model", "field"), COMPARISONS)
def test_field_matches_pybfoam_on_nonconservative_flux(
    model: str, field: str, stepped_cases: dict[str, Path]
) -> None:
    """One ``correct`` step on a non-conservative flux matches pybFoam."""
    bound = MODELS[model][1]
    case = stepped_cases[model]

    reference = np.load(case / f"reference_{field}.npy")
    result = np.load(case / f"subject_{field}.npy")

    peak = float(np.max(np.abs(reference))) or 1.0
    np.testing.assert_allclose(
        result,
        reference,
        rtol=0.0,
        atol=bound * peak,
        err_msg=f"{model}: {field} differs after one correct step on a non-conservative flux",
    )
