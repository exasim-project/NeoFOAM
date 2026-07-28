# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""One-role-per-process worker for the NeoN turbulence parity test.

pybFoam and NeoN both hang state off the global OpenFOAM ``objectRegistry``, so a
single process that constructs several ``Foam::Time`` objects (mesh generation,
the pybFoam reference read, the NeoN subject read) corrupts that registry and
segfaults intermittently. Each role here therefore runs in its own fresh process
— exactly **one** ``Foam::Time`` per process — and hands its result to the parent
through a ``.npy`` file. The parent (:mod:`test_neon_turbulence_parity`) only
orchestrates subprocesses and compares arrays; it never touches OpenFOAM.

Roles (``python _parity_worker.py <role> <case_dir>``):

* ``mesh``      — generate the block mesh only, for a case that already ships its
  own ``0`` fields (``setup`` would overwrite them).
* ``setup``     — generate the block mesh and seed **identical** random ``U`` /
  ``k`` / ``epsilon`` fields on disk (both backends then read the same bytes).
* ``reference`` — build the trusted pybFoam turbulence model, advance one
  ``correct`` step, write ``reference_<field>.npy`` for each of :data:`COMPARE_FIELDS`.
* ``subject``   — build the pure-Python NeoN ModelSpec model, advance one
  ``correct`` step, write ``subject_<field>.npy`` for each field it owns.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import neon._neon as nn
import numpy as np
import pybFoam as pyf
from pybFoam.turbulence import singlePhaseTransportModel

from neofoam import neofoam_bindings as nfb
from neofoam.framework.context import Context
from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.fallback import OpenFOAMTurbulenceModel
from neofoam.turbulence.selection import select_turbulence_model

os.environ.setdefault("FOAM_SIGFPE", "false")

# The single time step both backends advance (matches controlDict deltaT).
DELTA_T = 5.0e-5
# Deterministic seed → the random k / epsilon are reproducible across runs.
SEED = 20260711


def _foam_scalar_field(name: str, dimensions: str, values: np.ndarray) -> str:
    """Render an OpenFOAM ``volScalarField`` with a nonuniform internal field.

    The boundary is all-``zeroGradient`` (no wall functions); only the internal
    field carries the per-cell random values. Written by both-read semantics: the
    same file feeds pybFoam and NeoN, so the cell ordering need only be
    self-consistent, which OpenFOAM's own reader guarantees.
    """
    body = "\n".join(f"{v:.16e}" for v in values)
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      {name};
}}

dimensions      {dimensions};

internalField   nonuniform List<scalar>
{len(values)}
(
{body}
)
;

boundaryField
{{
    ".*"
    {{
        type            zeroGradient;
    }}
}}
"""


def _foam_vector_field(name: str, dimensions: str, values: np.ndarray) -> str:
    """Render an OpenFOAM ``volVectorField`` with a nonuniform internal field.

    Boundary is all-``zeroGradient``. On the Cartesian box this keeps ``createPhi``
    divergence-free for the seeded velocity (see :func:`role_setup`).
    """
    body = "\n".join(f"({v[0]:.16e} {v[1]:.16e} {v[2]:.16e})" for v in values)
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      {name};
}}

dimensions      {dimensions};

internalField   nonuniform List<vector>
{len(values)}
(
{body}
)
;

boundaryField
{{
    ".*"
    {{
        type            zeroGradient;
    }}
}}
"""


def role_mesh(case: Path) -> None:
    """Generate the block mesh, leaving the case's own ``0`` fields untouched."""
    time = pyf.Time(str(case.parent), case.name)
    block_dict = pyf.dictionary.read(str(case / "system" / "blockMeshDict"))
    pyf.meshing.generate_blockmesh(time, block_dict, False, "constant")


def role_setup(case: Path) -> None:
    """Generate the mesh and seed identical random ``U`` / ``k`` / ``epsilon`` on disk.

    ``U = (a y + b z, c x + d z, e x + f y)`` is analytically divergence-free (each
    component is independent of its own coordinate, so ``div U = 0`` exactly) yet has
    a full off-diagonal ``grad U`` — so turbulent production ``G = nut(grad U && …)``
    is genuinely exercised while ``createPhi(U)`` stays divergence-free (the
    ``div U`` dilatation terms OpenFOAM's kEpsilon adds then vanish identically on
    both backends, rather than being dropped only on the NeoN side).
    """
    time = pyf.Time(str(case.parent), case.name)
    block_dict = pyf.dictionary.read(str(case / "system" / "blockMeshDict"))
    pyf.meshing.generate_blockmesh(time, block_dict, False, "constant")
    mesh = pyf.fvMesh(time)
    n = mesh.nCells()

    rng = np.random.default_rng(SEED)
    centres = np.asarray(mesh.C().internalField())  # (n, 3) cell centres, mesh order
    x, y, z = centres[:, 0], centres[:, 1], centres[:, 2]
    a, b, c, d, e, f = rng.uniform(-2.0, 2.0, 6)
    u = np.stack([a * y + b * z, c * x + d * z, e * x + f * y], axis=1)
    # Positive, physically plausible spreads about the template uniform values.
    k = rng.uniform(0.2, 0.6, n)
    epsilon = rng.uniform(8.0, 20.0, n)
    nu_tilda = rng.uniform(1.0e-4, 2.0e-3, n)  # chi = nuTilda/nu ~ 10..200
    omega = rng.uniform(50.0, 500.0, n)  # omega ~ epsilon/(betaStar k)

    (case / "0" / "U").write_text(_foam_vector_field("U", "[0 1 -1 0 0 0 0]", u))
    (case / "0" / "k").write_text(_foam_scalar_field("k", "[0 2 -2 0 0 0 0]", k))
    (case / "0" / "epsilon").write_text(_foam_scalar_field("epsilon", "[0 2 -3 0 0 0 0]", epsilon))
    (case / "0" / "nuTilda").write_text(_foam_scalar_field("nuTilda", "[0 2 -1 0 0 0 0]", nu_tilda))
    (case / "0" / "omega").write_text(_foam_scalar_field("omega", "[0 0 -1 0 0 0 0]", omega))


# The fields compared cell-by-cell between the two backends after one ``correct``
# step. ``nut`` is universal (every model defines an eddy viscosity); ``k`` /
# ``epsilon`` are the closure's transport unknowns (absent for ``laminar``) — saved
# only when the model actually owns them, so both the derived viscosity *and* the
# transport solutions themselves are checked.
COMPARE_FIELDS = ("nut", "k", "epsilon", "nuTilda", "omega")


def role_reference(case: Path) -> None:
    """pybFoam turbulence model, one ``correct`` step → ``reference_<field>.npy``."""
    of_time = pyf.Time(str(case.parent), case.name)
    of_mesh = pyf.fvMesh(of_time)
    U = pyf.volVectorField.read_field(of_mesh, "U")
    phi = pyf.createPhi(U)
    transport = singlePhaseTransportModel(U, phi)
    of_model = OpenFOAMTurbulenceModel(U, phi, transport).build()
    of_model.correct()

    for name in COMPARE_FIELDS:
        registered = pyf.volScalarField.from_registry(of_mesh, name)
        if registered is not None:
            np.save(case / f"reference_{name}.npy", np.asarray(registered.internalField()))
        elif name == "nut":
            # laminar registers no nut; its nut() is a fresh (non-const) zero tmp, so
            # .ref() is valid here — unlike a RAS model's const nut() tmp above.
            np.save(
                case / "reference_nut.npy",
                np.asarray(of_model.nut().ref().internalField()),
            )


def role_subject(case: Path) -> None:
    """NeoN ModelSpec model, one ``correct`` step → ``subject_<field>.npy``."""
    nn.initialize(["neon"])

    cfg = TurbulencePropertiesConfig.load(case_dir=str(case))

    # The NeoN linear solve reads the fvSolution / fvSchemes dicts off the
    # Foam::Time registry; Foam::Time keeps a *raw* reference to the argList, so
    # both must stay alive for the whole solve (the two-arg Time(rootPath, case)
    # constructor doesn't wire that environment and segfaults inside solve()).
    arg_list = pyf.argList(["subject", "-case", str(case)])
    neon_time = pyf.Time(arg_list)
    rt = nfb.create_adapter_run_time(neon_time)
    rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
    solvers = rt.fv_solution_dict.subDict("solvers")
    for solver_name in (
        "k",
        "epsilon",
        "kFinal",
        "epsilonFinal",
        "nuTilda",
        "nuTildaFinal",
        "omega",
        "omegaFinal",
    ):
        if solvers.contains(solver_name):
            solvers.insert_dict(solver_name, nfb.map_fv_solution(solvers.subDict(solver_name)))
    rt.dt = DELTA_T

    U_neon = nfb.read_vector_volume_field(rt, "U")
    phi_neon = nfb.create_phi(rt, "U")
    nu = nfb.create_uniform_volume_field(rt, "nu", nfb.read_transport_viscosity(rt))
    turbulence = select_turbulence_model(cfg, fallback=False, runtime=rt, nu=nu, case_dir=case)
    turbulence.validate(U_neon)
    turbulence.correct(U_neon, phi_neon, rt)

    for name in COMPARE_FIELDS:
        try:
            field = turbulence.field(name)
        except KeyError:
            continue  # the model does not own this field (e.g. laminar has no k)
        np.save(
            case / f"subject_{name}.npy",
            np.asarray(field.internal_vector().copy_to_host()),
        )


def role_subject_fb(case: Path) -> None:
    """pybFoam-fallback handle via ``select(fallback=True)``, one op step → ``subject_fb_nut.npy``.

    Builds the model the way ``incompressibleFluid`` does — the merged-family
    :func:`select_turbulence_model` with ``fallback=True`` returns a
    :class:`FallbackHandle` whose ``.operations`` is the model file's
    ``fallback=True`` ``correct`` op. Running that op advances the wrapped pybFoam
    model's ``nut``. Because the fallback *is* pybFoam, this equals the reference —
    the point is to prove the handle + op-dispatch wiring actually advances ``nut``,
    not that two engines agree.
    """
    of_time = pyf.Time(str(case.parent), case.name)
    of_mesh = pyf.fvMesh(of_time)
    U = pyf.volVectorField.read_field(of_mesh, "U")
    phi = pyf.createPhi(U)
    transport = singlePhaseTransportModel(U, phi)
    cfg = TurbulencePropertiesConfig.load(case_dir=str(case))

    handle = select_turbulence_model(
        cfg, fallback=True, case_dir=str(case), U=U, phi=phi, transport=transport
    ).build()
    # The solver resolves the handle from ctx.models["turbulence"]; the fallback op
    # calls its no-arg correct().
    ctx = Context(fields={}, models={"turbulence": handle})
    for op in handle.operations:
        op.run(ctx)

    registered = pyf.volScalarField.from_registry(of_mesh, "nut")
    if registered is not None:
        np.save(case / "subject_fb_nut.npy", np.asarray(registered.internalField()))
    else:
        # laminar registers no nut; nut() is a fresh non-const zero tmp.
        np.save(
            case / "subject_fb_nut.npy",
            np.asarray(handle.nut().ref().internalField()),
        )


_ROLES = {
    "mesh": role_mesh,
    "setup": role_setup,
    "reference": role_reference,
    "subject": role_subject,
    "subject_fb": role_subject_fb,
}


def main() -> None:
    role, case_dir = sys.argv[1], Path(sys.argv[2])
    _ROLES[role](case_dir)
    # The role's result is already flushed to its .npy file. NeoN/Kokkos + OpenFOAM
    # teardown at normal interpreter exit is fragile (Kokkos finalize segfaults after
    # a solve), so skip atexit entirely with a hard, successful exit — a crash *here*
    # would otherwise mark the whole subprocess failed even though its work succeeded.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
