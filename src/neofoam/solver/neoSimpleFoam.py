# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
neoSimpleFoam.py — Steady-state incompressible SIMPLE/SIMPLEC solver with
turbulence, using NeoN + NeoFOAM bindings.

Python port of ``examples/neoSimpleFoam/neoSimpleFoam.cpp``: one pressure-
velocity coupling pass per outer iteration (no inner PISO corrector loop),
with an optional non-orthogonal corrector loop, the NeoN-backed analogue of
OpenFOAM's ``simpleFoam``.

The momentum stress follows the same ``divDevReff(U)`` decomposition as
``neoPimpleFoam.py``. ``nuEff``/``nut``/``gradU`` come from a runtime-selected
turbulence model (``laminar`` or LES ``SpalartAllmarasDDES``, per
``constant/turbulenceProperties``), corrected once per outer iteration.
SIMPLEC (``consistent`` switch in the SIMPLE subdict) swaps the plain
reciprocal diagonal rAU for the consistent rAtU and applies the matching
flux/HbyA corrections - see ``examples/neoSimpleFoam/neoSimpleFoam.cpp``.
"""

from __future__ import annotations

import sys
from typing import Any

import neon._neon as nn  # NeoN Python bindings
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
from neofoam.solver.neon_runtime import ensure_neon_initialized, requested_executor
from neofoam.solver.simpleControl import SimpleControl


def _read_int(d: Any, key: str, default: int) -> int:
    return int(d.get_int(key)) if d.contains(key) else default


def _read_switch(d: Any, key: str, default: bool) -> bool:
    """Read an OpenFOAM on/off switch, tolerating word or bool storage."""
    if not d.contains(key):
        return default
    try:
        return bool(d.get_bool(key))
    except Exception:
        return d.get_string(key).strip().lower() in ("yes", "true", "on", "1")


class NeoSimpleFoam:
    """Steady-state incompressible SIMPLE/SIMPLEC solver backed by NeoN."""

    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def run(self) -> None:
        ensure_neon_initialized(self._argv)
        self._run()

    def _run(self) -> None:
        argList = pyf.argList(self._argv)
        runTime = pyf.Time(argList)
        rt = nfb.create_adapter_run_time(runTime, requested_executor())

        simple_dict = rt.fv_solution_dict.subDict("SIMPLE")
        simple = SimpleControl(
            n_non_orthogonal_correctors=_read_int(simple_dict, "nNonOrthogonalCorrectors", 0),
            momentum_predictor=_read_switch(simple_dict, "momentumPredictor", True),
            consistent=_read_switch(simple_dict, "consistent", False),
        )

        rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
        solvers = rt.fv_solution_dict.subDict("solvers")
        for name in ("p", "U", "nuTilda"):
            if solvers.contains(name):
                solvers.insert_dict(name, nfb.map_fv_solution(solvers.subDict(name)))

        nu_value = nfb.read_transport_viscosity(rt)
        nu_vol = nfb.create_uniform_volume_field(rt, "nu", nu_value)

        p = nfb.read_scalar_volume_field(rt, "p")
        U = nfb.read_vector_volume_field(rt, "U")
        phi = nfb.create_phi(rt, "U")

        turb = nfb.create_turbulence_model(rt, nu_vol)
        turb.validate(U)
        grad_op = nfb.GaussGreenGrad(rt)

        p_ref_cell, p_ref_value, needs_ref = nfb.set_ref_cell(rt, "p", "SIMPLE")

        surf_interp = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )

        cumulative_cont_err = 0.0

        while runTime.loop():
            print(f"Time = {runTime.timeName()}")

            grad_u = grad_op.grad_tensor(U)

            UEqn = nfb.PDESolverVec3(
                nn.imp.div(phi, U)
                - nn.imp.laplacian(turb.nu_eff(), U)
                + nfb.viscous_stress(nu_vol, turb.nut(), grad_u),
                U,
                rt,
            )

            UEqn.relax()

            if simple.momentum_predictor():
                UEqn.solve_with_source(-1.0 * nn.exp.grad(p))
            else:
                UEqn.assemble_and_relax()

            # SIMPLE / SIMPLEC pressure-velocity coupling: a single pass, no inner
            # PISO loop.
            rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
            nfb.constrain_hbya(U, p, hByA)

            # SIMPLEC: rAtU absorbs off-diagonal coupling; for plain SIMPLE rAtU == rAU.
            consistent = simple.consistent()
            rAtU = nfb.compute_ratu(UEqn, rAU) if consistent else rAU

            rAUf = surf_interp.interpolate(rAtU)
            rAUf.name = "rAUf"

            phiHbyA = nfb.flux(hByA)

            if consistent:
                nfb.add_consistent_flux_correction(phiHbyA, rAU, rAtU, p, rt.fv_schemes_dict)
                nfb.subtract_consistent_hbya(hByA, rAU, rAtU, p)

            prev_p = nfb.field_relaxation_snapshot(p)

            while simple.correct_non_orthogonal():
                pEqn = nfb.PDESolverScalar(
                    nn.imp.laplacian(rAUf, p) - nn.exp.div(phiHbyA),
                    p,
                    rt,
                )

                if needs_ref:
                    pEqn.set_reference(p_ref_cell, p_ref_value)

                pEqn.solve()
                p.correct_boundary_conditions()

                if simple.final_non_orthogonal_iter():
                    nfb.update_face_velocity(phiHbyA, pEqn, phi)

            sum_local, global_err = nfb.compute_continuity_error(phi, rt)
            cumulative_cont_err += global_err
            print(
                f"time step continuity errors : sum local = {sum_local}, "
                f"global = {global_err}, cumulative = {cumulative_cont_err}"
            )

            nfb.apply_field_relaxation(
                p, prev_p, nfb.lookup_field_relaxation(rt.fv_solution_dict, p.name, False)
            )
            p.correct_boundary_conditions()

            nfb.update_velocity(hByA, rAtU, p, U, rt)
            U.correct_boundary_conditions()

            turb.correct(U, phi, rt)

            if runTime.outputTime():
                print("Writing fields")
                nfb.write_scalar_field(p, rt)
                nfb.write_vector_field(U, rt)
                turb.write(rt.mesh)

            runTime.printExecutionTime()

        print("End")


def main() -> None:
    argv = sys.argv if len(sys.argv) > 1 else ["neoSimpleFoam"]
    solver = NeoSimpleFoam(argv)
    solver.run()


if __name__ == "__main__":
    main()
