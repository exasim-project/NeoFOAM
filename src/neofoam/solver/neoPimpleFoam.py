# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
neoPimpleFoam.py — Transient incompressible PIMPLE solver using NeoN + NeoFOAM
bindings.

Python port of ``examples/neoPimpleFoam/neoPimpleFoam.cpp``: a PISO inner
corrector loop nested inside an outer, residual-driven PIMPLE loop. This is the
NeoN-backed analogue of OpenFOAM's ``pimpleFoam``.

The momentum stress follows ``pimpleFoam``'s ``divDevReff(U)`` decomposition,
``-laplacian(nuEff,U) - div(nuEff*dev2(T(grad(U))))``: the implicit laplacian
plus the explicit dev2 viscous-stress term. ``nuEff``/``nut``/``gradU`` come from
a runtime-selected turbulence model (``nf::TurbulenceModel::create``) chosen from
``constant/turbulenceProperties`` — ``laminar`` (nut = 0) or the LES
``SpalartAllmarasDDES`` model, corrected once per time step after the PIMPLE loop.
This mirrors ``examples/neoPimpleFoam/neoPimpleFoam.cpp``.
"""

from __future__ import annotations

import sys
from typing import Any

import neon._neon as nn  # NeoN Python bindings
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings

# Compatibility re-export: the process-wide Kokkos init guard lives in
# neofoam.solver.neon_runtime under its public name.
from neofoam.solver.neon_runtime import (  # noqa: F401
    ensure_neon_initialized as _ensure_neon_initialized,
)
from neofoam.solver.pisoControl import PisoControl


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


class NeoPimpleFoam:
    """Transient incompressible PIMPLE solver backed by NeoN."""

    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def run(self) -> None:
        _ensure_neon_initialized(self._argv)
        self._run()

    def _run(self) -> None:
        argList = pyf.argList(self._argv)
        runTime = pyf.Time(argList)
        rt = nfb.create_adapter_run_time(runTime)

        # Inner-corrector counts live in the "PIMPLE" subdict for a stock pimpleFoam
        # case (there is no "PISO" block); mirror neoPimpleFoam.cpp / pimpleParity.cpp.
        pimple_dict = rt.fv_solution_dict.subDict("PIMPLE")
        piso = PisoControl(
            n_correctors=_read_int(pimple_dict, "nCorrectors", 1),
            n_non_orthogonal_correctors=_read_int(pimple_dict, "nNonOrthogonalCorrectors", 0),
            # OpenFOAM's solutionControl defaults momentumPredictor to true.
            momentum_predictor=_read_switch(pimple_dict, "momentumPredictor", True),
        )

        # Map OpenFOAM dictionaries to NeoN/Ginkgo equivalents. Map the base solver
        # subdicts AND the *Final subdicts so the final outer-corrector pass selects a
        # converted dict.
        rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
        solvers = rt.fv_solution_dict.subDict("solvers")
        for name in ("p", "U", "pFinal", "UFinal", "nuTilda", "nuTildaFinal"):
            if solvers.contains(name):
                solvers.insert_dict(name, nfb.map_fv_solution(solvers.subDict(name)))

        nu_value = nfb.read_transport_viscosity(rt)
        # Volume nu for the explicit dev2 viscous-stress term (nuEff/nut come from
        # the turbulence model below).
        nu_vol = nfb.create_uniform_volume_field(rt, "nu", nu_value)
        grad_op = nfb.GaussGreenGrad(rt)

        p = nfb.read_scalar_volume_field(rt, "p")
        U = nfb.read_vector_volume_field(rt, "U")
        phi = nfb.create_phi(rt, "U")

        # Runtime-selected turbulence model (laminar or LES SpalartAllmarasDDES,
        # per constant/turbulenceProperties). Owns nut / nuEff / gradU; for laminar
        # nut = 0 and nuEff = nu, reproducing the inline laminar path.
        turb = nfb.create_turbulence_model(rt, nu_vol)
        turb.validate(U)

        p_ref_cell, p_ref_value, needs_ref = nfb.set_ref_cell(rt, "p", "PIMPLE")

        surf_interp = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )

        cumulative_cont_err = 0.0
        pimple_loop = nfb.PimpleControl(rt.fv_solution_dict)

        def reduce_u(stats: Any) -> tuple[float, float]:
            # Max-component reduction: Ux/Uy/Uz entries -> one {init, final} pair.
            mi = 0.0
            mf = 0.0
            for e in stats.entries:
                mi = max(mi, e.initial_residual)
                mf = max(mf, e.final_residual)
            return (mi, mf)

        while runTime.loop():
            print(f"Time = {runTime.timeName()}")

            nn.rotate_old_times(U)
            nn.rotate_old_times(phi)
            turb.rotate_old_times()

            max_co, mean_co = nn.compute_co_num(phi, rt.dt)
            print(f"Courant Number mean: {mean_co:.6f} max: {max_co:.6f}")
            nfb.sync_run_times(runTime, rt, max_co)

            residuals: dict[str, tuple[float, float]] = {}
            while pimple_loop.loop(residuals):
                final_iter = pimple_loop.final_iter()

                # Snapshot p at the top of each outer corrector to blend against
                # after the pressure solve (explicit field under-relaxation).
                prev_p = nfb.field_relaxation_snapshot(p)

                # grad(U) for the explicit dev2 viscous stress. Keep ``grad_u`` alive
                # until the equation is assembled (the operator holds a reference).
                grad_u = grad_op.grad_tensor(U)

                # Full pimpleFoam momentum stress divDevReff(U) =
                # -laplacian(nuEff,U) - div(nuEff*dev2(T(grad(U)))): the implicit
                # laplacian plus the explicit dev2 viscous-stress term. nuEff and nut
                # come from the turbulence model (fixed across the PIMPLE loop, as in
                # OpenFOAM, where the model is corrected once per time step).
                UEqn = nfb.PDESolverVec3(
                    nn.imp.ddt(U)
                    + nn.imp.div(phi, U)
                    - nn.imp.laplacian(turb.nu_eff(), U)
                    + nfb.viscous_stress(nu_vol, turb.nut(), grad_u),
                    U,
                    rt,
                )

                ddt_scheme = UEqn.ddt_scheme()
                if ddt_scheme == getattr(nfb.DdtScheme, "None"):
                    raise RuntimeError(
                        "neoPimpleFoam: steadyState ddt unsupported (BDF1/BDF2 only)"
                    )

                UEqn.set_final_iter(final_iter)
                # pimpleFoam/UEqn.H: UEqn.relax(). The pressure equation below is
                # never equation-relaxed (only the p *field* is, after the solve).
                UEqn.relax()

                if piso.momentum_predictor():
                    stats_u = UEqn.solve_with_source(-1.0 * nn.exp.grad(p))
                    residuals["U"] = reduce_u(stats_u)
                else:
                    # Relax unconditionally so computeRAUandHByA reads the relaxed
                    # diagonal even when the momentum predictor is disabled.
                    UEqn.assemble_and_relax()

                p_res: tuple[float, float] = (0.0, 0.0)
                have_p_res = False

                while piso.correct():
                    rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
                    nfb.constrain_hbya(U, p, hByA)

                    rAUf = surf_interp.interpolate(rAU)
                    rAUf.name = "rAUf"

                    phiHbyA = nfb.flux(hByA) + rAUf * nfb.ddt_flux_corr(U, phi, rt.dt, ddt_scheme)

                    while piso.correct_non_orthogonal():
                        pEqn = nfb.PDESolverScalar(
                            nn.imp.laplacian(rAUf, p) - nn.exp.div(phiHbyA),
                            p,
                            rt,
                        )
                        pEqn.set_final_iter(final_iter)

                        if needs_ref:
                            pEqn.set_reference(p_ref_cell, p_ref_value)

                        stats_p = pEqn.solve()
                        if not have_p_res:
                            entry = stats_p.entries[0]
                            p_res = (entry.initial_residual, entry.final_residual)
                            have_p_res = True
                        p.correct_boundary_conditions()

                        if piso.final_non_orthogonal_iter():
                            nfb.update_face_velocity(phiHbyA, pEqn, phi)

                    nfb.apply_field_relaxation(
                        p,
                        prev_p,
                        nfb.lookup_field_relaxation(rt.fv_solution_dict, p.name, final_iter),
                    )
                    p.correct_boundary_conditions()

                    sum_local, global_err = nfb.compute_continuity_error(phi, rt)
                    cumulative_cont_err += global_err
                    print(
                        f"time step continuity errors : sum local = {sum_local}, "
                        f"global = {global_err}, cumulative = {cumulative_cont_err}"
                    )

                    nfb.update_velocity(hByA, rAU, p, U)
                    U.correct_boundary_conditions()

                if have_p_res:
                    residuals["p"] = p_res

            # Update the turbulence model once per time step (after the PIMPLE loop):
            # solves the nuTilda transport PDE and refreshes nut/gradU for SA-DDES;
            # a no-op recompute for laminar.
            turb.correct(U, phi, rt)

            if runTime.outputTime():
                print("Writing fields")
                nfb.write_scalar_field(p, rt)
                nfb.write_vector_field(U, rt)
                turb.write(rt.mesh)

            runTime.printExecutionTime()

        print("End")


def main() -> None:
    argv = sys.argv if len(sys.argv) > 1 else ["neoPimpleFoam"]
    solver = NeoPimpleFoam(argv)
    solver.run()


if __name__ == "__main__":
    main()
