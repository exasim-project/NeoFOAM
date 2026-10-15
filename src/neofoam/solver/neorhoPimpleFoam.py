# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
neorhoPimpleFoam.py — Transient compressible PIMPLE solver using NeoN +
NeoFOAM bindings.

Minimal first version: laminar only, subsonic (no transonic pEqn branch), no
SIMPLEC/consistent variant. perfectGas equation of state, constant Cv,
constant transport (mu, Pr), read from constant/thermophysicalProperties.
Energy equation is solved for T directly (not he/e) since Cv is constant, so
EEqn divides through by Cv - this avoids needing to derive e's boundary
conditions from T's BCs, which NeoFOAM has no machinery for.

The pressure equation uses fvm::ddt(psi, p) in place of literal
rhoPimpleFoam's fvc::ddt(rho) + psi*correction(fvm::ddt(p)): NeoN's DSL has
no fvm::correction() equivalent, and this is a physically motivated
simplification (rho ~= psi*p with psi held fixed across the inner PISO loop,
same as OpenFOAM does internally). The literal ddtCorr flux correction and
the bounded thermo.correctRho() increment are both dropped for this first
version in favor of directly reassigning rho from the current psi/p.
"""

from __future__ import annotations

import sys
from typing import Any

import neon._neon as nn
import pybFoam as pyf

from neofoam import neofoam_bindings as nfb
from neofoam.solver.neon_runtime import (  # noqa: F401
    ensure_neon_initialized as _ensure_neon_initialized,
)
from neofoam.solver.neon_runtime import requested_executor
from neofoam.solver.pisoControl import PisoControl
from neofoam.thermo.perfectGas import PerfectGasConstTransport


def _read_int(d: Any, key: str, default: int) -> int:
    return int(d.get_int(key)) if d.contains(key) else default


def _read_switch(d: Any, key: str, default: bool) -> bool:
    if not d.contains(key):
        return default
    try:
        return bool(d.get_bool(key))
    except Exception:
        return d.get_string(key).strip().lower() in ("yes", "true", "on", "1")


class NeoRhoPimpleFoam:
    """Transient compressible PIMPLE solver backed by NeoN (laminar, minimal scope)."""

    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def run(self) -> None:
        _ensure_neon_initialized(self._argv)
        self._run()

    def _run(self) -> None:
        argList = pyf.argList(self._argv)
        runTime = pyf.Time(argList)
        rt = nfb.create_adapter_run_time(runTime, requested_executor())

        pimple_dict = rt.fv_solution_dict.subDict("PIMPLE")
        piso = PisoControl(
            n_correctors=_read_int(pimple_dict, "nCorrectors", 1),
            n_non_orthogonal_correctors=_read_int(pimple_dict, "nNonOrthogonalCorrectors", 0),
            momentum_predictor=_read_switch(pimple_dict, "momentumPredictor", True),
        )

        rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
        solvers = rt.fv_solution_dict.subDict("solvers")
        for name in ("p", "U", "rho", "T", "pFinal", "UFinal", "rhoFinal", "TFinal"):
            if solvers.contains(name):
                solvers.insert_dict(name, nfb.map_fv_solution(solvers.subDict(name)))

        gas = PerfectGasConstTransport(rt)

        p = nfb.read_scalar_volume_field(rt, "p")
        T = nfb.read_scalar_volume_field(rt, "T")
        U = nfb.read_vector_volume_field(rt, "U")

        psi_seed = nfb.create_uniform_volume_field(rt, "psi", 0.0)
        psi_seed.assign(gas.psi(T))
        psi = nfb.register_computed_scalar_field(rt, psi_seed, "psi")

        rho_seed = nfb.create_uniform_volume_field(rt, "rho", 0.0)
        rho_seed.assign(gas.rho(psi, p))
        rho = nfb.register_computed_scalar_field(rt, rho_seed, "rho")

        K_seed = nfb.create_uniform_volume_field(rt, "K", 0.0)
        K_seed.assign(0.5 * nn.mag_sqr(U))
        K = nfb.register_computed_scalar_field(rt, K_seed, "K")

        nn.old_time(psi)
        nn.old_time(rho)
        nn.old_time(K)

        mu_vol = nfb.create_uniform_volume_field(rt, "mu", gas.mu)
        zero_nut = nfb.create_uniform_volume_field(rt, "nut0", 0.0)
        alphaEff = nfb.create_uniform_surface_field(rt, "alphaEff", gas.alpha)

        grad_op = nfb.GaussGreenGrad(rt)
        surf_interp = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )

        rho_face = surf_interp.interpolate(rho)
        phi = rho_face * nfb.flux(U)
        phi.name = "phi"

        p_ref_cell, p_ref_value, needs_ref = nfb.set_ref_cell(rt, "p", "PIMPLE")
        pimple_loop = nfb.PimpleControl(rt.fv_solution_dict)

        def reduce_u(stats: Any) -> tuple[float, float]:
            mi = 0.0
            mf = 0.0
            for e in stats.entries:
                mi = max(mi, e.initial_residual)
                mf = max(mf, e.final_residual)
            return (mi, mf)

        while runTime.loop():
            print(f"Time = {runTime.timeName()}")

            nn.rotate_old_times(U)
            nn.rotate_old_times(T)
            nn.rotate_old_times(p)
            nn.rotate_old_times(rho)
            nn.rotate_old_times(psi)
            nn.rotate_old_times(K)

            max_co, mean_co = nn.compute_co_num(phi, rt.dt)
            print(f"Courant Number mean: {mean_co:.6f} max: {max_co:.6f}")
            nfb.sync_run_times(runTime, rt, max_co)

            residuals: dict[str, tuple[float, float]] = {}
            while pimple_loop.loop(residuals):
                final_iter = pimple_loop.final_iter()

                prev_p = nfb.field_relaxation_snapshot(p)

                grad_u = grad_op.grad_tensor(U)
                mu_face = surf_interp.interpolate(rho) * gas.mu
                mu_face.name = "mu"

                UEqn = nfb.PDESolverVec3(
                    nn.imp.ddt(rho, U)
                    + nn.imp.div(phi, U)
                    - nn.imp.laplacian(mu_face, U)
                    + nfb.viscous_stress(mu_vol, zero_nut, grad_u),
                    U,
                    rt,
                )

                ddt_scheme = UEqn.ddt_scheme()
                if ddt_scheme == getattr(nfb.DdtScheme, "None"):
                    raise RuntimeError(
                        "neorhoPimpleFoam: steadyState ddt unsupported (BDF1/BDF2 only)"
                    )

                UEqn.set_final_iter(final_iter)
                UEqn.relax()

                if piso.momentum_predictor():
                    stats_u = UEqn.solve_with_source(-1.0 * nn.exp.grad(p))
                    residuals["U"] = reduce_u(stats_u)
                    K.assign(0.5 * nn.mag_sqr(U))
                else:
                    UEqn.assemble_and_relax()

                phiv = phi / surf_interp.interpolate(rho)
                phiv.name = "phiv"
                EEqn = nfb.PDESolverScalar(
                    nn.imp.ddt(rho, T)
                    + nn.imp.div(phi, T)
                    + (1.0 / gas.Cv) * nn.exp.ddt(rho, K)
                    + (1.0 / gas.Cv) * nn.exp.div(phi, K)
                    + (1.0 / gas.Cv) * nn.exp.div(phiv, p)
                    - nn.imp.laplacian(alphaEff, T),
                    T,
                    rt,
                )
                EEqn.set_final_iter(final_iter)
                EEqn.relax()
                stats_t = EEqn.solve()
                residuals["T"] = (
                    stats_t.entries[0].initial_residual,
                    stats_t.entries[0].final_residual,
                )
                T.correct_boundary_conditions()
                psi.assign(gas.psi(T))
                rho.assign(gas.rho(psi, p))

                p_res: tuple[float, float] = (0.0, 0.0)
                have_p_res = False

                while piso.correct():
                    rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
                    nfb.constrain_hbya(U, p, hByA)

                    rho_face = surf_interp.interpolate(rho)
                    rAUf = surf_interp.interpolate(rAU)
                    rAUf.name = "rAUf"
                    rhorAUf = surf_interp.interpolate(rho * rAU)
                    rhorAUf.name = "rhorAUf"
                    phiHbyA = rho_face * nfb.flux(hByA)

                    while piso.correct_non_orthogonal():
                        pEqn = nfb.PDESolverScalar(
                            nn.imp.ddt(psi, p)
                            + nn.exp.div(phiHbyA)
                            - nn.imp.laplacian(rhorAUf, p),
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

                    nfb.update_velocity(hByA, rAU, p, U, rt)
                    U.correct_boundary_conditions()
                    rho.assign(gas.rho(psi, p))

                if have_p_res:
                    residuals["p"] = p_res

            if runTime.outputTime():
                print("Writing fields")
                nfb.write_scalar_field(p, rt)
                nfb.write_scalar_field(T, rt)
                nfb.write_scalar_field(rho, rt)
                nfb.write_vector_field(U, rt)

            runTime.printExecutionTime()

        print("End")


def main() -> None:
    argv = sys.argv if len(sys.argv) > 1 else ["neorhoPimpleFoam"]
    solver = NeoRhoPimpleFoam(argv)
    solver.run()


if __name__ == "__main__":
    main()
