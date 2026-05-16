# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
neoIcoFoam.py — Incompressible Navier-Stokes solver using NeoN + NeoFOAM bindings.
"""

import pybFoam as pyf
import neon._neon as nn  # NeoN Python bindings
from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings
from neofoam.solver.pisoControl import PisoControl


class NeoIcoFoam:
    """PISO-based incompressible Navier-Stokes solver backed by NeoN."""

    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def run(self) -> None:
        nn.initialize(self._argv)

        argList = pyf.argList(self._argv)

        runTime = pyf.Time(argList)

        rt = nfb.create_adapter_run_time(runTime)

        piso_dict = rt.fv_solution_dict.subDict("PISO")
        piso = PisoControl(
            n_correctors=piso_dict.get_int("nCorrectors"),
            n_non_orthogonal_correctors=piso_dict.get_int("nNonOrthogonalCorrectors"),
            momentum_predictor=piso_dict.get_string("momentumPredictor").lower()
            in ("yes", "true", "on", "1"),
        )

        # Map OpenFOAM dictionaries to NeoN equivalents
        solvers = rt.fv_solution_dict.subDict("solvers")
        solvers.insert_dict("p", nfb.map_fv_solution(solvers.subDict("p")))
        solvers.insert_dict("U", nfb.map_fv_solution(solvers.subDict("U")))
        rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)

        nu_value = nfb.read_transport_viscosity(rt)
        nu = nfb.create_uniform_surface_field(rt, "nu", nu_value)

        p = nfb.read_scalar_volume_field(rt, "p")
        U = nfb.read_vector_volume_field(rt, "U")
        phi = nfb.create_phi(rt, "U")

        p_ref_cell, p_ref_value, needs_ref = nfb.set_ref_cell(rt, "p", "PISO")

        while runTime.loop():
            print(f"Time = {runTime.timeName()}")

            nn.rotate_old_times(U)
            nn.rotate_old_times(phi)

            max_co, mean_co = nn.compute_co_num(
                rt.nf_mesh, phi.internal_vector(), rt.dt
            )
            print(f"Courant Number mean: {mean_co:.6f} max: {max_co:.6f}")

            nfb.sync_run_times(runTime, rt, max_co)

            UEqn = nfb.PDESolverVec3(
                nn.imp.ddt(U) + nn.imp.div(phi, U) - nn.imp.laplacian(nu, U),
                U,
                rt,
            )
            ddt_scheme = UEqn.ddt_scheme()

            if piso.momentum_predictor():
                UEqn.solve_with_source(-1.0 * nn.exp.grad(p))
            else:
                UEqn.assemble()

            while piso.correct():
                rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
                nfb.constrain_hbya(U, p, hByA)

                # Interpolate rAU to faces
                interp = nn.SurfaceInterpolationScalar(
                    rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
                )
                rAUf = interp.interpolate(rAU)
                rAUf.name = "rAUf"

                phiHbyA = nfb.flux(hByA) + rAUf * nfb.ddt_flux_corr(
                    U, phi, rt.dt, ddt_scheme
                )

                while piso.correct_non_orthogonal():
                    pEqn = nfb.PDESolverScalar(
                        nn.imp.laplacian(rAUf, p) - nn.exp.div(phiHbyA),
                        p,
                        rt,
                    )

                    if needs_ref:
                        pEqn.set_reference(p_ref_cell, p_ref_value)

                    pEqn.solve()
                    p.correct_boundary_conditions()

                    if piso.final_non_orthogonal_iter():
                        nfb.update_face_velocity(phiHbyA, pEqn, phi)

                nfb.update_velocity(hByA, rAU, p, U)
                U.correct_boundary_conditions()

            if runTime.outputTime():
                nfb.write_scalar_field(p, rt)
                nfb.write_vector_field(U, rt)

            runTime.printExecutionTime()

        print("End")
        nn.finalize()


def main() -> None:
    import sys

    argv = sys.argv if len(sys.argv) > 1 else ["neoIcoFoam"]
    solver = NeoIcoFoam(argv)
    solver.run()


if __name__ == "__main__":
    main()
