# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""neoInterFoam.py — NeoN-backed interFoam-style incompressible two-phase VoF solver.

Builds the run-time, reads the damBreak fields (alpha.water, U, p_rgh), creates phi,
rhoPhi and the initial two-phase property fields (rho, mu, gh, ghf), and advances the
phase fraction with the NeoN-core bounded FCT MULES limiter (no clamp), rebuilding
rho/mu/rhoPhi from the limited flux, then runs a density-weighted momentum + p_rgh
PISO correction each step. Mirrors neoPimpleFoam.py.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from typing import Any

import pybFoam as pyf
import neon._neon as nn  # NeoN Python bindings
from neofoam import neofoam_bindings as nfb  # NeoFOAM Python bindings

_neon_initialized = False


def _ensure_neon_initialized(argv: list[str]) -> None:
    global _neon_initialized
    if not _neon_initialized:
        nn.initialize(argv)
        _neon_initialized = True
        atexit.register(nn.finalize)


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


@dataclass
class VoFState:
    """Live handles for one VoF run. Holds arg_list/run_time to keep the
    Foam::Time -> argList raw reference alive (else field reads return nan)."""

    arg_list: Any
    run_time: Any
    rt: Any
    alpha1: Any
    U: Any
    p_rgh: Any
    p: Any
    phi: Any
    rhoPhi: Any
    rho: Any
    mu: Any
    gh: Any
    ghf: Any
    phase: dict[str, float]
    gravity: tuple[float, float, float]
    p_ref_cell: int
    p_ref_value: float
    needs_ref: bool
    n_correctors: int
    n_non_orth: int
    momentum_predictor: bool
    n_limiter_iter: int


class NeoInterFoam:
    """interFoam-style two-phase VoF solver backed by NeoN.

    MULES-limited alpha advection (no clamp), density-weighted momentum with gravity
    and surface tension, and a p_rgh PISO pressure correction. Mirrors neoPimpleFoam.py.
    """

    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def setup(self) -> VoFState:
        _ensure_neon_initialized(self._argv)

        arg_list = pyf.argList(self._argv)
        run_time = pyf.Time(arg_list)
        rt = nfb.create_adapter_run_time(run_time)

        # Dict mapping is only needed once we solve; map defensively what exists.
        rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
        solvers = rt.fv_solution_dict.subDict("solvers")

        # MULES limiter sweeps for the alpha equation. The alpha linear-solver subdict
        # is keyed by an OpenFOAM regex (damBreak: "alpha.water.*"); the NeoN dictionary
        # stores that pattern verbatim and does NOT do regex matching, so look the key
        # up explicitly by its "alpha.water" prefix (no bare-except silent default).
        n_limiter_iter = 3
        alpha_solver_key = next(
            (k for k in solvers.keys() if k.startswith("alpha.water")), None
        )
        if alpha_solver_key is not None:
            n_limiter_iter = _read_int(
                solvers.subDict(alpha_solver_key), "nLimiterIter", 3
            )

        for name in ("p_rgh", "p_rghFinal", "U", "UFinal"):
            if solvers.contains(name):
                solvers.insert_dict(name, nfb.map_fv_solution(solvers.subDict(name)))

        # The alpha step uses the MULES primitive directly (no fvm div solve), so no
        # alpha convection scheme / linear-solver subdict is required here.
        div_schemes = rt.fv_schemes_dict.subDict("divSchemes")
        # Momentum/pressure scheme keys resolved by the new implicit operators.
        div_schemes.insert_token_list(
            "div(rhoPhi,U)", nn.TokenList(["Gauss", "linear"])
        )
        lap = rt.fv_schemes_dict.subDict("laplacianSchemes")
        for key in ("laplacian(muf,U)", "laplacian(rAUf,p_rgh)"):
            lap.insert_token_list(key, nn.TokenList(["Gauss", "linear", "uncorrected"]))

        pimple = rt.fv_solution_dict.subDict("PIMPLE")
        n_correctors = _read_int(pimple, "nCorrectors", 1)
        n_non_orth = _read_int(pimple, "nNonOrthogonalCorrectors", 0)
        momentum_predictor = _read_switch(pimple, "momentumPredictor", True)
        p_ref_cell, p_ref_value, needs_ref = nfb.set_ref_cell(rt, "p_rgh", "PIMPLE")

        phase = nfb.read_two_phase_transport_properties(rt)
        gravity = nfb.read_gravity(rt)

        alpha1 = nfb.read_scalar_volume_field(rt, "alpha.water")
        U = nfb.read_vector_volume_field(rt, "U")
        p_rgh = nfb.read_scalar_volume_field(rt, "p_rgh")
        # damBreak ships no `p` field; the static pressure p = p_rgh + rho*gh is a derived
        # output. Create it with calculated BCs and fill it after the pressure solve.
        p = nfb.create_uniform_volume_field(rt, "p", 0.0)
        phi = nfb.create_phi(rt, "U")
        rho_phi = nfb.create_rho_phi(rt)

        rho = nfb.create_mixture_density(rt)
        mu = nfb.create_mixture_viscosity(rt)
        gh = nfb.create_gh(rt)
        ghf = nfb.create_ghf(rt)

        return VoFState(
            arg_list,
            run_time,
            rt,
            alpha1,
            U,
            p_rgh,
            p,
            phi,
            rho_phi,
            rho,
            mu,
            gh,
            ghf,
            phase,
            gravity,
            p_ref_cell,
            p_ref_value,
            needs_ref,
            n_correctors,
            n_non_orth,
            momentum_predictor,
            n_limiter_iter,
        )

    def advect_alpha(self, state: VoFState) -> None:
        """One MULES-limited phase-fraction step.

        Builds the high-order face flux alphaPhiUn = linearInterp(alpha1)*phi, then
        calls the NeoN-core FCT limiter nn.mules_explicit_solve which limits the flux
        in place AND advances alpha1 conservatively (no clamp — boundedness comes from
        the limiter). Recomputes rho/mu and rebuilds rhoPhi from the *limited* flux:
        rhoPhi = alphaPhi*(rho1-rho2) + phi*rho2. First MULES: nAlphaCorr=1, no
        subcycling, cAlpha=0 (no interface compression), MULESCorr=false.
        """
        nn.rotate_old_times(state.alpha1)
        surf = nn.SurfaceInterpolationScalar(
            state.rt.executor, state.rt.nf_mesh, nn.TokenList(["linear"])
        )
        alphaf = surf.interpolate(state.alpha1)  # high-order (linear) face alpha
        alpha_phi = alphaf * state.phi
        alpha_phi.name = "alphaPhi"
        nn.mules_explicit_solve(
            state.alpha1,
            state.phi,
            alpha_phi,
            state.rt.dt,
            1.0,
            0.0,
            state.n_limiter_iter,
        )
        state.alpha1.correct_boundary_conditions()

        rho1, rho2 = state.phase["rho1"], state.phase["rho2"]
        nfb.update_mixture_density(state.rho, state.alpha1, rho1, rho2)
        nfb.update_mixture_viscosity(
            state.mu, state.alpha1, rho1, rho2, state.phase["nu1"], state.phase["nu2"]
        )
        # Rebuild rhoPhi from the LIMITED alpha flux (conservation-consistent with
        # ddt(rho,U)): rhoPhi = alphaPhi*(rho1-rho2) + phi*rho2.
        rho_phi = alpha_phi * (rho1 - rho2) + rho2 * state.phi
        state.rhoPhi.assign(rho_phi)
        state.rho.correct_boundary_conditions()
        state.mu.correct_boundary_conditions()

    def momentum_pressure(self, state: VoFState) -> None:
        """Density-weighted momentum predictor + a PISO p_rgh pressure correction.

        UEqn = ddt(rho,U) + div(rhoPhi,U) - laplacian(muf,U), solved against the buoyant +
        capillary source reconstruct((fSigma - ghf*snGrad(rho) - snGrad(p_rgh))*magSf); then
        the p_rgh pressure loop with phig = (fSigma - ghf*snGrad(rho))*rAUf*magSf, where
        fSigma = interpolate(sigma*K)*snGrad(alpha1) is the surface-tension face force.
        Deferred: dev2 viscous stress, non-orthogonal correction. Recomputes p = p_rgh+rho*gh.
        """
        nn.rotate_old_times(state.U)
        nn.rotate_old_times(state.phi)

        surf = nn.SurfaceInterpolationScalar(
            state.rt.executor, state.rt.nf_mesh, nn.TokenList(["linear"])
        )
        muf = surf.interpolate(state.mu)
        muf.name = "muf"

        UEqn = nfb.PDESolverVec3(
            nn.imp.ddt(state.rho, state.U)
            + nn.imp.div(state.rhoPhi, state.U)
            - nn.imp.laplacian(muf, state.U),
            state.U,
            state.rt,
        )
        ddt_scheme = UEqn.ddt_scheme()

        sn_rho = nfb.sn_grad(state.rho)
        sn_prgh = nfb.sn_grad(state.p_rgh)
        magSf = nfb.mag_sf(state.rt)
        # Surface-tension face force fSigma = interpolate(sigma*K)*snGrad(alpha1), computed
        # once (curvature depends on alpha1, fixed across the corrector loop).
        fSigma = nfb.surface_tension_force(state.rt, state.alpha1, state.phase["sigma"])
        # Buoyant + capillary face force flux: (fSigma - ghf*snGrad(rho) - snGrad(p_rgh))*magSf.
        face_force = (fSigma + (-1.0 * state.ghf) * sn_rho - sn_prgh) * magSf
        src_field = nn.reconstruct(face_force)

        if state.momentum_predictor:
            UEqn.solve_with_source(nn.exp.source(src_field))
        else:
            UEqn.assemble_and_relax()

        for _ in range(state.n_correctors):
            rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
            nfb.constrain_hbya(state.U, state.p_rgh, hByA)
            rAUf = surf.interpolate(rAU)
            rAUf.name = "rAUf"
            # ddtCorr weight is interFoam's interpolate(rho*rAU) (not interpolate(rAU)):
            # near the interface rho jumps ~1000x, so the transient Rhie-Chow correction
            # must carry the density weight. The laplacian/phig terms keep plain rAUf.
            rho_rau = nfb.mul_scalar_volume(state.rho, rAU)
            rhorAUf = surf.interpolate(rho_rau)
            rhorAUf.name = "rhorAUf"
            phig = (fSigma + (-1.0 * state.ghf) * sn_rho) * rAUf * magSf
            phiHbyA = (
                nfb.flux(hByA)
                + rhorAUf
                * nfb.ddt_flux_corr(state.U, state.phi, state.rt.dt, ddt_scheme)
                + phig
            )

            pEqn = None
            for _ in range(state.n_non_orth + 1):
                pEqn = nfb.PDESolverScalar(
                    nn.imp.laplacian(rAUf, state.p_rgh) - nn.exp.div(phiHbyA),
                    state.p_rgh,
                    state.rt,
                )
                if state.needs_ref:
                    pEqn.set_reference(state.p_ref_cell, state.p_ref_value)
                pEqn.solve()
                state.p_rgh.correct_boundary_conditions()
            nfb.update_face_velocity(phiHbyA, pEqn, state.phi)

            sum_local, global_err = nfb.compute_continuity_error(state.phi, state.rt)
            print(f"continuity: local={sum_local}, global={global_err}")

            nfb.update_velocity(hByA, rAU, state.p_rgh, state.U)
            state.U.correct_boundary_conditions()

        # Static pressure p = p_rgh + rho*gh (derived output).
        nfb.update_static_pressure(state.p, state.p_rgh, state.rho, state.gh)

    def run(self) -> None:
        state = self.setup()
        print(f"phases: {state.phase}")
        print(f"g = {state.gravity}")
        print(f"nCells = {state.alpha1.size()}")
        while state.run_time.loop():
            print(f"Time = {state.run_time.timeName()}")
            max_co, mean_co = nn.compute_co_num(state.phi, state.rt.dt)
            print(f"Courant max {max_co:.4f} mean {mean_co:.4f}")
            # Rotate rho at the top of the step so oldTime(rho) is the previous-step
            # density before advect_alpha recomputes the current rho.
            nn.rotate_old_times(state.rho)
            self.advect_alpha(state)
            self.momentum_pressure(state)
            if state.run_time.outputTime():
                nfb.write_scalar_field(state.p_rgh, state.rt)
                nfb.write_vector_field(state.U, state.rt)
            state.run_time.printExecutionTime()
        print("End")


def main() -> None:
    import sys

    argv = sys.argv if len(sys.argv) > 1 else ["neoInterFoam"]
    NeoInterFoam(argv).run()


if __name__ == "__main__":
    main()
