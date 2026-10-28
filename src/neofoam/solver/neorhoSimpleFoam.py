# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
neorhoSimpleFoam.py — Steady-state compressible SIMPLE solver using NeoN +
NeoFOAM bindings.

Minimal first version, matching ``neorhoPimpleFoam.py``'s own scope: laminar
only, subsonic (no transonic pEqn branch), no SIMPLEC/consistent variant, no
closed-volume (``adjustPhi``/``initialMass``) handling. perfectGas equation
of state, constant Cv, constant transport (mu, Pr), read from
constant/thermophysicalProperties. Energy equation is solved for
he = Cv*(T - Tstd), not raw T — see neorhoPimpleFoam.py's module docstring
for why; the same he-space machinery (map_energy_boundary_from_temperature,
the fixedEnergy/gradientEnergy boundary types) is reused unchanged here.

Unlike literal rhoSimpleFoam's EEqn.H, which folds the pressure-work term
into a single explicit ``Ekp = 0.5*magSqr(U) + p/rho`` convected quantity,
this port keeps K and the pressure-work term separate (``div(phi,K) +
div(phiv,p)``), matching neorhoPimpleFoam.py's already-validated EEqn form
with its ddt terms simply dropped (steady state has no ddt(rho,he) or
ddt(rho,K)). This reuses NeoFOAM DSL expressions already exercised by the
PIMPLE solver instead of introducing a new discretization.

pEqn is built as +div(phiHbyA) - laplacian(rhorAUf,p) (no ddt(psi,p) term —
steady state has no d(psi*p)/dt), exactly mirroring literal rhoSimpleFoam's
own non-transonic pEqn.H and NeoFOAM's existing neorhoPimpleFoam.py
convention; update_face_velocity is called with the same
PEqnConvention.DivMinusLaplacian override.

There is no inner PISO/pressure-correction loop — SIMPLE solves momentum,
energy, and pressure exactly once per outer (time-loop) iteration, same
shape as neoSimpleFoam.py. Density updates directly from the EOS each
iteration (rho = gas.rho(psi, p)) with field relaxation, matching literal's
``rho = thermo.rho(); rho.relax();`` at the end of pEqn.H.
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
from neofoam.solver.simpleControl import SimpleControl
from neofoam.thermo.perfectGas import PerfectGasTransport

# Same he = Cv*(T - Tstd) rationale as neorhoPimpleFoam.py.
_TSTD = 298.15


def _read_int(d: Any, key: str, default: int) -> int:
    return int(d.get_int(key)) if d.contains(key) else default


def _read_switch(d: Any, key: str, default: bool) -> bool:
    if not d.contains(key):
        return default
    try:
        return bool(d.get_bool(key))
    except Exception:
        return d.get_string(key).strip().lower() in ("yes", "true", "on", "1")


class NeoRhoSimpleFoam:
    """Steady-state compressible SIMPLE solver backed by NeoN (laminar, minimal scope)."""

    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def run(self) -> None:
        _ensure_neon_initialized(self._argv)
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
        for name in ("p", "U", "rho", "T", "e"):  # "e": OpenFOAM's name of the internal energy
            if solvers.contains(name):
                solvers.insert_dict(name, nfb.map_fv_solution(solvers.subDict(name)))

        gas = PerfectGasTransport(rt)
        # The energy equation below diffuses the internal energy with a constant
        # kappa/Cv; Sutherland transport (T-dependent mu, kappa) and sensibleEnthalpy are
        # only wired up in neorhoPimpleFoam.py so far.
        if gas.transport_type != "const" or gas.is_enthalpy:
            raise NotImplementedError(
                "neorhoSimpleFoam supports only `transport const` with "
                "`energy sensibleInternalEnergy` (case has "
                f"`transport {gas.transport_type}`, `energy {gas.energy_type}`)"
            )

        p = nfb.read_scalar_volume_field(rt, "p")
        T = nfb.read_scalar_volume_field(rt, "T")
        U = nfb.read_vector_volume_field(rt, "U")

        # Same case-specific 4-patch assumption as neorhoPimpleFoam.py — see its
        # docstring/assert for the full rationale.
        assert rt.nf_mesh.n_boundaries() == 4, (
            "he's patch-type list (fixedEnergy/gradientEnergy/gradientEnergy/empty) "
            "assumes this case's exact 4-patch (inlet, outlet, walls, frontAndBack) "
            "blockMeshDict order; a different mesh needs this list reworked, not just "
            "run through unchanged"
        )
        he_raw = nfb.create_scalar_field_with_boundary_types(
            rt,
            "e",
            ["fixedEnergy", "gradientEnergy", "gradientEnergy", "empty"],
        )
        he = nfb.register_computed_scalar_field(rt, he_raw, "e")
        he.assign(gas.Cv * (T - _TSTD))
        nfb.map_energy_boundary_from_temperature(T, he, gas.Cv, _TSTD)
        he.correct_boundary_conditions()

        psi_seed = nfb.create_uniform_volume_field(rt, "psi", 0.0)
        psi_seed.assign(gas.psi(T))
        psi = nfb.register_computed_scalar_field(rt, psi_seed, "psi")

        rho_seed = nfb.create_uniform_volume_field(rt, "rho", 0.0)
        rho_seed.assign(gas.rho(psi, p))
        rho = nfb.register_computed_scalar_field(rt, rho_seed, "rho")

        K_seed = nfb.create_uniform_volume_field(rt, "K", 0.0)
        K_seed.assign(0.5 * nn.mag_sqr(U))
        K = nfb.register_computed_scalar_field(rt, K_seed, "K")

        mu_vol = nfb.create_uniform_volume_field(rt, "mu", gas.mu(T))
        zero_nut = nfb.create_uniform_volume_field(rt, "nut0", 0.0)
        # he (internal energy) is diffused here, not T — kappa/Cv (gas.alphaHe), not
        # kappa/Cp (gas.alpha); see neorhoPimpleFoam.py's identical field for the
        # derivation.
        alphaEff = nfb.create_uniform_surface_field(rt, "alphaEff", gas.alphaHe)

        grad_op = nfb.GaussGreenGrad(rt)
        surf_interp = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )

        rho_face = surf_interp.interpolate(rho)
        phi_seed = rho_face * nfb.flux(U)
        phi_seed.name = "phi"
        phi = nfb.register_computed_surface_field(rt, phi_seed, "phi")

        p_ref_cell, p_ref_value, needs_ref = nfb.set_ref_cell(rt, "p", "SIMPLE")

        cumulative_cont_err = 0.0

        while runTime.loop():
            print(f"Time = {runTime.timeName()}")

            # --- Momentum (no ddt term: steady state) ---
            grad_u = grad_op.grad_tensor(U)
            mu_face = surf_interp.interpolate(mu_vol)
            mu_face.name = "mu"

            UEqn = nfb.PDESolverVec3(
                nn.imp.div(phi, U)
                - nn.imp.laplacian(mu_face, U)
                + nfb.viscous_stress(mu_vol, zero_nut, grad_u, compressible=True),
                U,
                rt,
            )
            UEqn.relax()

            if simple.momentum_predictor():
                UEqn.solve_with_source(-1.0 * nn.exp.grad(p))
                K.assign(0.5 * nn.mag_sqr(U))
            else:
                UEqn.assemble_and_relax()

            # --- Energy (he-space; no ddt(rho,he)/ddt(rho,K): steady state) ---
            T.correct_boundary_conditions()
            nfb.map_energy_boundary_from_temperature(T, he, gas.Cv, _TSTD)
            he.correct_boundary_conditions()

            phiv = phi / surf_interp.interpolate(rho)
            phiv.name = "phiv"

            EEqn = nfb.PDESolverScalar(
                nn.imp.div(phi, he)
                - nn.imp.laplacian(alphaEff, he)
                + nn.exp.div(phi, K)
                + nn.exp.div(phiv, p),
                he,
                rt,
            )
            EEqn.relax()
            EEqn.solve()
            he.correct_boundary_conditions()

            T.assign((1.0 / gas.Cv) * he + _TSTD)
            psi.assign(gas.psi(T))

            # --- Pressure-velocity coupling (single pass: no inner PISO loop) ---
            rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
            nfb.constrain_hbya(U, p, hByA)

            rho_face = surf_interp.interpolate(rho)
            rhorAUf = surf_interp.interpolate(rho * rAU)
            rhorAUf.name = "rhorAUf"
            phiHbyA = rho_face * nfb.flux(hByA)

            prev_p = nfb.field_relaxation_snapshot(p)

            while simple.correct_non_orthogonal():
                pEqn = nfb.PDESolverScalar(
                    nn.exp.div(phiHbyA) - nn.imp.laplacian(rhorAUf, p),
                    p,
                    rt,
                )

                if needs_ref:
                    pEqn.set_reference(p_ref_cell, p_ref_value)

                pEqn.solve()
                p.correct_boundary_conditions()

                if simple.final_non_orthogonal_iter():
                    # Same convention as neorhoPimpleFoam.py's pEqn: +div(phiHbyA) -
                    # laplacian(rhorAUf,p), the opposite overall sign from NeoFOAM's
                    # incompressible solvers' default +laplacian(...) - div(...).
                    nfb.update_face_velocity(
                        phiHbyA, pEqn, phi, nfb.PEqnConvention.DivMinusLaplacian
                    )

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

            nfb.update_velocity(hByA, rAU, p, U, rt)
            U.correct_boundary_conditions()
            K.assign(0.5 * nn.mag_sqr(U))

            prev_rho = nfb.field_relaxation_snapshot(rho)
            rho.assign(gas.rho(psi, p))
            nfb.apply_field_relaxation(
                rho, prev_rho, nfb.lookup_field_relaxation(rt.fv_solution_dict, rho.name, False)
            )
            rho.correct_boundary_conditions()

            if runTime.outputTime():
                print("Writing fields")
                nfb.write_scalar_field(p, rt)
                nfb.write_scalar_field(T, rt)
                nfb.write_scalar_field(rho, rt)
                nfb.write_vector_field(U, rt)

            runTime.printExecutionTime()

        print("End")


def main() -> None:
    argv = sys.argv if len(sys.argv) > 1 else ["neorhoSimpleFoam"]
    solver = NeoRhoSimpleFoam(argv)
    solver.run()


if __name__ == "__main__":
    main()
