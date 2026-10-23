# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
neorhoPimpleFoam.py — Transient compressible PIMPLE solver using NeoN +
NeoFOAM bindings.

Minimal first version: laminar only, subsonic (no transonic pEqn branch), no
SIMPLEC/consistent variant. perfectGas equation of state, constant Cv-or-Cp,
const-or-sutherland transport, read from constant/thermophysicalProperties
(see neofoam.thermo.perfectGas.PerfectGasTransport). Energy equation is
solved for he = Cv*(T - Tstd) (sensibleInternalEnergy) or h = Cp*(T - Tstd)
(sensibleEnthalpy), not raw T (see _TSTD below for why) -- which one, and
whether the EEqn's pressure-coupling term is +div(phiv,p) or -dpdt, is
decided by the case's own thermoType.energy. The energy field's boundary is
kept live every outer iteration via real FixedEnergy/GradientEnergy BCs
(NeoN feat/he-temperature-boundary-mapping), mirroring OpenFOAM's own
fixedEnergyFvPatchScalarField/gradientEnergyFvPatchScalarField, mapped from
T's current boundary via map_energy_boundary_from_temperature rather than
derived from scratch, with per-patch types derived from T's own literal BC
types (any patch count/order) rather than a hardcoded list.

The pressure equation uses fvm::ddt(psi, p) in place of literal
rhoPimpleFoam's fvc::ddt(rho) + psi*correction(fvm::ddt(p)): NeoN's DSL has
no fvm::correction() equivalent. imp.ddt(psi, p) discretizes the full
conservative d(psi*p)/dt directly (psi_new weighting the diagonal), which is
algebraically the same as literal's (psi*p_new - rho_old)/dt, so no extra
p*ddt(psi) term is added or subtracted. (Subtracting one cancels the psi
change from T, which turns the acoustics isothermal.) phiHbyA also includes the density-weighted
ddtFluxCorr term (rho_f^0 * uf^0), matching rhoPimpleFoam's own phiHbyA
correction. The bounded thermo.correctRho() increment is dropped for this
first version in favor of directly reassigning rho from the current psi/p.
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
from neofoam.solver.neon_fv_options import NeonFvOptions
from neofoam.solver.pisoControl import PisoControl
from neofoam.thermo.perfectGas import PerfectGasTransport

# EEqn is solved for energy = coeff*(T - Tstd) (he/Cv or h/Cp, see
# PerfectGasTransport.energy_coeff), not raw T, to keep the energy equation's
# own reference scale (rather than T's absolute ~300K baseline) small
# relative to typical mid-PIMPLE-loop mass-conservation slip. energy's
# boundary is kept live every outer iteration via the real FixedEnergy/
# GradientEnergy BCs (feat/he-temperature-boundary-mapping), mirroring
# OpenFOAM's own fixedEnergyFvPatchScalarField/gradientEnergyFvPatchScalarField
# exactly, mapped from T's current boundary via
# map_energy_boundary_from_temperature rather than derived from scratch.
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


def _energy_patch_types(t_patch_types: list[str]) -> list[str]:
    """Derive the energy field's per-patch BC types from T's literal ones, mirroring
    OpenFOAM's own fixedEnergyFvPatchScalarField/gradientEnergyFvPatchScalarField
    derivation: fixedValue -> fixedEnergy, empty -> empty, anything else (zeroGradient,
    inletOutlet, ...) -> gradientEnergy. General over patch count/order (reads T's own
    case-file types rather than assuming a fixed layout)."""
    mapped = []
    for t in t_patch_types:
        if t == "fixedValue":
            mapped.append("fixedEnergy")
        elif t == "empty":
            mapped.append("empty")
        else:
            mapped.append("gradientEnergy")
    return mapped


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

        gas = PerfectGasTransport(rt)

        rt.fv_schemes_dict = nfb.map_fv_schemes(rt.fv_schemes_dict)
        solvers = rt.fv_solution_dict.subDict("solvers")
        for name in (
            "p",
            "U",
            "rho",
            "T",
            gas.energy_field_name,
            "pFinal",
            "UFinal",
            "rhoFinal",
            "TFinal",
            gas.energy_field_name + "Final",
        ):
            # map_solver_settings (not a plain contains()/subDict() lookup) honours
            # OpenFOAM's wildcard fvSolution keys (e.g. "rho.*", "(U|h)"), which a literal
            # name lookup would silently miss, leaving that field's solver/preconditioner
            # as the raw OpenFOAM name the Ginkgo factory doesn't recognise.
            nfb.map_solver_settings(solvers, name)

        p = nfb.read_scalar_volume_field(rt, "p")
        T = nfb.read_scalar_volume_field(rt, "T")
        U = nfb.read_vector_volume_field(rt, "U")

        # energy's (he for sensibleInternalEnergy, h for sensibleEnthalpy) patches are
        # real FixedEnergy/GradientEnergy BC objects (feat/he-temperature-boundary-mapping),
        # externally-pushed and refreshed from T's current boundary every outer iteration
        # via map_energy_boundary_from_temperature below -- not a value captured once at
        # setup. Patch types are derived from T's own literal per-patch BC types (see
        # _energy_patch_types), not a hardcoded list, so this works for any patch count/order.
        energy_name = gas.energy_field_name
        energy_patch_types = _energy_patch_types(nfb.read_patch_types(rt, "T"))
        energy_raw = nfb.create_scalar_field_with_boundary_types(
            rt, energy_name, energy_patch_types
        )
        # Register in the field database (same as psi/rho/K below) -- required for
        # nn.old_time()/imp.ddt() to find energy's old-time slot; a bare VolumeField
        # constructed directly isn't database-backed.
        energy = nfb.register_computed_scalar_field(rt, energy_raw, energy_name)
        energy.assign(gas.energy_coeff * (T - _TSTD))  # internal values; boundary set below
        nfb.map_energy_boundary_from_temperature(T, energy, gas.energy_coeff, _TSTD)
        energy.correct_boundary_conditions()
        nn.old_time(energy)

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

        # mu_vol/alpha_energy_vol are recomputed from T every outer iteration for
        # sutherland transport (see the UEqn/EEqn assembly below); for const transport
        # gas.mu(T)/gas.alphaEnergy(T) don't actually depend on T, so this initial fill is
        # also the value they keep for the whole run. gas.mu(T)/alphaEnergy(T) return a
        # per-cell field (not a plain float, which create_uniform_volume_field needs) for
        # sutherland, since they're evaluated from the current T there -- seed at 0 and
        # .assign() the real (possibly spatially-varying) value instead in that case.
        zero_nut = nfb.create_uniform_volume_field(rt, "nut0", 0.0)
        if gas.transport_type == "sutherland":
            mu_vol = nfb.create_uniform_volume_field(rt, "mu", 0.0)
            mu_vol.assign(gas.mu(T))
            # The energy variable being diffused (he or h) needs kappa/Cv or kappa/Cp
            # respectively, not the Pr-based kappa/Cp const transport uses for
            # *temperature* diffusion -- see gas.alphaEnergy's definition for the
            # derivation; matches OpenFOAM's own heThermo::alphahe()/alpha() exactly.
            alpha_energy_vol = nfb.create_uniform_volume_field(rt, "alphaEnergy", 0.0)
            alpha_energy_vol.assign(gas.alphaEnergy(T))
        else:
            mu_vol = nfb.create_uniform_volume_field(rt, "mu", gas.mu(T))
            alpha_energy_vol = nfb.create_uniform_volume_field(
                rt, "alphaEnergy", gas.alphaEnergy(T)
            )
        # Feature-for-feature port of literal rhoPimpleFoam's `psip0 = psi*p` /
        # `thermo.correctRho(psi*p - psip0, rhoMin, rhoMax)` -- captured once per PISO
        # pass, used to replace the plain EOS rho.assign() with literal's incremental
        # update. rhoMin=0/rhoMax=GREAT defaults (unset in this case's PIMPLE dict) make
        # the bound inactive, so the clamp itself is omitted.
        psip0 = nfb.create_uniform_volume_field(rt, "psip0", 0.0)
        # Literal rhoPimpleFoam's dpdt: zero-initialised, recomputed as fvc::ddt(p) at the
        # end of every pEqn and consumed by the NEXT EEqn (pressure work for the enthalpy
        # form). Evaluating ddt(p) inside EEqn instead would see p == p^n on the first
        # outer iteration, where BDF2 gives a spurious -0.5*(p^n - p^{n-1})/dt.
        dpdt = nfb.create_uniform_volume_field(rt, "dpdt", 0.0)
        n_steps = 0

        grad_op = nfb.GaussGreenGrad(rt)
        surf_interp = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )

        rho_face = surf_interp.interpolate(rho)
        phi_seed = rho_face * nfb.flux(U)
        phi_seed.name = "phi"
        phi = nfb.register_computed_surface_field(rt, phi_seed, "phi")
        nn.old_time(phi)

        # UEqn's `== fvOptions(rho, U)`; raises on option types not implemented for NeoN.
        fv_options = NeonFvOptions.from_case(rt, U)

        p_ref_cell, p_ref_value, needs_ref = nfb.set_ref_cell(rt, "p", "PIMPLE")
        pimple_loop = nfb.PimpleControl(rt.fv_solution_dict)

        def boundary_context() -> Any:
            # Built fresh at each call site from whatever phi/rho/psi/p currently hold --
            # matches literal OpenFOAM's own per-call BC evaluation, not a value cached once
            # per timestep. Covers every context-aware BC this solver's fields can carry:
            # inletOutlet/totalPressure (phi), waveTransmissive (phi, rho, psi, p's own
            # old-time, deltaT), pressureInletOutletVelocity (phi), a sine-forced fixedValue
            # (time). A BC that doesn't need a given key simply never looks it up.
            ctx = nn.BoundaryContext()
            ctx.insert("phi", phi)
            ctx.insert("rho", rho)
            ctx.insert("psi", psi)
            ctx.insert("p", p)
            ctx.insert_value("deltaT", rt.dt)
            ctx.insert_value("time", runTime.value())
            return ctx

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
            nn.rotate_old_times(energy)
            nn.rotate_old_times(p)
            nn.rotate_old_times(rho)
            nn.rotate_old_times(psi)
            nn.rotate_old_times(K)
            nn.rotate_old_times(phi)
            n_steps += 1

            max_co, mean_co = nn.compute_co_num(phi, rt.dt)
            print(f"Courant Number mean: {mean_co:.6f} max: {max_co:.6f}")
            nfb.sync_run_times(runTime, rt, max_co)

            residuals: dict[str, tuple[float, float]] = {}
            while pimple_loop.loop(residuals):
                final_iter = pimple_loop.final_iter()

                # Feature-for-feature port of literal rhoPimpleFoam.C's
                # `if (pimple.firstIter() && !pimple.SIMPLErho()) { #include "rhoEqn.H" }`,
                # which runs BEFORE UEqn.H/EEqn.H on the first outer iteration of every
                # timestep: fvm::ddt(rho) + fvc::div(phi) == 0, solved for rho using the
                # PREVIOUS timestep's converged phi. Advances rho via mass conservation
                # (a real continuity update, using the actual local div(phi) residual at
                # every cell) before anything else touches it this timestep -- distinct
                # from, and run before, the EOS-based rho.assign(gas.rho(psi,p)) later in
                # the loop. transient PIMPLE is never SIMPLErho, so this always runs.
                if pimple_loop.first_iter():
                    rhoEqn = nfb.PDESolverScalar(
                        nn.imp.ddt(rho) + nn.exp.div(phi),
                        rho,
                        rt,
                    )
                    rhoEqn.solve()
                    rho.correct_boundary_conditions()

                prev_p = nfb.field_relaxation_snapshot(p)

                grad_u = grad_op.grad_tensor(U)

                # Sutherland transport depends on T, so mu/alphaEnergy are refreshed from
                # the current T every outer iteration; const transport leaves them at
                # their initial, never-changing fill (see their creation above).
                if gas.transport_type == "sutherland":
                    mu_vol.assign(gas.mu(T))
                    alpha_energy_vol.assign(gas.alphaEnergy(T))

                # Literal's divDevRhoReff coefficient is alpha*rho*nuEff(), and for this
                # laminar (Stokes) model nuEff()==nu()==mu/rho, so the coefficient
                # collapses to mu -- NOT rho*mu -- for *both* the implicit laplacian and
                # the explicit dev2 correction below.
                mu_face = surf_interp.interpolate(mu_vol)
                mu_face.name = "mu"

                ueqn_expr = (
                    nn.imp.ddt(rho, U)
                    + nn.imp.div(phi, U)
                    - nn.imp.laplacian(mu_face, U)
                    + nfb.viscous_stress(mu_vol, zero_nut, grad_u)
                )
                ueqn_expr = fv_options.add_momentum_sources(ueqn_expr, rho, U, runTime.value())

                UEqn = nfb.PDESolverVec3(
                    ueqn_expr,
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

                prev_T = nfb.field_relaxation_snapshot(energy)

                # Live per-outer-iteration refresh, matching literal's actual
                # fixedEnergy/gradientEnergy mechanism (recomputed every call, not
                # captured once) -- push T's current boundary state onto energy's
                # refValue/refGrad, then materialise energy's value() from it.
                T.correct_boundary_conditions(boundary_context())
                nfb.map_energy_boundary_from_temperature(T, energy, gas.energy_coeff, _TSTD)
                energy.correct_boundary_conditions()

                alphaEff = surf_interp.interpolate(alpha_energy_vol)
                alphaEff.name = "alphaEff"

                # sensibleInternalEnergy (he): +div(phiv,p). sensibleEnthalpy (h): -dpdt
                # instead (thermo.dpdt() defaults true) -- literal rhoPimpleFoam's own
                # EEqn.H branch on he.name()=="e". phiv = phi/rho is only needed by the
                # he-space branch, but it's cheap enough to compute unconditionally.
                phiv = phi / surf_interp.interpolate(rho)
                phiv.name = "phiv"
                energy_pressure_term = (
                    -1.0 * nn.exp.source(dpdt) if gas.is_enthalpy else nn.exp.div(phiv, p)
                )

                eeqn_expr = (
                    nn.imp.ddt(rho, energy)
                    + nn.imp.div(phi, energy)
                    - nn.imp.laplacian(alphaEff, energy)
                    + nn.exp.ddt(rho, K)
                    + nn.exp.div(phi, K)
                    + energy_pressure_term
                )
                EEqn = nfb.PDESolverScalar(
                    eeqn_expr,
                    energy,
                    rt,
                )
                EEqn.set_final_iter(final_iter)
                EEqn.relax()

                stats_t = EEqn.solve()
                residuals["T"] = (
                    stats_t.entries[0].initial_residual,
                    stats_t.entries[0].final_residual,
                )
                nfb.apply_field_relaxation(
                    energy,
                    prev_T,
                    nfb.lookup_field_relaxation(rt.fv_solution_dict, energy.name, final_iter),
                )
                energy.correct_boundary_conditions()
                # Convert energy back to T for the EOS/reporting -- a plain affine field
                # expression, so it updates internal AND boundary values consistently in
                # one shot (elementwise field arithmetic already applies to both,
                # confirmed in volumeField.cpp's mulFieldField/addFieldField), with no
                # separate boundary-write call needed.
                T.assign((1.0 / gas.energy_coeff) * energy + _TSTD)
                psi.assign(gas.psi(T))
                rho.assign(gas.rho(psi, p))

                p_res: tuple[float, float] = (0.0, 0.0)
                have_p_res = False

                while piso.correct():
                    psip0.assign(psi * p)
                    rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
                    nfb.constrain_hbya(U, p, hByA)

                    rho_face = surf_interp.interpolate(rho)
                    rhorAUf = surf_interp.interpolate(rho * rAU)
                    rhorAUf.name = "rhorAUf"
                    phiHbyA = rho_face * nfb.flux(hByA)

                    # imp.ddt(psi, p) discretizes the full conservative d(psi*p)/dt
                    # (psi_new*p_new on the diagonal, psi_old*p_old on the rhs). That is
                    # literal rhoPimpleFoam's fvc::ddt(rho) + psi*correction(fvm::ddt(p))
                    # = (psi*p_new - rho_old)/dt with rho = psi*p, so no extra
                    # p*ddt(psi) term is needed. The psi change from the just-solved T
                    # must stay in this term: it is what makes the effective
                    # compressibility 1/(gamma*R*T) (isentropic sound speed) instead of
                    # 1/(R*T) (isothermal).
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
                        p.correct_boundary_conditions(boundary_context())

                        if piso.final_non_orthogonal_iter():
                            # pEqn above is assembled +ddt(psi,p) + div(phiHbyA) -
                            # laplacian(rhorAUf,p) (literal rhoPimpleFoam's own
                            # pDDtEqn - fvm::laplacian ordering, needed to keep the
                            # matrix positive-definite for p's DIC/IC preconditioner),
                            # the opposite overall sign from NeoFOAM's incompressible
                            # solvers' +laplacian(...) - div(...) convention. Must be
                            # told explicitly, or update_face_velocity's default
                            # (calibrated for the incompressible convention) would
                            # reconstruct phi with the wrong sign.
                            nfb.update_face_velocity(
                                phiHbyA, pEqn, phi, nfb.PEqnConvention.DivMinusLaplacian
                            )

                    nfb.apply_field_relaxation(
                        p,
                        prev_p,
                        nfb.lookup_field_relaxation(rt.fv_solution_dict, p.name, final_iter),
                    )
                    p.correct_boundary_conditions(boundary_context())

                    nfb.update_velocity(hByA, rAU, p, U, rt)
                    U.correct_boundary_conditions(boundary_context())
                    # Feature-for-feature port: literal pEqn.H also refreshes K here, from
                    # the just-pressure-corrected U -- not just once after the momentum
                    # predictor. Without this, K stays one-PISO-pass stale for the rest of
                    # this outer iteration and feeds EEqn's explicit K terms next outer
                    # iteration using pre-pressure-correction U.
                    K.assign(0.5 * nn.mag_sqr(U))
                    rho.assign(rho + psi * p - psip0)

                    # literal pEqn.H: dpdt = fvc::ddt(p), same BDF order as the implicit
                    # ddt terms (BDF1 until p^{n-1} exists)
                    if gas.is_enthalpy:
                        p0 = nn.old_time(p)
                        if ddt_scheme == nfb.DdtScheme.BDF2 and n_steps >= 2:
                            p00 = nn.old_time(p0)
                            dpdt.assign((1.5 / rt.dt) * p - (2.0 / rt.dt) * p0 + (0.5 / rt.dt) * p00)
                        else:
                            dpdt.assign((1.0 / rt.dt) * (p - p0))

                if have_p_res:
                    residuals["p"] = p_res

            fv_options.end_of_time_step(rt.dt)

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
