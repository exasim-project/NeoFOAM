# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
neorhoPimpleFoam.py — Transient compressible PIMPLE solver using NeoN +
NeoFOAM bindings.

Minimal first version: laminar only, subsonic (no transonic pEqn branch), no
SIMPLEC/consistent variant. perfectGas equation of state, constant Cv,
constant transport (mu, Pr), read from constant/thermophysicalProperties.
Energy equation is solved for he = Cv*(T - Tstd), not raw T (see _TSTD below
for why): he's boundary is kept live every outer iteration via real
FixedEnergy/GradientEnergy BCs (NeoN feat/he-temperature-boundary-mapping),
mirroring OpenFOAM's own fixedEnergyFvPatchScalarField/
gradientEnergyFvPatchScalarField, mapped from T's current boundary via
map_energy_boundary_from_temperature rather than derived from scratch.

The pressure equation uses fvm::ddt(psi, p) in place of literal
rhoPimpleFoam's fvc::ddt(rho) + psi*correction(fvm::ddt(p)): NeoN's DSL has
no fvm::correction() equivalent. imp.ddt(psi, p) discretizes the full
conservative d(psi*p)/dt directly (psi_new weighting the diagonal), which
already carries the psi-sensitivity that rhoPimpleFoam's explicit
p*fvc::ddt(psi) term supplies on top of its psi-frozen implicit part; adding
that term here would double-count it, so it is subtracted instead to cancel
the redundant implicit contribution. Getting this sign right (rather than
masking the resulting instability with field relaxation on T) is what lets
this solver match classic rhoPimpleFoam's robustness on the duct test case
with no relaxation of any kind. phiHbyA also includes the density-weighted
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
from neofoam.solver.pisoControl import PisoControl
from neofoam.thermo.perfectGas import PerfectGasConstTransport

# EEqn is solved for he = Cv*(T - Tstd), not raw T, to keep the energy
# equation's own reference scale (rather than T's absolute ~300K baseline)
# small relative to typical mid-PIMPLE-loop mass-conservation slip. he's
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
        for name in (
            "p", "U", "rho", "T", "he",
            "pFinal", "UFinal", "rhoFinal", "TFinal", "heFinal",
        ):
            if solvers.contains(name):
                solvers.insert_dict(name, nfb.map_fv_solution(solvers.subDict(name)))

        gas = PerfectGasConstTransport(rt)

        p = nfb.read_scalar_volume_field(rt, "p")
        T = nfb.read_scalar_volume_field(rt, "T")
        U = nfb.read_vector_volume_field(rt, "U")

        # he's patches are real FixedEnergy/GradientEnergy BC objects
        # (feat/he-temperature-boundary-mapping), externally-pushed and refreshed from
        # T's current boundary every outer iteration via
        # map_energy_boundary_from_temperature below -- not a value captured once at
        # setup. Patch order (inlet, outlet, walls, frontAndBack) is this case's own
        # blockMeshDict order; asserted below rather than assumed silently, since it's
        # this solver's one hardcoded, case-specific assumption (a general version would
        # need to derive patch types from T's own per-patch BC types instead, which
        # NeoFOAM doesn't currently expose a way to query).
        assert rt.nf_mesh.n_boundaries() == 4, (
            "he's patch-type list (fixedEnergy/gradientEnergy/gradientEnergy/empty) "
            "assumes this case's exact 4-patch (inlet, outlet, walls, frontAndBack) "
            "blockMeshDict order; a different mesh needs this list reworked, not just "
            "run through unchanged"
        )
        he_raw = nfb.create_scalar_field_with_boundary_types(
            rt,
            "he",
            ["fixedEnergy", "gradientEnergy", "gradientEnergy", "empty"],
        )
        # Register in the field database (same as psi/rho/K below) -- required for
        # nn.old_time()/imp.ddt() to find he's old-time slot; a bare VolumeField
        # constructed directly isn't database-backed.
        he = nfb.register_computed_scalar_field(rt, he_raw, "he")
        he.assign(gas.Cv * (T - _TSTD))  # internal values; boundary set properly below
        nfb.map_energy_boundary_from_temperature(T, he, gas.Cv, _TSTD)
        he.correct_boundary_conditions()
        nn.old_time(he)

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
        # he (internal energy) is being diffused here, not T, so this needs
        # kappa/Cv (gas.alphaHe), not kappa/Cp (gas.alpha) -- see gas.alphaHe's
        # definition for the derivation; matches OpenFOAM's own
        # heThermo::alphahe() exactly for this thermo type.
        alphaEff = nfb.create_uniform_surface_field(rt, "alphaEff", gas.alphaHe)
        # Feature-for-feature port of literal rhoPimpleFoam's `psip0 = psi*p` /
        # `thermo.correctRho(psi*p - psip0, rhoMin, rhoMax)` -- captured once per PISO
        # pass, used to replace the plain EOS rho.assign() with literal's incremental
        # update. rhoMin=0/rhoMax=GREAT defaults (unset in this case's PIMPLE dict) make
        # the bound inactive, so the clamp itself is omitted.
        psip0 = nfb.create_uniform_volume_field(rt, "psip0", 0.0)

        grad_op = nfb.GaussGreenGrad(rt)
        surf_interp = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )

        rho_face = surf_interp.interpolate(rho)
        phi_seed = rho_face * nfb.flux(U)
        phi_seed.name = "phi"
        phi = nfb.register_computed_surface_field(rt, phi_seed, "phi")
        nn.old_time(phi)

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
            nn.rotate_old_times(phi)

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

                # Literal's divDevRhoReff coefficient is alpha*rho*nuEff(), and for this
                # laminar (Stokes) model nuEff()==nu()==mu/rho, so the coefficient
                # collapses to mu -- NOT rho*mu -- for *both* the implicit laplacian and
                # the explicit dev2 correction below.
                mu_face = surf_interp.interpolate(mu_vol)
                mu_face.name = "mu"

                ueqn_expr = (
                    nn.imp.ddt(rho, U) + nn.imp.div(phi, U) - nn.imp.laplacian(mu_face, U)
                    + nfb.viscous_stress(mu_vol, zero_nut, grad_u)
                )

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

                prev_T = nfb.field_relaxation_snapshot(he)

                # Live per-outer-iteration refresh, matching literal's actual
                # fixedEnergy/gradientEnergy mechanism (recomputed every call, not
                # captured once) -- push T's current boundary state onto he's
                # refValue/refGrad, then materialise he's value() from it.
                T.correct_boundary_conditions()
                nfb.map_energy_boundary_from_temperature(T, he, gas.Cv, _TSTD)
                he.correct_boundary_conditions()

                phiv = phi / surf_interp.interpolate(rho)
                phiv.name = "phiv"

                eeqn_expr = (
                    nn.imp.ddt(rho, he) + nn.imp.div(phi, he) - nn.imp.laplacian(alphaEff, he)
                    + nn.exp.ddt(rho, K) + nn.exp.div(phi, K)
                    + nn.exp.div(phiv, p)
                )
                EEqn = nfb.PDESolverScalar(
                    eeqn_expr,
                    he,
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
                    he,
                    prev_T,
                    nfb.lookup_field_relaxation(rt.fv_solution_dict, he.name, final_iter),
                )
                he.correct_boundary_conditions()
                # Convert he back to T for the EOS/reporting -- a plain affine field
                # expression, so it updates internal AND boundary values consistently in
                # one shot (elementwise field arithmetic already applies to both,
                # confirmed in volumeField.cpp's mulFieldField/addFieldField), with no
                # separate boundary-write call needed.
                T.assign((1.0 / gas.Cv) * he + _TSTD)
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

                    # imp.ddt(psi, p) discretizes the FULL conservative d(rho)/dt =
                    # d(psi*p)/dt directly (psi_new on the diagonal, psi_old*p_old on the
                    # rhs) -- unlike literal rhoPimpleFoam's fvc::ddt(rho) +
                    # psi*correction(fvm::ddt(p)), which holds psi frozen for the implicit
                    # part and adds back p*fvc::ddt(psi) explicitly. Our imp.ddt(psi, p)
                    # already carries that psi-sensitivity implicitly, so adding
                    # p*fvc::ddt(psi) on top would double-count it; it is subtracted
                    # instead, to remove the redundant implicit-psi contribution.
                    ddt_psi = (psi - nn.old_time(psi)) / rt.dt
                    p_ddt_psi = p * ddt_psi

                    while piso.correct_non_orthogonal():
                        pEqn = nfb.PDESolverScalar(
                            nn.imp.ddt(psi, p) - nn.exp.source(p_ddt_psi)
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
                    # Feature-for-feature port: literal pEqn.H also refreshes K here, from
                    # the just-pressure-corrected U -- not just once after the momentum
                    # predictor. Without this, K stays one-PISO-pass stale for the rest of
                    # this outer iteration and feeds EEqn's explicit K terms next outer
                    # iteration using pre-pressure-correction U.
                    K.assign(0.5 * nn.mag_sqr(U))
                    rho.assign(rho + psi * p - psip0)

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
