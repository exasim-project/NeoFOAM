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

import os
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

# Diagnostic-only ablation (investigation thread, see neorhoPimpleFoam_investigation
# _summary.md §4.1): when set, the pEqn's rhorAUf coefficient is captured once (first
# PISO pass, first timestep) and reused unchanged for the rest of the run, severing the
# T -> rho -> rAU -> rhorAUf -> pEqn-diagonal feedback path at the corner cell so its
# contribution to the corner div(phi)/T growth can be isolated from everything else.
_FREEZE_RHORAUF = os.environ.get("NEORHO_FREEZE_RHORAUF") == "1"

# Same investigation, same purpose: ablate UEqn's explicit viscous_stress (dev2)
# correction term to isolate its contribution to the corner div(phi)/T growth.
_DROP_VISCOUS_STRESS = os.environ.get("NEORHO_DROP_VISCOUS_STRESS") == "1"

# Same investigation: ablate EEqn's explicit pressure-work term div(phiv,p)/Cv to
# validate (against the live solve, not a standalone diagnostic) whether it is
# actually as large as a standalone evaluate_explicit() reading suggested.
_DROP_PRESSURE_WORK = os.environ.get("NEORHO_DROP_PRESSURE_WORK") == "1"

# Same investigation: ablate EEqn's explicit K terms (ddt(rho,K) + div(phi,K)) ALONE,
# keeping the pressure-work term intact -- §4.4 only ever tested K bundled together
# with pressure-work, never isolated.
_DROP_K_TERMS = os.environ.get("NEORHO_DROP_K_TERMS") == "1"

# Modified-equation derivation (neorhoPimpleFoam_investigation_summary.md §4.5):
# imp.ddt(psi, p) ALONE already algebraically equals literal rhoPimpleFoam's full
# fvc::ddt(rho) + psi*correction(fvm::ddt(p)) (both reduce to diag=psi_c*V/dt,
# source=psi_oldT*p_oldT*V/dt once rho_c=psi_c*p_c is substituted in and the
# p_c cross-terms cancel) -- the explicit p_ddt_psi subtraction on top of it has no
# counterpart in literal and may be a spurious, non-cancelling extra term. Test
# zeroing it to see whether that moves the corner spread toward OpenFOAM's.
_ZERO_PDDT_PSI = os.environ.get("NEORHO_ZERO_PDDT_PSI") == "1"

# §4.15 follow-up: trace psi_c, psi_oldT, p_c at the corner cell (cell 0) on every
# outer/PISO pass (not just final_iter) to check, directly, whether the
# _ZERO_PDDT_PSI divergence (§4.5) shows the predicted runaway signature in
# psi_c/psi_oldT -- the closed-form per-cell shift p_ddt_psi contributes to each
# pEqn solve is p_c*(1 - psi_oldT/psi_c), which vanishes when psi (T) is steady
# and grows with psi's relative change; dropping it should matter most exactly
# when that ratio is moving fastest.
_TRACE_PSI_P = os.environ.get("NEORHO_TRACE_PSI_P") == "1"

# investigation_summary.md §4.34: dump the assembled pEqn's real diag/off-diag/rhs
# entries at the corner cell (0) and its two traced interior neighbours (1, 50) to
# build a local amplification-matrix eigenvalue check against literal's equivalent.
_TRACE_PEQN_MATRIX = os.environ.get("NEORHO_TRACE_PEQN_MATRIX") == "1"

# investigation_summary.md §4.35: replace the iterative pEqn solve with an exact
# direct solve (scipy.sparse.linalg.spsolve) on the same assembled matrix, to test
# whether NEORHO_ZERO_PDDT_PSI's divergence is a property of the discretization
# itself (an exact solve also diverges) or an artifact of the iterative solver's
# behaviour on an increasingly ill-conditioned matrix.
_DIRECT_PSOLVE = os.environ.get("NEORHO_DIRECT_PSOLVE") == "1"

# Per-term EEqn contribution printout, requested directly: report each of the
# six terms' real, live-assembled (diag, rhs) at cell 0, by constructing a
# separate single-term PDESolverScalar for each and calling .assemble() on it
# -- the same real assembly path validated for the pEqn CSR dump (§4.34-4.36),
# NOT the bare evaluate_implicit/evaluate_explicit bypass already shown (§4.4)
# to not reproduce live-solve scaling. Includes a self-check: the six terms'
# net contributions (diag*T[0]-rhs) must sum to the combined EEqn's own.
_TRACE_EEQN_TERMS = os.environ.get("NEORHO_TRACE_EEQN_TERMS") == "1"

# EEqn is solved for he = Cv*(T - Tstd), not raw T. T is an absolute Kelvin
# value (~300), while he's reference point Tstd sits close to the operating
# temperature, so the same intermediate (non-fully-converged outer-iteration)
# mass-conservation slip that's unavoidable mid-PIMPLE-loop gets multiplied by
# a much smaller base value in he-space than it would in T-space -- cuts the
# corner-cell T-spread on this investigation's duct case by ~215x (measured,
# not estimated: 0.6164K -> 0.002863K, vs. literal's 0.000185K). he's boundary
# is kept live every outer iteration via the real FixedEnergy/GradientEnergy
# BCs (feat/he-temperature-boundary-mapping), mirroring OpenFOAM's own
# fixedEnergyFvPatchScalarField/gradientEnergyFvPatchScalarField exactly,
# rather than this being a one-shot Python-side value correction.
_TSTD = 298.15


def _safe_idx(arr: Any, i: int) -> float:
    """Debug-probe helper: several corner-cell trace prints below hardcode
    indices (e.g. the +y neighbour at cell 50, a walls-patch boundary face at
    offset 40) specific to this investigation's 1000-cell duct case. Smaller
    meshes used for quick debugging don't have those indices; return NaN
    instead of raising so the trace stays informative rather than fatal."""
    return float(arr[i]) if i < len(arr) else float("nan")


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
        # the bound inactive, so the clamp itself is omitted; see §4.8.
        psip0 = nfb.create_uniform_volume_field(rt, "psip0", 0.0)
        # Scratch result buffer for per-term diagnostic evaluation (investigation thread,
        # see neorhoPimpleFoam_investigation_summary.md §4.4) -- never touched by the
        # real simulation, only used as an out-param for evaluate_implicit/evaluate_explicit.
        _scratch_term = nfb.create_uniform_volume_field(rt, "scratch_term", 0.0)

        grad_op = nfb.GaussGreenGrad(rt)
        surf_interp = nn.SurfaceInterpolationScalar(
            rt.executor, rt.nf_mesh, nn.TokenList(["linear"])
        )
        # §4.23 probe (neorhoPimpleFoam_investigation_summary.md): direct comparison of the
        # linearUpwindV-reconstructed face U value at the corner cell's two internal faces,
        # the one piece of the convection scheme not yet checked against literal at the
        # implementation level (the plain Gauss-Green grad(U) was already checked in §4.22
        # and matches closely).
        _luw_interp = nn.SurfaceInterpolationVec3(
            rt.executor, rt.nf_mesh, nn.TokenList(["linearUpwindV"])
        )
        # EEqn follow-up: direct comparison of the limitedLinear-reconstructed face T value
        # at the corner's two internal faces, same methodology as the linearUpwindV probe
        # above, matching this case's div(phi,T) scheme exactly.
        _ll_tokens = nn.TokenList()
        _ll_tokens.insert_string("limitedLinear")
        _ll_tokens.insert_double(1.0)
        _ll_interp = nn.SurfaceInterpolationScalar(rt.executor, rt.nf_mesh, _ll_tokens)

        rho_face = surf_interp.interpolate(rho)
        phi_seed = rho_face * nfb.flux(U)
        phi_seed.name = "phi"
        phi = nfb.register_computed_surface_field(rt, phi_seed, "phi")
        nn.old_time(phi)

        p_ref_cell, p_ref_value, needs_ref = nfb.set_ref_cell(rt, "p", "PIMPLE")
        pimple_loop = nfb.PimpleControl(rt.fv_solution_dict)
        frozen_rhorAUf: Any = None

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
            _outer_counter = 0
            while pimple_loop.loop(residuals):
                _outer_counter += 1
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
                if _outer_counter == 1:
                    rhoEqn = nfb.PDESolverScalar(
                        nn.imp.ddt(rho) + nn.exp.div(phi),
                        rho,
                        rt,
                    )
                    rhoEqn.solve()
                    rho.correct_boundary_conditions()

                prev_p = nfb.field_relaxation_snapshot(p)

                grad_u = grad_op.grad_tensor(U)

                if os.environ.get("NEORHO_TRACE_GRAD_U") == "1" and runTime.timeIndex() <= 1:
                    import numpy as _np

                    gU = _np.asarray(grad_u.internal_vector().copy_to_host())
                    print(
                        f"TRACE_GRADU t={runTime.timeName()} outer={_outer_counter} "
                        f"gradU[0]={gU[0]}",
                        flush=True,
                    )

                if os.environ.get("NEORHO_TRACE_DEV2") == "1" and runTime.timeIndex() <= 1:
                    import numpy as _np

                    vs_op = nfb.viscous_stress(mu_vol, zero_nut, grad_u)
                    vs_result = nn.VectorVector(rt.executor, U.size(), nn.Vec3(0.0, 0.0, 0.0))
                    nfb.evaluate_explicit(vs_op, rt.fv_schemes_dict, vs_result)
                    vs = _np.asarray(vs_result.copy_to_host())
                    gU = _np.asarray(grad_u.internal_vector().copy_to_host())
                    print(
                        f"TRACE_DEV2 t={runTime.timeName()} outer={_outer_counter} "
                        f"gradU[0]={gU[0]} viscous_stress[0]={vs[0]}",
                        flush=True,
                    )

                if os.environ.get("NEORHO_TRACE_LUW") == "1" and runTime.timeIndex() <= 1:
                    import numpy as _np

                    U.correct_boundary_conditions()
                    _luw_dst = nn.VectorSurfaceField(rt.executor, "U_luw_probe", rt.nf_mesh)
                    _luw_interp.interpolate_into(phi, U, _luw_dst)
                    luw = _np.asarray(_luw_dst.internal_vector().copy_to_host())
                    print(
                        f"TRACE_LUW t={runTime.timeName()} outer={_outer_counter} "
                        f"U_luw[face0,+x]={luw[0]} U_luw[face1,+y]={luw[1]}",
                        flush=True,
                    )

                # Investigation fix (neorhoPimpleFoam_investigation_summary.md §4.19):
                # literal's divDevRhoReff coefficient is alpha*rho*nuEff(), and for this
                # laminar (Stokes) model nuEff()==nu()==mu/rho, so the coefficient
                # collapses to mu -- NOT rho*mu -- for *both* the implicit laplacian and
                # the explicit dev2 correction below. The previous rho_face*gas.mu here
                # introduced a spurious extra density factor the explicit viscous_stress
                # term (which correctly uses mu_vol, unweighted) does not have, making
                # the two halves of what should be one consistent operator disagree.
                mu_face = surf_interp.interpolate(mu_vol)
                mu_face.name = "mu"

                ueqn_expr = (
                    nn.imp.ddt(rho, U) + nn.imp.div(phi, U) - nn.imp.laplacian(mu_face, U)
                )
                if not _DROP_VISCOUS_STRESS:
                    ueqn_expr = ueqn_expr + nfb.viscous_stress(mu_vol, zero_nut, grad_u)

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

                if os.environ.get("NEORHO_TRACE_LL_T") == "1" and runTime.timeIndex() <= 8:
                    import numpy as _np

                    T.correct_boundary_conditions()
                    # Sanity check T's own boundary value independent of the gradient machinery:
                    # walls is plain zeroGradient, so T_boundary[walls] must equal T[owner]
                    # exactly (fixedGradient=0 => value = iVector[owner] + 0/deltaCoeff), a
                    # trivial, unit-independent correctness check on the *input* before blaming
                    # the gradient operator itself.
                    Tb = _np.asarray(T.boundary_data_value().copy_to_host())
                    T_full = _np.asarray(T.internal_vector().copy_to_host())
                    print(
                        f"TRACE_LL_T_BC t={runTime.timeName()} outer={_outer_counter} "
                        f"T_boundary[inlet]={Tb[0]} T_boundary[walls]={Tb[40]} "
                        f"T[0]={T_full[0]} T[1]={T_full[1]} T[50]={T_full[50]}",
                        flush=True,
                    )
                    _ll_dst = nn.ScalarSurfaceField(rt.executor, "T_ll_probe", rt.nf_mesh)
                    _ll_interp.interpolate_into(phi, T, _ll_dst)
                    llv = _np.asarray(_ll_dst.internal_vector().copy_to_host())
                    Tarr = _np.asarray(T.internal_vector().copy_to_host())
                    # Report the *effective blend weight*, not the raw value, so the comparison
                    # against literal's "e"-based reconstruction (he = Cv*(T-Tref)-ish, not T
                    # directly) needs no unit conversion: w_eff = (recon - T_nei)/(T_own - T_nei)
                    # recovers the scheme's actual blend fraction regardless of the field's
                    # affine units, since the TVD ratio r = gradcf/gradf is itself affine-invariant.
                    w0 = (llv[0] - Tarr[1]) / (Tarr[0] - Tarr[1])
                    w1 = (llv[1] - Tarr[50]) / (Tarr[0] - Tarr[50])
                    gT = _np.asarray(grad_op.grad_scalar(T).internal_vector().copy_to_host())
                    print(
                        f"TRACE_LL_T t={runTime.timeName()} outer={_outer_counter} "
                        f"T_ll[face0,+x]={llv[0]} T_ll[face1,+y]={llv[1]} "
                        f"w_eff[face0]={w0} w_eff[face1]={w1} "
                        f"gradT[0]={gT[0]} gradT[1]={gT[1]} gradT[50]={gT[50]}",
                        flush=True,
                    )

                phiv = phi / surf_interp.interpolate(rho)
                phiv.name = "phiv"
                # Live per-outer-iteration refresh, matching literal's actual
                # fixedEnergy/gradientEnergy mechanism (recomputed every call, not
                # captured once) -- push T's current boundary state onto he's
                # refValue/refGrad, then materialise he's value() from it.
                T.correct_boundary_conditions()
                nfb.map_energy_boundary_from_temperature(T, he, gas.Cv, _TSTD)
                he.correct_boundary_conditions()
                eeqn_expr = (
                    nn.imp.ddt(rho, he) + nn.imp.div(phi, he) - nn.imp.laplacian(alphaEff, he)
                )
                if not _DROP_K_TERMS:
                    eeqn_expr = eeqn_expr + nn.exp.ddt(rho, K) + nn.exp.div(phi, K)
                if not _DROP_PRESSURE_WORK:
                    eeqn_expr = eeqn_expr + nn.exp.div(phiv, p)
                EEqn = nfb.PDESolverScalar(
                    eeqn_expr,
                    he,
                    rt,
                )
                EEqn.set_final_iter(final_iter)
                EEqn.relax()

                if _TRACE_EEQN_TERMS and runTime.timeIndex() <= 1:
                    import numpy as _np

                    def _row_residual(expr: Any, T_current: Any) -> float:
                        # Full row-0 residual = sum_k(M[0,k]*T_current[k]) - rhs[0] --
                        # NOT just diag[0]*T0-rhs[0], which silently drops any
                        # off-diagonal (neighbour-coupling) entries row 0 may have for
                        # convection/diffusion terms. Caught via a uniform-field sanity
                        # check: div(phi,T) must be exactly 0 when T is uniform and
                        # local continuity holds, which the diagonal-only formula failed.
                        full_expr = nn.ExpressionScalar(rt.executor) + expr
                        solver = nfb.PDESolverScalar(full_expr, T, rt)
                        solver.assemble()
                        ls = solver.linear_system()
                        mat = ls.matrix()
                        vals = _np.asarray(mat.values().copy_to_host())
                        cols = _np.asarray(mat.col_idxs().copy_to_host())
                        rowp = _np.asarray(mat.row_offs().copy_to_host())
                        rhs = _np.asarray(ls.rhs().copy_to_host())
                        row_sum = 0.0
                        for k in range(int(rowp[0]), int(rowp[1])):
                            row_sum += float(vals[k]) * float(T_current[int(cols[k])])
                        return row_sum - float(rhs[0])

                    T0 = float(_np.asarray(T.internal_vector().copy_to_host())[0])
                    _phi_int = _np.asarray(phi.internal_vector().copy_to_host())
                    _phi_b = _np.asarray(phi.boundary_data_value().copy_to_host())
                    _T_full = _np.asarray(T.internal_vector().copy_to_host())
                    print(
                        f"EEQN_TERM_INPUTS t={runTime.timeName()} "
                        f"phi[inlet]={_phi_b[0]:.9e} phi[face0,+x,0-1]={_phi_int[0]:.9e} "
                        f"phi[face1,+y,0-5]={_phi_int[1]:.9e} "
                        f"T[0]={_T_full[0]:.9f} T[1]={_T_full[1]:.9f} T[5]={_T_full[5]:.9f}",
                        flush=True,
                    )
                    terms = {
                        "ddt(rho,T)": nn.imp.ddt(rho, T),
                        "div(phi,T)": nn.imp.div(phi, T),
                        "-laplacian(alphaEff,T)": -1.0 * nn.imp.laplacian(alphaEff, T),
                        "ddt(rho,K)/Cv": (1.0 / gas.Cv) * nn.exp.ddt(rho, K),
                        "div(phi,K)/Cv": (1.0 / gas.Cv) * nn.exp.div(phi, K),
                        "div(phiv,p)/Cv": (1.0 / gas.Cv) * nn.exp.div(phiv, p),
                    }
                    nets = {}
                    for name, expr in terms.items():
                        net = _row_residual(expr, _T_full)
                        nets[name] = net
                        print(
                            f"EEQN_TERM t={runTime.timeName()} name={name!r} "
                            f"net[0]={net:.9e}",
                            flush=True,
                        )
                    net_full = _row_residual(eeqn_expr, _T_full)
                    sum_nets = sum(nets.values())
                    print(
                        f"EEQN_TERM_CHECK t={runTime.timeName()} T0={T0:.9f} "
                        f"combined_net[0]={net_full:.9e} sum_of_6_terms={sum_nets:.9e} "
                        f"diff={net_full - sum_nets:.3e}",
                        flush=True,
                    )

                if final_iter:
                    import numpy as _np

                    def _term(op, explicit: bool) -> float:
                        if explicit:
                            nfb.evaluate_explicit(op, rt.fv_schemes_dict, _scratch_term.internal_vector())
                        else:
                            nfb.evaluate_implicit(
                                op, rt.fv_schemes_dict, T, _scratch_term.internal_vector()
                            )
                        return float(
                            _np.asarray(_scratch_term.internal_vector().copy_to_host())[0]
                        )

                    # ddt(...) returns a TemporalOperator, not a SpatialOperator --
                    # evaluate_implicit/evaluate_explicit only bind SpatialOperator, so
                    # the ddt terms can't be evaluated standalone this way; limited to
                    # the spatial (div/laplacian) terms.
                    div_phi_T = _term(nn.imp.div(phi, T), False)
                    lap_T = _term(nn.imp.laplacian(alphaEff, T), False)
                    div_K = _term((1.0 / gas.Cv) * nn.exp.div(phi, K), True)
                    div_phiv_p = _term((1.0 / gas.Cv) * nn.exp.div(phiv, p), True)

                    # Sanity check for the diagnostic itself: does the scheme lookup via
                    # rt.fv_schemes_dict actually resolve to "div(phiv,p)"'s configured
                    # limitedLinear 1 (not some silent default)? Re-evaluate the same term
                    # forcing an explicit, different scheme and compare.
                    nfb.evaluate_explicit(
                        (1.0 / gas.Cv) * nn.exp.div(phiv, p),
                        nn.TokenList(["Gauss", "upwind"]),
                        _scratch_term.internal_vector(),
                    )
                    div_phiv_p_upwind = float(
                        _np.asarray(_scratch_term.internal_vector().copy_to_host())[0]
                    )

                    phiv_b = _np.asarray(phiv.boundary_data_value().copy_to_host())
                    p_b = _np.asarray(p.boundary_data_value().copy_to_host())
                    p_arr = _np.asarray(p.internal_vector().copy_to_host())
                    print(
                        f"DEBUG t={runTime.timeName()} EEqn_terms[0] "
                        f"div_phi_T={div_phi_T:.6e} "
                        f"lap_T={lap_T:.6e} div_K={div_K:.6e} "
                        f"div_phiv_p={div_phiv_p:.6e} "
                        f"div_phiv_p_upwind={div_phiv_p_upwind:.6e} "
                        f"phiv[inlet]={phiv_b[0]:.6e} phiv[walls]={_safe_idx(phiv_b, 40):.6e} "
                        f"p[inlet]={p_b[0]:.6e} p[walls]={_safe_idx(p_b, 40):.6e} p[0]={p_arr[0]:.6f} "
                        f"p_b_full[0:3]={p_b[0:3]} p_b_full[20:23]={p_b[20:23]} "
                        f"p_b_full[40:43]={p_b[40:43]} p_b.size={p_b.size}",
                        flush=True,
                    )

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
                _piso_counter = 0

                while piso.correct():
                    _piso_counter += 1
                    psip0.assign(psi * p)
                    rAU, hByA = nfb.compute_rau_and_hbya(UEqn)
                    nfb.constrain_hbya(U, p, hByA)

                    rho_face = surf_interp.interpolate(rho)
                    rAUf = surf_interp.interpolate(rAU)
                    rAUf.name = "rAUf"
                    rhorAUf = surf_interp.interpolate(rho * rAU)
                    rhorAUf.name = "rhorAUf"
                    phiHbyA = rho_face * nfb.flux(hByA)

                    if _FREEZE_RHORAUF:
                        if frozen_rhorAUf is None:
                            frozen_rhorAUf = nfb.create_uniform_surface_field(rt, "rhorAUf0", 0.0)
                            frozen_rhorAUf.assign(rhorAUf)
                        pEqn_rhorAUf = frozen_rhorAUf
                    else:
                        pEqn_rhorAUf = rhorAUf

                    if final_iter:
                        import numpy as _np

                        phba = _np.asarray(phiHbyA.boundary_data_value().copy_to_host())
                        rauf = _np.asarray(rhorAUf.boundary_data_value().copy_to_host())
                        raufOnly = _np.asarray(rAUf.boundary_data_value().copy_to_host())
                        rhof = _np.asarray(rho_face.boundary_data_value().copy_to_host())
                        # boundary_data_value is flat in patch order (inlet, outlet, walls,
                        # frontAndBack per constant/polyMesh/boundary); local offset 0 = cell
                        # 0's inlet face, local offset 40 = cell 0's walls face (both owned by
                        # cell 0, confirmed from constant/polyMesh/owner at global faces
                        # 1930/1970 with nInternalFaces=1930, inlet nFaces=20, outlet nFaces=20).
                        print(
                            f"DEBUG t={runTime.timeName()} corner "
                            f"phiHbyA[inlet]={phba[0]:.6e} phiHbyA[walls]={_safe_idx(phba, 40):.6e} "
                            f"rhorAUf[inlet]={rauf[0]:.6e} rhorAUf[walls]={_safe_idx(rauf, 40):.6e} "
                            f"rAUf[inlet]={raufOnly[0]:.6e} rAUf[walls]={_safe_idx(raufOnly, 40):.6e} "
                            f"rho_face[inlet]={rhof[0]:.6e} rho_face[walls]={_safe_idx(rhof, 40):.6e}",
                            flush=True,
                        )

                    # imp.ddt(psi, p) discretizes the FULL conservative d(rho)/dt =
                    # d(psi*p)/dt directly (psi_new on the diagonal, psi_old*p_old on the
                    # rhs) -- unlike literal rhoPimpleFoam's fvc::ddt(rho) +
                    # psi*correction(fvm::ddt(p)), which holds psi frozen for the implicit
                    # part and adds back p*fvc::ddt(psi) explicitly. Our imp.ddt(psi, p)
                    # already carries that psi-sensitivity implicitly, so adding
                    # p*fvc::ddt(psi) on top double-counts it (and was the actual source
                    # of the instability T-relaxation was masking); it must be subtracted
                    # instead, to remove the redundant implicit-psi contribution.
                    ddt_psi = (psi - nn.old_time(psi)) / rt.dt
                    p_ddt_psi = p * ddt_psi

                    if _TRACE_PSI_P:
                        import numpy as _np

                        psi_c0 = float(
                            _np.asarray(psi.internal_vector().copy_to_host())[0]
                        )
                        psi_old0 = float(
                            _np.asarray(
                                nn.old_time(psi).internal_vector().copy_to_host()
                            )[0]
                        )
                        p_c0 = float(_np.asarray(p.internal_vector().copy_to_host())[0])
                        T_c0 = float(_np.asarray(T.internal_vector().copy_to_host())[0])
                        pdp0 = float(
                            _np.asarray(p_ddt_psi.internal_vector().copy_to_host())[0]
                        )
                        ratio = psi_old0 / psi_c0
                        shift0 = p_c0 * (1.0 - ratio)
                        print(
                            f"TRACE t={runTime.timeName()} step={runTime.timeIndex()} "
                            f"outer={_outer_counter} piso={_piso_counter} "
                            f"psi_c[0]={psi_c0:.9e} psi_oldT[0]={psi_old0:.9e} "
                            f"ratio={ratio:.9f} p_c[0]={p_c0:.6f} T_c[0]={T_c0:.6f} "
                            f"p_ddt_psi[0]={pdp0:.6e} shift[0]={shift0:.6f}",
                            flush=True,
                        )

                    if final_iter:
                        import numpy as _np

                        pdp = _np.asarray(p_ddt_psi.internal_vector().copy_to_host())
                        Tarr = _np.asarray(T.internal_vector().copy_to_host())
                        parr = _np.asarray(p.internal_vector().copy_to_host())
                        print(
                            f"DEBUG t={runTime.timeName()} p_ddt_psi[0]={pdp[0]:.6e} "
                            f"p_ddt_psi[950]={_safe_idx(pdp, 950):.6e} T[0]={Tarr[0]:.6f} "
                            f"p[0]={parr[0]:.4f}",
                            flush=True,
                        )

                    pEqn_ddt_psi_term = (
                        nn.imp.ddt(psi, p)
                        if _ZERO_PDDT_PSI
                        else nn.imp.ddt(psi, p) - nn.exp.source(p_ddt_psi)
                    )

                    while piso.correct_non_orthogonal():
                        pEqn = nfb.PDESolverScalar(
                            pEqn_ddt_psi_term
                            + nn.exp.div(phiHbyA)
                            - nn.imp.laplacian(pEqn_rhorAUf, p),
                            p,
                            rt,
                        )
                        pEqn.set_final_iter(final_iter)

                        if needs_ref:
                            pEqn.set_reference(p_ref_cell, p_ref_value)

                        if _TRACE_PEQN_MATRIX and runTime.timeIndex() <= 6:
                            import numpy as _np

                            pEqn.assemble()
                            ls = pEqn.linear_system()
                            mat = ls.matrix()
                            vals = _np.asarray(mat.values().copy_to_host())
                            cols = _np.asarray(mat.col_idxs().copy_to_host())
                            rowp = _np.asarray(mat.row_offs().copy_to_host())
                            rhs = _np.asarray(ls.rhs().copy_to_host())

                            def _row(r: int) -> dict[str, float]:
                                out = {f"rhs_{r}": float(rhs[r])}
                                for k in range(int(rowp[r]), int(rowp[r + 1])):
                                    out[f"M_{r}_{int(cols[k])}"] = float(vals[k])
                                return out

                            blk: dict[str, float] = {}
                            for cell in (0, 1, 50):
                                blk.update(_row(cell))
                            psi_oldT_arr = _np.asarray(
                                nn.old_time(psi).internal_vector().copy_to_host()
                            )
                            p_oldT_arr = _np.asarray(
                                nn.old_time(p).internal_vector().copy_to_host()
                            )
                            vols = _np.asarray(p.mesh().cell_volumes().copy_to_host())
                            extra = {
                                "psi_oldT_0": float(psi_oldT_arr[0]),
                                "psi_oldT_1": float(psi_oldT_arr[1]),
                                "psi_oldT_50": float(psi_oldT_arr[50]),
                                "p_oldT_0": float(p_oldT_arr[0]),
                                "p_oldT_1": float(p_oldT_arr[1]),
                                "p_oldT_50": float(p_oldT_arr[50]),
                                "V_0": float(vols[0]),
                                "V_1": float(vols[1]),
                                "V_50": float(vols[50]),
                                "dt": float(rt.dt),
                            }
                            print(
                                f"TRACE_PEQN_MATRIX t={runTime.timeName()} "
                                f"step={runTime.timeIndex()} outer={_outer_counter} "
                                f"piso={_piso_counter} final={final_iter} blk={blk} "
                                f"extra={extra}",
                                flush=True,
                            )

                        if _DIRECT_PSOLVE:
                            import numpy as _np
                            from scipy.sparse import csr_matrix as _csr_matrix
                            from scipy.sparse.linalg import spsolve as _spsolve

                            pEqn.assemble()
                            ls = pEqn.linear_system()
                            mat = ls.matrix()
                            vals = _np.asarray(mat.values().copy_to_host())
                            cols = _np.asarray(mat.col_idxs().copy_to_host())
                            rowp = _np.asarray(mat.row_offs().copy_to_host())
                            rhs_vec = _np.asarray(ls.rhs().copy_to_host())
                            n = mat.n_rows()
                            A = _csr_matrix((vals, cols, rowp), shape=(n, n))
                            p_new = _spsolve(A, rhs_vec)
                            if not _np.all(_np.isfinite(p_new)):
                                print(
                                    f"DIRECT_PSOLVE t={runTime.timeName()} "
                                    f"step={runTime.timeIndex()} NON-FINITE direct "
                                    f"solve result (nan/inf) -- matrix likely singular "
                                    f"or indefinite",
                                    flush=True,
                                )
                            nfb.set_scalar_field_values(p, p_new.tolist())
                            if not have_p_res:
                                p_res = (0.0, 0.0)
                                have_p_res = True
                        else:
                            stats_p = pEqn.solve()
                            if not have_p_res:
                                entry = stats_p.entries[0]
                                p_res = (entry.initial_residual, entry.final_residual)
                                have_p_res = True
                        p.correct_boundary_conditions()

                        if piso.final_non_orthogonal_iter():
                            nfb.update_face_velocity(phiHbyA, pEqn, phi)

                            if runTime.timeIndex() <= 1:
                                import numpy as _np

                                phi_b = _np.asarray(phi.boundary_data_value().copy_to_host())
                                phi_int = _np.asarray(phi.internal_vector().copy_to_host())
                                p_full = _np.asarray(p.internal_vector().copy_to_host())
                                rauf_int = _np.asarray(rhorAUf.internal_vector().copy_to_host())
                                print(
                                    f"DEBUG t={runTime.timeName()} "
                                    f"outer={_outer_counter} piso={_piso_counter} "
                                    f"PHI_POST_CORRECTION "
                                    f"phi[inlet]={phi_b[0]:.9e} phi[walls]={_safe_idx(phi_b, 40):.9e} "
                                    f"phi_int[face0,+x]={phi_int[0]:.9e} "
                                    f"phi_int[face1,+y]={phi_int[1]:.9e} "
                                    f"p[0]={p_full[0]:.9f} p[1]={p_full[1]:.9f} "
                                    f"p[50]={_safe_idx(p_full, 50):.9f} "
                                    f"rhorAUf_int[face0]={rauf_int[0]:.9e} "
                                    f"rhorAUf_int[face1]={rauf_int[1]:.9e}",
                                    flush=True,
                                )

                    nfb.apply_field_relaxation(
                        p,
                        prev_p,
                        nfb.lookup_field_relaxation(rt.fv_solution_dict, p.name, final_iter),
                    )
                    p.correct_boundary_conditions()

                    if final_iter:
                        import numpy as _np

                        pb_check = _np.asarray(p.boundary_data_value().copy_to_host())
                        p_int_check = _np.asarray(p.internal_vector().copy_to_host())
                        print(
                            f"DEBUG t={runTime.timeName()} POST-CORRECT p[0]={p_int_check[0]:.6f} "
                            f"p_boundary[inlet]={pb_check[0]:.6f} p_boundary[walls]={_safe_idx(pb_check, 40):.6f}",
                            flush=True,
                        )

                    nfb.update_velocity(hByA, rAU, p, U, rt)
                    U.correct_boundary_conditions()
                    # Feature-for-feature port: literal pEqn.H also refreshes K here, from
                    # the just-pressure-corrected U -- not just once after the momentum
                    # predictor. Without this, K stays one-PISO-pass stale for the rest of
                    # this outer iteration and feeds EEqn's explicit K terms next outer
                    # iteration using pre-pressure-correction U.
                    K.assign(0.5 * nn.mag_sqr(U))
                    rho.assign(rho + psi * p - psip0)

                    if final_iter:
                        import numpy as _np

                        divphi = _np.asarray(nfb.compute_div_phi(phi, rt))
                        print(
                            f"DEBUG t={runTime.timeName()} divphi[0]={divphi[0]:.6e} "
                            f"divphi[950]={_safe_idx(divphi, 950):.6e}",
                            flush=True,
                        )

                if have_p_res:
                    residuals["p"] = p_res

                print(f"DEBUG t={runTime.timeName()} residuals={residuals}", flush=True)

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
