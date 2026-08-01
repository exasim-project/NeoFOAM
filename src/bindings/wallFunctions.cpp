// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Turbulence wall-function helpers (steady SIMPLE parity for kEpsilon/kOmegaSST).
// The pure-Python closures own the transport equations; these expose the
// *model-side* half of the epsilon/omega/nut wall functions that cannot be
// written in the field DSL (boundary-face views, atomic per-cell scatter,
// matrix-cell pins):
//   * read_wall_distance / build_near_wall_dist — the wall-distance inputs,
//   * correct_scalar_bc_ctx[_u]  — correctBoundaryConditions with the (k,nu,y)
//                                  / (U,nu,y) context the WF BCs need,
//   * epsilon_wall_production    — near-wall production (G) override,
//   * pin_epsilon_wall_cells / pin_omega_wall_cells — the near-wall matrix-cell
//                                  pin (the cell half of manipulateMatrix),
//   * refresh_omega_wall_cells   — the cell-write half of updateCoeffs (D2).
// Top-level (neofoam) only; the NeoN submodule is untouched.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

// NeoN headers
#include "NeoN/NeoN.hpp"

// OpenFOAM headers
#include "wallDist.H" // Foam::wallDist for read_wall_distance / build_near_wall_dist

// NeoFOAM headers
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/datastructures/pde.hpp" // PDE<scalar> for the epsilon/omega wall-cell pin
#include "NeoFOAM/fvcc/boundary/volume/omegaWallFunction.hpp" // OMEGA_WF_OMEGA_MAX for the omega pin
#include "NeoFOAM/auxiliary/readers.hpp" // constructFrom (Foam field -> NeoN field)

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

using NeoN::localIdx;
using NeoN::scalar;

namespace NeoFOAM::bindings
{

void registerWallFunctions(nb::module_& m)
{
    m.def(
        "read_wall_distance",
        [](nf::RunTime& rt) -> fvcc::VolumeField<NeoN::scalar>
        {
            Foam::wallDist y(rt.mesh);
            return NeoFOAM::constructFrom(rt.exec, rt.nfMesh, y.y());
        },
        "runtime"_a,
        "Wall-distance field y (Foam::wallDist to the nearest wall patch)"
    );

    // -------------------------------------------------------------------
    // The pure-Python kEpsilon closure owns the transport equations; these expose
    // the *model-side* half of the epsilon/k/nut wall functions it cannot write in
    // the field DSL (boundary-face views, atomic per-cell scatter):
    //   * build_near_wall_dist  — nearWallDist field (boundary faces = owner-cell y),
    //   * correct_scalar_bc_ctx — correctBoundaryConditions with a (U,k,nu,y) context
    //                             so the WF BCs actually set their face values,
    //   * epsilon_wall_production — the near-wall production (G) override the
    //                             epsilonWallFunction depends on (mirrors
    //                             KEpsilon::correct, kEpsilon.cpp).
    // -------------------------------------------------------------------
    m.def(
        "build_near_wall_dist",
        [](nf::RunTime& rt) -> fvcc::VolumeField<scalar>
        {
            Foam::wallDist y(rt.mesh);
            const auto wallDist = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, y.y());
            fvcc::VolumeField<scalar> nearWallDist(
                rt.exec,
                "nearWallDist",
                rt.nfMesh,
                fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(rt.nfMesh)
            );
            const auto wd = wallDist.internalVector().view();
            const auto owners = rt.nfMesh.boundaryMesh().faceOwners().view();
            auto nwdB = nearWallDist.boundaryData().value().view();
            const auto nBF = static_cast<localIdx>(nwdB.size());
            NeoN::parallelFor(
                rt.exec,
                {0, nBF},
                NEON_LAMBDA(const localIdx i) { nwdB[i] = wd[owners[i]]; },
                "build_near_wall_dist"
            );
            return nearWallDist;
        },
        "runtime"_a,
        "nearWallDist field: boundary faces hold the owner-cell wall distance (WF input)"
    );

    m.def(
        "correct_scalar_bc_ctx",
        [](fvcc::VolumeField<scalar>& target,
           const fvcc::VolumeField<scalar>& k,
           const fvcc::VolumeField<scalar>& nu,
           const fvcc::VolumeField<scalar>& nearWallDist)
        {
            fvcc::BoundaryContext ctx;
            ctx.insert("k", k);
            ctx.insert("nu", nu);
            ctx.insert("nearWallDist", nearWallDist);
            target.correctBoundaryConditions(ctx);
        },
        "target"_a,
        "k"_a,
        "nu"_a,
        "near_wall_dist"_a,
        "correctBoundaryConditions with a (k,nu,nearWallDist) context for the epsilon/nutk/kqR WFs"
    );

    m.def(
        "correct_scalar_bc_ctx_u",
        [](fvcc::VolumeField<scalar>& target,
           const fvcc::VolumeField<NeoN::Vec3>& U,
           const fvcc::VolumeField<scalar>& nu,
           const fvcc::VolumeField<scalar>& nearWallDist)
        {
            // nutUSpaldingWallFunction reads U (not k): (U,nu,nearWallDist) context.
            fvcc::BoundaryContext ctx;
            ctx.insert("U", U);
            ctx.insert("nu", nu);
            ctx.insert("nearWallDist", nearWallDist);
            target.correctBoundaryConditions(ctx);
        },
        "target"_a,
        "U"_a,
        "nu"_a,
        "near_wall_dist"_a,
        "correctBoundaryConditions with a (U,nu,nearWallDist) context for the nutUSpalding WF"
    );

    m.def(
        "epsilon_wall_production",
        [](const fvcc::VolumeField<scalar>& G,
           const fvcc::VolumeField<scalar>& epsilon,
           const fvcc::VolumeField<NeoN::Vec3>& U,
           const fvcc::VolumeField<scalar>& k,
           const fvcc::VolumeField<scalar>& nu,
           const fvcc::VolumeField<scalar>& nut,
           const fvcc::VolumeField<scalar>& nearWallDist,
           const nf::RunTime& rt,
           double cmu,
           double kappa,
           const std::string& wall_patch) -> fvcc::VolumeField<scalar>
        {
            // Copy the bulk production; override only the wall cells (the k equation
            // consumes this; the epsilon/omega equation keeps the un-overridden bulk G).
            // Reused for kOmegaSST (wall_patch="omegaWallFunction", cmu=betaStar) — its
            // near-wall G formula is identical to kEpsilon's.
            fvcc::VolumeField<scalar> Gk = G;
            const auto& exec = rt.exec;
            const auto& mesh = rt.nfMesh;
            const scalar cmu25 = Kokkos::pow(static_cast<scalar>(cmu), scalar(0.25));
            const scalar kappa_ = static_cast<scalar>(kappa);
            const auto epsilonBCs = epsilon.boundaryConditions();
            const auto faceOwnersV = mesh.boundaryMesh().faceOwners().view();
            const auto deltaCoeffsV = mesh.boundaryMesh().deltaCoeffs().view();
            const auto uInternalV = U.internalVector().view();
            const auto uBoundaryV = U.boundaryData().value().view();
            const auto nuBoundaryV = nu.boundaryData().value().view();
            const auto nutBoundaryV = nut.boundaryData().value().view();
            const auto yBoundaryV = nearWallDist.boundaryData().value().view();
            const auto kInternalV = k.internalVector().view();
            auto gkV = Gk.internalVector().view();
            const auto nCells = static_cast<localIdx>(mesh.nCells());

            // cornerWeight[c] = 1/(count of epsilonWallFunction faces on cell c), else 0.
            NeoN::Vector<scalar> cornerWeight(exec, nCells, scalar(0));
            auto cwBuild = cornerWeight.view();
            for (localIdx patchID = 0; patchID < static_cast<localIdx>(epsilonBCs.size());
                 ++patchID)
            {
                if (epsilonBCs[static_cast<size_t>(patchID)].name() != wall_patch) continue;
                const auto [start, end] = epsilon.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        Kokkos::atomic_add(&cwBuild[faceOwnersV[i]], scalar(1));
                    },
                    "epsilon_wall_production::countFaces"
                );
            }
            NeoN::parallelFor(
                exec,
                {0, nCells},
                NEON_LAMBDA(const localIdx c) {
                    if (cwBuild[c] > scalar(0)) cwBuild[c] = scalar(1) / cwBuild[c];
                },
                "epsilon_wall_production::invertCount"
            );
            const auto cornerWeightV = cornerWeight.view();

            // Zero bulk production at wall cells before accumulating the WF form.
            NeoN::parallelFor(
                exec,
                {0, nCells},
                NEON_LAMBDA(const localIdx c) {
                    if (cornerWeightV[c] > scalar(0)) gkV[c] = scalar(0);
                },
                "epsilon_wall_production::zeroWall"
            );
            NeoN::fence(exec);

            for (localIdx patchID = 0; patchID < static_cast<localIdx>(epsilonBCs.size());
                 ++patchID)
            {
                if (epsilonBCs[static_cast<size_t>(patchID)].name() != wall_patch) continue;
                const auto [start, end] = epsilon.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        const auto owner = faceOwnersV[i];
                        const scalar cw = cornerWeightV[owner];
                        const NeoN::Vec3 uOwn = uInternalV[owner];
                        const NeoN::Vec3 uWall = uBoundaryV[i];
                        const scalar deltaInv = deltaCoeffsV[i];
                        const NeoN::Vec3 snGradU = (uWall - uOwn) * deltaInv;
                        const scalar magGradUw = NeoN::mag(snGradU);
                        const scalar nuw = nuBoundaryV[i];
                        const scalar nutw = nutBoundaryV[i];
                        const scalar yv = yBoundaryV[i];
                        const scalar kc = Kokkos::max(kInternalV[owner], scalar(0));
                        const scalar gWall =
                            (nutw + nuw) * magGradUw * cmu25 * Kokkos::sqrt(kc) / (kappa_ * yv);
                        Kokkos::atomic_add(&gkV[owner], cw * gWall);
                    },
                    "epsilon_wall_production::gFeedback"
                );
            }
            return Gk;
        },
        "G"_a,
        "epsilon"_a,
        "U"_a,
        "k"_a,
        "nu"_a,
        "nut"_a,
        "near_wall_dist"_a,
        "runtime"_a,
        "cmu"_a = 0.09,
        "kappa"_a = 0.41,
        "wall_patch"_a = "epsilonWallFunction",
        "k-equation production with the epsilon/omega wall-function near-wall G override"
    );

    m.def(
        "pin_epsilon_wall_cells",
        [](nf::PDE<scalar>& epsPde,
           const fvcc::VolumeField<scalar>& epsilon,
           const fvcc::VolumeField<scalar>& k,
           const fvcc::VolumeField<scalar>& nearWallDist,
           const nf::RunTime& rt,
           double cmu,
           double kappa)
        {
            // OpenFOAM's epsilonWallFunction::manipulateMatrix pins the near-wall CELL
            // epsilon to the log-law value (STEPWISE blender, lowRe off):
            //   epsilon0 = Cmu^0.75 k^1.5 / (kappa y),
            // corner-weight-averaged over the wall faces touching each cell. The BC only
            // sets the wall *face*; this is the cell half. Mirrors the omegaWallFunction
            // pin in kOmegaSST::correct.
            const auto& exec = rt.exec;
            const auto& mesh = rt.nfMesh;
            const scalar Cmu75 = Kokkos::pow(static_cast<scalar>(cmu), scalar(0.75));
            const scalar kappa_ = static_cast<scalar>(kappa);
            const auto epsilonBCs = epsilon.boundaryConditions();
            const auto faceOwnersV = mesh.boundaryMesh().faceOwners().view();
            const auto yBoundaryV = nearWallDist.boundaryData().value().view();
            const auto kInternalV = k.internalVector().view();
            const auto nCells = static_cast<localIdx>(mesh.nCells());

            // cornerWeight[c] = 1/(count of epsilonWallFunction faces on cell c), else 0.
            NeoN::Vector<scalar> cornerWeight(exec, nCells, scalar(0));
            auto cwBuild = cornerWeight.view();
            for (localIdx patchID = 0; patchID < static_cast<localIdx>(epsilonBCs.size());
                 ++patchID)
            {
                if (epsilonBCs[static_cast<size_t>(patchID)].name() != "epsilonWallFunction")
                    continue;
                const auto [start, end] = epsilon.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        Kokkos::atomic_add(&cwBuild[faceOwnersV[i]], scalar(1));
                    },
                    "pin_epsilon_wall_cells::countFaces"
                );
            }
            NeoN::parallelFor(
                exec,
                {0, nCells},
                NEON_LAMBDA(const localIdx c) {
                    if (cwBuild[c] > scalar(0)) cwBuild[c] = scalar(1) / cwBuild[c];
                },
                "pin_epsilon_wall_cells::invertCount"
            );
            const auto cornerWeightV = cornerWeight.view();

            NeoN::Vector<scalar> mask(exec, nCells, scalar(0));
            NeoN::Vector<scalar> values(exec, nCells, scalar(0));
            auto maskV = mask.view();
            auto valuesV = values.view();
            NeoN::fence(exec);

            for (localIdx patchID = 0; patchID < static_cast<localIdx>(epsilonBCs.size());
                 ++patchID)
            {
                if (epsilonBCs[static_cast<size_t>(patchID)].name() != "epsilonWallFunction")
                    continue;
                const auto [start, end] = epsilon.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        const auto owner = faceOwnersV[i];
                        const scalar cw = cornerWeightV[owner];
                        const scalar y = yBoundaryV[i];
                        const scalar kc = Kokkos::max(kInternalV[owner], scalar(0));
                        const scalar eLog = Cmu75 * Kokkos::pow(kc, scalar(1.5)) / (kappa_ * y);
                        Kokkos::atomic_add(&valuesV[owner], cw * eLog);
                        maskV[owner] = scalar(1);
                    },
                    "pin_epsilon_wall_cells::accumulate"
                );
            }
            epsPde.setConstraintsOwned(mask, values);
        },
        "epsilon_pde"_a,
        "epsilon"_a,
        "k"_a,
        "near_wall_dist"_a,
        "runtime"_a,
        "cmu"_a = 0.09,
        "kappa"_a = 0.41,
        "Pin the epsilon equation's near-wall cells to the epsilonWallFunction log-law value"
    );

    m.def(
        "pin_omega_wall_cells",
        [](nf::PDE<scalar>& omegaPde,
           const fvcc::VolumeField<scalar>& omega,
           const fvcc::VolumeField<scalar>& k,
           const fvcc::VolumeField<scalar>& nu,
           const fvcc::VolumeField<scalar>& nearWallDist,
           const nf::RunTime& rt,
           double beta1,
           double cmu,
           double kappa)
        {
            // OpenFOAM's omegaWallFunction::manipulateMatrix pins the near-wall CELL
            // omega to the blended (BINOMIAL n=2) value:
            //   wVis = 6 nu / (beta1 y^2), wLog = sqrt(k) / (Cmu^0.25 kappa y),
            //   omega0 = min(sqrt(wVis^2 + wLog^2), OMEGA_WF_OMEGA_MAX),
            // corner-weight-averaged over the wall faces touching each cell. The BC only
            // sets the wall *face*; this is the cell half (mirrors kOmegaSST::correct).
            const auto& exec = rt.exec;
            const auto& mesh = rt.nfMesh;
            const scalar cmu25 = Kokkos::pow(static_cast<scalar>(cmu), scalar(0.25));
            const scalar beta1_ = static_cast<scalar>(beta1);
            const scalar kappa_ = static_cast<scalar>(kappa);
            const scalar omegaMax = fvcc::volumeBoundary::detail::OMEGA_WF_OMEGA_MAX;
            const auto omegaBCs = omega.boundaryConditions();
            const auto faceOwnersV = mesh.boundaryMesh().faceOwners().view();
            const auto yBoundaryV = nearWallDist.boundaryData().value().view();
            const auto nuBoundaryV = nu.boundaryData().value().view();
            const auto kInternalV = k.internalVector().view();
            const auto nCells = static_cast<localIdx>(mesh.nCells());

            // cornerWeight[c] = 1/(count of omegaWallFunction faces on cell c), else 0.
            NeoN::Vector<scalar> cornerWeight(exec, nCells, scalar(0));
            auto cwBuild = cornerWeight.view();
            for (localIdx patchID = 0; patchID < static_cast<localIdx>(omegaBCs.size()); ++patchID)
            {
                if (omegaBCs[static_cast<size_t>(patchID)].name() != "omegaWallFunction") continue;
                const auto [start, end] = omega.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        Kokkos::atomic_add(&cwBuild[faceOwnersV[i]], scalar(1));
                    },
                    "pin_omega_wall_cells::countFaces"
                );
            }
            NeoN::parallelFor(
                exec,
                {0, nCells},
                NEON_LAMBDA(const localIdx c) {
                    if (cwBuild[c] > scalar(0)) cwBuild[c] = scalar(1) / cwBuild[c];
                },
                "pin_omega_wall_cells::invertCount"
            );
            const auto cornerWeightV = cornerWeight.view();

            NeoN::Vector<scalar> mask(exec, nCells, scalar(0));
            NeoN::Vector<scalar> values(exec, nCells, scalar(0));
            auto maskV = mask.view();
            auto valuesV = values.view();
            NeoN::fence(exec);

            for (localIdx patchID = 0; patchID < static_cast<localIdx>(omegaBCs.size()); ++patchID)
            {
                if (omegaBCs[static_cast<size_t>(patchID)].name() != "omegaWallFunction") continue;
                const auto [start, end] = omega.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        const auto owner = faceOwnersV[i];
                        const scalar cw = cornerWeightV[owner];
                        const scalar y = yBoundaryV[i];
                        const scalar nuw = nuBoundaryV[i];
                        const scalar kc = Kokkos::max(kInternalV[owner], scalar(0));
                        const scalar wVis = scalar(6) * nuw / (beta1_ * y * y);
                        const scalar wLog = Kokkos::sqrt(kc) / (cmu25 * kappa_ * y);
                        const scalar wOmega =
                            Kokkos::min(Kokkos::sqrt(wVis * wVis + wLog * wLog), omegaMax);
                        Kokkos::atomic_add(&valuesV[owner], cw * wOmega);
                        maskV[owner] = scalar(1);
                    },
                    "pin_omega_wall_cells::accumulate"
                );
            }
            omegaPde.setConstraintsOwned(mask, values);
        },
        "omega_pde"_a,
        "omega"_a,
        "k"_a,
        "nu"_a,
        "near_wall_dist"_a,
        "runtime"_a,
        "beta1"_a = 0.075,
        "cmu"_a = 0.09,
        "kappa"_a = 0.41,
        "Pin the omega equation's near-wall cells to the omegaWallFunction blended value"
    );

    m.def(
        "refresh_omega_wall_cells",
        [](fvcc::VolumeField<scalar>& omega,
           const fvcc::VolumeField<scalar>& k,
           const fvcc::VolumeField<scalar>& nu,
           const fvcc::VolumeField<scalar>& nearWallDist,
           const nf::RunTime& rt,
           double beta1,
           double cmu,
           double kappa)
        {
            // The cell-write half of omegaWallFunction::updateCoeffs
            // (kOmegaSSTBase.C:541 — "omegaWallFunctions change the cell value!"):
            // overwrite the near-wall omega CELL with the blended (BINOMIAL n=2)
            // omega0 evaluated from the CURRENT k, *before* blend/CDkOmega/F1 read it.
            // Sibling to pin_omega_wall_cells (which pins the matrix); this refreshes
            // the field so the pre-solve blending coefficients use current-k near-wall
            // omega instead of last iteration's pinned value (removes the D2 lag).
            const auto& exec = rt.exec;
            const auto& mesh = rt.nfMesh;
            const scalar cmu25 = Kokkos::pow(static_cast<scalar>(cmu), scalar(0.25));
            const scalar beta1_ = static_cast<scalar>(beta1);
            const scalar kappa_ = static_cast<scalar>(kappa);
            const scalar omegaMax = fvcc::volumeBoundary::detail::OMEGA_WF_OMEGA_MAX;
            const auto omegaBCs = omega.boundaryConditions();
            const auto faceOwnersV = mesh.boundaryMesh().faceOwners().view();
            const auto yBoundaryV = nearWallDist.boundaryData().value().view();
            const auto nuBoundaryV = nu.boundaryData().value().view();
            const auto kInternalV = k.internalVector().view();
            const auto nCells = static_cast<localIdx>(mesh.nCells());

            // cornerWeight[c] = 1/(count of omegaWallFunction faces on cell c), else 0.
            NeoN::Vector<scalar> cornerWeight(exec, nCells, scalar(0));
            auto cwBuild = cornerWeight.view();
            for (localIdx patchID = 0; patchID < static_cast<localIdx>(omegaBCs.size()); ++patchID)
            {
                if (omegaBCs[static_cast<size_t>(patchID)].name() != "omegaWallFunction") continue;
                const auto [start, end] = omega.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        Kokkos::atomic_add(&cwBuild[faceOwnersV[i]], scalar(1));
                    },
                    "refresh_omega_wall_cells::countFaces"
                );
            }
            NeoN::parallelFor(
                exec,
                {0, nCells},
                NEON_LAMBDA(const localIdx c) {
                    if (cwBuild[c] > scalar(0)) cwBuild[c] = scalar(1) / cwBuild[c];
                },
                "refresh_omega_wall_cells::invertCount"
            );
            const auto cornerWeightV = cornerWeight.view();

            NeoN::Vector<scalar> values(exec, nCells, scalar(0));
            auto valuesV = values.view();
            NeoN::fence(exec);
            for (localIdx patchID = 0; patchID < static_cast<localIdx>(omegaBCs.size()); ++patchID)
            {
                if (omegaBCs[static_cast<size_t>(patchID)].name() != "omegaWallFunction") continue;
                const auto [start, end] = omega.boundaryData().range(patchID);
                NeoN::parallelFor(
                    exec,
                    {start, end},
                    NEON_LAMBDA(const localIdx i) {
                        const auto owner = faceOwnersV[i];
                        const scalar cw = cornerWeightV[owner];
                        const scalar y = yBoundaryV[i];
                        const scalar nuw = nuBoundaryV[i];
                        const scalar kc = Kokkos::max(kInternalV[owner], scalar(0));
                        const scalar wVis = scalar(6) * nuw / (beta1_ * y * y);
                        const scalar wLog = Kokkos::sqrt(kc) / (cmu25 * kappa_ * y);
                        const scalar wOmega =
                            Kokkos::min(Kokkos::sqrt(wVis * wVis + wLog * wLog), omegaMax);
                        Kokkos::atomic_add(&valuesV[owner], cw * wOmega);
                    },
                    "refresh_omega_wall_cells::accumulate"
                );
            }
            // Overwrite the near-wall cell omega with omega0 (cells with a WF face).
            auto omegaInternalV = omega.internalVector().view();
            NeoN::parallelFor(
                exec,
                {0, nCells},
                NEON_LAMBDA(const localIdx c) {
                    if (cornerWeightV[c] > scalar(0)) omegaInternalV[c] = valuesV[c];
                },
                "refresh_omega_wall_cells::write"
            );
        },
        "omega"_a,
        "k"_a,
        "nu"_a,
        "near_wall_dist"_a,
        "runtime"_a,
        "beta1"_a = 0.075,
        "cmu"_a = 0.09,
        "kappa"_a = 0.41,
        "Refresh near-wall omega CELL values to the omegaWallFunction omega0 (current k)"
        " — the cell-write half of updateCoeffs (removes the D2 lag)"
    );
}

} // namespace NeoFOAM::bindings
