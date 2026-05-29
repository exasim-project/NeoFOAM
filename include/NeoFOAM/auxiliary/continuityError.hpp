// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 - 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"

namespace NeoFOAM
{

struct ContinuityErrors
{
    NeoN::scalar sumLocal;
    NeoN::scalar global;
};

/**
 * @brief Compute time-step continuity errors from the face flux field.
 *
 * Matches the OpenFOAM continuityErrs.H formulation:
 *   sumLocal  = dt * volWeightedAverage( |div(phi)| )   (abs, MPI-reduced)
 *   global    = dt * volWeightedAverage(  div(phi)  )   (signed, MPI-reduced)
 *
 * @param phi   Face-flux NeoN SurfaceField
 * @param rt    NeoFOAM RunTime (provides exec, mesh, MPI env, dt)
 */
inline ContinuityErrors computeContinuityError(
    const NeoN::finiteVolume::cellCentred::SurfaceField<NeoN::scalar>& phi,
    const RunTime& rt
)
{
    const NeoN::scalar dt = rt.dt;
    const auto nCells = static_cast<NeoN::localIdx>(rt.nfMesh.nCells());

    NeoN::dsl::Expression<NeoN::scalar> divExpr(rt.exec);
    divExpr.addOperator(NeoN::dsl::exp::div(phi));
    NeoN::Vector<NeoN::scalar> divPhi = divExpr.explicitOperation(nCells);

    const auto divPhiView = divPhi.view();
    const auto volView = rt.nfMesh.cellVolumes().view();

    NeoN::scalar localAbsVolSum = 0.0;
    NeoN::parallelReduce(
        rt.exec,
        {0, static_cast<size_t>(nCells)},
        NEON_LAMBDA(const size_t i, NeoN::scalar& s) {
            s += Kokkos::abs(divPhiView[i]) * volView[i];
        },
        localAbsVolSum
    );

    NeoN::scalar localSignedVolSum = 0.0;
    NeoN::parallelReduce(
        rt.exec,
        {0, static_cast<size_t>(nCells)},
        NEON_LAMBDA(const size_t i, NeoN::scalar& s) { s += divPhiView[i] * volView[i]; },
        localSignedVolSum
    );

    NeoN::scalar localTotalVol = 0.0;
    NeoN::parallelReduce(
        rt.exec,
        {0, static_cast<size_t>(nCells)},
        NEON_LAMBDA(const size_t i, NeoN::scalar& s) { s += volView[i]; },
        localTotalVol
    );

    NeoN::scalar globalAbsVolSum = localAbsVolSum;
    NeoN::scalar globalSignedVolSum = localSignedVolSum;
    NeoN::scalar totalVol = localTotalVol;

#ifdef NF_WITH_MPI_SUPPORT
    if (rt.nfMesh.boundaryMesh().isDistributed())
    {
        NeoN::mpi::Environment env;
        NeoN::scalar sums[3] = {globalAbsVolSum, globalSignedVolSum, totalVol};
        MPI_Allreduce(
            MPI_IN_PLACE,
            sums,
            3,
            NeoN::mpi::getType<NeoN::scalar>(),
            MPI_SUM,
            env.comm()
        );
        globalAbsVolSum = sums[0];
        globalSignedVolSum = sums[1];
        totalVol = sums[2];
    }
#endif

    return {dt * globalAbsVolSum / totalVol, dt * globalSignedVolSum / totalVol};
}

/**
 * @brief Compute, accumulate, and log time-step continuity errors.
 *
 * @param phi           Face-flux NeoN SurfaceField
 * @param rt            NeoFOAM RunTime (provides exec, mesh, MPI env, dt)
 * @param cumulative    Running cumulative error — caller must initialise to 0
 */
inline void reportContinuityError(
    const NeoN::finiteVolume::cellCentred::SurfaceField<NeoN::scalar>& phi,
    const RunTime& rt,
    NeoN::scalar& cumulative
)
{
    const auto errs = computeContinuityError(phi, rt);
    cumulative += errs.global;

    NeoN::Logging::info(
        "time step continuity errors : sum local = {}, global = {}, cumulative = {}",
        errs.sumLocal,
        errs.global,
        cumulative
    );
}

} // namespace NeoFOAM
