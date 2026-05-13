// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 - 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"

namespace NeoFOAM
{

/**
 * @brief Compute and log time-step continuity errors from the face flux field.
 *
 * Computes sum-local, global, and cumulative continuity errors matching the
 * OpenFOAM continuityErrs.H formulation:
 *   sumLocal  = dt * volWeightedAverage( |div(phi)| )   (abs, MPI-reduced)
 *   global    = dt * volWeightedAverage(  div(phi)  )   (signed, MPI-reduced)
 *   cumulative accumulates global across all PISO iterations
 *
 * @param phi           Face-flux NeoN SurfaceField
 * @param rt            NeoFOAM RunTime (provides exec, mesh, MPI env, dt)
 * @param cumulative    Running cumulative error — caller must initialise to 0
 */
inline std::tuple<NeoN::scalar, NeoN::scalar, NeoN::scalar> continuityError(
    const NeoN::finiteVolume::cellCentred::SurfaceField<NeoN::scalar>& phi,
    const RunTime& rt,
    NeoN::scalar& cumulative
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

    if (rt.mpiEnvironment.isInitialized() && rt.mpiEnvironment.sizeRank() > 1)
    {
        NeoN::mpi::allReduce(
            globalAbsVolSum, NeoN::mpi::ReduceOp::Sum, rt.mpiEnvironment.comm()
        );
        NeoN::mpi::allReduce(
            globalSignedVolSum, NeoN::mpi::ReduceOp::Sum, rt.mpiEnvironment.comm()
        );
        NeoN::mpi::allReduce(totalVol, NeoN::mpi::ReduceOp::Sum, rt.mpiEnvironment.comm());
    }

    const NeoN::scalar sumLocalContErr = dt * globalAbsVolSum / totalVol;
    const NeoN::scalar globalContErr = dt * globalSignedVolSum / totalVol;
    cumulative += globalContErr;

    return  {sumLocalContErr, globalContErr, cumulative};
}

} // namespace NeoFOAM
