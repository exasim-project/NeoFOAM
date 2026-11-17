// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "NeoN/NeoN.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include "NeoFOAM/algorithms/vof/interface.hpp"
#include "foamGeometry.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

using NeoN::localIdx;
using NeoN::scalar;

void interfaceCurvature(const fvcc::SurfaceField<scalar>& nHatf, fvcc::VolumeField<scalar>& K)
{
    const auto& mesh = nHatf.mesh();
    const auto exec = nHatf.exec();
    auto kI = K.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) { kI[c] = 0.0; },
        "vof::kZero"
    );
    integrateFluxInto(mesh, nHatf.internalVector().view(), nHatf.boundaryData().value().view(), kI);
    NeoN::parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) { kI[c] = -kI[c]; },
        "vof::kNegate"
    );
    // extrapolatedCalculated: every boundary face takes its owner-cell value.
    const auto bOwn = mesh.boundaryMesh().faceOwners().view();
    auto kB = K.boundaryData().value().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { kB[bf] = kI[bOwn[bf]]; },
        "vof::kBoundary"
    );
}

} // namespace NeoFOAM
