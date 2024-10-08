// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/stencil/basicGeometryScheme.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

BasicGeometryScheme::BasicGeometryScheme(const UnstructuredMesh& mesh)
    : GeometrySchemeFactory(mesh), mesh_(mesh)
{}

void BasicGeometryScheme::updateWeights(const Executor& exec, SurfaceField<scalar>& weights)
{
    const auto owner = mesh_.faceOwner().span();
    const auto neighbour = mesh_.faceNeighbour().span();

    const auto cf = mesh_.faceCentres().span();
    const auto c = mesh_.cellCentres().span();
    const auto sf = mesh_.faceAreas().span();

    auto w = weights.internalField().span();

    parallelFor(
        exec,
        {0, mesh_.nInternalFaces()},
        KOKKOS_LAMBDA(const size_t facei) {
            // Note: mag in the dot-product.
            // For all valid meshes, the non-orthogonality will be less than
            // 90 deg and the dot-product will be positive.  For invalid
            // meshes (d & s <= 0), this will stabilise the calculation
            // but the result will be poor.
            scalar sfdOwn = mag(sf[facei] & (cf[facei] - c[static_cast<size_t>(owner[facei])]));
            scalar sfdNei = mag(sf[facei] & (c[static_cast<size_t>(neighbour[facei])] - cf[facei]));

            if (std::abs(sfdOwn + sfdNei) > ROOTVSMALL)
            {
                w[facei] = sfdNei / (sfdOwn + sfdNei);
            }
            else
            {
                w[facei] = 0.5;
            }
        }
    );

    parallelFor(
        exec,
        {mesh_.nInternalFaces(), w.size()},
        KOKKOS_LAMBDA(const size_t facei) { w[facei] = 1.0; }
    );
}

void BasicGeometryScheme::updateDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<scalar>& deltaCoeffs
)
{
    NF_ERROR_EXIT("Not implemented");
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    NF_ERROR_EXIT("Not implemented");
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    [[maybe_unused]] const Executor& exec, [[maybe_unused]] SurfaceField<Vector>& nonOrthDeltaCoeffs
)
{
    NF_ERROR_EXIT("Not implemented");
}

} // namespace NeoFOAM
