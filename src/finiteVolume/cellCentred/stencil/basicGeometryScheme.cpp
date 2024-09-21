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
    const auto [owner, neighbour] = spans(mesh_.faceOwner(), mesh_.faceNeighbour());

    const auto [cf, c, sf] = spans(mesh_.faceCentres(), mesh_.cellCentres(), mesh_.faceAreas());

    auto w = weights.internalField().span();

    parallelFor(
        exec,
        {0, mesh_.nInternalFaces()},
        KOKKOS_LAMBDA(const size_t facei) {
            // stabelizes the scheme for poor meshes --> mag
            scalar sfdOwn = mag(sf[facei] & (cf[facei] - c[owner[facei]]));
            scalar sfdNei = mag(sf[facei] & (c[neighbour[facei]] - cf[facei]));

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
        exec, {mesh_.nInternalFaces(), w.size()}, KOKKOS_LAMBDA(const int facei) { w[facei] = 1.0; }
    );
}

void BasicGeometryScheme::updateDeltaCoeffs(const Executor& exec, SurfaceField<scalar>& deltaCoeffs)
{
    NF_ERROR_EXIT("Not implemented");
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    const Executor& exec, SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    NF_ERROR_EXIT("Not implemented");
}


void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    const Executor& exec, SurfaceField<Vector>& nonOrthDeltaCoeffs
)
{
    NF_ERROR_EXIT("Not implemented");
}

} // namespace NeoFOAM
