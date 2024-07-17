// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/stencil/basicFvccGeometryScheme.hpp"

namespace NeoFOAM
{

BasicGeometryScheme::BasicGeometryScheme(const UnstructuredMesh& mesh)
    : GeometrySchemeFactory(mesh), mesh_(mesh)
{
    // Constructor implementation here...
}

void BasicGeometryScheme::updateWeights(
    const CPUExecutor& exec, fvcc::SurfaceField<scalar>& weights
)
{
    const auto owner = mesh_.faceOwner().span();
    const auto neighbour = mesh_.faceNeighbour().span();

    const auto Cf = mesh_.faceCentres().span();
    const auto C = mesh_.cellCentres().span();
    const auto Sf = mesh_.faceAreas().span();

    auto w = weights.internalField().span();

    for (label facei = 0; facei < mesh_.nInternalFaces(); facei++)
    {
        // Note: mag in the dot-product.
        // For all valid meshes, the non-orthogonality will be less than
        // 90 deg and the dot-product will be positive.  For invalid
        // meshes (d & s <= 0), this will stabilise the calculation
        // but the result will be poor.
        scalar SfdOwn = mag(Sf[facei] & (Cf[facei] - C[owner[facei]]));
        scalar SfdNei = mag(Sf[facei] & (C[neighbour[facei]] - Cf[facei]));

        if (std::abs(SfdOwn + SfdNei) > ROOTVSMALL)
        {
            w[facei] = SfdNei / (SfdOwn + SfdNei);
        }
        else
        {
            w[facei] = 0.5;
        }
    }

    // TODO: other boundary condition requires other weights which is not implemented yet
    //  and requires the implementation of the mesh functionality
    for (label facei = mesh_.nInternalFaces(); facei < w.size(); facei++)
    {
        w[facei] = 1.0;
    }
}

void BasicGeometryScheme::updateWeights(
    const OMPExecutor& exec, fvcc::SurfaceField<scalar>& weights
)
{
    using executor = OMPExecutor::exec;
    const auto owner = mesh_.faceOwner().span();
    const auto neighbour = mesh_.faceNeighbour().span();

    const auto Cf = mesh_.faceCentres().span();
    const auto C = mesh_.cellCentres().span();
    const auto Sf = mesh_.faceAreas().span();

    auto w = weights.internalField().span();

    Kokkos::parallel_for(
        "BasicFcccGeometryScheme::updateWeights",
        Kokkos::RangePolicy<executor>(0, mesh_.nInternalFaces()),
        KOKKOS_LAMBDA(const int facei) {
            // Note: mag in the dot-product.
            // For all valid meshes, the non-orthogonality will be less than
            // 90 deg and the dot-product will be positive.  For invalid
            // meshes (d & s <= 0), this will stabilise the calculation
            // but the result will be poor.
            scalar SfdOwn = mag(Sf[facei] & (Cf[facei] - C[owner[facei]]));
            scalar SfdNei = mag(Sf[facei] & (C[neighbour[facei]] - Cf[facei]));

            if (std::abs(SfdOwn + SfdNei) > ROOTVSMALL)
            {
                w[facei] = SfdNei / (SfdOwn + SfdNei);
            }
            else
            {
                w[facei] = 0.5;
            }
        }
    );

    // TODO: other boundary condition requires other weights which is not implemented yet
    //  and requires the implementation of the mesh functionality
    Kokkos::parallel_for(
        "BasicFcccGeometryScheme::updateWeightsBC",
        Kokkos::RangePolicy<executor>(mesh_.nInternalFaces(), w.size()),
        KOKKOS_LAMBDA(const int facei) { w[facei] = 1.0; }
    );
}

void BasicGeometryScheme::updateWeights(
    const GPUExecutor& exec, fvcc::SurfaceField<scalar>& weights
)
{
    using executor = GPUExecutor::exec;
    const auto owner = mesh_.faceOwner().span();
    const auto neighbour = mesh_.faceNeighbour().span();

    const auto Cf = mesh_.faceCentres().span();
    const auto C = mesh_.cellCentres().span();
    const auto Sf = mesh_.faceAreas().span();

    auto w = weights.internalField().span();

    Kokkos::parallel_for(
        "BasicFcccGeometryScheme::updateWeights",
        Kokkos::RangePolicy<executor>(0, mesh_.nInternalFaces()),
        KOKKOS_LAMBDA(const int facei) {
            // Note: mag in the dot-product.
            // For all valid meshes, the non-orthogonality will be less than
            // 90 deg and the dot-product will be positive.  For invalid
            // meshes (d & s <= 0), this will stabilise the calculation
            // but the result will be poor.
            scalar SfdOwn = mag(Sf[facei] & (Cf[facei] - C[owner[facei]]));
            scalar SfdNei = mag(Sf[facei] & (C[neighbour[facei]] - Cf[facei]));

            if (std::abs(SfdOwn + SfdNei) > ROOTVSMALL)
            {
                w[facei] = SfdNei / (SfdOwn + SfdNei);
            }
            else
            {
                w[facei] = 0.5;
            }
        }
    );

    // TODO: other boundary condition requires other weights which is not implemented yet
    //  and requires the implementation of the mesh functionality
    Kokkos::parallel_for(
        "BasicFcccGeometryScheme::updateWeightsBC",
        Kokkos::RangePolicy<executor>(mesh_.nInternalFaces(), w.size()),
        KOKKOS_LAMBDA(const int facei) { w[facei] = 1.0; }
    );
}

void BasicGeometryScheme::updateDeltaCoeffs(
    const CPUExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs
)
{
    // Implementation here...
}

void BasicGeometryScheme::updateDeltaCoeffs(
    const OMPExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs
)
{
    // Implementation here...
}

void BasicGeometryScheme::updateDeltaCoeffs(
    const GPUExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs
)
{
    // Implementation here...
}

void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    const CPUExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    // Implementation here...
}

void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    const OMPExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    // Implementation here...
}

void BasicGeometryScheme::updateNonOrthDeltaCoeffs(
    const GPUExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    // Implementation here...
}

void BasicGeometryScheme::updateNonOrthCorrectionVectors(
    const CPUExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
)
{
    // Implementation here...
}

void BasicGeometryScheme::updateNonOrthCorrectionVectors(
    const OMPExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
)
{
    // Implementation here...
}

void BasicGeometryScheme::updateNonOrthCorrectionVectors(
    const GPUExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
)
{
    // Implementation here...
}

} // namespace NeoFOAM
