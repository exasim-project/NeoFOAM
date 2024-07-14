// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/stencil/basicFvccGeometryScheme.hpp"

namespace NeoFOAM
{

BasicFvccGeometryScheme::BasicFvccGeometryScheme(const UnstructuredMesh& uMesh)
    : FvccGeometrySchemeKernel(uMesh), uMesh_(uMesh)
{
    // Constructor implementation here...
}

void BasicFvccGeometryScheme::updateWeights(
    const CPUExecutor& exec, fvcc::SurfaceField<scalar>& weights
)
{
    const auto owner = uMesh_.faceOwner().span();
    const auto neighbour = uMesh_.faceNeighbour().span();

    const auto Cf = uMesh_.faceCentres().span();
    const auto C = uMesh_.cellCentres().span();
    const auto Sf = uMesh_.faceAreas().span();

    auto w = weights.internalField().span();

    for (label facei = 0; facei < uMesh_.nInternalFaces(); facei++)
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
    for (label facei = uMesh_.nInternalFaces(); facei < w.size(); facei++)
    {
        w[facei] = 1.0;
    }
}

void BasicFvccGeometryScheme::updateWeights(
    const OMPExecutor& exec, fvcc::SurfaceField<scalar>& weights
)
{
    using executor = OMPExecutor::exec;
    const auto owner = uMesh_.faceOwner().span();
    const auto neighbour = uMesh_.faceNeighbour().span();

    const auto Cf = uMesh_.faceCentres().span();
    const auto C = uMesh_.cellCentres().span();
    const auto Sf = uMesh_.faceAreas().span();

    auto w = weights.internalField().span();

    Kokkos::parallel_for(
        "BasicFcccGeometryScheme::updateWeights",
        Kokkos::RangePolicy<executor>(0, uMesh_.nInternalFaces()),
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
        Kokkos::RangePolicy<executor>(uMesh_.nInternalFaces(), w.size()),
        KOKKOS_LAMBDA(const int facei) { w[facei] = 1.0; }
    );
}

void BasicFvccGeometryScheme::updateWeights(
    const GPUExecutor& exec, fvcc::SurfaceField<scalar>& weights
)
{
    using executor = GPUExecutor::exec;
    const auto owner = uMesh_.faceOwner().span();
    const auto neighbour = uMesh_.faceNeighbour().span();

    const auto Cf = uMesh_.faceCentres().span();
    const auto C = uMesh_.cellCentres().span();
    const auto Sf = uMesh_.faceAreas().span();

    auto w = weights.internalField().span();

    Kokkos::parallel_for(
        "BasicFcccGeometryScheme::updateWeights",
        Kokkos::RangePolicy<executor>(0, uMesh_.nInternalFaces()),
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
        Kokkos::RangePolicy<executor>(uMesh_.nInternalFaces(), w.size()),
        KOKKOS_LAMBDA(const int facei) { w[facei] = 1.0; }
    );
}

void BasicFvccGeometryScheme::updateDeltaCoeffs(
    const CPUExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs
)
{
    // Implementation here...
}

void BasicFvccGeometryScheme::updateDeltaCoeffs(
    const OMPExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs
)
{
    // Implementation here...
}

void BasicFvccGeometryScheme::updateDeltaCoeffs(
    const GPUExecutor& exec, fvcc::SurfaceField<scalar>& deltaCoeffs
)
{
    // Implementation here...
}

void BasicFvccGeometryScheme::updateNonOrthDeltaCoeffs(
    const CPUExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    // Implementation here...
}

void BasicFvccGeometryScheme::updateNonOrthDeltaCoeffs(
    const OMPExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    // Implementation here...
}

void BasicFvccGeometryScheme::updateNonOrthDeltaCoeffs(
    const GPUExecutor& exec, fvcc::SurfaceField<scalar>& nonOrthDeltaCoeffs
)
{
    // Implementation here...
}

void BasicFvccGeometryScheme::updateNonOrthCorrectionVectors(
    const CPUExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
)
{
    // Implementation here...
}

void BasicFvccGeometryScheme::updateNonOrthCorrectionVectors(
    const OMPExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
)
{
    // Implementation here...
}

void BasicFvccGeometryScheme::updateNonOrthCorrectionVectors(
    const GPUExecutor& exec, fvcc::SurfaceField<Vector>& nonOrthCorrectionVectors
)
{
    // Implementation here...
}

} // namespace NeoFOAM
