// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoFOAM/NeoFOAM.hpp"
#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace la = NeoN::la;

// Fused implicit kernel: ddt + div + laplacian in single assembly pass
template<typename ValueType>
void fusedImplicitAssembly(
    la::LinearSystem<ValueType, NeoN::localIdx>& ls,
    const fvcc::VolumeField<ValueType>& field,
    const NeoN::Vector<ValueType>& oldTimeField,
    const fvcc::SurfaceField<NeoN::scalar>& faceFlux,
    const NeoN::Vector<NeoN::scalar>& weights,
    const fvcc::SurfaceField<NeoN::scalar>& gamma,
    const NeoN::Vector<NeoN::scalar>& deltaCoeffs,
    const la::SparsityPattern& sparsityPattern,
    NeoN::scalar dt
)
{
    using namespace NeoN;
    const auto& mesh = field.mesh();
    const auto exec = field.exec();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nCells = mesh.nCells();

    // Get views for all data
    const auto [vol, magFaceArea, owner, neighbour, surfFaceCells] = views(
        mesh.cellVolumes(),
        mesh.magFaceAreas(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.boundaryMesh().faceCells()
    );

    const auto [diagOffs, ownOffs, neiOffs] = views(
        sparsityPattern.diagOffset(),
        sparsityPattern.ownerOffset(),
        sparsityPattern.neighbourOffset()
    );

    const auto [faceFluxV, weightsV, gammaV, deltaCoeffsV, oldVector] = views(
        faceFlux.internalVector(),
        weights,
        gamma.internalVector(),
        deltaCoeffs,
        oldTimeField
    );

    auto [matrix, rhs] = ls.view();

    // Pass 1: Cell-based ddt assembly (BDF1)
    const scalar a0 = 1.0 / dt;
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            const auto idx = matrix.rowOffs[celli] + diagOffs[celli];
            const auto coeff = a0 * vol[celli];
            matrix.values[idx] += coeff * one<ValueType>();
            rhs[celli] += coeff * oldVector[celli];
        },
        "fusedKernel::ddtPass"
    );

    // Pass 2: Face-based div + laplacian assembly
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto flux = faceFluxV[facei];
            const auto weight = weightsV[facei];
            const auto own = owner[facei];
            const auto nei = neighbour[facei];

            const auto rowOwnStart = matrix.rowOffs[own];
            const auto rowNeiStart = matrix.rowOffs[nei];

            // DIV + LAPLACIAN contributions (combined)
            const auto lapFlux = deltaCoeffsV[facei] * gammaV[facei] * magFaceArea[facei];

            // Combined contributions for neighbour column in owner row
            const auto combinedValueNei = (-weight * flux + lapFlux) * one<ValueType>();
            matrix.values[rowNeiStart + neiOffs[facei]] += combinedValueNei;
            Kokkos::atomic_sub(&matrix.values[rowOwnStart + diagOffs[own]], combinedValueNei);
            // matrix.values[rowOwnStart + diagOffs[own]] -= combinedValueNei;


            // Combined contributions for owner column in neighbour row
            const auto combinedValueOwn = (flux * (1.0 - weight) + lapFlux) * one<ValueType>();
            matrix.values[rowOwnStart + ownOffs[facei]] += combinedValueOwn;
            Kokkos::atomic_sub(&matrix.values[rowNeiStart + diagOffs[nei]], combinedValueOwn);
            // matrix.values[rowNeiStart + diagOffs[nei]] -= combinedValueOwn;
        },
        "fusedKernel::divLaplacianPass"
    );

    // Pass 3: Boundary contributions
    const auto [refGradient, value, valueFraction, refValue, bDeltaCoeffs] = views(
        field.boundaryData().refGrad(),
        field.boundaryData().value(),
        field.boundaryData().valueFraction(),
        field.boundaryData().refValue(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    auto& bcCoeffs =
        ls.auxiliaryCoefficients().template get<la::BoundaryCoefficients<ValueType, localIdx>>(
            "boundaryCoefficients"
        );
    auto [boundValues, rhsBoundValues] = views(bcCoeffs.matrixValues, bcCoeffs.rhsValues);

    parallelFor(
        exec,
        {nInternalFaces, faceFluxV.size()},
        NEON_LAMBDA(const localIdx facei) {
            const auto bcfacei = facei - nInternalFaces;
            const auto own = surfFaceCells[bcfacei];
            const auto rowOwnStart = matrix.rowOffs[own];

            const auto valFrac1 = valueFraction[bcfacei];
            const auto valFrac2 = 1.0 - valFrac1;

            // Combined DIV + LAPLACIAN boundary contribution
            const auto divFlux = weightsV[facei] * faceFluxV[facei];
            const auto lapFlux = gammaV[facei] * magFaceArea[facei];

            const auto combinedValueMat =
                (divFlux * valFrac2 - lapFlux * valFrac1 * deltaCoeffsV[facei]) * one<ValueType>();
            const auto combinedValueRhs =
                (divFlux * valFrac1 * refValue[bcfacei]
                 + valFrac2 * refGradient[bcfacei] * (1.0 / bDeltaCoeffs[bcfacei]))
                + lapFlux
                      * (valFrac1 * deltaCoeffsV[facei] * refValue[bcfacei]
                         + valFrac2 * refGradient[bcfacei]);

            // Single atomic operations
            Kokkos::atomic_add(&matrix.values[rowOwnStart + diagOffs[own]], combinedValueMat);
            Kokkos::atomic_sub(&rhs[own], combinedValueRhs);

            boundValues[bcfacei] = combinedValueMat;
            rhsBoundValues[bcfacei] = combinedValueRhs;
        },
        "fusedKernel::boundaryPass"
    );
}
