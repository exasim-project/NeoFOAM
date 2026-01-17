// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoFOAM/NeoFOAM.hpp"
#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace la = NeoN::la;

// Struct to hold pre-computed face flux values (non-symmetric)
template<typename ValueType>
struct FaceFluxData
{
    NeoN::Vector<ValueType> upper;  // Flux from neighbour to owner (in owner's row)
    NeoN::Vector<ValueType> lower;  // Flux from owner to neighbour (in neighbour's row)
};

// Struct to hold cell-to-faces connectivity and pre-computed data (for flux storage variant)
struct CellBasedDataWithFluxStorage
{
    NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> cellFaces;  // faces per cell
    NeoN::Vector<NeoN::localIdx> faceNeighbour;                        // neighbour cell for each face (or -1 if boundary)
    NeoN::Vector<NeoN::scalar> faceSign;                               // sign for face contribution (+1 if owner, -1 if neighbour)
    NeoN::Vector<NeoN::localIdx> matrixColumnIdx;                      // pre-computed column index in matrix
};

// Function to pre-compute cell-based connectivity (for flux storage variant)
CellBasedDataWithFluxStorage computeCellBasedDataWithFluxStorage(
    const NeoN::Executor& exec,
    const la::SparsityPattern& sparsityPattern,
    const NeoN::Vector<NeoN::localIdx>& owner,
    const NeoN::Vector<NeoN::localIdx>& neighbour,
    NeoN::size_t nCells,
    NeoN::size_t nInternalFaces
)
{
    using namespace NeoN;
    
    // Count faces per cell (owner and neighbour contributions)
    Vector<localIdx> facesPerCell(exec, nCells, localIdx(0));
    auto facesPerCellV = facesPerCell.view();
    
    const auto [ownerV, neighbourV] = views(owner, neighbour);
    
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            Kokkos::atomic_inc(&facesPerCellV[ownerV[facei]]);
            Kokkos::atomic_inc(&facesPerCellV[neighbourV[facei]]);
        },
        "countFacesPerCell"
    );
    
    // Create segmented vector for cell-to-faces connectivity
    SegmentedVector<localIdx, localIdx> cellFaces(facesPerCell);
    
    // Reset counters to use as insertion indices
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            facesPerCellV[celli] = 0;
        },
        "resetCounters"
    );
    
    // Populate cell-to-faces connectivity
    auto [cellFacesValues, cellFacesSegments] = cellFaces.views();
    
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto own = ownerV[facei];
            const auto nei = neighbourV[facei];
            
            // Add face to owner cell
            const auto ownPos = cellFacesSegments[own] + Kokkos::atomic_fetch_inc(&facesPerCellV[own]);
            cellFacesValues[ownPos] = facei;
            
            // Add face to neighbour cell
            const auto neiPos = cellFacesSegments[nei] + Kokkos::atomic_fetch_inc(&facesPerCellV[nei]);
            cellFacesValues[neiPos] = facei;
        },
        "populateCellFaces"
    );
    
    // Pre-compute face neighbour, sign, and matrix column indices
    const auto totalFaceConnections = cellFaces.size();
    Vector<localIdx> faceNeighbour(exec, totalFaceConnections);
    Vector<scalar> faceSign(exec, totalFaceConnections);
    Vector<localIdx> matrixColumnIdx(exec, totalFaceConnections);
    
    const auto [diagOffs, ownOffs, neiOffs] = views(
        sparsityPattern.diagOffset(),
        sparsityPattern.ownerOffset(),
        sparsityPattern.neighbourOffset()
    );
    
    const auto [matrixRowOffs] = views(sparsityPattern.rowOffs());
    
    auto [faceNeighbourV, faceSignV, matrixColumnIdxV] = views(
        faceNeighbour,
        faceSign,
        matrixColumnIdx
    );
    
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            const auto faces = cellFacesSegments[celli + 1] - cellFacesSegments[celli];
            const auto startIdx = cellFacesSegments[celli];
            
            for (localIdx i = 0; i < faces; ++i)
            {
                const auto faceIdx = cellFacesValues[startIdx + i];
                const auto own = ownerV[faceIdx];
                const auto nei = neighbourV[faceIdx];
                
                const auto isOwner = (own == celli);
                const auto neiCell = isOwner ? nei : own;
                
                faceNeighbourV[startIdx + i] = neiCell;
                faceSignV[startIdx + i] = isOwner ? 1.0 : -1.0;
                
                // Compute matrix column index
                const auto rowStart = matrixRowOffs[celli];
                if (isOwner)
                {
                    matrixColumnIdxV[startIdx + i] = rowStart + neiOffs[faceIdx];
                }
                else
                {
                    matrixColumnIdxV[startIdx + i] = rowStart + ownOffs[faceIdx];
                }
            }
        },
        "computeFaceData"
    );
    
    return CellBasedDataWithFluxStorage {
        std::move(cellFaces),
        std::move(faceNeighbour),
        std::move(faceSign),
        std::move(matrixColumnIdx)
    };
}

// Function to pre-compute face flux values (upper and lower for non-symmetric matrix)
template<typename ValueType>
FaceFluxData<ValueType> computeFaceFluxData(
    const NeoN::Executor& exec,
    const fvcc::SurfaceField<NeoN::scalar>& faceFlux,
    const NeoN::Vector<NeoN::scalar>& weights,
    const fvcc::SurfaceField<NeoN::scalar>& gamma,
    const NeoN::Vector<NeoN::scalar>& deltaCoeffs,
    const NeoN::Vector<NeoN::scalar>& magFaceArea,
    NeoN::size_t nInternalFaces
)
{
    using namespace NeoN;
    
    FaceFluxData<ValueType> fluxData {
        Vector<ValueType>(exec, nInternalFaces),
        Vector<ValueType>(exec, nInternalFaces)
    };
    
    const auto [faceFluxV, weightsV, gammaV, deltaCoeffsV, magFaceAreaV] = views(
        faceFlux.internalVector(),
        weights,
        gamma.internalVector(),
        deltaCoeffs,
        magFaceArea
    );
    
    auto [upperV, lowerV] = views(fluxData.upper, fluxData.lower);
    
    // Pre-compute upper and lower face fluxes
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto flux = faceFluxV[facei];
            const auto weight = weightsV[facei];
            
            // DIV + LAPLACIAN contributions
            const auto lapFlux = deltaCoeffsV[facei] * gammaV[facei] * magFaceAreaV[facei];
            
            // Upper: contribution from neighbour to owner (in owner's row)
            upperV[facei] = (-weight * flux + lapFlux) * one<ValueType>();
            
            // Lower: contribution from owner to neighbour (in neighbour's row)
            lowerV[facei] = (flux * (1.0 - weight) + lapFlux) * one<ValueType>();
        },
        "computeFaceFluxData"
    );
    
    return fluxData;
}

// Fused implicit kernel with cell-based loop using pre-computed fluxes
template<typename ValueType>
void fusedImplicitAssemblyCellBasedWithFluxStorage(
    la::LinearSystem<ValueType, NeoN::localIdx>& ls,
    const fvcc::VolumeField<ValueType>& field,
    const NeoN::Vector<ValueType>& oldTimeField,
    const fvcc::SurfaceField<NeoN::scalar>& faceFlux,
    const NeoN::Vector<NeoN::scalar>& weights,
    const fvcc::SurfaceField<NeoN::scalar>& gamma,
    const NeoN::Vector<NeoN::scalar>& deltaCoeffs,
    const la::SparsityPattern& sparsityPattern,
    CellBasedDataWithFluxStorage& cellData,
    FaceFluxData<ValueType>& fluxData,
    NeoN::scalar dt
)
{
    using namespace NeoN;
    const auto& mesh = field.mesh();
    const auto exec = field.exec();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nCells = mesh.nCells();

    // Pass 1: Compute face fluxes (upper and lower)
    {
        const auto [faceFluxV, weightsV, gammaV, deltaCoeffsV, magFaceAreaV] = views(
            faceFlux.internalVector(),
            weights,
            gamma.internalVector(),
            deltaCoeffs,
            mesh.magFaceAreas()
        );
        
        auto [upperV, lowerV] = views(fluxData.upper, fluxData.lower);
        
        parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const localIdx facei) {
                const auto flux = faceFluxV[facei];
                const auto weight = weightsV[facei];
                
                // DIV + LAPLACIAN contributions
                const auto lapFlux = deltaCoeffsV[facei] * gammaV[facei] * magFaceAreaV[facei];
                
                // Upper: contribution from neighbour to owner (in owner's row)
                upperV[facei] = (-weight * flux + lapFlux) * one<ValueType>();
                
                // Lower: contribution from owner to neighbour (in neighbour's row)
                lowerV[facei] = (flux * (1.0 - weight) + lapFlux) * one<ValueType>();
            },
            "fusedKernelCellBasedWithFluxStorage::computeFluxes"
        );
    }

    // Pass 2: Cell-based assembly using pre-computed fluxes
    // Get views for all data
    const auto [vol, magFaceArea, owner, neighbour, surfFaceCells] = views(
        mesh.cellVolumes(),
        mesh.magFaceAreas(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        mesh.boundaryMesh().faceCells()
    );

    const auto [diagOffs] = views(sparsityPattern.diagOffset());

    const auto [oldVector] = views(oldTimeField);
    
    const auto [cellFacesValues, cellFacesSegments] = cellData.cellFaces.views();
    const auto [faceNeighbourV, faceSignV, matrixColumnIdxV] = views(
        cellData.faceNeighbour,
        cellData.faceSign,
        cellData.matrixColumnIdx
    );
    
    const auto [upperFluxV, lowerFluxV] = views(
        fluxData.upper,
        fluxData.lower
    );

    auto [matrix, rhs] = ls.view();

    // Single cell-based loop: ddt + div + laplacian assembly
    const scalar a0 = 1.0 / dt;
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            // DDT contribution to diagonal
            const auto diagIdx = matrix.rowOffs[celli] + diagOffs[celli];
            const auto coeff = a0 * vol[celli];
            auto diagValue = coeff * one<ValueType>();
            auto rhsValue = coeff * oldVector[celli];
            
            // Loop over faces of this cell
            const auto numFaces = cellFacesSegments[celli + 1] - cellFacesSegments[celli];
            const auto startIdx = cellFacesSegments[celli];
            
            for (localIdx i = 0; i < numFaces; ++i)
            {
                const auto faceIdx = cellFacesValues[startIdx + i];
                const auto neiCell = faceNeighbourV[startIdx + i];
                const auto sign = faceSignV[startIdx + i];
                
                // Use pre-computed flux (upper if owner, lower if neighbour)
                const auto combinedFlux = (sign > 0.0) ? upperFluxV[faceIdx] : lowerFluxV[faceIdx];
                
                const auto offDiagValue = combinedFlux;
                matrix.values[matrixColumnIdxV[startIdx + i]] += offDiagValue;
                
                // Contribution to diagonal (subtract off-diagonal)
                diagValue -= offDiagValue;
            }
            
            // Write diagonal and RHS
            matrix.values[diagIdx] += diagValue;
            rhs[celli] += rhsValue;
        },
        "fusedKernelCellBasedWithFluxStorage::cellLoop"
    );

    // Pass 3: Boundary contributions
    const auto [faceFluxV, weightsV, gammaV, deltaCoeffsV] = views(
        faceFlux.internalVector(),
        weights,
        gamma.internalVector(),
        deltaCoeffs
    );
    
    const auto [refGradient, value, valueFraction, refValue, bDeltaCoeffs] = views(
        field.boundaryData().refGrad(),
        field.boundaryData().value(),
        field.boundaryData().valueFraction(),
        field.boundaryData().refValue(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    auto& bcCoeffs = ls.auxiliaryCoefficients().template get<la::BoundaryCoefficients<ValueType, localIdx>>("boundaryCoefficients");
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

            const auto combinedValueMat = (divFlux * valFrac2 - lapFlux * valFrac1 * deltaCoeffsV[facei]) * one<ValueType>();
            const auto combinedValueRhs = (divFlux * valFrac1 * refValue[bcfacei] + valFrac2 * refGradient[bcfacei] * (1.0 / bDeltaCoeffs[bcfacei]))
                                         + lapFlux * (valFrac1 * deltaCoeffsV[facei] * refValue[bcfacei] + valFrac2 * refGradient[bcfacei]);

            // Single atomic operations
            Kokkos::atomic_add(&matrix.values[rowOwnStart + diagOffs[own]], combinedValueMat);
            Kokkos::atomic_sub(&rhs[own], combinedValueRhs);

            boundValues[bcfacei] = combinedValueMat;
            rhsBoundValues[bcfacei] = combinedValueRhs;
        },
        "fusedKernelCellBasedWithFluxStorage::boundaryPass"
    );
}
