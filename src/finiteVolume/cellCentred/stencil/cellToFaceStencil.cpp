// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/stencil/cellToFaceStencil.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

CellToFaceStencil::CellToFaceStencil(const UnstructuredMesh& mesh) : mesh_(mesh) {}

SegmentedField<localIdx, localIdx> CellToFaceStencil::computeStencil() const
{
    const auto exec = mesh_.exec();
    const localIdx nCells = mesh_.nCells();
    const auto faceOwner = mesh_.faceOwner().span();
    const auto faceNeighbour = mesh_.faceNeighbour().span();
    const auto faceFaceCells = mesh_.boundaryMesh().faceCells().span();
    const size_t nInternalFaces = mesh_.nInternalFaces();

    Field<localIdx> nFacesPerCell(exec, nCells, 0);
    std::span<localIdx> nFacesPerCellSpan = nFacesPerCell.span();

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t i) {
            Kokkos::atomic_increment(&nFacesPerCellSpan[static_cast<size_t>(faceOwner[i])]
            ); // hit on performance on serial
            Kokkos::atomic_increment(&nFacesPerCellSpan[static_cast<size_t>(faceNeighbour[i])]);
        }
    );

    parallelFor(
        exec,
        {0, faceFaceCells.size()},
        KOKKOS_LAMBDA(const size_t i) {
            Kokkos::atomic_increment(&nFacesPerCellSpan[faceFaceCells[i]]);
        }
    );

    SegmentedField<localIdx, localIdx> stencil(nFacesPerCell); // guessed
    auto [stencilValues, segment] = stencil.spans();

    fill(nFacesPerCell, 0); // reset nFacesPerCell

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            size_t owner = static_cast<size_t>(faceOwner[facei]);
            size_t neighbour = static_cast<size_t>(faceNeighbour[facei]);

            // return the oldValues
            size_t segIdxOwn = Kokkos::atomic_fetch_add(
                &nFacesPerCellSpan[owner], 1
            ); // hit on performance on serial
            size_t segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellSpan[neighbour], 1);

            size_t startSegOwn = segment[owner];
            size_t startSegNei = segment[neighbour];
            Kokkos::atomic_assign(&stencilValues[startSegOwn + segIdxOwn], facei);
            Kokkos::atomic_assign(&stencilValues[startSegNei + segIdxNei], facei);
        }
    );

    parallelFor(
        exec,
        {nInternalFaces, nInternalFaces + faceFaceCells.size()},
        KOKKOS_LAMBDA(const size_t facei) {
            size_t owner = static_cast<size_t>(faceFaceCells[facei - nInternalFaces]);
            // return the oldValues
            size_t segIdxOwn = Kokkos::atomic_fetch_add(
                &nFacesPerCellSpan[owner], 1
            ); // hit on performance on serial
            size_t startSegOwn = segment[owner];
            Kokkos::atomic_assign(&stencilValues[startSegOwn + segIdxOwn], facei);
        }
    );

    // // print elements per cell
    // auto stencilView = stencil.view();
    // for (size_t i = 0; i < nCells; i++)
    // {
    //     std::cout << "Cell " << i << " has " << nFacesPerCellSpan[i] << " faces" << std::endl;
    //     for (auto& values: stencilView.span(i))
    //     {
    //         std::cout << "Face " << values << std::endl;
    //     }
    // }

    return stencil;
}

} // namespace NeoFOAM::finiteVolume::cellCentred