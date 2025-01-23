// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/operators/fvccSparsityPattern.hpp"
#include "NeoFOAM/fields/segmentedField.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

SparsityPattern::SparsityPattern(const UnstructuredMesh& mesh) : mesh_(mesh) {}

la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> SparsityPattern::compute() const
{
    const auto exec = mesh_.exec();
    const localIdx nCells = mesh_.nCells();
    const auto faceOwner = mesh_.faceOwner().span();
    const auto faceNeighbour = mesh_.faceNeighbour().span();
    const auto faceFaceCells = mesh_.boundaryMesh().faceCells().span();
    const size_t nInternalFaces = mesh_.nInternalFaces();

    // start with one to include the diagonal
    Field<localIdx> nFacesPerCell(exec, nCells, 1);
    std::span<localIdx> nFacesPerCellSpan = nFacesPerCell.span();

    // only the internalfaces define the sparsity pattern
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t i) {
            Kokkos::atomic_increment(&nFacesPerCellSpan[static_cast<size_t>(faceOwner[i])]
            ); // hit on performance on serial
            Kokkos::atomic_increment(&nFacesPerCellSpan[static_cast<size_t>(faceNeighbour[i])]);
        }
    );

    Field<localIdx> rowPtrs(exec, nCells + 1, 0);
    auto nEntries = NeoFOAM::segmentsFromIntervals(nFacesPerCell, rowPtrs);
    NeoFOAM::Field<NeoFOAM::scalar> values(exec, nEntries, 0.0);
    NeoFOAM::Field<NeoFOAM::localIdx> colIdx(exec, nEntries, 0);
    std::span<localIdx> sColIdx = colIdx.span();

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

            size_t startSegOwn = rowPtrs[owner];
            size_t startSegNei = rowPtrs[neighbour];

            Kokkos::atomic_assign(&sColIdx[startSegOwn + segIdxOwn], owner);
            Kokkos::atomic_assign(&sColIdx[startSegNei + segIdxNei], neighbour);
        }
    );

    SegmentedField<localIdx, localIdx> sPattern(colIdx, rowPtrs); // guessed
    auto sPatternView = sPattern.view();

    // parallelFor(
    //     exec,
    //     {0, sPattern.numSegments()},
    //     KOKKOS_LAMBDA(const size_t segI) {
    //         auto vals = sPatternView.span(segI);
    //         // std::sort(vals.begin(), vals.end());
    //     }
    // );

    // NeoFOAM::Field<NeoFOAM::scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    // NeoFOAM::Field<NeoFOAM::localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
    // NeoFOAM::Field<NeoFOAM::localIdx> rowPtrs(exec, {0, 3, 6, 9});
    NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> csrMatrix(values, colIdx, rowPtrs);

    NeoFOAM::Field<NeoFOAM::scalar> rhs(exec, nCells, 0.0);
    NeoFOAM::la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> linearSystem(csrMatrix, rhs);
    // parallelFor(
    //     exec,
    //     {0, faceFaceCells.size()},
    //     KOKKOS_LAMBDA(const size_t i) {
    //         Kokkos::atomic_increment(&nFacesPerCellSpan[faceFaceCells[i]]);
    //     }
    // );

    // SegmentedField<localIdx, localIdx> stencil(nFacesPerCell); // guessed
    // auto [stencilValues, segment] = stencil.spans();

    // fill(nFacesPerCell, 0); // reset nFacesPerCell

    // parallelFor(
    //     exec,
    //     {0, nInternalFaces},
    //     KOKKOS_LAMBDA(const size_t facei) {
    //         size_t owner = static_cast<size_t>(faceOwner[facei]);
    //         size_t neighbour = static_cast<size_t>(faceNeighbour[facei]);

    //         // return the oldValues
    //         size_t segIdxOwn = Kokkos::atomic_fetch_add(&nFacesPerCellSpan[owner], 1); // hit on
    //         performance on serial size_t segIdxNei =
    //         Kokkos::atomic_fetch_add(&nFacesPerCellSpan[neighbour], 1);

    //         size_t startSegOwn = segment[owner];
    //         size_t startSegNei = segment[neighbour];
    //         Kokkos::atomic_assign(&stencilValues[startSegOwn + segIdxOwn], facei);
    //         Kokkos::atomic_assign(&stencilValues[startSegNei + segIdxNei], facei);
    //     }
    // );

    // parallelFor(
    //     exec,
    //     {nInternalFaces, nInternalFaces + faceFaceCells.size()},
    //     KOKKOS_LAMBDA(const size_t facei) {
    //         size_t owner = static_cast<size_t>(faceFaceCells[facei - nInternalFaces]);
    //         // return the oldValues
    //         size_t segIdxOwn = Kokkos::atomic_fetch_add(&nFacesPerCellSpan[owner], 1); // hit on
    //         performance on serial size_t startSegOwn = segment[owner];
    //         Kokkos::atomic_assign(&stencilValues[startSegOwn + segIdxOwn], facei);
    //     }
    // );

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

    return linearSystem;
}

} // namespace NeoFOAM::finiteVolume::cellCentred
