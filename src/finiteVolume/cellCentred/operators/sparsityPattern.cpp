// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/finiteVolume/cellCentred/operators/sparsityPattern.hpp"
#include "NeoFOAM/fields/segmentedField.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

SparsityPattern::SparsityPattern(const UnstructuredMesh& mesh)
    : mesh_(mesh), ls_(mesh_.exec()), ownerOffset_(mesh_.exec(), mesh_.nInternalFaces(), 0),
      neighbourOffset_(mesh_.exec(), mesh_.nInternalFaces(), 0),
      diagOffset_(mesh_.exec(), mesh_.nCells(), 0)
{
    update();
}

const std::shared_ptr<SparsityPattern> SparsityPattern::readOrCreate(const UnstructuredMesh& mesh)
{
    StencilDataBase& stencilDb = mesh.stencilDB();
    if (!stencilDb.contains("SparsityPattern"))
    {
        stencilDb.insert(std::string("SparsityPattern"), std::make_shared<SparsityPattern>(mesh));
    }
    return stencilDb.get<std::shared_ptr<SparsityPattern>>("SparsityPattern");
}


void SparsityPattern::update()
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
    std::span<uint8_t> neighbourOffsetSpan = neighbourOffset_.span();
    std::span<uint8_t> ownerOffsetSpan = ownerOffset_.span();
    std::span<uint8_t> diagOffsetSpan = diagOffset_.span();

    // only the internalfaces define the sparsity pattern
    // get the number of faces per cell to allocate the correct size
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            // hit on performance on serial
            size_t owner = static_cast<size_t>(faceOwner[facei]);
            size_t neighbour = static_cast<size_t>(faceNeighbour[facei]);

            Kokkos::atomic_increment(&nFacesPerCellSpan[owner]);
            Kokkos::atomic_increment(&nFacesPerCellSpan[neighbour]);
        }
    );


    Field<localIdx> rowPtrs(exec, nCells + 1, 0);
    auto nEntries = NeoFOAM::segmentsFromIntervals(nFacesPerCell, rowPtrs);
    NeoFOAM::Field<NeoFOAM::scalar> values(exec, nEntries, 0.0);
    NeoFOAM::Field<NeoFOAM::localIdx> colIdx(exec, nEntries, 0);
    std::span<localIdx> sColIdx = colIdx.span();
    fill(nFacesPerCell, 0); // reset nFacesPerCell

    // compute the lower triangular part of the matrix
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            size_t neighbour = static_cast<size_t>(faceNeighbour[facei]);
            size_t owner = static_cast<size_t>(faceOwner[facei]);

            // return the oldValues
            // hit on performance on serial
            size_t segIdxNei = Kokkos::atomic_fetch_add(&nFacesPerCellSpan[neighbour], 1);
            neighbourOffsetSpan[facei] = segIdxNei;

            size_t startSegNei = rowPtrs[neighbour];
            // neighbour --> current cell
            // colIdx --> needs to be store the owner
            Kokkos::atomic_assign(&sColIdx[startSegNei + segIdxNei], owner);
        }
    );

    map(
        nFacesPerCell,
        KOKKOS_LAMBDA(const size_t celli) {
            size_t nFaces = nFacesPerCellSpan[static_cast<size_t>(celli)];
            diagOffsetSpan[celli] = nFaces;
            sColIdx[rowPtrs[celli] + nFaces] = celli;
            return nFaces + 1;
        }
    );

    // compute the upper triangular part of the matrix
    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t facei) {
            size_t neighbour = static_cast<size_t>(faceNeighbour[facei]);
            size_t owner = static_cast<size_t>(faceOwner[facei]);

            // return the oldValues
            // hit on performance on serial
            size_t segIdxOwn = Kokkos::atomic_fetch_add(&nFacesPerCellSpan[owner], 1);
            ownerOffsetSpan[facei] = segIdxOwn;

            size_t startSegOwn = rowPtrs[owner];
            // owner --> current cell
            // colIdx --> needs to be store the neighbour
            Kokkos::atomic_assign(&sColIdx[startSegOwn + segIdxOwn], neighbour);
        }
    );

    NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, NeoFOAM::localIdx> csrMatrix(values, colIdx, rowPtrs);

    NeoFOAM::Field<NeoFOAM::scalar> rhs(exec, nCells, 0.0);
    ls_ = NeoFOAM::la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx>(csrMatrix, rhs);
}

const NeoFOAM::la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx>&
SparsityPattern::linearSystem() const
{
    return ls_;
}

const NeoFOAM::Field<uint8_t>& SparsityPattern::ownerOffset() const { return ownerOffset_; }

const NeoFOAM::Field<uint8_t>& SparsityPattern::neighbourOffset() const { return neighbourOffset_; }

const NeoFOAM::Field<uint8_t>& SparsityPattern::diagOffset() const { return diagOffset_; }

} // namespace NeoFOAM::finiteVolume::cellCentred
