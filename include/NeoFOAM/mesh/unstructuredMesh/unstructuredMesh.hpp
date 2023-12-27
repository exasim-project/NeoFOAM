// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <array>
#include <vector>

#include "NeoFOAM/blas/adjacency.hpp"
#include "NeoFOAM/blas/fields.hpp"

#include "../cellDefinitions.hpp"
#include "../cellUtities.hpp"

namespace NeoFOAM
{

class Stencil<int8_t>; // 0 is dynamic

struct unstructuredMesh
{

    unstructuredMesh(NeoFOAM::vectorField Sf, NeoFOAM::labelField owner, NeoFOAM::labelField neighbour, NeoFOAM::scalarField V, int32_t nCells, int32_t nInternalFaces)
        : Sf_(Sf)
        , owner_(owner)
        , neighbour_(neighbour)
        , V_(V)
        , nCells_(nCells)
        , nInternalFaces_(nInternalFaces){

        };

    vectorField Sf_; // area vector

    labelField owner_;     // owner cell
    labelField neighbour_; // neighbour cell

    scalarField V_; // cell volume

    int32_t nCells_; // number of cells
    int32_t nInternalFaces_; // number of internal faces

    kokkos::View<NeoFOAM::cellType*> cellTypes_; // cell types

    localAdjacency<false> face_cell_;

    localAdjacency<false> cell_vertex_;
    localAdjacency<false> cellFace_boundary_;
    
    localAdjacency<true> cellCell;
    localAdjacency<false> cellFace; 
};

void createUnstructuredConnectivity(const localAdjacency<false>& cell_vertex, const kokkos::View<NeoFOAM::cellType*>& cell_type) {
    
    // first invert the cell_vertex_ adjacency - reduces n^2 cell-cell connectivity construction to local (nlogn or something like it).
    localAdjacency<false> vertex_cell = createVertexCellAdjacency(cell_vertex);

    // create cell cell connectivity
    createFaceCellAdjacency(vertex_cell, const kokkos::View<NeoFOAM::cellType*>& cell_type);
    
}

localAdjacency<false> createVertexCellAdjacency(const localAdjacency<false>& cell_vertex){
    return cell_vertex.transpose(); // need to make this still

}

// very inefficient algorithm, but it works. - we will search for cell - cell connection multiple times, even after we know there is no connection.
localAdjacency<true> createFaceCellAdjacency(const localAdjacency<false>& vertex_cell, const localAdjacency<false>& cell_vertex, 
                                             const kokkos::View<NeoFOAM::cellType*>& celltype) {
    localAdjacency<true> cellCell;
    cell_cell.resize(cell_type.size());

    for(auto i_vertex = 0; i_vertex < vertex_cell.size(); ++i_vertex){
        for(auto i_cell = 1; i_cell < vertex_cell[i_vertex].size(); ++i_cell){
            
            const auto& cell0 = vertex_cell(i_vertex)(i_cell - 1);
            const auto& cell1 = vertex_cell(i_vertex)(i_cell);
            if(cellCell.contains({cell0, cell1})) continue; // we already know this cell-cell connection exists.
            CellGlobalFace cell0(cell_vertex(cell0), celltype(cell0));
            CellGlobalFace cell1(cell_vertex(cell1), celltype(cell1));
            if(CellShareFace(cell0, cell1)) cellCell.insert(cell0, cell1);
        }
    }


    return cellCell;
}

} // namespace NeoFOAM