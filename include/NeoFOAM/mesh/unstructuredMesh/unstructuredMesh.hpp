// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <array>
#include <vector>

#include "NeoFOAM/blas/fields.hpp"



namespace NeoFOAM
{

class Stencil<int8_t>; // 0 is dynamic

struct unstructuredMesh
{
    vectorField Sf_;  // area vector

    labelField owner_;  // owner cell
    labelField neighbour_;  // neighbour cell

    scalarField V_;  // cell volume

    int32_t nCells_;  // number of cells
    
    int32_t nInternalFaces_;  // number of internal faces






    // connectivity information - we need a 'sparse vector' to make this memory efficient.
    Kokkos::vector<Kokkos::vector<index> > cell_cell;  
    Kokkos::vector<Kokkos::vector<index> > cell_face; 
    Kokkos::vector<Kokkos::vector<index> > cell_vertex;
    Kokkos::vector<std::array<index, 2> > face_cell;
    Kokkos::vector<Kokkos::vector<index> > face_vertex;
    Kokkos::vector<Kokkos::vector<index> > vertex_face; // might be helpful for PLIC reconstruction - but otherwise is helper.
    Kokkos::vector<Kokkos::vector<index> > vertex_cell; // might be helpful for PLIC reconstruction - but otherwise is helper.
    Kokkos::vector<Kokkos::vector<index> > cell_boundary_zone;
};



// --------------------------
// move below to separate file.
// --------------------------

enum class cellType{
    triangle,
    quad,
    tetra,
    hexa
}

/**
 * @brief 
 * 
 * @param[in] cell_vertex 
 * @param[out] vertex_cell 
 */
void build_vertex_cell(const Kokkos::vector<Kokkos::vector<index> >& cell_vertex,
                       Kokkos::vector<Kokkos::vector<index> >& vertex_cell, index size_vertex) {
    vertex_cell.resize(size_vertex);
    Kokkos::parallel_for(
        cell_vertex.size(), KOKKOS_CLASS_LAMBDA(const index i_cell) {
            vertex_cell[cell_vertex[vertex]].push_back(i_cell);
        });
}



// The following data forms the primitive mesh -> minimal data to be read in (excl. boundary zone information.)
// const std::vector<NeoFOAM::vector>& points, 
// const std::vector<std::vector<index> >& cell_vertex,
// const std::vector<cellType>& cell_type


void unstructuredMesh_factory(const std::vector<NeoFOAM::vector>& points, const std::vector<std::vector<index> >& cell_vertex,
                              const std::vector<cellType>& cell_type, unstructuredMesh& mesh) {

    unstructuredMesh new_mesh;
    new_mesh.cell_vertex = cell_vertex;

    // setup connectivity
    // 1. vertex_cell   -> invert cell vertex.
    build_vertex_cell(new_mesh.cell_vertex, new_mesh.vertex_cell, points.size());
    // 2. cell_cell     -> if cells share #dim vertices they are 1st order neighbours
    // 3. face_cell, cell_face.  -> Loop cells, and construct.
    // 4. vertex face
    // 5. cell_boundary_zone     .


    // setup geometry


    // set up boundary (how to handle period like boundaries -> effects connectivity.) ?

                            

};


}  // namespace NeoFOAM