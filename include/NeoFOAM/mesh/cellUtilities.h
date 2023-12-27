// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/blas/primative/label.hpp"

#include "cellDefinitions.hpp"

namespace NeoFOAM
{      
    template<typename label_t>
    [[no_discard]] inline bool isVertexFaceVertex(const FaceGlobal<label_t> face, const label_t& vertex)
    {
        for(auto local_vertex = 0; local_vertex < face.size(); ++local_vertex) if(face[local_vertex] == vertex) return true;
        return false;
    }

    template<typename label_t>
    [[no_discard]] inline bool isFaceCellFace(const CellGlobalFace<label_t>& cell_global_faces, const FaceGlobal<label_t>& face) {

        for(auto i_face = 0; i_face < cell_global_faces.size(); ++i_face) {
            if(cell_global_faces[i_face].size() != face.size()) continue;
           
            bool all_vertex_on_face = true;
            for(auto i_face_vertex = 0; i_face_vertex < face.size() : ++i_face_vertex) {
                if(!isVertexFaceVertex(cell_global_faces[i_face], face[i_face_vertex])) {
                    all_vertex_on_face = false;
                    break;
                }
            }
            if(all_vertex_on_face) return true;
        }
        return false;
    }    

    template<typename label_t>
    [[no_discard]] inline bool CellShareFace(const CellGlobalFace<label_t>& cell0_global_faces, const CellGlobalFace<label_t>& cell1_global_faces)
    {   
        for(auto i_face = 0; i_face < cell1_global_faces.size(); ++i_face)
            if(isFaceCellFace(cell0_global_faces, cell1_global_faces[i_face]])) return true;
        return false;
    }

}
