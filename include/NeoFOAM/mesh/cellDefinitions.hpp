// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <array>
#include <vector>
#include <string>

#include "cellTypes.h"
#include "NeoFOAM/blas/primitives/label.hpp"

namespace NeoFOAM {

inline const std::array<std::string, totalCellTypes> cellNames = {
    "edge",
    "quad",
    "tri",
    "hexah",
    "tetra",
    "prism",
    "pyram",
};


// cell to edges
using CellEdgeTable std::vector<std::array<localIdx, 2> >;
inline const CellEdgeTable EdgeEdge = {{0, 1}};
inline const CellEdgeTable QuadEdge = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
inline const CellEdgeTable TriEdge = {{0, 1}, {1, 2}, {2, 0}};     
inline const CellEdgeTable HexahEdge = {{0, 4}, {0, 1}, {1, 5}, {4, 5}, {1, 3}, {3, 7}, {5, 7}, {2, 3}, {2, 6}, {6, 7}, {0, 2}, {4, 6}};
inline const CellEdgeTable PrismEdge = {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}};
inline const CellEdgeTable TetraEdge = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};
inline const CellEdgeTable PyramEdge = {{0, 1}, {1, 3}, {3, 2}, {0, 2}, {0, 4}, {1, 4}, {3, 4}, {2, 4}};

struct CellLocalEdgeHelper
{
    [[no_discard]] static const CellEdgeTable& operator[](const cellTypes cellType) const {
        switch (cellType) {
            case cellTypes::edge:
                return EdgeEdge;
            case cellTypes::quad:
                return QuadEdge;
            case cellTypes::tri:
                return TriEdge;
            case cellTypes::hexah:
                return HexahEdge;
            case cellTypes::prism:
                return PrismEdge;
            case cellTypes::tetra:
                return TetraEdge;
            case cellTypes::pyram:
                return PyramEdge;
            default:
                throw std::runtime_error("Invalid cell type");
        }
    }
};
const inline CellLocalEdge = CellLocalEdgeHelper();

// cell to faces
using CellFaceTable std::vector<std::vector<localIdx> >;
inline const CellFaceTable EdgeFace = {};
inline const CellFaceTable QuadFace = {};
inline const CellFaceTable TriFace = {};
inline const CellFaceTable HexahFace = {{0, 1, 5, 4}, {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}, {1, 0, 2, 3}, {4, 5, 7, 6}};
inline const CellFaceTable PrismFace = {{0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}, {0, 2, 1}, {3, 4, 5}};
inline const CellFaceTable TetraFace = {{1, 0, 2}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
inline const CellFaceTable PyramFace = {{0, 2, 3, 1}, {0, 1, 4}, {1, 3, 4}, {3, 2, 4}, {2, 0, 4}};

template<bool 2d_edge_as_face>
struct CellLocalFaceHelper
{
    [[no_discard]] static const CellFaceTable& operator[](const cellTypes cellType) const {
        switch (cellType) {
            case cellTypes::edge:
                return 2d_edge_as_face ? EdgeEdge : EdgeFace;
            case cellTypes::quad:
                return 2d_edge_as_face ? QuadEdge : QuadFace;
            case cellTypes::tri:
                return 2d_edge_as_face ? TriEdge : TriFace;
            case cellTypes::hexah:
                return HexahFace;
            case cellTypes::prism:
                return PrismFace;
            case cellTypes::tetra:
                return TetraFace;
            case cellTypes::pyram:
                return PyramFace;
            default:
                throw std::runtime_error("Invalid cell type");
        }
    }
};
const inline CellLocalFace = CellLocalFaceHelper<false>();
const inline CellLocalEdgeasFace = CellLocalFaceHelper<true>();

// local to global
template<typename label_t>
struct FaceGlobal{

    LocalToGlobal() = delete; // no default construction that would break shit.
    LocalToGlobal(const std::span<label_t>& global, const std::vector<label_t>& local_face) : global_(global), local_face_(local_face) {};

    std::size_t size() {return local_face_.size()}; // number of vertices on the face

    [[no_discard]] static const label_t& operator[](const label_t i_local) const  // given a local vertex index returns the global vertex index
    { 
        return global_[local_face_[i_local]];
    }

    private:
    const std::span<label_t>& global_;
    const std::vector<label_t>& local_face_; 
};

template<typename label_t, bool 2d_edge_as_face>
struct LocalToGlobalFaceHelper
{   
    LocalToGlobal() = delete; // no default construction that would break shit.
    LocalToGlobal(const std::span<label_t>& global, const cellTypes type) : global_(global), type_(type) {};

    std::size_t size(){helper[type_].size()}; // total number of faces

    [[no_discard]] static const FaceGlobalHelper<>& operator[](const label_t i_face) const { // returns a global face helper.
        return {global_, helper[type_][i_face]};
    }

    private:
    const CellLocalFaceHelper<2d_edge_as_face> helper;
    const std::span<label_t>& global_;
    const cellTypes& type_;
};
template<typename label_t>
using CellGlobalFace = LocalToGlobalFaceHelper<label_t, cell_t, false>();
template<typename label_t>
using CellGlobalEdgeasFace = LocalToGlobalEdgeasFaceHelper<label_t, cell_t, true>();

} // namespace 
