// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"
// #include "NeoFOAM/linearAlgebra/linearSystem.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/* @class SparsityPattern
 * @brief row and column index representation of a mesh
 *
 * This class implements the finite volume 3/5/7 pt stencil specific generation
 * of sparsity patterns from a given unstructured mesh
 *
 */
class SparsityPattern
{
public:

    SparsityPattern(const UnstructuredMesh& mesh);

    void update();

    // const la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx>& linearSystem() const;

    // FIXME rename upperOffset
    const Field<uint8_t>& ownerOffset() const;

    // FIXME rename lowerOffset
    const Field<uint8_t>& neighbourOffset() const;

    const Field<uint8_t>& diagOffset() const;

    const UnstructuredMesh& mesh() const { return mesh_; };
    /* @brief given faceIdxs t
     *
     */
    [[nodiscard]] Field<localIdx> columnIndex() const;

    // add selection mechanism via dictionary later
    static const std::shared_ptr<SparsityPattern> readOrCreate(const UnstructuredMesh& mesh);

private:

    const UnstructuredMesh& mesh_;

    // TODO: currently sparsity pattern owns a linear system
    // this should be changed and sparsityPattern should only compute row and column idxs
    // la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> ls_;

    Field<uint8_t> ownerOffset_; //! mapping from faceId to lower index in a row

    Field<uint8_t> neighbourOffset_; //! mapping from faceId to upper index in a row

    Field<uint8_t> diagOffset_; //! mapping from faceId to column index in a row
};

} // namespace NeoFOAM::finiteVolume::cellCentred
