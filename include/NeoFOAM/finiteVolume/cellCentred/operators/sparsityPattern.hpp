// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"

namespace la = NeoFOAM::la;

namespace NeoFOAM::finiteVolume::cellCentred
{


class SparsityPattern
{
public:

    SparsityPattern(const UnstructuredMesh& mesh);

    void update();

    const la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx>& linearSystem() const;

    const Field<uint8_t>& ownerOffset() const;

    const Field<uint8_t>& neighbourOffset() const;

    const Field<uint8_t>& diagOffset() const;

    // add selection mechanism via dictionary later
    static const std::shared_ptr<SparsityPattern> readOrCreate(const UnstructuredMesh& mesh);

private:

    const UnstructuredMesh& mesh_;
    la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> ls_;
    Field<uint8_t> ownerOffset_;
    Field<uint8_t> neighbourOffset_;
    Field<uint8_t> diagOffset_;
};

} // namespace NeoFOAM::finiteVolume::cellCentred
