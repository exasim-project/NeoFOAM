// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoFOAM/mesh/unstructured.hpp"
#include "NeoFOAM/linearAlgebra.hpp"

namespace la = NeoFOAM::la;

namespace NeoFOAM::finiteVolume::cellCentred
{


class SparsityPattern
{
public:

    SparsityPattern(const UnstructuredMesh& mesh);

    la::LinearSystem<NeoFOAM::scalar, NeoFOAM::localIdx> compute() const;

private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoFOAM::finiteVolume::cellCentred
