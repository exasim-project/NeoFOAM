// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "catch2_common.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

// FIXME make sure that this is in NeoFOAM/NeoFOAM.hpp
#include "NeoFOAM/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"


namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

using SparsityPattern = fvcc::SparsityPattern;

namespace NeoFOAM
{

TEST_CASE("SparsityPattern")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto nFaces = 9;
    auto mesh = create1DUniformMesh(exec, nCells);

    SECTION("Can construct sparsity pattern " + execName)
    {
        auto sp = SparsityPattern {mesh};
        // some basic sanity checks
        REQUIRE(sp.ownerOffset().size() == nFaces);
        REQUIRE(sp.neighbourOffset().size() == nFaces);
        REQUIRE(sp.diagOffset().size() == nCells);
    }
}

}
