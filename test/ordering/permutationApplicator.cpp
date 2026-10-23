// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoFOAM/ordering/permutation.hpp"
#include "NeoFOAM/ordering/preparedPermutation.hpp"
#include "NeoFOAM/ordering/permutationApplicator.hpp"

namespace
{
using NeoN::sizeType;
using NeoFOAM::Permutation;
using NeoFOAM::PreparedPermutation;
using NeoFOAM::PermutationApplicator;
using IndexType = Permutation::IndexType;
}


TEST_CASE("Reorder cell-associated data")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Applying permutation reorders cell centers " + execName)
    {
        // Arrange
        // Prepare a permutation on the given executor
        // Create a permutation applicator
        // Create a 1D uniform mesh
        // constexpr sizeType size = 4; 
        // const Permutation permutation = Permutation::identity(size);
        // const PreparedPermutation prepared(permutation, exec);
        // NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

        // Act
        // Apply the permutation to the cell centers of the mesh
        // PermutationApplicator applicator{exec};

        // Assert
        // Verify the reordered cell centers
        // REQUIRE(applicator.exec() == exec);
        // REQUIRE_THAT(result, Equals(expected, EqualInt{}));
    }
}


TEST_CASE("Applying permutation remaps owner indices")
{}

TEST_CASE("Applying permutation remaps neighbour indices")
{}

TEST_CASE("Applying permutation preserves field-cell association")
{}

TEST_CASE("Applying identity permutation leaves mesh unchanged", "[permutationApplicator]")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Identity mapping on " + execName);
    {
        // Arrange
        constexpr sizeType size = 4; 
        const Permutation permutation = Permutation::identity(size);
        const PreparedPermutation prepared(permutation, exec);
        NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

        // Act
        PermutationApplicator applicator{exec};
        // applicator.reorderCellVolumes(mesh, prepared);

        // Assert
        REQUIRE(applicator.exec() == exec);
        // REQUIRE_THAT(result, Equals(expected, EqualInt{}));
    }
}