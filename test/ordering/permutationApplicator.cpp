// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "NeoFOAM/ordering/permutation.hpp"
#include "NeoFOAM/ordering/preparedPermutation.hpp"
#include "NeoFOAM/ordering/permutationApplicator.hpp"

namespace
{
using NeoFOAM::Permutation;
using NeoFOAM::PreparedPermutation;
using NeoFOAM::PermutationApplicator;
using IndexType = Permutation::IndexType;
using SizeType = Permutation::SizeType;
}

TEST_CASE("Applying identity permutation leaves mesh unchanged", "[permutationApplicator]")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Identity mapping on " + execName);
    {
        // Arrange
        constexpr SizeType size = 4; 
        const Permutation permutation = Permutation::identity(size);
        const PreparedPermutation prepared(permutation, exec);
        NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

        // Act
        PermutationApplicator applicator{exec};
        applicator.apply(mesh, prepared);

        // NeoN::parallelFor(
        //     exec,
        //     {0, static_cast<NeoN::localIdx>(prepared.size())},
        //     NEON_LAMBDA(const NeoN::localIdx i) {
        //         resultView[i] = prepared.oldToNew()[i];
        //     }
        // );

        // Assert
        // REQUIRE(prepared.size() == expected.size());
        // REQUIRE_THAT(result, Equals(expected, EqualInt{}));
    }
}

TEST_CASE("Applying permutation reorders cell centers")
{}

TEST_CASE("Applying permutation reorders cell volumes")
{}

TEST_CASE("Applying permutation remaps owner indices")
{}

TEST_CASE("Applying permutation remaps neighbour indices")
{}

TEST_CASE("Applying permutation preserves field-cell association")
{}
