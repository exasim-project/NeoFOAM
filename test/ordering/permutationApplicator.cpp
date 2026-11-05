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

extern Foam::Time* timePtr; // A single time object

TEST_CASE("Reorder cell-associated data")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Applying permutation reorders scalar cell data " + execName)
    {
        // --- Arrange ---
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};

        const std::vector<NeoN::scalar> data {1.0, 2.0, 3.0, 4.0};
        NeoN::scalarVector cellData(exec, data);

        const std::vector<NeoN::scalar> expectedData {2.0, 4.0, 1.0, 3.0};
        const NeoN::scalarVector expectedCellData(exec, expectedData);

        const PermutationApplicator applicator(exec);

        // --- Act ---
        applicator.reorderCellData(cellData, prepared);

        // --- Assert ---
        REQUIRE(applicator.exec() == exec);
        REQUIRE_THAT(cellData, Equals(expectedCellData, EqualInt {}));
    }

    SECTION("Applying permutation reorders Vec3 cell data " + execName)
    {
        // --- Arrange ---
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};

        const std::vector<NeoN::Vec3>
            data {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}};
        NeoN::vectorVector cellData(exec, data);

        const std::vector<NeoN::Vec3>
            expectedData {{4.0, 5.0, 6.0}, {10.0, 11.0, 12.0}, {1.0, 2.0, 3.0}, {7.0, 8.0, 9.0}};
        const NeoN::vectorVector expectedCellData(exec, expectedData);

        const PermutationApplicator applicator(exec);

        // --- Act ---
        applicator.reorderCellData(cellData, prepared);

        // --- Assert ---
        REQUIRE(applicator.exec() == exec);
        REQUIRE_THAT(cellData, Equals(expectedCellData, EqualInt {}));
    }

    SECTION("Applying permutation reorders cell volumes " + execName)
    {
        // --- Arrange ---
        // Set up a mesh
        Foam::Time& runTime = *timePtr;
        auto rt = NeoFOAM::createAdapterRunTime(runTime, exec);
        Foam::scalar startTime = 0.0;
        Foam::label startTimeIndex = 0;
        runTime.setTime(startTime, startTimeIndex);

        std::unique_ptr<NeoFOAM::MeshAdapter> meshAdapterPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshAdapterPtr;
        NeoN::UnstructuredMesh& nfMesh = mesh.nfMesh();

        // Prepare a permutation and the applicator
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};
        const PermutationApplicator applicator(exec);

        // Prepare expected results
        // original cell volumes of cell 0, 1, 2, 3
        // {0.0111111111111111, 0.0222222222222222, 0.0222222222222222, 0.0444444444444444}
        const std::vector<NeoN::scalar> expectedCellData {0.0222222222222222, 0.0444444444444444, 
                                                    0.0111111111111111, 0.0222222222222222};
        const NeoN::scalarVector expectedCellVolumes(exec, expectedCellData);

        // --- Act ---
        applicator.reorderCellData(nfMesh.cellVolumes(), prepared);

        // --- Assert ---
        REQUIRE(nfMesh.nCells() == 4);
        REQUIRE_THAT(nfMesh.cellVolumes(), Equals(expectedCellVolumes, ApproxScalar{1e-12}));
    }
}


TEST_CASE("Applying permutation remaps owner indices") {}

TEST_CASE("Applying permutation remaps neighbour indices") {}

TEST_CASE("Applying permutation preserves field-cell association") {}

TEST_CASE("Applying identity permutation leaves mesh unchanged", "[permutationApplicator]")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Identity mapping on " + execName);
    {
        // Arrange
        constexpr sizeType size = 4;
        const Permutation permutation = Permutation::identity(size);
        const PreparedPermutation prepared(exec, permutation);
        NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

        // Act
        PermutationApplicator applicator {exec};
        // applicator.reorderCellVolumes(mesh, prepared);

        // Assert
        REQUIRE(applicator.exec() == exec);
        // REQUIRE_THAT(result, Equals(expected, EqualInt{}));
    }
}
