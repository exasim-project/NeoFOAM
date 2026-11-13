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

    // Set up a mesh: 2x2x1 non-uniform mesh with 4 cells
    Foam::Time& runTime = *timePtr;
    auto rt = NeoFOAM::createAdapterRunTime(runTime, exec);
    Foam::scalar startTime = 0.0;
    Foam::label startTimeIndex = 0;
    runTime.setTime(startTime, startTimeIndex);

    std::unique_ptr<NeoFOAM::MeshAdapter> meshAdapterPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshAdapterPtr;
    NeoN::UnstructuredMesh& nfMesh = mesh.nfMesh();

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
        // Prepare a permutation and the applicator
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};
        const PermutationApplicator applicator(exec);

        // Prepare expected results
        // The original cell volumes of cell 0, 1, 2, 3
        // {0.0111111111111111, 0.0222222222222222, 0.0222222222222222, 0.0444444444444444}
        const std::vector<NeoN::scalar> expectedCellData {
            0.0222222222222222,
            0.0444444444444444,
            0.0111111111111111,
            0.0222222222222222
        };
        const NeoN::scalarVector expectedCellVolumes(exec, expectedCellData);

        // --- Act ---
        applicator.reorderCellData(nfMesh.cellVolumes(), prepared);

        // --- Assert ---
        REQUIRE(nfMesh.nCells() == 4);
        REQUIRE_THAT(nfMesh.cellVolumes(), Equals(expectedCellVolumes, ApproxScalar {1e-12}));
    }

    SECTION("Applying permutation reorders cell centers " + execName)
    {
        // --- Arrange ---
        // Prepare a permutation and the applicator
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};
        const PermutationApplicator applicator(exec);

        // Prepare expected results
        // The original cell centers of cell 0, 1, 2, 3
        // cell 0: {0.166666666666667, 0.166666666666667, 0.05},
        // cell 1: {0.666666666666667, 0.166666666666667, 0.05},
        // cell 2: {0.166666666666667, 0.666666666666667, 0.05},
        // cell 3: {0.666666666666667, 0.666666666666667, 0.05}
        const std::vector<NeoN::Vec3> expectedCellData {
            {0.666666666666667, 0.166666666666667, 0.05},
            {0.666666666666667, 0.666666666666667, 0.05},
            {0.166666666666667, 0.166666666666667, 0.05},
            {0.166666666666667, 0.666666666666667, 0.05}
        };
        const NeoN::vectorVector expectedCellCenters(exec, expectedCellData);

        // --- Act ---
        applicator.reorderCellData(nfMesh.cellCenters(), prepared);

        // --- Assert ---
        REQUIRE(nfMesh.nCells() == 4);
        REQUIRE_THAT(nfMesh.cellCenters(), Equals(expectedCellCenters, ApproxVector {1e-12}));
    }

    SECTION("Applying permutation reorders a mesh " + execName)
    {
        // --- Arrange ---
        // Prepare a permutation and the applicator
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};
        const PermutationApplicator applicator(exec);

        // Prepare expected results
        const std::vector<NeoN::scalar> cellVolumes {
            0.0222222222222222,
            0.0444444444444444,
            0.0111111111111111,
            0.0222222222222222
        };
        const NeoN::scalarVector expectedCellVolumes(exec, cellVolumes);

        const std::vector<NeoN::Vec3> cellCenters {
            {0.666666666666667, 0.166666666666667, 0.05},
            {0.666666666666667, 0.666666666666667, 0.05},
            {0.166666666666667, 0.166666666666667, 0.05},
            {0.166666666666667, 0.666666666666667, 0.05}
        };
        const NeoN::vectorVector expectedCellCenters(exec, cellCenters);

        // --- Act ---
        applicator.apply(nfMesh, prepared);

        // --- Assert ---
        REQUIRE(nfMesh.nCells() == 4);
        REQUIRE_THAT(nfMesh.cellVolumes(), Equals(expectedCellVolumes, ApproxScalar {1e-12}));
        REQUIRE_THAT(nfMesh.cellCenters(), Equals(expectedCellCenters, ApproxVector {1e-12}));
    }
}


TEST_CASE("Applying permutation remaps connectivity")
{

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Set up a mesh
    Foam::Time& runTime = *timePtr;
    auto rt = NeoFOAM::createAdapterRunTime(runTime, exec);
    Foam::scalar startTime = 0.0;
    Foam::label startTimeIndex = 0;
    runTime.setTime(startTime, startTimeIndex);

    std::unique_ptr<NeoFOAM::MeshAdapter> meshAdapterPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshAdapterPtr;
    NeoN::UnstructuredMesh& nfMesh = mesh.nfMesh();

    SECTION(
        "Applying permutation remaps owner and neighbor cell indices of internal faces " + execName
    )
    {
        // --- Arrange ---
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};

        // Original owner cell indices of internal faces: {0, 0, 1, 2}
        // Original neighbor cell indices of internal faces: {1, 2, 3, 3}
        const std::vector<NeoN::label> expectedOwners {2, 2, 0, 3};
        const std::vector<NeoN::label> expectedNeighbors {0, 3, 1, 1};

        const PermutationApplicator applicator(exec);

        // --- Act ---
        applicator.remapConnectivity(nfMesh, prepared);

        // --- Assert ---
        REQUIRE(applicator.exec() == exec);
        REQUIRE_THAT(nfMesh.faceOwners(), EqualsRange(expectedOwners, {0, 4}, EqualInt {}));
        REQUIRE_THAT(nfMesh.faceNeighbors(), EqualsRange(expectedNeighbors, {0, 4}, EqualInt {}));
        // REQUIRE_THAT(cellData, Equals(expectedCellData, EqualInt {}));
    }

    SECTION("Applying permutation remaps owner cell indices of boundary faces " + execName)
    {
        // --- Arrange ---
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};

        // Original owner cell indices of boundary faces:
        // { 2, 3, 0, 2,
        //   1, 3, 0, 1 };
        const std::vector<NeoN::label> expectedOwners {3, 1, 2, 3, 0, 1, 2, 0};

        const PermutationApplicator applicator(exec);

        // --- Act ---
        applicator.remapConnectivity(nfMesh, prepared);

        // --- Assert ---
        REQUIRE(applicator.exec() == exec);
        REQUIRE(nfMesh.nBoundaryFaces() == 8);
        REQUIRE_THAT(nfMesh.boundaryMesh().faceOwners(), Equals(expectedOwners, EqualInt {}));
    }
}

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
