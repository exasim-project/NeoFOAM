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

TEST_CASE("Applying permutation reorders cell-associated mesh data")
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

    SECTION("Applying permutation reorders cell volumes " + execName)
    {
        // --- Arrange ---
        // Prepare a permutation and the applicator
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};
        const PermutationApplicator applicator {exec};

        // Prepare expected results
        // The original cell volumes of cell 0, 1, 2, 3
        // {0.0111111111111111, 0.0222222222222222, 0.0222222222222222, 0.0444444444444444}
        const sizeType expectedCellCount {4};
        const std::vector<NeoN::scalar> expectedCellVolumes {
            0.0222222222222222,
            0.0444444444444444,
            0.0111111111111111,
            0.0222222222222222
        };

        // --- Act ---
        applicator.apply(prepared, nfMesh);

        // --- Assert ---
        REQUIRE(nfMesh.nCells() == expectedCellCount);
        REQUIRE_THAT(nfMesh.cellVolumes(), Equals(expectedCellVolumes, ApproxScalar {1e-12}));
    }

    SECTION("Applying permutation reorders cell centers " + execName)
    {
        // --- Arrange ---
        // Prepare a permutation and the applicator
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};
        const PermutationApplicator applicator {exec};

        // Prepare expected results
        // The original cell centers of cell 0, 1, 2, 3
        // cell 0: {0.166666666666667, 0.166666666666667, 0.05},
        // cell 1: {0.666666666666667, 0.166666666666667, 0.05},
        // cell 2: {0.166666666666667, 0.666666666666667, 0.05},
        // cell 3: {0.666666666666667, 0.666666666666667, 0.05}
        const sizeType expectedCellCount {4};
        const std::vector<NeoN::Vec3> expectedCellCenters {
            {0.666666666666667, 0.166666666666667, 0.05},
            {0.666666666666667, 0.666666666666667, 0.05},
            {0.166666666666667, 0.166666666666667, 0.05},
            {0.166666666666667, 0.666666666666667, 0.05}
        };

        // --- Act ---
        applicator.apply(prepared, nfMesh);

        // --- Assert ---
        REQUIRE(nfMesh.nCells() == expectedCellCount);
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
        const sizeType expectedInternalFaces {4};
        const std::vector<NeoN::label> expectedOwners {0, 2, 0, 1};
        const std::vector<NeoN::label> expectedNeighbors {2, 3, 1, 3};

        const PermutationApplicator applicator {exec};

        // --- Act ---
        applicator.apply(prepared, nfMesh);

        // --- Assert ---
        REQUIRE(applicator.exec() == exec);
        REQUIRE(nfMesh.nInternalFaces() == expectedInternalFaces);
        REQUIRE_THAT(
            nfMesh.faceOwners(),
            EqualsRange(expectedOwners, {0, NeoN::toLocalIdx(expectedInternalFaces)}, EqualInt {})
        );
        REQUIRE_THAT(
            nfMesh.faceNeighbors(),
            EqualsRange(
                expectedNeighbors,
                {0, NeoN::toLocalIdx(expectedInternalFaces)},
                EqualInt {}
            )
        );
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
        const sizeType expectedBoundaryFaces = 8;
        const std::vector<NeoN::label> expectedOwners {3, 1, 2, 3, 0, 1, 2, 0};

        const PermutationApplicator applicator {exec};

        // --- Act ---
        applicator.apply(prepared, nfMesh);

        // --- Assert ---
        REQUIRE(applicator.exec() == exec);
        REQUIRE(nfMesh.nBoundaryFaces() == expectedBoundaryFaces);
        REQUIRE_THAT(nfMesh.boundaryMesh().faceOwners(), Equals(expectedOwners, EqualInt {}));
    }
}

TEST_CASE("Applying permutation preserves face-cell connectivity")
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

    SECTION("Applying permutation keeps internal faces associated with the same cells " + execName)
    {
        // --- Arrange ---
        // Prepare a permutation and the applicator
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};
        const PermutationApplicator applicator {exec};

        // Prepare expected results
        // The original cell centers of cell 0, 1, 2, 3
        // cell 0: {0.166666666666667, 0.166666666666667, 0.05},
        // cell 1: {0.666666666666667, 0.166666666666667, 0.05},
        // cell 2: {0.166666666666667, 0.666666666666667, 0.05},
        // cell 3: {0.666666666666667, 0.666666666666667, 0.05}
        const std::vector<NeoN::Vec3> expectedFaceOwnerCenters {
            {0.666666666666667, 0.166666666666667, 0.05},
            {0.166666666666667, 0.166666666666667, 0.05},
            {0.666666666666667, 0.166666666666667, 0.05},
            {0.666666666666667, 0.666666666666667, 0.05}
        };

        const std::vector<NeoN::Vec3> expectedFaceNeighborCenters {
            {0.166666666666667, 0.166666666666667, 0.05},
            {0.166666666666667, 0.666666666666667, 0.05},
            {0.666666666666667, 0.666666666666667, 0.05},
            {0.166666666666667, 0.666666666666667, 0.05}
        };

        // --- Act ---
        applicator.apply(prepared, nfMesh);

        // --- Assert ---
        // Prepare actual face owner centers
        const NeoN::vectorVector cellCenters = nfMesh.cellCenters().copyToHost();
        auto cellCentersView = cellCenters.view();

        const NeoN::labelVector faceOwners = nfMesh.faceOwners().copyToHost();
        auto faceOwnersView = faceOwners.view();

        const std::vector<NeoN::Vec3> faceOwnerCenters {
            cellCentersView[faceOwnersView[0]],
            cellCentersView[faceOwnersView[1]],
            cellCentersView[faceOwnersView[2]],
            cellCentersView[faceOwnersView[3]]
        };

        // Prepare actual face neighbor centers
        const NeoN::labelVector faceNeighbors = nfMesh.faceNeighbors().copyToHost();
        auto faceNeighborsView = faceNeighbors.view();

        const std::vector<NeoN::Vec3> faceNeighborCenters {
            cellCentersView[faceNeighborsView[0]],
            cellCentersView[faceNeighborsView[1]],
            cellCentersView[faceNeighborsView[2]],
            cellCentersView[faceNeighborsView[3]]
        };
        REQUIRE_THAT(
            faceNeighborCenters,
            Equals(expectedFaceNeighborCenters, ApproxVector {1e-12})
        );
    }

    SECTION("Applying permutation keeps boundary faces associated with the same cells " + execName)
    {
        // --- Arrange ---
        // Prepare a permutation and the applicator
        const std::vector<IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};
        const PermutationApplicator applicator {exec};

        // Prepare expected results
        // The original cell centers of cell 0, 1, 2, 3
        // cell 0: {0.166666666666667, 0.166666666666667, 0.05},
        // cell 1: {0.666666666666667, 0.166666666666667, 0.05},
        // cell 2: {0.166666666666667, 0.666666666666667, 0.05},
        // cell 3: {0.666666666666667, 0.666666666666667, 0.05}
        // boundary face owners after reordering: {3, 1, 2, 3, 0, 1, 2, 0};
        const std::vector<NeoN::Vec3> expectedBoundaryFaceOwnerCenters {
            {0.166666666666667, 0.666666666666667, 0.05},
            {0.666666666666667, 0.666666666666667, 0.05},
            {0.166666666666667, 0.166666666666667, 0.05},
            {0.166666666666667, 0.666666666666667, 0.05},
            {0.666666666666667, 0.166666666666667, 0.05},
            {0.666666666666667, 0.666666666666667, 0.05},
            {0.166666666666667, 0.166666666666667, 0.05},
            {0.666666666666667, 0.166666666666667, 0.05}
        };

        // --- Act ---
        applicator.apply(prepared, nfMesh);

        // --- Assert ---
        // Prepare actual boundary face owner centers
        const NeoN::vectorVector cellCenters = nfMesh.cellCenters().copyToHost();
        auto cellCentersView = cellCenters.view();

        const NeoN::labelVector boundaryFaceOwners =
            nfMesh.boundaryMesh().faceOwners().copyToHost();
        auto boundaryFaceOwnersView = boundaryFaceOwners.view();

        const std::vector<NeoN::Vec3> boundaryFaceOwnerCenters {
            cellCentersView[boundaryFaceOwnersView[0]],
            cellCentersView[boundaryFaceOwnersView[1]],
            cellCentersView[boundaryFaceOwnersView[2]],
            cellCentersView[boundaryFaceOwnersView[3]],
            cellCentersView[boundaryFaceOwnersView[4]],
            cellCentersView[boundaryFaceOwnersView[5]],
            cellCentersView[boundaryFaceOwnersView[6]],
            cellCentersView[boundaryFaceOwnersView[7]]
        };

        REQUIRE_THAT(
            boundaryFaceOwnerCenters,
            Equals(expectedBoundaryFaceOwnerCenters, ApproxVector {1e-12})
        );
    }
}

TEST_CASE("Applying identity permutation leaves mesh unchanged")
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

    // --- Arrange ---
    // Prepare a permutation and the applicator
    const Permutation permutation = Permutation::identity(nfMesh.nCells());
    const PreparedPermutation prepared {exec, permutation};
    const PermutationApplicator applicator {exec};

    // Prepare expected results
    const sizeType expectedCellCount {4};
    const sizeType expectedInternalFaces {4};

    const NeoN::scalarVector originalCellVolumes = nfMesh.cellVolumes().copyToHost();
    auto originalCellVolulmesView = originalCellVolumes.view();

    const NeoN::vectorVector originalCellCenters = nfMesh.cellCenters().copyToHost();
    auto originalCellCentersView = originalCellCenters.view();

    const NeoN::labelVector originalFaceOwners = nfMesh.faceOwners().copyToHost();
    auto originalInternalFaceOwners = originalFaceOwners.view({0, expectedInternalFaces});

    const NeoN::labelVector originalFaceNeighbors = nfMesh.faceNeighbors().copyToHost();
    auto originalInternalFaceNeighbors = originalFaceNeighbors.view({0, expectedInternalFaces});

    const NeoN::labelVector originalBoundaryFaceOwners =
        nfMesh.boundaryMesh().faceOwners().copyToHost();
    auto originalBoundaryFaceOwnersView = originalBoundaryFaceOwners.view();

    // --- Act ---
    applicator.apply(prepared, nfMesh);

    // --- Assert ---
    REQUIRE(nfMesh.nCells() == expectedCellCount);
    REQUIRE_THAT(nfMesh.cellVolumes(), Equals(originalCellVolulmesView, ApproxScalar {1e-12}));
    REQUIRE_THAT(nfMesh.cellCenters(), Equals(originalCellCentersView, ApproxVector {1e-12}));
    REQUIRE_THAT(
        nfMesh.faceOwners(),
        EqualsRange(
            originalInternalFaceOwners,
            {0, NeoN::toLocalIdx(expectedInternalFaces)},
            EqualInt {}
        )
    );
    REQUIRE_THAT(
        nfMesh.faceNeighbors(),
        EqualsRange(
            originalInternalFaceNeighbors,
            {0, NeoN::toLocalIdx(expectedInternalFaces)},
            EqualInt {}
        )
    );
    REQUIRE_THAT(
        nfMesh.boundaryMesh().faceOwners(),
        Equals(originalBoundaryFaceOwnersView, EqualInt {})
    );
}

TEST_CASE("Applying an incompatible permutation to a mesh throws an exception")
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

    // --- Arrange ---
    // Prepare an incompatible permutation and the applicator
    const std::vector<Permutation::IndexType> oldToNew {2, 0, 1};
    const Permutation permutation {oldToNew};
    const PreparedPermutation prepared {exec, permutation};
    const PermutationApplicator applicator {exec};

    // --- Act & Assert ---
    REQUIRE_THROWS_AS(applicator.apply(prepared, nfMesh), std::invalid_argument);
}

TEST_CASE("Applying permutation preserves field-cell association") {}
