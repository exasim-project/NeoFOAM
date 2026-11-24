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

    // Create a block mesh of dimension (1 1 0.1) with (2 2 1) blocks and simple grading (2 2 1)
    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
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
        // The original cell volumes
        // cell 0: 1.0/3.0 * 1.0/3.0 * 0.1
        // cell 1: 2.0/3.0 * 1.0/3.0 * 0.1
        // cell 2:  1.0/3.0 * 2.0/3.0 * 0.1
        // cell 3:  2.0/3.0 * 2.0/3.0 * 0.1
        const sizeType expectedCellCount {4};
        const std::vector<NeoN::scalar> expectedCellVolumes {
            2.0 / 3.0 * 1.0 / 3.0 * 0.1,
            2.0 / 3.0 * 2.0 / 3.0 * 0.1,
            1.0 / 3.0 * 1.0 / 3.0 * 0.1,
            1.0 / 3.0 * 2.0 / 3.0 * 0.1
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
        // The original cell centers
        // cell 0: {1.0/3.0 * 0.5, 1.0/3.0 * 0.5, 0.1 * 0.5},
        // cell 1: {1.0/3.0 + 2.0/3.0 * 0.5, 1.0/3.0 * 0.5, 0.1 * 0.5},
        // cell 2: {1.0/3.0 * 0.5, 1.0/3.0 + 2.0/3.0 * 0.5, 0.1 * 0.5},
        // cell 3: {1.0/3.0 + 2.0/3.0 * 0.5, 1.0/3.0 + 2.0/3.0 * 0.5, 0.1 * 0.5}
        const sizeType expectedCellCount {4};
        const std::vector<NeoN::Vec3> expectedCellCenters {
            {1.0 / 3.0 + 2.0 / 3.0 * 0.5, 1.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 + 2.0 / 3.0 * 0.5, 1.0 / 3.0 + 2.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 * 0.5, 1.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 * 0.5, 1.0 / 3.0 + 2.0 / 3.0 * 0.5, 0.1 * 0.5}
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

    // Create a block mesh of dimension (1 1 0.1) with (2 2 1) blocks and simple grading (2 2 1)
    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
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

    // Create a block mesh of dimension (1 1 0.1) with (2 2 1) blocks and simple grading (2 2 1)
    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
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
        // The original cell centers
        // cell 0: {1.0/3.0 * 0.5, 1.0/3.0 * 0.5, 0.1 * 0.5},
        // cell 1: {1.0/3.0 + 2.0/3.0 * 0.5, 1.0/3.0 * 0.5, 0.1 * 0.5},
        // cell 2: {1.0/3.0 * 0.5, 1.0/3.0 + 2.0/3.0 * 0.5, 0.1 * 0.5},
        // cell 3: {1.0/3.0 + 2.0/3.0 * 0.5, 1.0/3.0 + 2.0/3.0 * 0.5, 0.1 * 0.5}
        const std::vector<NeoN::Vec3> expectedFaceOwnerCenters {
            {1.0 / 3.0 + 2.0 / 3.0 * 0.5, 1.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 * 0.5, 1.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 + 2.0 / 3.0 * 0.5, 1.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 + 2.0 / 3.0 * 0.5, 1.0 / 3.0 + 2.0 / 3.0 * 0.5, 0.1 * 0.5}
        };

        const std::vector<NeoN::Vec3> expectedFaceNeighborCenters {
            {1.0 / 3.0 * 0.5, 1.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 * 0.5, 1.0 / 3.0 + 2.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 + 2.0 / 3.0 * 0.5, 1.0 / 3.0 + 2.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 * 0.5, 1.0 / 3.0 + 2.0 / 3.0 * 0.5, 0.1 * 0.5}
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
        // The original cell centers
        // cell 0: {1.0/3.0 * 0.5, 1.0/3 * 0.5, 0.1 * 0.5},
        // cell 1: {1.0/3.0 + 2.0/3.0 * 0.5, 1.0/3.0 * 0.5, 0.1 * 0.5},
        // cell 2: {1.0/3.0 * 0.5, 1.0/3.0 + 2.0/3.0 * 0.5, 0.1 * 0.5},
        // cell 3: {1.0/3.0 + 2.0/3.0 * 0.5, 1.0/3.0 + 2.0/3.0 * 0.5, 0.1 * 0.5}
        // boundary face owners after reordering: {3, 1, 2, 3, 0, 1, 2, 0};
        const std::vector<NeoN::Vec3> expectedBoundaryFaceOwnerCenters {
            {1.0 / 3.0 * 0.5, 1.0 / 3.0 + 2.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 + 2.0 / 3.0 * 0.5, 1.0 / 3.0 + 2.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 * 0.5, 1.0 / 3 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 * 0.5, 1.0 / 3.0 + 2.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 + 2.0 / 3.0 * 0.5, 1.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 + 2.0 / 3.0 * 0.5, 1.0 / 3.0 + 2.0 / 3.0 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 * 0.5, 1.0 / 3 * 0.5, 0.1 * 0.5},
            {1.0 / 3.0 + 2.0 / 3.0 * 0.5, 1.0 / 3.0 * 0.5, 0.1 * 0.5}
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

TEST_CASE("Applying permutation may change the direction of face normals")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Create a block mesh of dimension (1 1 0.1) with (2 2 1) blocks and simple grading (2 2 1)
    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    NeoN::UnstructuredMesh& nfMesh = mesh.nfMesh();

    // --- Arrange ---
    // Prepare a permutation and the applicator
    const std::vector<IndexType> oldToNew {2, 0, 3, 1};
    const Permutation permutation {oldToNew};
    const PreparedPermutation prepared {exec, permutation};
    const PermutationApplicator applicator {exec};

    // Prepare expected results
    const std::vector<NeoN::Vec3> expectedInternalFaceNormals {
        {-1.0 / 30.0, 0.0, 0.0},
        {0.0, 1.0 / 30.0, 0.0},
        {0.0, 2.0 / 30.0, 0.0},
        {-2.0 / 30.0, 0.0, 0.0}
    };

    const sizeType expectedInternalFaces {4};

    // --- Act ---
    applicator.apply(prepared, nfMesh);

    // --- Assert ---
    REQUIRE_THAT(
        nfMesh.faceNormals(),
        EqualsRange(
            expectedInternalFaceNormals,
            {0, NeoN::toLocalIdx(expectedInternalFaces)},
            ApproxVector {1e-12}
        )
    );
}

TEST_CASE("Applying identity permutation leaves mesh unchanged")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Create a block mesh of dimension (1 1 0.1) with (2 2 1) blocks and simple grading (2 2 1)
    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
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

    // Create a block mesh of dimension (1 1 0.1) with (2 2 1) blocks and simple grading (2 2 1)
    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
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

TEST_CASE("Applying an incompatible permutation to a field throws an exception")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Create a block mesh of dimension (1 1 0.1) with (2 2 1) blocks and simple grading (2 2 1)
    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    NeoN::UnstructuredMesh& nfMesh = mesh.nfMesh();

    // Create a cell-centered scalar field on the mesh
    auto ofT = NeoFOAM::indexVolScalarField(runTime, mesh, "T");
    auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);

    // --- Arrange ---
    // Prepare an incompatible permutation and the applicator
    const std::vector<Permutation::IndexType> oldToNew {2, 0, 1};
    const Permutation permutation {oldToNew};
    const PreparedPermutation prepared {exec, permutation};
    const PermutationApplicator applicator {exec};

    // --- Act & Assert ---
    REQUIRE_THROWS_AS(applicator.apply(prepared, nfT), std::invalid_argument);
}

TEST_CASE("Applying a permutation to a volume field reorders cell-associated physical field")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Create a block mesh of dimension (1 1 0.1) with (2 2 1) blocks and simple grading (2 2 1)
    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    NeoN::UnstructuredMesh& nfMesh = mesh.nfMesh();

    SECTION("Applying a permutation reorders volume scalar field " + execName)
    {
        // Create a cell-centered scalar field on the mesh
        auto ofT = NeoFOAM::indexVolScalarField(runTime, mesh, "T");
        auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);

        // --- Arrange ---
        // Prepare an incompatible permutation and the applicator
        const std::vector<Permutation::IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};
        const PermutationApplicator applicator {exec};

        // The original field T: {0.0, 1.0, 2.0, 3.0}
        const std::vector<NeoN::scalar> expectedScalarField {1.0, 3.0, 0.0, 2.0};

        // --- Act ---
        applicator.apply(prepared, nfT);

        // --- Assert ---
        REQUIRE_THAT(nfT.internalVector(), Equals(expectedScalarField, ApproxScalar {1e-12}));
    }

    SECTION("Applying a permutation reorders volume vector field " + execName)
    {
        // Create a cell-centered scalar field on the mesh
        auto ofU = NeoFOAM::indexVolVectorField(runTime, mesh, "U");
        auto nfU = NeoFOAM::constructFrom(exec, nfMesh, ofU);

        // --- Arrange ---
        // Prepare an incompatible permutation and the applicator
        const std::vector<Permutation::IndexType> oldToNew {2, 0, 3, 1};
        const Permutation permutation {oldToNew};
        const PreparedPermutation prepared {exec, permutation};
        const PermutationApplicator applicator {exec};

        // The original field U:
        // {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}}}
        const std::vector<NeoN::Vec3> expectedVectorField {
            {1.0, 1.0, 1.0},
            {3.0, 3.0, 3.0},
            {0.0, 0.0, 0.0},
            {2.0, 2.0, 2.0}
        };

        // --- Act ---
        applicator.apply(prepared, nfU);

        // --- Assert ---
        REQUIRE_THAT(nfU.internalVector(), Equals(expectedVectorField, ApproxVector {1e-12}));
    }
}
TEST_CASE("Applying permutation preserves field-cell association") {}
