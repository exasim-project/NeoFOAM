// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <unordered_set>

#include "common.hpp"

#include "gaussConvectionScheme.H"


namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object


TEST_CASE("cell To Face Stencil")
{
    Foam::Time& runTime = *timePtr;
    Foam::argList& args = *argsPtr;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    auto& nfMesh = mesh.nfMesh();

    SECTION("cellToFaceStencil_" + execName)
    {
        fvcc::CellToFaceStencil cellToFaceStencil(nfMesh);
        NeoN::SegmentedVector<NeoN::localIdx, NeoN::localIdx> stencil =
            cellToFaceStencil.computeStencil();

        auto hostStencil = stencil.copyToHost();
        auto stencilView = hostStencil.view();

        for (auto celli = 0; celli < mesh.cells().size(); celli++)
        {
            std::unordered_set<Foam::label> faceSet;
            REQUIRE(stencilView.view(celli).size() == mesh.cells()[celli].size());
            for (auto facei : mesh.cells()[celli])
            {
                faceSet.insert(facei);
            }

            for (auto facei : stencilView.view(celli))
            {
                REQUIRE(faceSet.contains(facei));
            }
        }
    }
}
