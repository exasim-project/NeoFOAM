// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "NeoFOAM/functionObjects/forces.hpp"
#include "NeoFOAM/functionObjects/forceCoeffs.hpp"

#include "forces.H"
#include "forceCoeffs.H"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace nf = NeoFOAM;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace fs = std::filesystem;

extern Foam::Time* timePtr;


// ── helpers (mirrors test_forceCoeffs.cpp) ─────────────────────────────────────

namespace
{
// Parse the last non-comment data row; treat OpenFOAM "(...)" as whitespace.
std::vector<double> readLastDataRow(const fs::path& filePath)
{
    std::ifstream in(filePath);
    REQUIRE(in.is_open());
    std::string line, last;
    while (std::getline(in, line))
        if (!line.empty() && line.front() != '#') last = line;
    for (char& c : last)
        if (c == '(' || c == ')') c = ' ';
    std::istringstream ss(last);
    std::vector<double> vals;
    double v;
    while (ss >> v)
        vals.push_back(v);
    return vals;
}

void compareDataFiles(const fs::path& nfPath, const fs::path& ofPath, double margin)
{
    auto nfRow = readLastDataRow(nfPath);
    auto ofRow = readLastDataRow(ofPath);
    REQUIRE(nfRow.size() == ofRow.size());
    for (std::size_t i = 0; i < nfRow.size(); ++i)
        CHECK(nfRow[i] == Catch::Approx(ofRow[i]).margin(margin));
}
} // namespace


// ── TEST: distributed Forces against OpenFOAM's parallel forces functionObject ─────────────
//
// neoForces now supports MPI: each rank integrates its local slice of the wall
// patches and execute() globally sums the contribution. OpenFOAM's own forces
// function object performs the equivalent reduction, so it is used as the
// reference on the same decomposed mesh. Only the master rank writes/compares.

TEST_CASE("DistributedForces - parallel force/moment match OpenFOAM reference")
{
    SECTION("Parallel sanity check")
    {
        REQUIRE(Foam::Pstream::parRun());
        REQUIRE(Foam::Pstream::nProcs() == 3);
    }

    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("uniform pressure, fixedWalls" + execName)
    {
        if (Foam::Pstream::master())
        {
            fs::remove_all("postProcessing/neoForces");
            fs::remove_all("postProcessing/ofForces");
        }
        Foam::UPstream::barrier(Foam::UPstream::worldComm);

        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& mesh = rt.mesh;

        Foam::volScalarField ofP(
            Foam::IOobject(
                "p",
                runTime.timeName(),
                mesh,
                Foam::IOobject::MUST_READ,
                Foam::IOobject::NO_WRITE
            ),
            mesh
        );
        ofP.primitiveFieldRef() = Foam::scalar {1.0};
        ofP.correctBoundaryConditions();

        fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        nf::constructAndRegister(vc, rt, ofP, false);

        Foam::dictionary dict;
        dict.add("patches", Foam::wordList {"fixedWalls"});
        dict.add("pName", Foam::word {"p"});
        dict.add("rhoInf", Foam::scalar {1.0});
        dict.add("pRef", Foam::scalar {0.0});
        dict.add("CofR", Foam::vector(0, 0, 0));

        nf::Forces forces("neoForces", runTime, dict);
        REQUIRE(forces.execute());
        REQUIRE(forces.write());

        // OF reference (forces functionObject reduces across ranks internally)
        Foam::volVectorField zeroU(
            Foam::IOobject(
                "U",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ),
            mesh,
            Foam::dimensionedVector("U", Foam::dimVelocity, Foam::vector::zero)
        );
        Foam::IOdictionary transportProps(Foam::IOobject(
            "transportProperties",
            runTime.constant(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ));
        transportProps.add("nu", Foam::scalar {0.0});

        Foam::dictionary ofDict;
        ofDict.add("patches", Foam::wordList {"fixedWalls"});
        ofDict.add("rho", Foam::word("rhoInf"));
        ofDict.add("rhoInf", Foam::scalar {1.0});
        ofDict.add("pRef", Foam::scalar {0.0});
        ofDict.add("CofR", Foam::vector(0, 0, 0));

        Foam::functionObjects::forces ofForces("ofForces", runTime, ofDict);
        REQUIRE(ofForces.execute());
        REQUIRE(ofForces.write());

        Foam::UPstream::barrier(Foam::UPstream::worldComm);
        if (Foam::Pstream::master())
        {
            const double tol = 1e-10;
            compareDataFiles(
                "postProcessing/neoForces/0/force.dat",
                "postProcessing/ofForces/0/force.dat",
                tol
            );
            compareDataFiles(
                "postProcessing/neoForces/0/moment.dat",
                "postProcessing/ofForces/0/moment.dat",
                tol
            );
            fs::remove_all("postProcessing/neoForces");
            fs::remove_all("postProcessing/ofForces");
        }
    }

    SECTION("random U + non-zero viscous forces, fixedWalls" + execName)
    {
        if (Foam::Pstream::master())
        {
            fs::remove_all("postProcessing/neoForces");
            fs::remove_all("postProcessing/ofForces");
        }
        Foam::UPstream::barrier(Foam::UPstream::worldComm);

        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& mesh = rt.mesh;

        auto ofP = randomScalarField(runTime, mesh, "p");
        ofP.correctBoundaryConditions();
        auto ofU = randomVectorField(runTime, mesh, "U");
        ofU.correctBoundaryConditions();

        fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        nf::constructAndRegister(vc, rt, ofP, false);
        nf::constructAndRegister(vc, rt, ofU, false);

        Foam::IOdictionary transportProps(Foam::IOobject(
            "transportProperties",
            runTime.constant(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ));
        const Foam::scalar nu = 1e-5;
        transportProps.add("nu", nu);

        Foam::dictionary dict;
        dict.add("patches", Foam::wordList {"fixedWalls"});
        dict.add("pName", Foam::word {"p"});
        dict.add("rhoInf", Foam::scalar {1.225});
        dict.add("pRef", Foam::scalar {0.0});
        dict.add("CofR", Foam::vector(0, 0, 0));

        nf::Forces forces("neoForces", runTime, dict);
        REQUIRE(forces.execute());
        REQUIRE(forces.write());

        Foam::dictionary ofDict;
        ofDict.add("patches", Foam::wordList {"fixedWalls"});
        ofDict.add("rho", Foam::word("rhoInf"));
        ofDict.add("rhoInf", Foam::scalar {1.225});
        ofDict.add("pRef", Foam::scalar {0.0});
        ofDict.add("CofR", Foam::vector(0, 0, 0));

        Foam::functionObjects::forces ofForces("ofForces", runTime, ofDict);
        REQUIRE(ofForces.execute());
        REQUIRE(ofForces.write());

        Foam::UPstream::barrier(Foam::UPstream::worldComm);
        if (Foam::Pstream::master())
        {
            const double tol = 1e-8;
            compareDataFiles(
                "postProcessing/neoForces/0/force.dat",
                "postProcessing/ofForces/0/force.dat",
                tol
            );
            compareDataFiles(
                "postProcessing/neoForces/0/moment.dat",
                "postProcessing/ofForces/0/moment.dat",
                tol
            );
            fs::remove_all("postProcessing/neoForces");
            fs::remove_all("postProcessing/ofForces");
        }
    }
}
