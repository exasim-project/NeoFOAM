// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "NeoFOAM/functionObjects/forces.hpp"
#include "NeoFOAM/functionObjects/forceCoeffs.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"

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


// ── helpers ──────────────────────────────────────────────────────────────────

/**
 * Parse the last non-comment data row from a function-object output file.
 * Parentheses (OpenFOAM vector notation) are treated as whitespace so both
 * "v1 v2 v3" and "(v1 v2 v3)" styles parse into flat scalar lists.
 */
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

/**
 * Column-by-column comparison of two function-object dat files.
 * Both NeoFOAM (flat scalars) and OpenFOAM (vector notation) formats are handled.
 */
void compareDataFiles(const fs::path& nfPath, const fs::path& ofPath, double margin)
{
    auto nfRow = readLastDataRow(nfPath);
    auto ofRow = readLastDataRow(ofPath);
    REQUIRE(nfRow.size() == ofRow.size());
    for (std::size_t i = 0; i < nfRow.size(); ++i)
        CHECK(nfRow[i] == Catch::Approx(ofRow[i]).margin(margin));
}

/**
 * Register a zero U field and a minimal transportProperties dictionary so that
 * OpenFOAM's forces function object can compute (zero) viscous contributions
 * without a solver or turbulence model being present.
 *
 * Both objects are RAII: they de-register from the mesh's objectRegistry on
 * destruction when the enclosing scope ends.
 */
struct OfForcesSetup
{
    Foam::volVectorField zeroU;
    Foam::IOdictionary transportProps;

    explicit OfForcesSetup(Foam::fvMesh& mesh, const Foam::Time& runTime)
        : zeroU(
            Foam::IOobject(
                "U",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ),
            mesh,
            Foam::dimensionedVector("U", Foam::dimVelocity, Foam::vector::zero)
        )
        , transportProps(Foam::IOobject(
              "transportProperties",
              runTime.constant(),
              mesh,
              Foam::IOobject::NO_READ,
              Foam::IOobject::NO_WRITE
          ))
    {
        // nu = 0 → zero gradient on uniform U → zero viscous forces
        transportProps.add("nu", Foam::scalar {0.0});
    }
};


// ── TEST: Forces ──────────────────────────────────────────────────────────────

TEST_CASE("Forces - pressure force and moment match OpenFOAM reference")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("uniform pressure, fixedWalls" + execName)
    {
        fs::remove_all("postProcessing/neoForces");
        fs::remove_all("postProcessing/ofForces");

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

        // ── NeoFOAM Forces ───────────────────────────────────────────────────
        Foam::dictionary dict;
        dict.add("patches", Foam::wordList {"fixedWalls"});
        dict.add("pName", Foam::word {"p"});
        dict.add("rhoInf", Foam::scalar {1.0});
        dict.add("pRef", Foam::scalar {0.0});
        dict.add("CofR", Foam::vector(0, 0, 0));

        nf::Forces forces("neoForces", runTime, dict);
        REQUIRE(forces.execute());
        REQUIRE(forces.write());

        // ── OF forces reference ──────────────────────────────────────────────
        OfForcesSetup ofSetup(mesh, runTime);

        Foam::dictionary ofDict;
        ofDict.add("patches", Foam::wordList {"fixedWalls"});
        ofDict.add("rho", Foam::word("rhoInf"));
        ofDict.add("rhoInf", Foam::scalar {1.0});
        ofDict.add("pRef", Foam::scalar {0.0});
        ofDict.add("CofR", Foam::vector(0, 0, 0));

        Foam::functionObjects::forces ofForces("ofForces", runTime, ofDict);
        REQUIRE(ofForces.execute());
        REQUIRE(ofForces.write());

        // ── File comparison ──────────────────────────────────────────────────
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

    SECTION("random pressure field, fixedWalls" + execName)
    {
        fs::remove_all("postProcessing/neoForces");
        fs::remove_all("postProcessing/ofForces");

        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& mesh = rt.mesh;

        auto ofP = randomScalarField(runTime, mesh, "p");
        ofP.correctBoundaryConditions();

        fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        nf::constructAndRegister(vc, rt, ofP, false);

        Foam::dictionary dict;
        dict.add("patches", Foam::wordList {"fixedWalls"});
        dict.add("pName", Foam::word {"p"});
        dict.add("rhoInf", Foam::scalar {1.225});
        dict.add("pRef", Foam::scalar {0.5});
        dict.add("CofR", Foam::vector(0.1, 0.2, 0.3));

        nf::Forces forces("neoForces", runTime, dict);
        REQUIRE(forces.execute());
        REQUIRE(forces.write());

        OfForcesSetup ofSetup(mesh, runTime);

        Foam::dictionary ofDict;
        ofDict.add("patches", Foam::wordList {"fixedWalls"});
        ofDict.add("rho", Foam::word("rhoInf"));
        ofDict.add("rhoInf", Foam::scalar {1.225});
        ofDict.add("pRef", Foam::scalar {0.5});
        ofDict.add("CofR", Foam::vector(0.1, 0.2, 0.3));

        Foam::functionObjects::forces ofForces("ofForces", runTime, ofDict);
        REQUIRE(ofForces.execute());
        REQUIRE(ofForces.write());

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

    SECTION("multiple patches: fixedWalls + inlet + outlet" + execName)
    {
        fs::remove_all("postProcessing/neoForces");
        fs::remove_all("postProcessing/ofForces");

        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& mesh = rt.mesh;

        auto ofP = randomScalarField(runTime, mesh, "p");
        ofP.correctBoundaryConditions();

        fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        nf::constructAndRegister(vc, rt, ofP, false);

        Foam::dictionary dict;
        dict.add("patches", Foam::wordList {"fixedWalls", "inlet", "outlet"});
        dict.add("pName", Foam::word {"p"});
        dict.add("rhoInf", Foam::scalar {1.0});
        dict.add("pRef", Foam::scalar {0.0});
        dict.add("CofR", Foam::vector(0, 0, 0));

        nf::Forces forces("neoForces", runTime, dict);
        REQUIRE(forces.execute());
        REQUIRE(forces.write());

        OfForcesSetup ofSetup(mesh, runTime);

        Foam::dictionary ofDict;
        ofDict.add("patches", Foam::wordList {"fixedWalls", "inlet", "outlet"});
        ofDict.add("rho", Foam::word("rhoInf"));
        ofDict.add("rhoInf", Foam::scalar {1.0});
        ofDict.add("pRef", Foam::scalar {0.0});
        ofDict.add("CofR", Foam::vector(0, 0, 0));

        Foam::functionObjects::forces ofForces("ofForces", runTime, ofDict);
        REQUIRE(ofForces.execute());
        REQUIRE(ofForces.write());

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


// ── TEST: ForceCoeffs ─────────────────────────────────────────────────────────

TEST_CASE("ForceCoeffs - normalised coefficients match OpenFOAM reference")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("random pressure, all coefficients" + execName)
    {
        fs::remove_all("postProcessing/neoForceCoeffs");
        fs::remove_all("postProcessing/ofForceCoeffs");

        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& mesh = rt.mesh;

        auto ofP = randomScalarField(runTime, mesh, "p");
        ofP.correctBoundaryConditions();

        fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        nf::constructAndRegister(vc, rt, ofP, false);

        const Foam::scalar rhoRef = 1.225;
        const Foam::scalar magUInf = 10.0;
        const Foam::scalar lRef = 0.5;
        const Foam::scalar Aref = 0.25;
        const Foam::scalar pRef = 0.0;
        const Foam::vector cofR = Foam::vector::zero;

        // ── NeoFOAM ForceCoeffs ──────────────────────────────────────────────
        Foam::dictionary dict;
        dict.add("patches", Foam::wordList {"fixedWalls"});
        dict.add("pName", Foam::word {"p"});
        dict.add("rhoInf", rhoRef);
        dict.add("pRef", pRef);
        dict.add("magUInf", magUInf);
        dict.add("lRef", lRef);
        dict.add("Aref", Aref);
        dict.add("CofR", cofR);

        dict.add("dragDir", Foam::vector(1, 0, 0));
        dict.add("liftDir", Foam::vector(0, 0, 1));
        dict.add("pitchAxis", Foam::vector(0, 1, 0));

        nf::ForceCoeffs fc("neoForceCoeffs", runTime, dict);
        REQUIRE(fc.execute());
        REQUIRE(fc.write());

        // ── OF forceCoeffs reference ─────────────────────────────────────────
        OfForcesSetup ofSetup(mesh, runTime);

        Foam::dictionary ofDict;
        ofDict.add("patches", Foam::wordList {"fixedWalls"});
        ofDict.add("rho", Foam::word("rhoInf"));
        ofDict.add("rhoInf", rhoRef);
        ofDict.add("pRef", pRef);
        ofDict.add("magUInf", magUInf);
        ofDict.add("lRef", lRef);
        ofDict.add("Aref", Aref);
        ofDict.add("CofR", cofR);
        ofDict.add("dragDir", Foam::vector(1, 0, 0));
        ofDict.add("liftDir", Foam::vector(0, 0, 1));
        ofDict.add("pitchAxis", Foam::vector(0, 1, 0));

        Foam::functionObjects::forceCoeffs ofFc("ofForceCoeffs", runTime, ofDict);
        REQUIRE(ofFc.execute());
        REQUIRE(ofFc.write());

        // ── Compare coefficient.dat ──────────────────────────────────────────
        // Both NeoFOAM and OF write 12 coefficient columns in alphabetical order:
        // Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) CmPitch CmRoll CmYaw Cs Cs(f) Cs(r)
        const double tol = 1e-10;
        compareDataFiles(
            "postProcessing/neoForceCoeffs/0/coefficient.dat",
            "postProcessing/ofForceCoeffs/0/coefficient.dat",
            tol
        );

        fs::remove_all("postProcessing/neoForceCoeffs");
        fs::remove_all("postProcessing/ofForceCoeffs");
    }
}
