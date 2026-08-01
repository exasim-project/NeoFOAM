// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "NeoFOAM/functionObjects/forces.hpp"
#include "NeoFOAM/functionObjects/forceCoeffs.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "NeoFOAM/fvcc/boundary/volume/nutWallFunction.hpp"

#include "forces.H"
#include "forceCoeffs.H"

#include <cmath>
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

    explicit OfForcesSetup(Foam::fvMesh& mesh, const Foam::Time& runTime, Foam::scalar nu = 0.0)
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
        transportProps.add("nu", nu);
    }
};


// ── TEST: Forces ──────────────────────────────────────────────────────────────

TEST_CASE("Forces - pressure force and moment match OpenFOAM reference")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("uniform pressure, fixedWalls" + execName)
    {
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/neoForces"));
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/ofForces"));

        {
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

            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(rt.db, "VectorCollection");
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
        }

        // ── File comparison ──────────────────────────────────────────────────
        INFO("Comparing NeoFOAM and OpenFOAM force output files");
        const double tol = 1e-10;
        compareDataFiles(
            "postProcessing/neoForces/0/force.dat",
            "postProcessing/ofForces/0/force.dat",
            tol
        );

        INFO("Comparing NeoFOAM and OpenFOAM moment output files");
        compareDataFiles(
            "postProcessing/neoForces/0/moment.dat",
            "postProcessing/ofForces/0/moment.dat",
            tol
        );

        INFO("Cleaning up output directories");
        CHECK_NOTHROW(fs::remove_all("postProcessing/neoForces"));
        CHECK_NOTHROW(fs::remove_all("postProcessing/ofForces"));
    }

    SECTION("random pressure field, fixedWalls" + execName)
    {
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/neoForces"));
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/ofForces"));

        {
            auto rt = nf::createAdapterRunTime(runTime, exec);
            auto& mesh = rt.mesh;

            auto ofP = randomScalarField(runTime, mesh, "p");
            ofP.correctBoundaryConditions();

            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(rt.db, "VectorCollection");
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
        }

        INFO("Comparing NeoFOAM and OpenFOAM force output files");
        const double tol = 1e-10;
        compareDataFiles(
            "postProcessing/neoForces/0/force.dat",
            "postProcessing/ofForces/0/force.dat",
            tol
        );

        INFO("Comparing NeoFOAM and OpenFOAM moment output files");
        compareDataFiles(
            "postProcessing/neoForces/0/moment.dat",
            "postProcessing/ofForces/0/moment.dat",
            tol
        );

        INFO("Cleaning up output directories");
        CHECK_NOTHROW(fs::remove_all("postProcessing/neoForces"));
        CHECK_NOTHROW(fs::remove_all("postProcessing/ofForces"));
    }

    SECTION("multiple patches: fixedWalls + inlet + outlet" + execName)
    {
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/neoForces"));
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/ofForces"));

        {
            auto rt = nf::createAdapterRunTime(runTime, exec);
            auto& mesh = rt.mesh;

            auto ofP = randomScalarField(runTime, mesh, "p");
            ofP.correctBoundaryConditions();

            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(rt.db, "VectorCollection");
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
        }

        INFO("Comparing NeoFOAM and OpenFOAM force output files");
        const double tol = 1e-10;
        compareDataFiles(
            "postProcessing/neoForces/0/force.dat",
            "postProcessing/ofForces/0/force.dat",
            tol
        );

        INFO("Comparing NeoFOAM and OpenFOAM moment output files");
        compareDataFiles(
            "postProcessing/neoForces/0/moment.dat",
            "postProcessing/ofForces/0/moment.dat",
            tol
        );

        INFO("Cleaning up output directories");
        CHECK_NOTHROW(fs::remove_all("postProcessing/neoForces"));
        CHECK_NOTHROW(fs::remove_all("postProcessing/ofForces"));
    }

    SECTION("random U + non-zero viscous forces, fixedWalls" + execName)
    {
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/neoForces"));
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/ofForces"));

        {
            auto rt = nf::createAdapterRunTime(runTime, exec);
            auto& mesh = rt.mesh;

            auto ofP = randomScalarField(runTime, mesh, "p");
            ofP.correctBoundaryConditions();
            auto ofU = randomVectorField(runTime, mesh, "U");
            ofU.correctBoundaryConditions();

            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(rt.db, "VectorCollection");
            nf::constructAndRegister(vc, rt, ofP, false);
            nf::constructAndRegister(vc, rt, ofU, false);

            // transportProperties with non-zero nu registered in Foam::Time registry
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

            // OF reference — U is already registered (ofU), pass nu to transportProperties
            Foam::dictionary ofDict;
            ofDict.add("patches", Foam::wordList {"fixedWalls"});
            ofDict.add("rho", Foam::word("rhoInf"));
            ofDict.add("rhoInf", Foam::scalar {1.225});
            ofDict.add("pRef", Foam::scalar {0.0});
            ofDict.add("CofR", Foam::vector(0, 0, 0));

            Foam::functionObjects::forces ofForces("ofForces", runTime, ofDict);
            REQUIRE(ofForces.execute());
            REQUIRE(ofForces.write());
        }

        INFO("Comparing NeoFOAM and OpenFOAM force output files");
        const double tol = 1e-8;
        compareDataFiles(
            "postProcessing/neoForces/0/force.dat",
            "postProcessing/ofForces/0/force.dat",
            tol
        );

        INFO("Comparing NeoFOAM and OpenFOAM moment output files");
        compareDataFiles(
            "postProcessing/neoForces/0/moment.dat",
            "postProcessing/ofForces/0/moment.dat",
            tol
        );

        INFO("Cleaning up output directories");
        CHECK_NOTHROW(fs::remove_all("postProcessing/neoForces"));
        CHECK_NOTHROW(fs::remove_all("postProcessing/ofForces"));
    }

    SECTION("random U + turbulent nuEff (nu + nut), fixedWalls" + execName)
    {
        // TODO: This test assumes nuEff in OpenFOAM is calculated as nu + nut on each patch face,
        // we never test if the NeoFOAM nuEff is actually equalt to OpenFOAMs nuEff
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/neoForces"));
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/ofForces"));

        const Foam::scalar nu = 1e-5;
        const Foam::scalar nutConst = 5e-4;

        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& mesh = rt.mesh;

        auto ofP = randomScalarField(runTime, mesh, "p");
        ofP.correctBoundaryConditions();
        auto ofU = randomVectorField(runTime, mesh, "U");
        ofU.correctBoundaryConditions();

        // Constant nut field — uniform value on all faces including boundary
        Foam::volScalarField ofNut(
            Foam::IOobject(
                "nut",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar("nut", Foam::dimViscosity, nutConst)
        );
        ofNut.correctBoundaryConditions();

        fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(rt.db, "VectorCollection");
        nf::constructAndRegister(vc, rt, ofP, false);
        nf::constructAndRegister(vc, rt, ofU, false);
        nf::constructAndRegister(vc, rt, ofNut, false);

        Foam::dictionary dict;
        dict.add("patches", Foam::wordList {"fixedWalls"});
        dict.add("pName", Foam::word {"p"});
        dict.add("rhoInf", Foam::scalar {1.225});
        dict.add("pRef", Foam::scalar {0.0});
        dict.add("CofR", Foam::vector(0, 0, 0));

        // NeoFOAM scope: transportProperties with laminar nu only.
        // Forces reads nu here; adds nut per boundary face from VectorCollection.
        {
            Foam::IOdictionary transportProps(Foam::IOobject(
                "transportProperties",
                runTime.constant(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ));
            transportProps.add("nu", nu);

            nf::Forces forces("neoForces", runTime, dict);
            REQUIRE(forces.execute());
            REQUIRE(forces.write());
        } // transportProps deregisters here

        // OF scope: transportProperties with nuEff = nu + nut.
        // OF forces has no turbulence model, so effective viscosity must be set directly.
        {
            Foam::IOdictionary ofTransportProps(Foam::IOobject(
                "transportProperties",
                runTime.constant(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE
            ));
            ofTransportProps.add("nu", nu + nutConst);

            Foam::dictionary ofDict;
            ofDict.add("patches", Foam::wordList {"fixedWalls"});
            ofDict.add("rho", Foam::word("rhoInf"));
            ofDict.add("rhoInf", Foam::scalar {1.225});
            ofDict.add("pRef", Foam::scalar {0.0});
            ofDict.add("CofR", Foam::vector(0, 0, 0));

            Foam::functionObjects::forces ofForces("ofForces", runTime, ofDict);
            REQUIRE(ofForces.execute());
            REQUIRE(ofForces.write());
        } // ofTransportProps deregisters here

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

        CHECK_NOTHROW(fs::remove_all("postProcessing/neoForces"));
        CHECK_NOTHROW(fs::remove_all("postProcessing/ofForces"));
    }
}


// ── TEST: ForceCoeffs ─────────────────────────────────────────────────────────

TEST_CASE("ForceCoeffs - normalised coefficients match OpenFOAM reference")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("random pressure, all coefficients" + execName)
    {
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/neoForceCoeffs"));
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/ofForceCoeffs"));

        {
            auto rt = nf::createAdapterRunTime(runTime, exec);
            auto& mesh = rt.mesh;

            auto ofP = randomScalarField(runTime, mesh, "p");
            ofP.correctBoundaryConditions();

            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(rt.db, "VectorCollection");
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
        }

        // ── Compare coefficient.dat ──────────────────────────────────────────
        // Both NeoFOAM and OF write 12 coefficient columns in alphabetical order:
        // Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) CmPitch CmRoll CmYaw Cs Cs(f) Cs(r)
        INFO("Comparing NeoFOAM and OpenFOAM coefficient output files");
        const double tol = 1e-10;
        compareDataFiles(
            "postProcessing/neoForceCoeffs/0/coefficient.dat",
            "postProcessing/ofForceCoeffs/0/coefficient.dat",
            tol
        );

        INFO("Cleaning up output directories");
        CHECK_NOTHROW(fs::remove_all("postProcessing/neoForceCoeffs"));
        CHECK_NOTHROW(fs::remove_all("postProcessing/ofForceCoeffs"));
    }

    SECTION("random U + non-zero viscous forces, all coefficients" + execName)
    {
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/neoForceCoeffs"));
        REQUIRE_NOTHROW(fs::remove_all("postProcessing/ofForceCoeffs"));

        {
            auto rt = nf::createAdapterRunTime(runTime, exec);
            auto& mesh = rt.mesh;

            auto ofP = randomScalarField(runTime, mesh, "p");
            ofP.correctBoundaryConditions();
            auto ofU = randomVectorField(runTime, mesh, "U");
            ofU.correctBoundaryConditions();

            fvcc::VectorCollection& vc =
                fvcc::VectorCollection::instance(rt.db, "VectorCollection");
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

            const Foam::scalar rhoRef = 1.225;
            const Foam::scalar magUInf = 10.0;
            const Foam::scalar lRef = 0.5;
            const Foam::scalar Aref = 0.25;
            const Foam::scalar pRef = 0.0;
            const Foam::vector cofR = Foam::vector::zero;

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
        }

        INFO("Comparing NeoFOAM and OpenFOAM coefficient output files");
        const double tol = 1e-8;
        compareDataFiles(
            "postProcessing/neoForceCoeffs/0/coefficient.dat",
            "postProcessing/ofForceCoeffs/0/coefficient.dat",
            tol
        );

        INFO("Cleaning up output directories");
        CHECK_NOTHROW(fs::remove_all("postProcessing/neoForceCoeffs"));
        CHECK_NOTHROW(fs::remove_all("postProcessing/ofForceCoeffs"));
    }
}


// ── TEST: nutUSpaldingWallFunction (viscous-force input) ───────────────────────
//
// The viscous part of the force integral is driven by the wall eddy viscosity nut,
// which on wall patches is set by the Spalding wall function. The existing viscous
// tests above feed Forces a uniform/constant nut and never run the wall function, so
// a regression in the Spalding nut would silently corrupt viscous (skin-friction)
// drag while leaving pressure forces correct. This test validates the wall-function
// math that produces that nut, independently of any mesh / turbulence model.
//
// Reference: solve Spalding's law of the wall for the friction velocity uTau,
//   y+ = u+ + (1/E) * ( exp(K u+) - 1 - K u+ - (K u+)^2/2 - (K u+)^3/6 )
// with u+ = magUp/uTau and y+ = uTau*y/nu, using an independent bisection (the NeoN
// implementation uses Newton). Then nut = uTau^2 / magGradU - nu, exactly as
// setNutUSpaldingWallFunction assigns it (magGradU = magUp * deltaCoeff, and at a
// wall face deltaCoeff = 1/y, so magGradU = magUp/y).

namespace
{
namespace wf = NeoN::finiteVolume::cellCentred::volumeBoundary::detail;

// Independent Spalding solve for uTau. f(uTau) = y+ - spalding(u+) is monotonically
// increasing in uTau (y+ rises, spalding falls), so a single root exists; bisect it.
// K and E must match the constants in nutWallFunction.hpp (KAPPA = 0.41, E = 9.8).
double spaldingUTauReference(double magUp, double y, double nu)
{
    constexpr double k = 0.41;
    constexpr double e = 9.8;
    auto f = [&](double uTau)
    {
        const double up = magUp / uTau;
        const double yp = uTau * y / nu;
        const double kup = k * up;
        const double spald =
            up + (1.0 / e) * (std::exp(kup) - 1.0 - kup - 0.5 * kup * kup - kup * kup * kup / 6.0);
        return yp - spald;
    };
    double lo = 1e-9; // f(lo) < 0  (y+ -> 0, spalding -> +inf)
    double hi = 1e3;  // f(hi) > 0  (y+ -> large, spalding -> 0)
    for (int it = 0; it < 100; ++it)
    {
        const double mid = 0.5 * (lo + hi);
        if (f(mid) > 0.0)
        {
            hi = mid;
        }
        else
        {
            lo = mid;
        }
    }
    return 0.5 * (lo + hi);
}
}

TEST_CASE("nutUSpaldingWallFunction - uTau and nut match independent Spalding reference")
{
    struct Case
    {
        double magUp; // |U_cell - U_wall|  [m/s]
        double y;     // nearWallDist        [m]
        double nu;    // laminar viscosity   [m2/s]
    };

    // Spread of near-wall states from the buffer layer up into the log layer.
    const std::vector<Case> cases = {
        {0.10, 1.0e-3, 1.0e-5},
        {0.50, 1.0e-3, 1.0e-5},
        {1.00, 5.0e-4, 1.0e-5},
        {2.00, 1.0e-4, 1.0e-5},
        {5.00, 2.0e-4, 1.5e-5},
    };

    for (const auto& c : cases)
    {
        const double magGradU = c.magUp / c.y; // = magUp * deltaCoeff, deltaCoeff = 1/y at the wall

        // Drive the Newton solve to convergence (tolerance well below the 1e-6
        // check): this validates the Spalding f/df math itself. Production stops
        // at OpenFOAM's default tolerance (wf::TOLERANCE = 0.01), which leaves an
        // O(1e-5) residual by design — that path is covered by the wiring test.
        NeoN::scalar err = 0.0;
        const NeoN::scalar utNeo = wf::computeUTau(
            magGradU,
            c.magUp,
            c.y,
            c.nu,
            /*nutw0*/ 0.0,
            err,
            50,
            /*tolerance*/ 1e-10
        );
        const double utRef = spaldingUTauReference(c.magUp, c.y, c.nu);

        INFO(
            "magUp=" << c.magUp << " y=" << c.y << " nu=" << c.nu << " y+=" << utRef * c.y / c.nu
                     << " utNeo=" << utNeo << " utRef=" << utRef
        );

        // Newton (NeoN) vs bisection (reference) on the same Spalding equation.
        CHECK(static_cast<double>(utNeo) == Catch::Approx(utRef).epsilon(1e-6));

        // nut assignment, exactly as setNutUSpaldingWallFunction computes it.
        const double nutNeo = static_cast<double>(utNeo) * static_cast<double>(utNeo) / magGradU
                            - static_cast<double>(c.nu);
        const double nutRef = utRef * utRef / magGradU - c.nu;
        CHECK(nutNeo == Catch::Approx(nutRef).epsilon(1e-6));
        CHECK(nutNeo > 0.0); // these states are turbulent -> positive eddy viscosity
    }
}


// ── TEST: nutUSpaldingWallFunction full BC-path wiring ─────────────────────────
//
// The test above validates the wall-function math in isolation. This one validates
// the field/context WIRING end-to-end: build a nut field carrying the
// nutUSpaldingWallFunction BC on the wall patch, feed it U / nu / nearWallDist via a
// BoundaryContext exactly as the turbulence model does (correctNut ->
// nutField.correctBoundaryConditions(ctx)), then confirm the wall nut written into
// the field equals a per-face host recompute that replicates setNutUSpaldingWallFunction
// from the SAME inputs and the mesh geometry (deltaCoeffs). This is the path the solver
// and Forces actually use; it catches a wrong owner/U/nearWallDist/deltaCoeffs lookup or
// a value not landing in the registered boundary — none of which the math-only test sees.

TEST_CASE("nutUSpaldingWallFunction - wall nut via correctBoundaryConditions matches recompute")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("end-to-end BC wiring on fixedWalls" + execName)
    {
        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& mesh = rt.mesh;
        const NeoN::UnstructuredMesh& nfMesh = rt.nfMesh;

        const Foam::label wallPatchID = mesh.boundaryMesh().findPatchID("fixedWalls");
        REQUIRE(wallPatchID >= 0);

        // nut carries the Spalding wall function on the wall patch, calculated elsewhere.
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> nutBCs;
        for (NeoN::localIdx p = 0; p < nfMesh.nBoundaries(); ++p)
        {
            const std::string type = (p == static_cast<NeoN::localIdx>(wallPatchID))
                                       ? std::string("nutUSpaldingWallFunction")
                                       : std::string("calculated");
            NeoN::Dictionary d({{"type", type}});
            nutBCs.emplace_back(nfMesh, d, p);
        }
        fvcc::VolumeField<NeoN::scalar> nut(exec, "nut", nfMesh, nutBCs);

        fvcc::VolumeField<NeoN::Vec3> U(
            exec,
            "U",
            nfMesh,
            fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(nfMesh)
        );
        fvcc::VolumeField<NeoN::scalar> nu(
            exec,
            "nu",
            nfMesh,
            fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(nfMesh)
        );
        fvcc::VolumeField<NeoN::scalar> nearWallDist(
            exec,
            "nearWallDist",
            nfMesh,
            fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(nfMesh)
        );

        const NeoN::scalar Ux = 1.0;
        const NeoN::scalar nuVal = 1.0e-5;
        const NeoN::scalar yVal = 1.0e-3;

        NeoN::fill(U.internalVector(), NeoN::Vec3(Ux, 0.0, 0.0));
        NeoN::fill(U.boundaryData().value(), NeoN::Vec3(0.0, 0.0, 0.0)); // no-slip wall
        NeoN::fill(nu.internalVector(), nuVal);
        NeoN::fill(nu.boundaryData().value(), nuVal);
        NeoN::fill(nearWallDist.internalVector(), yVal);
        NeoN::fill(nearWallDist.boundaryData().value(), yVal);
        NeoN::fill(nut.internalVector(), NeoN::scalar(0.0));
        NeoN::fill(nut.boundaryData().value(), NeoN::scalar(0.0)); // cold start

        // Drive the wall function exactly as SpalartAllmarasDDES::correctNut does.
        fvcc::BoundaryContext ctx;
        ctx.insert("U", U);
        ctx.insert("nu", nu);
        ctx.insert("nearWallDist", nearWallDist);
        nut.correctBoundaryConditions(ctx);

        // Independent per-face recompute from the same inputs + mesh deltaCoeffs.
        auto nutHost = nut.boundaryData().value().copyToHost();
        auto uIntHost = U.internalVector().copyToHost();
        auto uBndHost = U.boundaryData().value().copyToHost();
        auto yHost = nearWallDist.boundaryData().value().copyToHost();
        auto ownerHost = nfMesh.boundaryMesh().faceOwners().copyToHost();
        auto deltaHost = nfMesh.boundaryMesh().deltaCoeffs().copyToHost();

        auto nutV = nutHost.view();
        auto uIntV = uIntHost.view();
        auto uBndV = uBndHost.view();
        auto yV = yHost.view();
        auto ownerV = ownerHost.view();
        auto deltaV = deltaHost.view();

        const auto [start, end] =
            nut.boundaryData().range(static_cast<NeoN::localIdx>(wallPatchID));
        REQUIRE(end > start); // wall patch must carry faces

        // Per-face reference replicates setNutUSpaldingWallFunction exactly (same computeUTau,
        // same preserve/clamp branch, cold-start currentNut = 0). This isolates the WIRING: it
        // passes iff the kernel reads the correct owner U, nu, nearWallDist and mesh deltaCoeffs
        // for each face and writes the result back to that face's boundary value. (Numerical
        // correctness of the Spalding solve itself is covered by the math-only test above.)
        int checked = 0;
        for (NeoN::localIdx i = start; i < end; ++i)
        {
            const auto owner = ownerV[i];
            const NeoN::Vec3 diff = uIntV[owner] - uBndV[i];
            const NeoN::scalar magUp = NeoN::mag(diff);
            const NeoN::scalar magGradU = magUp * deltaV[i];
            const NeoN::scalar y = yV[i];

            NeoN::scalar err = 0.0;
            NeoN::scalar errOneIter = 0.0;
            const NeoN::scalar uTau = wf::computeUTau(
                magGradU,
                magUp,
                y,
                nuVal,
                /*currentNut*/ 0.0,
                err,
                wf::MAX_ITER,
                wf::TOLERANCE
            );
            wf::computeUTau(
                magGradU,
                magUp,
                y,
                nuVal,
                /*currentNut*/ 0.0,
                errOneIter,
                1,
                wf::TOLERANCE
            );
            const NeoN::scalar cand = (uTau * uTau) / (magGradU + NeoN::ROOTVSMALL) - nuVal;
            const NeoN::scalar candClamped = cand > 0.0 ? cand : 0.0;
            const NeoN::scalar nutRef =
                (errOneIter < wf::TOLERANCE) ? NeoN::scalar(0.0) : candClamped;

            INFO(
                "face " << i << " magUp=" << magUp << " magGradU=" << magGradU << " y=" << y
                        << " nutWF=" << nutV[i] << " nutRef=" << nutRef
            );
            CHECK(
                static_cast<double>(nutV[i])
                == Catch::Approx(static_cast<double>(nutRef)).epsilon(1e-9)
            );
            ++checked;
        }
        CHECK(checked > 0);
    }
}
