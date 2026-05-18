// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#ifdef NeoN_WITH_JULIA
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"
#include "constrainHbyA.H"
#include <julia.h>
#include <fmt/core.h>
JULIA_DEFINE_FAST_TLS; // only define this once, in an executable (not in a
// shared library) if you want fast code.
namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace dsl = NeoN::dsl;
namespace nnfvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr; // A single time object

static bool check_julia_exception(const char* where)
{
    jl_value_t* exc = jl_exception_occurred();
    if (exc)
    {
        std::cerr << where << ": Julia exception: " << jl_typeof_str(exc) << std::endl;
        return true;
    }
    return false;
}

TEST_CASE("Julia Momentum")
{
    jl_init();

    float epsilon = 1e-32;
    Foam::Time& runTime = *timePtr;
    // auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto exec = NeoN::Executor(NeoN::SerialExecutor {});

    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    //std::string path = std::format("include(\"{}\")", JULIA_MODULE_INIT);
    std::string pre = "include(\""; 
    std::string inf = JULIA_MODULE_INIT;
    std::string suf = "\")";
    std::string path = pre + inf + suf;
    auto tst = fmt::format(
					fmt::runtime("include(\"{}\")"),
					JULIA_MODULE_INIT
	); 
	std::cout << "path to init: "<< path << std::endl;
    jl_eval_string(path.c_str());

    if (jl_exception_occurred())
    {
        const char* p = jl_string_ptr(
            jl_eval_string("sprint(showerror, ccall(:jl_exception_occurred, Any, ()))")
        );

        fprintf(stderr, "%s%s\n", "error: ", p);
    }
    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);
    std::cout << schemesDict << std::endl;
    auto divs = schemesDict.subDict("divSchemes");
    auto phiu = divs.get<NeoN::TokenList>("div(phi,U)");

    auto ofU = randomVectorField(runTime, mesh, "U");
    auto ofp = randomScalarField(runTime, mesh, "p");
    ofp.correctBoundaryConditions();
    ofU.correctBoundaryConditions();
    auto& oldOfU = ofU.oldTime();
    oldOfU.primitiveFieldRef() = Foam::vector(0.0, 0.0, 0.0);
    oldOfU.correctBoundaryConditions();
    auto& vectorCollection = nnfvcc::VectorCollection::instance(rt.db, "VectorCollection");
    auto& nfP = NeoFOAM::constructAndRegister(vectorCollection, rt, ofp);

    Foam::surfaceScalarField ofPhi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        fvc::flux(ofU)
    );

    Foam::surfaceScalarField ofNu(
        Foam::IOobject(
            "nu",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("nu", Foam::dimensionSet(0, 2, -1, 0, 0), 0.01)
    );

    auto faceFlux = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi);
    auto gamma = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofNu);

    auto& phi = NeoFOAM::constructAndRegister(vectorCollection, rt, ofU);
    auto& nfOldPhi = fvcc::oldTime(phi);
    NeoN::fill(nfOldPhi.internalVector(), NeoN::Vec3(0.0, 0.0, 0.0));
    nfOldPhi.correctBoundaryConditions();

    SECTION("Solve transient momentum without grad(p) on ")
    {

        Foam::fvVectorMatrix ofUEqn(
            fvm::ddt(ofU) + fvm::div(ofPhi, ofU) - fvm::laplacian(ofNu, ofU)
        );

        nf::PDESolver<NeoN::Vec3> nfUEqn(
            dsl::imp::ddt(phi) +
            dsl::imp::div(faceFlux, phi) - 5 * dsl::imp::laplacian(gamma, phi), // expr
            phi,                                                                // volumefield
            rt                                                                  // runtime
        );

        nf::PDESolver<NeoN::Vec3> juliaUEqn(
            dsl::imp::ddt(phi) +
            dsl::imp::div(faceFlux, phi) - 5 * dsl::imp::laplacian(gamma, phi), // expr
            phi,                                                                // volumefield
            rt                                                                  // runtime
        );
        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

        std::chrono::steady_clock::time_point beginnf = std::chrono::steady_clock::now();
        nfUEqn.assemble();

        std::chrono::steady_clock::time_point endnf = std::chrono::steady_clock::now();

        std::cout << "Assembly time (NEON) = "
                  << std::chrono::duration_cast<std::chrono::microseconds>(endnf - beginnf).count()
                  << "[ns]" << std::endl;

        std::chrono::steady_clock::time_point beginj = std::chrono::steady_clock::now();
        juliaUEqn.warmupFaceBased();
        juliaUEqn.juliaFaceBased(faceFlux, phi, gamma);
        std::chrono::steady_clock::time_point endj = std::chrono::steady_clock::now();
        std::cout << "Assembly time (JULIA) = "
                  << std::chrono::duration_cast<std::chrono::microseconds>(endj - beginj).count()
                  << "[ns]" << std::endl;

        if (jl_exception_occurred())
        {
            const char* p = jl_string_ptr(
                jl_eval_string("sprint(showerror, ccall(:jl_exception_occurred, Any, ()))")
            );

            fprintf(stderr, "%s%s\n", "error: ", p);
        }

        REQUIRE(!jl_exception_occurred());
    }
    jl_atexit_hook(0);
}

#endif
