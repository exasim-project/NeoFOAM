// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#ifdef NeoN_WITH_JULIA
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"
#include "constrainHbyA.H"
#include <catch2/catch_approx.hpp>

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
    auto exec = NeoN::SerialExecutor {};
    std::cout << "exec: " << exec.name() << std::endl;
    auto path = fmt::format(
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
    auto gamma2 = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofNu);

    auto& phi = NeoFOAM::constructAndRegister(vectorCollection, rt, ofU);
    auto& phi2 = NeoFOAM::constructAndRegister(vectorCollection, rt, ofU);
    auto& nfOldPhi = fvcc::oldTime(phi);
    NeoN::fill(nfOldPhi.internalVector(), NeoN::Vec3(0.0, 0.0, 0.0));
    nfOldPhi.correctBoundaryConditions();
    std::cout << "cells: "<< mesh.nCells() << std::endl;
    SECTION("Interfaced Cellbased Assembly creates the same LS as NeoN")
    {
        auto lsFaceBased = NeoN::la::createEmptyLinearSystem<NeoN::Vec3>(rt.nfMesh);

        auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
        auto lsCellBased = NeoN::la::createEmptyLinearSystem<NeoN::Vec3>(rt.nfMesh, cellIterator);
        auto jlsCellBased = NeoN::la::createEmptyLinearSystem<NeoN::Vec3>(rt.nfMesh, cellIterator);
        nf::PDESolver<NeoN::Vec3> nfUEqn_c(
            // dsl::imp::ddt(phi) + 
            dsl::imp::div(faceFlux, phi)+
            dsl::imp::laplacian(gamma, phi), // expr
            phi,                                       // volumefield
            rt,                                         // runtime
            lsCellBased
        );
        nf::PDESolver<NeoN::Vec3> nfUEqn_f(
            // dsl::imp::ddt(phi) + 
            dsl::imp::div(faceFlux, phi)+
            dsl::imp::laplacian(gamma, phi), // expr
            phi,                                       // volumefield
            rt,                                         // runtime
            lsFaceBased
        );

        nf::PDESolver<NeoN::Vec3> juliaUEqn_c(
        //    dsl::imp::ddt(phi) +
            dsl::imp::div(faceFlux, phi)+
            dsl::imp::laplacian(gamma, phi), // expr
            phi,                                       // volumefield
            rt,                                         // runtime
            jlsCellBased
        );
        nf::PDESolver<NeoN::Vec3> juliaUEqn_f(
        //    dsl::imp::ddt(phi) +
            dsl::imp::div(faceFlux, phi)+
            dsl::imp::laplacian(gamma, phi), // expr
            phi,                                       // volumefield
            rt,                                         // runtime
            jlsCellBased
        );
        nfUEqn_c.assemble();

        for(size_t i = 0; i < 3; i++){
            std::chrono::steady_clock::time_point beginj2 = std::chrono::steady_clock::now();
            nfUEqn_c.assemble();
            std::chrono::steady_clock::time_point endj2 = std::chrono::steady_clock::now();
            auto result = fmt::format(
                fmt::runtime("Assembly time (NEON{}) = {}[ns]"),
                i,
                std::chrono::duration_cast<std::chrono::microseconds>(endj2 - beginj2).count()
            ); 
            std::cout << result << std::endl;

        }
        nfUEqn_f.assemble();
        // auto fv = lsFaceBased.matrix().values().view();
        // auto cv = lsCellBased.matrix().values().view();
        juliaUEqn_c.juliaCellbased(faceFlux, phi, gamma);
        for(size_t i = 0; i < 3; i++){
            std::chrono::steady_clock::time_point beginj2 = std::chrono::steady_clock::now();
            juliaUEqn_c.juliaCellbased(faceFlux, phi, gamma);
            std::chrono::steady_clock::time_point endj2 = std::chrono::steady_clock::now();
            auto result = fmt::format(
                fmt::runtime("Assembly time (JULIA{}) = {}[ns]"),
                i,
                std::chrono::duration_cast<std::chrono::microseconds>(endj2 - beginj2).count()
            ); 
            std::cout << result << std::endl;

        }
        // juliaUEqn_f.juliaFaceBased(faceFlux, phi, gamma);
        // auto jcv = jlsCellBased.matrix().values().view();
        REQUIRE(!jl_exception_occurred());
    }
    jl_atexit_hook(0);
}



#endif
