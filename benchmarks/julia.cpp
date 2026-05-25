// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoFOAM/NeoFOAM.hpp"

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "../test/common.hpp"
#ifdef NeoN_WITH_JULIA
#include <julia.h>
#include <fmt/core.h>
JULIA_DEFINE_FAST_TLS; // only define this once, in an executable (not in a
// shared library) if you want fast code.
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;
namespace dsl = NeoN::dsl;
namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

#include "fvc.H"
#include "fvm.H"
#include "fvMatrices.H"

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object
TEST_CASE("julia")
{
    jl_init();

    Foam::Time& runTime = *timePtr;
    std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
    Foam::fvMesh& mesh = *meshPtr;

    auto ofU = nf::randomVectorField(runTime, mesh, "U");
    auto ofP = nf::randomScalarField(runTime, mesh, "p");
    
    auto exec = NeoN::CPUExecutor {};
    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);
    auto& nfMesh = rt.mesh;
    auto& fieldCollection = fvcc::VectorCollection::instance(rt.db, "fieldCollection");
    auto& vectorCollection = nnfvcc::VectorCollection::instance(rt.db, "VectorCollection");
    
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
    
    nf::PDESolver<NeoN::Vec3> UEQN(
        dsl::imp::div(faceFlux, phi) - dsl::imp::laplacian(gamma, phi),
        phi,
        rt
    );
    std::cout << "benchmarkg?\n";
    SECTION("NeoN With Julia")
    {
        std::string path = fmt::format(
            fmt::runtime(
                "include(\"{}\")"
            ),
            JULIA_MODULE_INIT
        );
        jl_eval_string(path.c_str());
        UEQN.warmupFaceBased();
        BENCHMARK("SpatialOperators")
        {
            UEQN.juliaFaceBased(faceFlux, phi, gamma);
            return;
        };
    }
    // SECTION(std::string("NeoN"))
    // {
        //     BENCHMARK("SpatialOperators")
        //     {
            //         UEQN.assemble();
            //         NeoN::fence(exec);
            //         return;
            //     };
            // }
    jl_atexit_hook(0);
}
#endif()
