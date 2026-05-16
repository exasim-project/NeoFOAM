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
#include <julia.h>
JULIA_DEFINE_FAST_TLS; // only define this once, in an executable (not in a
// shared library) if you want fast code.
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;
namespace dsl = NeoN::dsl;

#include "fvc.H"
#include "fvm.H"
#include "fvMatrices.H"

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

TEST_CASE("julia")
{
    Foam::Time& runTime = *timePtr;
    std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
    Foam::fvMesh& mesh = *meshPtr;

    auto ofU = nf::randomVectorField(runTime, mesh, "U");
    auto ofP = nf::randomScalarField(runTime, mesh, "p");
    auto ofPhi = nf::randDimField<Foam::surfaceScalarField>(mesh, {0, 3, -1, 0, 0}, "phi");
    auto ofGamma = nf::randDimField<Foam::surfaceScalarField>(mesh, {0, 2, -1, 0, 0}, "Gamma");

    SECTION("NeoN With Julia")
    {
        jl_init();

        // std::format would be nice and I implore you to try and get it to run on the cluster. 
        // Alas, I couldnt.
        std::string path = "include(\"{" + std::string(JULIA_MODULE_INIT) + "}\")";
        jl_eval_string(path.c_str());

        std::cout << "path: " << path << std::endl;
        auto exec = NeoN::Executor(NeoN::SerialExecutor {});
        auto rt = nf::createAdapterRunTime(runTime, exec);
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(schemesDict);
        auto& nfMesh = rt.mesh;
        auto& fieldCollection = fvcc::VectorCollection::instance(rt.db, "fieldCollection");
        auto& nfU = constructAndRegister(fieldCollection, rt, ofU);
        auto [nfP, nfPhi, nfGamma] =
            NeoFOAM::constFromMany(rt.exec, rt.nfMesh, ofP, ofPhi, ofGamma);
        
            SECTION(std::string("[NEON] SpatialOperators only"))
        {
            nf::PDESolver<NeoN::Vec3> eqn(
                dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfGamma, nfU),
                nfU,
                rt
            );
            BENCHMARK("CPUExecutor")
            {
                eqn.assemble();
                NeoN::fence(exec);
                return;
            };
        }
        SECTION(std::string("[JULIA] SpatialOperators, no explicit warmup"))
        {
            nf::PDESolver<NeoN::Vec3> juliaUEqn(
                dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfGamma, nfU),
                nfU,
                rt
            );
            BENCHMARK("CPUExecutor")
            {
                juliaUEqn.juliaFaceBased(nfPhi, nfU, nfGamma);
                return;
            };
        }
        jl_atexit_hook(0);
    }
}
