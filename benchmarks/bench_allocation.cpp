// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#define CATCH_CONFIG_RUNNER

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "common.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

#include "fvCFD.H"

extern Foam::Time* timePtr;
extern Foam::argList* argsPtr;
extern Foam::fvMesh* meshPtr;

// TEST_CASE("VectorAllocation")
// {
//     // Test different sizes to see scaling behavior
//     auto size = GENERATE(10, 100, 1000, 10000, 100000);

//     SECTION("OpenFOAM List (with init)")
//     {
//         BENCHMARK("OpenFOAM List init size=" + std::to_string(size))
//         {
//             Foam::List<Foam::scalar> data(size, 0.0);
//             return data[0];
//         };
//     }

//     SECTION("OpenFOAM List (no init)")
//     {
//         BENCHMARK("OpenFOAM List no-init size=" + std::to_string(size))
//         {
//             Foam::List<Foam::scalar> data(size);
//             return data[0];
//         };
//     }

//     SECTION("std::vector (with init)")
//     {
//         BENCHMARK("std::vector init size=" + std::to_string(size))
//         {
//             std::vector<double> data(size, 0.0);
//             return data[0];
//         };
//     }

//     SECTION("std::vector (no init)")
//     {
//         BENCHMARK("std::vector no-init size=" + std::to_string(size))
//         {
//             std::vector<double> data(size);
//             return data[0];
//         };
//     }

//     SECTION("Kokkos::View HostSpace (with init)")
//     {
//         BENCHMARK("Kokkos::View HostSpace init size=" + std::to_string(size))
//         {
//             Kokkos::View<double*, Kokkos::HostSpace> data("data", size);
//             Kokkos::deep_copy(data, 0.0);
//             return data(0);
//         };
//     }

//     SECTION("Kokkos::View HostSpace (no init)")
//     {
//         BENCHMARK("Kokkos::View HostSpace no-init size=" + std::to_string(size))
//         {
//             Kokkos::View<double*, Kokkos::HostSpace> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), size);
//             return data(0);
//         };
//     }

//     SECTION("NeoN SerialExecutor (with init)")
//     {
//         NeoN::SerialExecutor exec;
        
//         BENCHMARK("NeoN SerialExecutor init size=" + std::to_string(size))
//         {
//             NeoN::Vector<NeoN::scalar> data(exec, size, 0.0);
//             return data.size();
//         };
//     }

//     SECTION("NeoN SerialExecutor (no init)")
//     {
//         NeoN::SerialExecutor exec;
        
//         BENCHMARK("NeoN SerialExecutor no-init size=" + std::to_string(size))
//         {
//             NeoN::Vector<NeoN::scalar> data(exec, size);
//             return data.size();
//         };
//     }

//     SECTION("NeoN CPUExecutor (with init)")
//     {
//         NeoN::CPUExecutor exec;
        
//         BENCHMARK("NeoN CPUExecutor init size=" + std::to_string(size))
//         {
//             NeoN::Vector<NeoN::scalar> data(exec, size, 0.0);
//             return data.size();
//         };
//     }

//     SECTION("NeoN CPUExecutor (no init)")
//     {
//         NeoN::CPUExecutor exec;
        
//         BENCHMARK("NeoN CPUExecutor no-init size=" + std::to_string(size))
//         {
//             NeoN::Vector<NeoN::scalar> data(exec, size);
//             return data.size();
//         };
//     }
// }

TEST_CASE("VolScalarFieldAllocation")
{
    Foam::Time& runTime = *timePtr;

    SECTION("OpenFOAM")
    {
        std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
        Foam::fvMesh& mesh = *meshPtr;

        BENCHMARK("OpenFOAM volScalarField allocation")
        {
            Foam::volScalarField field(
                Foam::IOobject(
                    "testField",
                    runTime.timeName(),
                    mesh,
                    Foam::IOobject::NO_READ,
                    Foam::IOobject::NO_WRITE
                ),
                mesh,
                Foam::dimensionedScalar("zero", Foam::dimless, 0.0)
            );
            return field.size();
        };
    }

    SECTION("OpenFOAM - with calculated BCs")
    {
        std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
        Foam::fvMesh& mesh = *meshPtr;

        BENCHMARK("OpenFOAM volScalarField with calculated BCs")
        {
            Foam::volScalarField field(
                Foam::IOobject(
                    "testField",
                    runTime.timeName(),
                    mesh,
                    Foam::IOobject::NO_READ,
                    Foam::IOobject::NO_WRITE
                ),
                mesh,
                Foam::dimensionedScalar("zero", Foam::dimless, 0.0),
                Foam::calculatedFvPatchScalarField::typeName
            );
            return field.size();
        };
    }

    SECTION("NeoN - Full Field SerialExecutor")
    {
        NeoN::SerialExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();

        BENCHMARK("NeoN Full VolumeField SerialExecutor")
        {
            fvcc::VolumeField<NeoN::scalar> field(
                exec,
                "testField",
                nfMesh,
                fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(nfMesh)
            );
            return field.size();
        };
    }

    SECTION("NeoN - Full Field CPUExecutor")
    {
        NeoN::CPUExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();

        BENCHMARK("NeoN Full VolumeField CPUExecutor")
        {
            fvcc::VolumeField<NeoN::scalar> field(
                exec,
                "testField",
                nfMesh,
                fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(nfMesh)
            );
            return field.size();
        };
    }

    SECTION("NeoN - Just Internal Vector")
    {
        NeoN::SerialExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        
        const auto nCells = nfMesh.nCells();

        BENCHMARK("NeoN Just Internal Vector")
        {
            NeoN::Vector<NeoN::scalar> internalField(exec, nCells, 0.0);
            return internalField.size();
        };
    }

    SECTION("NeoN - Just BCs Creation")
    {
        NeoN::SerialExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();

        BENCHMARK("NeoN Just createCalculatedBCs")
        {
            auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(nfMesh);
            return bcs.size();
        };
    }

    SECTION("NeoN - Just Vector Reserve")
    {
        NeoN::SerialExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();

        BENCHMARK("NeoN Just std::vector reserve")
        {
            std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs;
            bcs.reserve(static_cast<std::size_t>(nfMesh.nBoundaries()));
            return bcs.size();
        };
    }

    SECTION("NeoN - Single BC Construction")
    {
        NeoN::SerialExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        
        NeoN::Dictionary patchDict({{"type", std::string("calculated")}});

        BENCHMARK("NeoN Single VolumeBoundary construction")
        {
            fvcc::VolumeBoundary<NeoN::scalar> bc(nfMesh, patchDict, 0);
            return bc.patchID();
        };
    }

    SECTION("NeoN - Internal + Boundary Vectors")
    {
        NeoN::SerialExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        
        const auto nCells = nfMesh.nCells();
        const auto nBoundaries = nfMesh.nBoundaries();
        const auto& offset = nfMesh.boundaryMesh().offset();

        BENCHMARK("NeoN Internal + Empty Boundary Vectors")
        {
            NeoN::Vector<NeoN::scalar> internalField(exec, nCells, 0.0);
            std::vector<NeoN::Vector<NeoN::scalar>> boundaryFields;
            for (NeoN::localIdx i = 0; i < nBoundaries; ++i)
            {
                auto patchSize = offset[i + 1] - offset[i];
                boundaryFields.emplace_back(exec, patchSize, 0.0);
            }
            return internalField.size();
        };
    }

    SECTION("NeoN - Just BoundaryData Construction")
    {
        NeoN::SerialExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        
        const auto nBoundaryFaces = nfMesh.nBoundaryFaces();
        const auto nBoundaries = nfMesh.nBoundaries();

        BENCHMARK("NeoN BoundaryData construction (6 vectors)")
        {
            NeoN::BoundaryData<NeoN::scalar> boundaryData(exec, nBoundaryFaces, nBoundaries);
            return boundaryData.nBoundaries();
        };
    }

    SECTION("NeoN - BoundaryData Copy")
    {
        NeoN::SerialExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        
        const auto nBoundaryFaces = nfMesh.nBoundaryFaces();
        const auto nBoundaries = nfMesh.nBoundaries();
        
        NeoN::BoundaryData<NeoN::scalar> original(exec, nBoundaryFaces, nBoundaries);

        BENCHMARK("NeoN BoundaryData copy (6 vector copies)")
        {
            NeoN::BoundaryData<NeoN::scalar> copy(exec, original);
            return copy.nBoundaries();
        };
    }

    SECTION("NeoN - Field Construction")
    {
        NeoN::SerialExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        
        const auto nCells = nfMesh.nCells();
        const auto nBoundaryFaces = nfMesh.nBoundaryFaces();
        const auto nBoundaries = nfMesh.nBoundaries();
        
        NeoN::Vector<NeoN::scalar> internalVec(exec, nCells, 0.0);
        NeoN::BoundaryData<NeoN::scalar> boundaryData(exec, nBoundaryFaces, nBoundaries);

        BENCHMARK("NeoN Field construction (copies internal + boundary)")
        {
            NeoN::Field<NeoN::scalar> field(exec, internalVec, boundaryData);
            return field.internalVector().size();
        };
    }

    SECTION("NeoN - Field with Offsets")
    {
        NeoN::SerialExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        
        const auto nCells = nfMesh.nCells();
        const auto& offsets = nfMesh.boundaryMesh().offset();
        
        BENCHMARK("NeoN Field(exec, nCells, offsets)")
        {
            NeoN::Field<NeoN::scalar> field(exec, nCells, offsets);
            return field.internalVector().size();
        };
    }

    SECTION("NeoN - DomainMixin Construction")
    {
        NeoN::SerialExecutor exec;
        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        
        const auto nCells = nfMesh.nCells();
        const auto& offsets = nfMesh.boundaryMesh().offset();
        
        BENCHMARK("NeoN DomainMixin construction (with move)")
        {
            fvcc::DomainMixin<NeoN::scalar> domain(
                exec,
                "testField",
                nfMesh,
                NeoN::Field<NeoN::scalar>(exec, nCells, offsets)
            );
            return domain.internalVector().size();
        };
    }
}

// TEST_CASE("SurfaceScalarFieldAllocation")
// {
//     Foam::Time& runTime = *timePtr;

//     SECTION("OpenFOAM")
//     {
//         std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
//         Foam::fvMesh& mesh = *meshPtr;

//         BENCHMARK("OpenFOAM surfaceScalarField allocation")
//         {
//             Foam::surfaceScalarField field(
//                 Foam::IOobject(
//                     "testField",
//                     runTime.timeName(),
//                     mesh,
//                     Foam::IOobject::NO_READ,
//                     Foam::IOobject::NO_WRITE
//                 ),
//                 mesh,
//                 Foam::dimensionedScalar("zero", Foam::dimless, 0.0)
//             );
//             return field.size();
//         };
//     }

//     SECTION("NeoN")
//     {
//         auto [execName, exec] = GENERATE(allAvailableExecutor());

//         std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
//         NeoFOAM::MeshAdapter& mesh = *meshPtr;
//         const auto& nfMesh = mesh.nfMesh();

//         BENCHMARK(std::string(execName) + " SurfaceField<scalar> allocation")
//         {
//             fvcc::SurfaceField<NeoN::scalar> field(
//                 exec,
//                 "testField",
//                 nfMesh,
//                 fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(nfMesh)
//             );
//             return field.size();
//         };
//     }
// }

// TEST_CASE("BoundaryConditionsAllocation")
// {
//     Foam::Time& runTime = *timePtr;

//     SECTION("NeoN createCalculatedBCs")
//     {
//         auto [execName, exec] = GENERATE(allAvailableExecutor());

//         std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
//         NeoFOAM::MeshAdapter& mesh = *meshPtr;
//         const auto& nfMesh = mesh.nfMesh();

//         BENCHMARK(std::string(execName) + " createCalculatedBCs")
//         {
//             auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(nfMesh);
//             return bcs.size();
//         };
//     }
// }

// TEST_CASE("CompleteFieldWithBCAllocation")
// {
//     Foam::Time& runTime = *timePtr;

//     SECTION("OpenFOAM surfaceScalarField with BCs")
//     {
//         std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
//         Foam::fvMesh& mesh = *meshPtr;

//         BENCHMARK("OpenFOAM surfaceScalarField + BCs")
//         {
//             Foam::surfaceScalarField field(
//                 Foam::IOobject(
//                     "testField",
//                     runTime.timeName(),
//                     mesh,
//                     Foam::IOobject::NO_READ,
//                     Foam::IOobject::NO_WRITE
//                 ),
//                 mesh,
//                 Foam::dimensionedScalar("zero", Foam::dimless, 0.0),
//                 Foam::calculatedFvsPatchScalarField::typeName
//             );
//             return field.size();
//         };
//     }

//     SECTION("NeoN SurfaceField with createCalculatedBCs")
//     {
//         auto [execName, exec] = GENERATE(allAvailableExecutor());

//         std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
//         NeoFOAM::MeshAdapter& mesh = *meshPtr;
//         const auto& nfMesh = mesh.nfMesh();

//         BENCHMARK(std::string(execName) + " SurfaceField + createCalculatedBCs")
//         {
//             fvcc::SurfaceField<NeoN::scalar> field(
//                 exec,
//                 "testField",
//                 nfMesh,
//                 fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(nfMesh)
//             );
//             return field.size();
//         };
//     }

//     SECTION("NeoN SurfaceField with pre-allocated BCs")
//     {
//         auto [execName, exec] = GENERATE(allAvailableExecutor());

//         std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
//         NeoFOAM::MeshAdapter& mesh = *meshPtr;
//         const auto& nfMesh = mesh.nfMesh();

//         // Pre-allocate BCs once
//         auto bcs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(nfMesh);

//         BENCHMARK(std::string(execName) + " SurfaceField with cached BCs")
//         {
//             fvcc::SurfaceField<NeoN::scalar> field(
//                 exec,
//                 "testField",
//                 nfMesh,
//                 bcs
//             );
//             return field.size();
//         };
//     }
// }
