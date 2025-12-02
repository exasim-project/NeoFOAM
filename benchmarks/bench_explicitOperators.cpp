// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "common.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;

#include "fv.H"
#include "fvc.H"
#include "gaussGrad.H"
#include "gaussConvectionScheme.H"
#include "gaussLaplacianScheme.H"

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

TEST_CASE("DivOperator")
{
    Foam::Time& runTime = *timePtr;

    SECTION("OpenFOAM")
    {
        std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
        Foam::fvMesh& mesh = *meshPtr;

        auto ofT = randomScalarField(runTime, mesh, "T");
        Foam::surfaceScalarField ofPhi(
            Foam::IOobject(
                "phi",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::AUTO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar("phi", Foam::dimless, 0.0)
        );
        forAll(ofPhi, facei)
        {
            ofPhi[facei] = facei;
        }
        SECTION("with Allocation")
        {
            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::IStringStream is("linear");
                Foam::fv::gaussConvectionScheme<Foam::scalar> foamDivScalar(mesh, ofPhi, is);
                Foam::volScalarField ofDivT("ofDivT", foamDivScalar.fvcDiv(ofPhi, ofT));
                return;
            };
        }
    }


    SECTION("NeoN")
    {
        auto [execName, exec] = GENERATE(allAvailableExecutor());

        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        // linear interpolation hardcoded for now


        auto ofT = randomScalarField(runTime, mesh, "T");
        auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);
        nfT.correctBoundaryConditions();

        Foam::surfaceScalarField ofPhi(
            Foam::IOobject(
                "phi",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::AUTO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar("phi", Foam::dimless, 0.0)
        );
        forAll(ofPhi, facei)
        {
            ofPhi[facei] = facei;
        }

        auto nfPhi = NeoFOAM::constructSurfaceField(exec, nfMesh, ofPhi);

        SECTION("with Allocation")
        {
            NeoN::TokenList scheme({std::string("linear")});

            BENCHMARK(std::string(execName))
            {
                fvcc::VolumeField<NeoN::scalar> divPhiT =
                    fvcc::GaussGreenDiv<NeoN::scalar>(exec, nfMesh, scheme)
                        .div(nfPhi, nfT, NeoN::dsl::Coeff(1.0));
                return;
            };
        }

        SECTION("No allocation")
        {
            auto nfDivT = NeoFOAM::constructFrom(exec, nfMesh, ofT);
            NeoN::TokenList scheme({std::string("linear")});

            BENCHMARK(std::string(execName))
            {
                NeoN::fill(nfDivT.internalVector(), 0.0);
                NeoN::fill(nfDivT.boundaryData().value(), 0.0);
                fvcc::GaussGreenDiv<NeoN::scalar>(exec, nfMesh, scheme)
                    .div(nfDivT, nfPhi, nfT, dsl::Coeff(1.0));
                if (execName == "GPUExecutor")
                {
                    Kokkos::fence();
                }
                return;
            };
        }

        // TODO: dsl
    }
}


TEST_CASE("LaplacianOperator")
{
    Foam::Time& runTime = *timePtr;

    SECTION("OpenFOAM")
    {
        std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
        Foam::fvMesh& mesh = *meshPtr;

        auto ofT = randomScalarField(runTime, mesh, "T");
        Foam::surfaceScalarField ofGamma(
            Foam::IOobject(
                "Gamma",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::AUTO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar("Gamma", Foam::dimless, 1.0)
        );

        SECTION("with Allocation")
        {
            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::IStringStream is("linear uncorrected");
                Foam::fv::gaussLaplacianScheme<Foam::scalar, Foam::scalar> foamLapScalar(mesh, is);
                Foam::volScalarField ofLapT("ofLapT", foamLapScalar.fvcLaplacian(ofGamma, ofT));
                return;
            };
        }
    }


    SECTION("NeoN")
    {
        auto [execName, exec] = GENERATE(allAvailableExecutor());

        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();

        auto ofT = randomScalarField(runTime, mesh, "T");
        auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);
        nfT.correctBoundaryConditions();

        Foam::surfaceScalarField ofGamma(
            Foam::IOobject(
                "Gamma",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::AUTO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar("Gamma", Foam::dimless, 1.0)
        );

        auto nfGamma = NeoFOAM::constructSurfaceField(exec, nfMesh, ofGamma);

        SECTION("with Allocation")
        {
            NeoN::TokenList scheme({std::string("linear"), std::string("uncorrected")});

            BENCHMARK(std::string(execName))
            {
                fvcc::VolumeField<NeoN::scalar> lapT =
                    fvcc::GaussGreenLaplacian<NeoN::scalar>(exec, nfMesh, scheme)
                        .laplacian(nfGamma, nfT, dsl::Coeff(1.0));
                return;
            };
        }

        SECTION("No allocation")
        {
            auto nfLapT = NeoFOAM::constructFrom(exec, nfMesh, ofT);
            NeoN::TokenList scheme({std::string("linear"), std::string("uncorrected")});

            BENCHMARK(std::string(execName))
            {
                NeoN::fill(nfLapT.internalVector(), 0.0);
                NeoN::fill(nfLapT.boundaryData().value(), 0.0);
                fvcc::GaussGreenLaplacian<NeoN::scalar>(exec, nfMesh, scheme)
                    .laplacian(nfLapT, nfGamma, nfT, dsl::Coeff(1.0));
                if (execName == "GPUExecutor")
                {
                    Kokkos::fence();
                }
                return;
            };
        }

        // TODO: dsl
    }
}

TEST_CASE("GradOperator")
{
    Foam::Time& runTime = *timePtr;

    SECTION("OpenFOAM")
    {
        std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
        Foam::fvMesh& mesh = *meshPtr;

        auto ofT = randomScalarField(runTime, mesh, "T");

        SECTION("with Allocation")
        {
            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::IStringStream is("linear");
                Foam::fv::gaussGrad<Foam::scalar> foamGrad(mesh, is);
                Foam::volVectorField ofGradT("ofGradT", foamGrad.calcGrad(ofT, "ofGradT"));
                return;
            };
        }
    }


    SECTION("NeoN")
    {
        auto [execName, exec] = GENERATE(allAvailableExecutor());

        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        // linear interpolation hardcoded for now


        auto ofT = randomScalarField(runTime, mesh, "T");
        auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);

        SECTION("with Allocation")
        {
            NeoN::TokenList scheme({std::string("linear")});

            BENCHMARK(std::string(execName))
            {
                fvcc::VolumeField<NeoN::Vec3> nfGradT =
                    fvcc::GaussGreenGrad(exec, nfMesh).grad(nfT);
                return;
            };
        }

        SECTION("No allocation")
        {
            fvcc::VolumeField<NeoN::Vec3> nfGradT = fvcc::GaussGreenGrad(exec, nfMesh).grad(nfT);

            BENCHMARK(std::string(execName))
            {
                NeoN::fill(nfGradT.internalVector(), NeoN::Vec3(0, 0, 0));
                NeoN::fill(nfGradT.boundaryData().value(), NeoN::Vec3(0, 0, 0));
                fvcc::GaussGreenGrad(exec, nfMesh).grad(nfT, NeoN::dsl::Coeff(), nfGradT);
                if (execName == "GPUExecutor")
                {
                    Kokkos::fence();
                }
                return;
            };
        }

        // TODO: dsl
    }
}


TEST_CASE("FaceInterpolation")
{
    Foam::Time& runTime = *timePtr;

    SECTION("OpenFOAM")
    {
        std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
        Foam::fvMesh& mesh = *meshPtr;

        auto ofT = randomScalarField(runTime, mesh, "T");
        Foam::surfaceScalarField ofPhi(
            Foam::IOobject(
                "phi",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::AUTO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar("phi", Foam::dimless, 0.0)
        );
        for (auto facei = 0; facei < ofPhi.size(); facei++)
        {
            ofPhi[facei] = facei;
        }

        SECTION("with Allocation")
        {
            // warmup
            Foam::IStringStream istmp("linear");
            Foam::fvc::interpolate(ofT, ofPhi, istmp);

            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::IStringStream is("linear");
                auto Tf = Foam::fvc::interpolate(ofT, ofPhi, is);
                return;
            };
        }
    }


    SECTION("NeoN")
    {
        auto [execName, exec] = GENERATE(allAvailableExecutor());

        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        // linear interpolation hardcoded for now

        auto ofT = randomScalarField(runTime, mesh, "T");
        auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);
        nfT.correctBoundaryConditions();

        Foam::surfaceScalarField ofPhi(
            Foam::IOobject(
                "phi",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::AUTO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar("phi", Foam::dimless, 0.0)
        );
        for (auto facei = 0; facei < ofPhi.size(); facei++)
        {
            ofPhi[facei] = facei;
        }

        auto nfPhi = NeoFOAM::constructSurfaceField(exec, nfMesh, ofPhi);

        SECTION("with Allocation")
        {
            NeoN::TokenList scheme({std::string("linear")});

            // warmup
            fvcc::SurfaceInterpolation<NeoN::scalar>(exec, nfMesh, scheme).interpolate(nfPhi, nfT);

            BENCHMARK(std::string(execName))
            {
                fvcc::SurfaceField<NeoN::scalar> Tf =
                    fvcc::SurfaceInterpolation<NeoN::scalar>(exec, nfMesh, scheme)
                        .interpolate(nfPhi, nfT);
                return;
            };
        }
        SECTION("No allocation")
        {
            NeoN::TokenList scheme({std::string("linear")});

            fvcc::SurfaceField<NeoN::scalar> nfTf(nfPhi);
            nfTf.name = "Tf";

            BENCHMARK(std::string(execName))
            {
                fvcc::SurfaceInterpolation<NeoN::scalar>(exec, nfMesh, scheme)
                    .interpolate(nfPhi, nfT, nfTf);
                if (execName == "GPUExecutor")
                {
                    Kokkos::fence();
                }
                return;
            };
        }

        // TODO: dsl
    }
}


TEST_CASE("FaceNormalGradient")
{
    Foam::Time& runTime = *timePtr;

    SECTION("OpenFOAM")
    {
        std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
        Foam::fvMesh& mesh = *meshPtr;

        auto ofT = randomScalarField(runTime, mesh, "T");

        SECTION("with Allocation")
        {
            // warmup
            Foam::IStringStream istmp("uncorrected");
            auto tmp = Foam::fv::snGradScheme<Foam::scalar>::New(mesh, istmp);
            tmp->snGrad(ofT);

            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::IStringStream is("uncorrected");
                auto snGradScheme = Foam::fv::snGradScheme<Foam::scalar>::New(mesh, is);
                snGradScheme->snGrad(ofT);
                return;
            };
        }
    }


    SECTION("NeoN")
    {
        auto [execName, exec] = GENERATE(allAvailableExecutor());

        std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, runTime);
        NeoFOAM::MeshAdapter& mesh = *meshPtr;
        const auto& nfMesh = mesh.nfMesh();
        // linear interpolation hardcoded for now

        auto ofT = randomScalarField(runTime, mesh, "T");
        auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);
        nfT.correctBoundaryConditions();

        SECTION("with Allocation")
        {
            NeoN::TokenList scheme({std::string("uncorrected")});
            // warmup
            fvcc::FaceNormalGradient<NeoN::scalar>(exec, nfMesh, scheme).faceNormalGrad(nfT);

            BENCHMARK(std::string(execName))
            {
                // fvcc::SurfaceField<NeoN::scalar> Tf =
                fvcc::FaceNormalGradient<NeoN::scalar>(exec, nfMesh, scheme).faceNormalGrad(nfT);
                return;
            };
        }
        SECTION("No allocation")
        {
            NeoN::TokenList scheme({std::string("uncorrected")});

            std::string nameFaceGrad = "faceGrad_" + nfT.name;
            fvcc::SurfaceField<NeoN::scalar> faceGradT(
                exec,
                nameFaceGrad,
                nfMesh,
                fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(nfMesh)
            );

            BENCHMARK(std::string(execName))
            {
                fvcc::FaceNormalGradient<NeoN::scalar>(exec, nfMesh, scheme)
                    .faceNormalGrad(nfT, faceGradT);
                if (execName == "GPUExecutor")
                {
                    Kokkos::fence();
                }
                return;
            };
        }

        // TODO: dsl
    }
}
