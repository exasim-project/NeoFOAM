// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"
#include "../test/common.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;
namespace nf = NeoFOAM;

#include "fv.H"
#include "fvc.H"
#include "gaussGrad.H"
#include "gaussConvectionScheme.H"
#include "gaussLaplacianScheme.H"


TEST_CASE("DivOperator")
{
    Foam::Time& runTime = *timePtr;
    Foam::argList& args = *argsPtr;
    std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
    Foam::fvMesh& mesh = *meshPtr;

    auto ofT = nf::randomScalarField(mesh, "T");
    auto ofPhi = nf::randDimField<Foam::surfaceScalarField>(mesh, {0, 3, -1, 0, 0}, "phi");

    SECTION("OpenFOAM")
    {
        SECTION("with Allocation")
        {
            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::IStringStream is("linear");
                Foam::fv::gaussConvectionScheme<Foam::scalar> foamDivScalar(mesh, ofPhi, is);
                auto fvmDivT = foamDivScalar.fvmDiv(ofPhi, ofT);
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
        auto [nfT, nfPhi] = NeoFOAM::constFromMany(exec, nfMesh, ofT, ofPhi);
        NeoN::TokenList scheme({std::string("linear")});
        auto mi = NeoN::la::createSparsityPatternMatrixIterator<NeoN::localIdx>(nfMesh);

        SECTION("with Allocation")
        {
            BENCHMARK(std::string(execName))
            {
                auto ls = la::createEmptyLinearSystem<
                    NeoN::scalar,
                    NeoN::la::SparsityPattern<NeoN::localIdx>>(
                    nfMesh,
                    mi.sparsityPattern(),
                    mi.boundarySparsityPattern()
                );
                fvcc::GaussGreenDiv<NeoN::scalar>(exec, nfMesh, scheme)
                    .div(ls, mi, nfPhi, nfT, NeoN::dsl::Coeff(1.0));
                NeoN::fence(exec);
                return;
            };
        }

        SECTION("No allocation")
        {
            auto ls = la::
                createEmptyLinearSystem<NeoN::scalar, NeoN::la::SparsityPattern<NeoN::localIdx>>(
                    nfMesh,
                    mi.sparsityPattern(),
                    mi.boundarySparsityPattern()
                );
            NeoN::TokenList scheme({std::string("linear")});

            BENCHMARK(std::string(execName))
            {
                NeoN::fill(ls.matrix().values(), 0.0);
                NeoN::fill(ls.rhs(), 0.0);
                fvcc::GaussGreenDiv<NeoN::scalar>(exec, nfMesh, scheme)
                    .div(ls, mi, nfPhi, nfT, dsl::Coeff(1.0));
                NeoN::fence(exec);
                return;
            };
        }

        // TODO dsl
    }
}


TEST_CASE("LaplacianOperator")
{
    Foam::Time& runTime = *timePtr;
    Foam::argList& args = *argsPtr;

    std::unique_ptr<Foam::fvMesh> meshPtr = NeoFOAM::createMesh(runTime);
    Foam::fvMesh& mesh = *meshPtr;

    auto ofT = nf::randomScalarField(mesh, "T");
    auto ofGamma = nf::randDimField<Foam::surfaceScalarField>(mesh, {0, 2, -1, 0, 0}, "Gamma");

    SECTION("OpenFOAM")
    {
        SECTION("with Allocation")
        {
            BENCHMARK(std::string("OpenFOAM"))
            {
                Foam::IStringStream is("linear uncorrected");
                Foam::fv::gaussLaplacianScheme<Foam::scalar, Foam::scalar> foamLapScalar(mesh, is);
                auto fvmLapT = foamLapScalar.fvmLaplacian(ofGamma, ofT);
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
        auto mi = NeoN::la::createSparsityPatternMatrixIterator<NeoN::localIdx>(nfMesh);
        auto [nfT, nfGamma] = NeoFOAM::constFromMany(exec, nfMesh, ofT, ofGamma);
        NeoN::TokenList scheme({std::string("linear"), std::string("uncorrected")});

        SECTION("with Allocation")
        {
            BENCHMARK(std::string(execName))
            {
                auto mi = NeoN::la::createSparsityPatternMatrixIterator<NeoN::localIdx>(nfMesh);
                auto ls = NeoN::la::createEmptyLinearSystem<
                    NeoN::scalar,
                    NeoN::la::SparsityPattern<NeoN::localIdx>>(
                    nfMesh,
                    mi.sparsityPattern(),
                    mi.boundarySparsityPattern()
                );
                fvcc::GaussGreenLaplacian<NeoN::scalar>(exec, nfMesh, scheme)
                    .laplacian(ls, mi, nfGamma, nfT, dsl::Coeff(1.0));
                NeoN::fence(exec);
                return;
            };
        }

        SECTION("No allocation")
        {
            auto ls = NeoN::la::
                createEmptyLinearSystem<NeoN::scalar, NeoN::la::SparsityPattern<NeoN::localIdx>>(
                    nfMesh,
                    mi.sparsityPattern(),
                    mi.boundarySparsityPattern()
                );
            NeoN::TokenList scheme({std::string("linear"), std::string("uncorrected")});

            BENCHMARK(std::string(execName))
            {
                NeoN::fill(ls.matrix().values(), 0.0);
                NeoN::fill(ls.rhs(), 0.0);
                fvcc::GaussGreenLaplacian<NeoN::scalar>(exec, nfMesh, scheme)
                    .laplacian(ls, mi, nfGamma, nfT, dsl::Coeff(1.0));
                NeoN::fence(exec);
                return;
            };
        }

        // TODO dsl
    }
}
