// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

#include <cstddef>
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

#include "gaussConvectionScheme.H"


namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

TEST_CASE("matrix multiplication")
{
    Foam::Time& runTime = *timePtr;
    Foam::argList& args = *argsPtr;

    NeoN::Database db;
    fvcc::VectorCollection& fieldCol = fvcc::VectorCollection::instance(db, "VectorCollection");

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    auto nfMesh = mesh.nfMesh();
    const auto sparsityPattern = NeoN::la::createSparsity(nfMesh);

    runTime.setDeltaT(1);

    SECTION("ddt_" + execName)
    {
        auto ofT = NeoFOAM::randomScalarField(runTime, mesh, "T");
        ofT.correctBoundaryConditions();

        fvcc::VolumeField<NeoN::scalar>& nfT =
            fieldCol.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                NeoFOAM::CreateFromFoamField<Foam::volScalarField> {
                    .exec = exec,
                    .nfMesh = nfMesh,
                    .foamField = ofT,
                    .name = "nfT"
                }
            );
        auto& nfTOld = fvcc::oldTime(nfT);
        const auto nfTOldView = nfTOld.internalVector().view();
        NeoN::map(
            nfTOld.internalVector(),
            KOKKOS_LAMBDA(const std::size_t celli) { return nfTOldView[celli] - 1.0; }
        );

        ofT.oldTime() -= Foam::dimensionedScalar("value", Foam::dimTemperature, 1);
        ofT.oldTime().correctBoundaryConditions();

        Foam::fvScalarMatrix matrix(Foam::fvm::ddt(ofT));
        Foam::volScalarField ddt(
            "ddt",
            matrix & ofT
        ); // we should get a uniform field with a value of 1
        // ddt.write();
        fvcc::DdtOperator ddtOp(dsl::Operator::Type::Implicit, nfT);

        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar, NeoN::localIdx>(
            nfMesh,
            sparsityPattern
        );
        ddtOp.implicitOperation(ls, runTime.value(), runTime.deltaTValue());

        // check rhs
        auto rhs = ls.rhs().copyToHost();
        auto rhsView = rhs.view();
        for (size_t celli = 0; celli < rhsView.size(); celli++)
        {
            REQUIRE(rhsView[celli] == Catch::Approx(matrix.source()[celli]).margin(1e-16));
        }

        // check diag
        auto diag = NeoFOAM::diag(ls, sparsityPattern);
        auto diagHost = diag.copyToHost();

        for (size_t celli = 0; celli < diagHost.size(); celli++)
        {
            REQUIRE(diagHost.view()[celli] == Catch::Approx(matrix.diag()[celli]).margin(1e-16));
        }

        auto result = NeoFOAM::applyOperator(ls, nfT);
        auto resultHost = result.internalVector().copyToHost();
        for (size_t celli = 0; celli < resultHost.size(); celli++)
        {
            REQUIRE(
                resultHost.view()[celli]
                == Catch::Approx(ddt[celli] * mesh.V()[celli]).margin(1e-16)
            );
        }
    }

    SECTION("sourceterm_" + execName)
    {
        NeoN::scalar coeff = 2.0;
        auto ofT = NeoFOAM::randomScalarField(runTime, mesh, "T");
        fvcc::VolumeField<NeoN::scalar> nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);

        NeoN::map(
            nfT.internalVector(),
            KOKKOS_LAMBDA(const std::size_t celli) { return celli; }
        );
        auto coefficients = nfT;
        NeoN::fill(coefficients.internalVector(), coeff);
        fvcc::SourceTerm sourceTerm(dsl::Operator::Type::Implicit, coefficients, nfT);
        NeoN::Vector<NeoN::scalar> source(nfT.exec(), nfT.internalVector().size(), 0.0);
        sourceTerm.explicitOperation(source);

        auto sourceHost = source.copyToHost();
        ;
        auto nftHost = nfT.internalVector().copyToHost();
        for (size_t i = 0; i < sourceHost.size(); i++)
        {
            REQUIRE(sourceHost.view()[i] == coeff * nftHost.view()[i]);
        }

        // the sourceterm operator implicit
        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar, NeoN::localIdx>(
            nfMesh,
            sparsityPattern
        );
        auto cellVolumes = nfMesh.cellVolumes().copyToHost();
        sourceTerm.implicitOperation(ls);

        // check diag
        auto diag = NeoFOAM::diag(ls, sparsityPattern);
        auto diagHost = diag.copyToHost();

        for (size_t celli = 0; celli < diagHost.size(); celli++)
        {
            REQUIRE(diagHost.view()[celli] == coeff * cellVolumes.view()[celli]);
        }

        auto result = NeoFOAM::applyOperator(ls, nfT);
        auto resultHost = result.internalVector().copyToHost();
        for (size_t celli = 0; celli < resultHost.size(); celli++)
        {
            REQUIRE(
                resultHost.view()[celli]
                == nftHost.view()[celli] * coeff * cellVolumes.view()[celli]
            );
        }
    }

    SECTION("div_" + execName)
    {
        auto ofT = NeoFOAM::randomScalarField(runTime, mesh, "T");
        forAll(ofT, celli)
        {
            ofT[celli] = celli;
        }
        ofT.correctBoundaryConditions();

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
            Foam::dimensionedScalar("phi", Foam::dimless, 0.1)
        );
        forAll(ofPhi, facei)
        {
            ofPhi[facei] = 1;
        }

        auto nfPhi = NeoFOAM::constructFrom(exec, nfMesh, ofPhi);

        Foam::fvScalarMatrix matrix(Foam::fvm::div(ofPhi, ofT));
        Foam::volScalarField divT("divT", matrix & ofT);

        NeoN::TokenList input = {std::string("Gauss"), std::string("linear")};
        fvcc::DivOperator<NeoN::scalar> divOp(dsl::Operator::Type::Implicit, nfPhi, nfT, input);

        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar, NeoN::localIdx>(
            nfMesh,
            sparsityPattern
        );
        divOp.implicitOperation(ls);

        // diag and rhs differ from the foam matrix as openfoam does not added the boundary values
        // to the matrix therefore we only check the operator results

        auto result = NeoFOAM::applyOperator(ls, nfT);
        auto resultHost = result.internalVector().copyToHost();
        for (size_t celli = 0; celli < resultHost.size(); celli++)
        {
            REQUIRE(
                resultHost.view()[celli]
                == Catch::Approx(divT[celli] * mesh.V()[celli]).margin(1e-14)
            );
        }
    }

    SECTION("laplacian_" + execName)
    {
        auto ofT = NeoFOAM::randomScalarField(runTime, mesh, "T");
        forAll(ofT, celli)
        {
            ofT[celli] = celli;
        }
        ofT.correctBoundaryConditions();

        auto nfT = NeoFOAM::constructFrom(exec, nfMesh, ofT);
        nfT.correctBoundaryConditions();

        Foam::surfaceScalarField ofNuf(
            Foam::IOobject(
                "ofNuf",
                runTime.timeName(),
                mesh,
                Foam::IOobject::NO_READ,
                Foam::IOobject::AUTO_WRITE
            ),
            mesh,
            Foam::dimensionedScalar("phi", Foam::dimless, 0.1)
        );

        auto nfNuf = NeoFOAM::constructFrom(exec, nfMesh, ofNuf);

        Foam::fvScalarMatrix matrix(Foam::fvm::laplacian(ofNuf, ofT));
        Foam::volScalarField laplacian("laplacian", matrix & ofT);

        NeoN::TokenList input =
            {std::string("Gauss"), std::string("linear"), std::string("uncorrected")};
        fvcc::LaplacianOperator<NeoN::scalar>
            laplacianOp(dsl::Operator::Type::Implicit, nfNuf, nfT, input);

        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar, NeoN::localIdx>(
            nfMesh,
            sparsityPattern
        );
        laplacianOp.implicitOperation(ls);

        // diag and rhs differ from the foam matrix as openfoam does not added the boundary values
        // to the matrix therefore we only check the operator results

        auto result = NeoFOAM::applyOperator(ls, nfT);
        auto resultHost = result.internalVector().copyToHost();
        for (size_t celli = 0; celli < resultHost.size(); celli++)
        {
            REQUIRE(
                resultHost.view()[celli]
                == Catch::Approx(laplacian[celli] * mesh.V()[celli]).margin(1e-14)
            );
        }
    }
}
