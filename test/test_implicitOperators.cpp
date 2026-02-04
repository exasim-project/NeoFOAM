// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

#include <cstddef>
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

#include "gaussConvectionScheme.H"


namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

TEST_CASE("matrix multiplication")
{
    float epsilon = 1e-32;
    Foam::Time& runTime = *timePtr;
    Foam::argList& args = *argsPtr;

    auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto rt = nf::createAdapterRunTime(runTime, exec);
    fvcc::VectorCollection& fieldCol = fvcc::VectorCollection::instance(rt.db, "VectorCollection");

    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    auto mi = NeoN::la::createSparsityPatternMatrixIterator<NeoN::localIdx>(rt.nfMesh);

    runTime.setDeltaT(1);

    SECTION("ddt_" + execName)
    {
        auto ofT = NeoFOAM::randomScalarField(runTime, mesh, "T");
        ofT.correctBoundaryConditions();

        auto& nfT = NeoFOAM::constructAndRegister(fieldCol, rt, ofT);
        auto& nfTOld = fvcc::oldTime(nfT);
        const auto nfTOldView = nfTOld.internalVector().view();

        NeoN::map(
            nfTOld.internalVector(),
            NEON_LAMBDA(const std::size_t celli) { return nfTOldView[celli] - 1.0; }
        );

        ofT.oldTime() -= Foam::dimensionedScalar("value", Foam::dimTemperature, 1);
        ofT.oldTime().correctBoundaryConditions();

        Foam::fvScalarMatrix matrix(Foam::fvm::ddt(ofT));
        Foam::volScalarField ddt("ddt", matrix & ofT);

        // we should get a uniform field with a value of 1
        fvcc::DdtOperator ddtOp(dsl::Operator::Type::Implicit, nfT);

        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(T)", std::string("BDF1"));
        rt.fvSchemesDict.insert("ddtSchemes", ddtSchemes);
        ddtOp.read(rt.fvSchemesDict);

        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(
            rt.nfMesh,
            mi.sparsityPattern(),
            mi.boundarySparsityPattern()
        );
        ddtOp.implicitOperation(ls, mi, runTime.value(), runTime.deltaTValue());

        // check rhs
        nf::compare(ls.rhs(), matrix.source(), ApproxScalar(epsilon));

        // check diag
        auto diag = ls.matrix().diag();
        nf::compare(diag, matrix.diag(), ApproxScalar(epsilon));
        auto diagHost = diag.copyToHost();

        auto result = NeoFOAM::applyOperator(ls, nfT);
        auto ddtV = ddt * mesh.V();
        nf::compare(result.internalVector(), ddtV(), ApproxScalar(epsilon));
    }

    SECTION("sourceterm_" + execName)
    {
        NeoN::scalar coeff = 2.0;
        auto ofT = NeoFOAM::randomScalarField(runTime, mesh, "T");
        auto nfT = NeoFOAM::constructFrom(exec, rt.nfMesh, ofT);

        NeoN::map(
            nfT.internalVector(),
            NEON_LAMBDA(const std::size_t celli) { return celli; }
        );
        auto coefficients = nfT;
        NeoN::fill(coefficients.internalVector(), coeff);
        fvcc::SourceTerm sourceTerm(dsl::Operator::Type::Implicit, coefficients, nfT);
        NeoN::Vector<NeoN::scalar> source(nfT.exec(), nfT.internalVector().size(), 0.0);
        sourceTerm.explicitOperation(source);

        auto sourceHost = source.copyToHost();

        auto nftHost = nfT.internalVector().copyToHost();
        for (size_t i = 0; i < sourceHost.size(); i++)
        {
            REQUIRE(sourceHost.view()[i] == coeff * nftHost.view()[i]);
        }

        // the sourceterm operator implicit
        auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(
            rt.nfMesh,
            mi.sparsityPattern(),
            mi.boundarySparsityPattern()
        );
        auto cellVolumes = rt.nfMesh.cellVolumes().copyToHost();
        sourceTerm.implicitOperation(ls, mi);

        // check diag
        auto diag = ls.matrix().diag();
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
        auto ofPhi = NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, Foam::dimless, "phi");
        Foam::fvScalarMatrix matrix(Foam::fvm::div(ofPhi, ofT));
        Foam::volScalarField divT("divT", matrix & ofT);
        auto divV = divT * mesh.V();

        auto [nfT, nfPhi] = NeoFOAM::constFromMany(exec, rt.nfMesh, ofT, ofPhi);
        auto nfPDE = NeoFOAM::PDESolver<NeoN::scalar>(
            NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::div(nfPhi, nfT)),
            nfT,
            rt
        );
        nfPDE.assemble();

        // diag and rhs differ from the foam matrix as openfoam does not added the boundary values
        // to the matrix therefore we only check the operator results
        auto result = NeoFOAM::applyOperator(nfPDE.linearSystem(), nfT);
        nf::compare(result.internalVector(), divV(), ApproxScalar(epsilon));

        nf::compare(nfPDE.linearSystem().matrix().diag(), matrix.diag(), ApproxScalar(epsilon));
        nf::compare(
            NeoN::la::upper(nfPDE.linearSystem().matrix()),
            matrix.upper(),
            ApproxScalar(epsilon)
        );
    }

    SECTION("laplacian_" + execName)
    {
        auto ofT = NeoFOAM::randomScalarField(runTime, mesh, "T");
        auto ofNuf = NeoFOAM::randDimField<Foam::surfaceScalarField>(mesh, Foam::dimless, "Gamma");
        Foam::fvScalarMatrix matrix(Foam::fvm::laplacian(ofNuf, ofT));
        Foam::volScalarField laplacian("laplacian", matrix & ofT);
        auto lapV = laplacian * mesh.V();

        auto [nfT, nfNuf] = NeoFOAM::constFromMany(exec, rt.nfMesh, ofT, ofNuf);
        auto nfPDE = NeoFOAM::PDESolver<NeoN::scalar>(
            NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::laplacian(nfNuf, nfT)),
            nfT,
            rt
        );
        nfPDE.assemble();

        // diag and rhs differ from the foam matrix as openfoam does not added the boundary values
        // to the matrix therefore we only check the operator results
        auto result = NeoFOAM::applyOperator(nfPDE.linearSystem(), nfT);
        nf::compare(result.internalVector(), lapV(), ApproxScalar(epsilon));
        nf::compare(nfPDE.linearSystem().matrix().diag(), matrix.diag(), ApproxScalar(epsilon));
        nf::compare(
            NeoN::la::upper(nfPDE.linearSystem().matrix()),
            matrix.upper(),
            ApproxScalar(epsilon)
        );
    }
}
