// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER
#include "common.hpp"

#include "fv.H"
#include "fvMatrix.H"
#include "gaussConvectionScheme.H"
#include "fvmLaplacian.H"

#include "NeoFOAM/NeoFOAM.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl  = NeoN::dsl;
using Foam::label;
using NeoN::localIdx;

extern Foam::Time*    timePtr;
extern Foam::argList* argsPtr;
extern Foam::fvMesh*  meshPtr;

TEST_CASE("MatrixRelaxation_div_laplacian")
{
    Foam::Time& runTime = *timePtr;
    const NeoN::scalar t  = runTime.value();
    const NeoN::scalar dt = runTime.deltaTValue();

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    INFO("Executor = " << execName);

    auto meshAdapter = NeoFOAM::createMesh(exec, runTime);
    auto& mesh = *meshAdapter;
    auto& nfMesh = mesh.nfMesh();

    // ------------------------------------------------------------------
    // Fields
    // ------------------------------------------------------------------
    Foam::volScalarField ofT =
        randomScalarField(runTime, mesh, "T");

    Foam::surfaceScalarField ofPhi
    (
        Foam::IOobject
        (
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("phi", Foam::dimVolume/Foam::dimTime, 1.0)
    );

    Foam::surfaceScalarField ofGamma
    (
        Foam::IOobject
        (
            "ofGamma",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("ofGamma", Foam::dimArea/Foam::dimTime, 1.0)
    );

    auto nfT   = NeoFOAM::constructFrom(exec, nfMesh, ofT);
    auto nfPhi = NeoFOAM::constructFrom(exec, nfMesh, ofPhi);
    auto nfGamma = NeoFOAM::constructFrom(exec, nfMesh, ofGamma);

    // ------------------------------------------------------------------
    // OpenFOAM assembly
    // ------------------------------------------------------------------
    Foam::IStringStream is("linear");
    Foam::fv::gaussConvectionScheme<Foam::scalar> divScheme(mesh, ofPhi, is);

    Foam::fvMatrix<Foam::scalar> ofEqn
    (
     //     divScheme.fvmDiv(ofPhi, ofT)
          Foam::fvm::div(ofPhi, ofT)
     //   - Foam::fvm::laplacian(ofGamma, ofT)
    );

    // Copy diagonal before relaxation
    const auto& lAddr = mesh.lduAddr().lowerAddr();
    const auto& uAddr = mesh.lduAddr().upperAddr();
    Foam::scalarField ofLower = ofEqn.lower();
    Foam::scalarField ofUpper = ofEqn.upper();
    Foam::scalarField ofDiag0 = ofEqn.diag();

    // Apply under-relaxation
    const scalar alpha = 0.7;
//    ofEqn.relax(alpha);

//    Foam::scalarField ofDiag = ofEqn.diag();
//    Foam::scalarField ofRhs  = ofEqn.source();

    // ------------------------------------------------------------------
    // NeoN assembly - use new API
    // ------------------------------------------------------------------
    auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(nfMesh);
    const auto matIt = ls.faceToMatrixAddress();

    // implicit div
    dsl::Expression<NeoN::scalar> expr(exec);
    NeoN::TokenList divSchemeTokens = NeoN::TokenList({std::string("Gauss"), std::string("linear")});

    // build implicit div with scheme
    auto divOp = dsl::imp::div(nfPhi, nfT);
    divOp.read(divSchemeTokens);
    expr.addOperator(std::move(divOp));

    // laplacian may also need a scheme in your framework; if it asserts similarly, do the same:
    NeoN::TokenList lapSchemeTokens = NeoN::TokenList({std::string("Gauss"), std::string("linear"), std::string("uncorrected")}); // or your laplacian token set
    auto lapOp = dsl::imp::laplacian(nfGamma, nfT);
    lapOp.read(lapSchemeTokens);
    //expr.addOperator(-1.0 * std::move(lapOp));

    expr.assemble(t, dt, ls);

    auto& matrix = ls.matrix();
    auto& rhs    = ls.rhs();

    // Snapshot diagonal before relaxation using new API
    auto diagIdxArr = matIt->diagOffset().copyToHost();
    auto rowOffsArr = matrix.rowOffs().copyToHost();
    auto ownOffsArr = matIt->ownerOffset().copyToHost();
    auto neiOffsArr = matIt->neighbourOffset().copyToHost();
    auto matVals = matrix.values().copyToHost();

    auto matValsV  = matVals.view();
    auto rowOffsV  = rowOffsArr.view();
    auto diagIdxV  = diagIdxArr.view();
    auto ownOffsV  = ownOffsArr.view();
    auto neiOffsV  = neiOffsArr.view();

    for (label facei = 0; facei < mesh.nInternalFaces(); ++facei)
{
    label own = lAddr[facei];
    label nei = uAddr[facei];

    // -----------------------------
    // OpenFOAM coefficients
    // -----------------------------
    scalar ofLowerVal = ofLower[facei];
    scalar ofUpperVal = ofUpper[facei];

    // -----------------------------
    // NeoN coefficients
    // -----------------------------
    scalar neonLower =
        matValsV[rowOffsV[own] + ownOffsV[facei]];

    scalar neonUpper =
        matValsV[rowOffsV[nei] + neiOffsV[facei]];

    // -----------------------------
    // Diagonal (checked once per cell)
    // -----------------------------
    scalar ofDiagOwn  = ofDiag0[own];
    scalar ofDiagNei  = ofDiag0[nei];

    scalar neonDiagOwn =
        matValsV[rowOffsV[own] + diagIdxV[own]];

    scalar neonDiagNei =
        matValsV[rowOffsV[nei] + diagIdxV[nei]];

    // -----------------------------
    // Logging
    // -----------------------------
    NeoN::Logging::info(
        "face {} | own {} nei {} | "
        "OF: L={} U={} | "
        "NeoN: L={} U={}",
        facei, own, nei,
        ofLowerVal, ofUpperVal,
        neonLower, neonUpper
    );

    NeoN::Logging::info(
        "diag own {} | OF={} NeoN={}",
        own, ofDiagOwn, neonDiagOwn
    );

    NeoN::Logging::info(
        "diag nei {} | OF={} NeoN={}",
        nei, ofDiagNei, neonDiagNei
    );

    // -----------------------------
    // Assertions (tight, meaningful)
    // -----------------------------
    REQUIRE(neonLower == Catch::Approx(-ofLowerVal).margin(1e-12));
    REQUIRE(neonUpper == Catch::Approx(-ofUpperVal).margin(1e-12));

    REQUIRE(neonDiagOwn == Catch::Approx(ofDiagOwn).margin(1e-12));
    REQUIRE(neonDiagNei == Catch::Approx(ofDiagNei).margin(1e-12));
}

    ofEqn.relax(alpha);

    Foam::scalarField ofDiag = ofEqn.diag();
    Foam::scalarField ofRhs  = ofEqn.source();
    // ------------------------------------------------------------------
    // Apply NeoN relaxation - use new API (no SparsityPattern param)
    // ------------------------------------------------------------------
    NeoN::dsl::detail::applyMatrixRelaxation<fvcc::VolumeField<NeoN::scalar>>(
        ls,
        nfT,
        scalar(0.7)
    );

    // Copy results back
    auto matValsAfter = matrix.values().copyToHost();
    auto rhsAfter     = rhs.copyToHost();
    auto matValsAfterV  = matValsAfter.view();

    // ------------------------------------------------------------------
    // Comparison
    // ------------------------------------------------------------------
    for (label celli = 0; celli < mesh.nCells(); ++celli)
    {
        scalar nfDiag =
            matValsAfterV[rowOffsV[celli] + diagIdxV[celli]];

	scalar nfDiag0 = matValsV[rowOffsV[celli] + diagIdxV[celli]];
	NeoN::Logging::info("nfDiag0 = {}, ofDiag0 = {}", nfDiag0,ofDiag0[celli]);
	NeoN::Logging::info("nfDiag = {}, ofDiag = {}", nfDiag,ofDiag[celli]);

        REQUIRE(nfDiag0 == Catch::Approx(ofDiag0[celli]).margin(1e-12));
        REQUIRE(nfDiag == Catch::Approx(ofDiag[celli]).margin(1e-12));
        REQUIRE(rhsAfter.view()[celli] == Catch::Approx(ofRhs[celli]).margin(1e-12));
    }
}

