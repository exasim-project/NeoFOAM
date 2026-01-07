// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors
#include <cstddef>
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "common.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

inline void bumpCurrentOF(Foam::volScalarField& ofT, const Foam::scalar a)
{
    ofT -= Foam::dimensionedScalar("bump", ofT.dimensions(), a);
    ofT.correctBoundaryConditions();
}

TEST_CASE("(backward) ddt implicit matches OpenFOAM", "[ddt][backward]")
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

    SECTION("ddtScheme backward on " + execName)
    {
        runTime.setTime(0.0, 0);

        auto ofT = NeoFOAM::randomScalarField(runTime, mesh, "T");
        ofT.correctBoundaryConditions();
        ofT.oldTime();
        ofT.oldTime().oldTime();

        auto& nfT = fieldCol.registerVector<fvcc::VolumeField<NeoN::scalar>>(
            NeoFOAM::CreateFromFoamField<Foam::volScalarField> {
                .exec = exec,
                .nfMesh = nfMesh,
                .foamField = ofT,
                .name = "nfT"
            }
        );
        fvcc::DdtOperator ddtOp(dsl::Operator::Type::Implicit, nfT);

        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(nfT)", std::string("BDF2"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        ddtOp.read(NeoN::Input {fvSchemes});

        //  =========================================================================
        //  Step 1: timeIndex == 1 → backward startup (implicit Euler)
        //  =========================================================================
        runTime++;
        ofT.storeOldTimes();
        fvcc::rotate(nfT);
        bumpCurrentOF(ofT, 1.0);
        nfT -= scalar(1.0);
        nfT.correctBoundaryConditions();

        Foam::fvScalarMatrix matrix1(Foam::fvm::ddt(ofT));
        Foam::volScalarField ddt1("ddt1", matrix1 & ofT);

        auto ls1 = NeoN::la::createEmptyLinearSystem<NeoN::scalar, NeoN::localIdx>(
            nfMesh,
            sparsityPattern
        );

        ddtOp.implicitOperation(ls1, runTime.value(), runTime.deltaTValue());

        // --- rhs ---
        {
            auto rhs = ls1.rhs().copyToHost();
            forAll(rhs.view(), celli)
            {
                REQUIRE(rhs.view()[celli] == Catch::Approx(matrix1.source()[celli]).margin(1e-16));
            }
        }

        // --- diag ---
        {
            auto diag = NeoFOAM::diag(ls1, sparsityPattern).copyToHost();
            forAll(diag.view(), celli)
            {
                REQUIRE(diag.view()[celli] == Catch::Approx(matrix1.diag()[celli]).margin(1e-16));
            }
        }

        // --- operator application ---
        {
            auto result = NeoFOAM::applyOperator(ls1, nfT).internalVector().copyToHost();

            forAll(result.view(), celli)
            {
                REQUIRE(
                    result.view()[celli]
                    == Catch::Approx(ddt1[celli] * mesh.V()[celli]).margin(1e-16)
                );
            }
        }

        // =========================================================================
        // Step 2: timeIndex == 2 → true backward (BDF2)
        // =========================================================================
        runTime++;
        ofT.storeOldTimes();
        fvcc::rotate(nfT);
        bumpCurrentOF(ofT, 2.0);
        nfT -= scalar(2.0);
        nfT.correctBoundaryConditions();

        Foam::fvScalarMatrix matrix2(Foam::fvm::ddt(ofT));
        Foam::volScalarField ddt2("ddt2", matrix2 & ofT);

        auto ls2 = NeoN::la::createEmptyLinearSystem<NeoN::scalar, NeoN::localIdx>(
            nfMesh,
            sparsityPattern
        );

        ddtOp.implicitOperation(ls2, runTime.value(), runTime.deltaTValue());

        // --- rhs ---
        {
            auto rhs = ls2.rhs().copyToHost();
            forAll(rhs.view(), celli)
            {
                REQUIRE(rhs.view()[celli] == Catch::Approx(matrix2.source()[celli]).margin(1e-16));
            }
        }

        // --- diag ---
        {
            auto diag = NeoFOAM::diag(ls2, sparsityPattern).copyToHost();
            forAll(diag.view(), celli)
            {
                REQUIRE(diag.view()[celli] == Catch::Approx(matrix2.diag()[celli]).margin(1e-16));
            }
        }

        // --- operator application ---
        {
            auto result = NeoFOAM::applyOperator(ls2, nfT).internalVector().copyToHost();

            forAll(result.view(), celli)
            {
                REQUIRE(
                    result.view()[celli]
                    == Catch::Approx(ddt2[celli] * mesh.V()[celli]).margin(1e-16)
                );
            }
        }
    }
}
