// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 FoamAdapter authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"
#include "constrainHbyA.H"

using Foam::Info;
using Foam::endl;
using Foam::nl;
namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace dsl = NeoN::dsl;
namespace nnfvcc = NeoN::finiteVolume::cellCentred;
namespace nf = FoamAdapter;

extern Foam::Time* timePtr; // A single time object


TEST_CASE("PressureVelocityCoupling")
{
    Foam::Time& runTime = *timePtr;


    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;

    auto ofU = randomVectorField(runTime, mesh, "ofU");
    auto ofp = randomScalarField(runTime, mesh, "ofp");
    // forAll(ofp, celli)
    // {
    //     ofp[celli] = celli;
    // }
    ofp.correctBoundaryConditions();
    // a predictable field is simpler to debug
    // forAll(ofU, celli)
    // {
    //     ofU[celli] = Foam::vector(celli, celli, celli);
    // }
    ofU.correctBoundaryConditions();
    auto& oldOfU = ofU.oldTime();
    oldOfU.primitiveFieldRef() = Foam::vector(0.0, 0.0, 0.0);
    oldOfU.correctBoundaryConditions();

    Info << "creating FoamAdapter velocity fields" << endl;
    auto& vectorCollection = nnfvcc::VectorCollection::instance(rt.db, "VectorCollection");
    nnfvcc::VolumeField<NeoN::Vec3>& nfU =
        vectorCollection.registerVector<nnfvcc::VolumeField<NeoN::Vec3>>(
            FoamAdapter::CreateFromFoamField<Foam::volVectorField> {
                .exec = rt.exec,
                .nfMesh = rt.nfMesh,
                .foamField = ofU,
                .name = "nfU"
            }
        );
    Info << "creating FoamAdapter pressure fields" << endl;
    auto nfp = vectorCollection.registerVector<nnfvcc::VolumeField<NeoN::scalar>>(
        FoamAdapter::CreateFromFoamField<Foam::volScalarField> {
            .exec = rt.exec,
            .nfMesh = rt.nfMesh,
            .foamField = ofp,
            .name = "nfp"
        }
    );
    nfp.correctBoundaryConditions();

    auto& nfOldU = fvcc::oldTime(nfU);
    NeoN::fill(nfOldU.internalVector(), NeoN::Vec3(0.0, 0.0, 0.0));
    nfOldU.correctBoundaryConditions();

    Foam::surfaceScalarField ofPhi(
        Foam::IOobject(
            "ofPhi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        fvc::flux(ofU)
    );
    auto nfPhi = FoamAdapter::constructSurfaceField(rt.exec, rt.nfMesh, ofPhi);
    nfPhi.name = "nfPhi";

    Foam::surfaceScalarField ofNu(
        Foam::IOobject(
            "ofNu",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("ofNu", Foam::dimensionSet(0, 2, -1, 0, 0), 0.01)
    );

    auto nfNu = FoamAdapter::constructSurfaceField(rt.exec, rt.nfMesh, ofNu);
    nfNu.name = "nfNu";
    NeoN::fill(nfNu.boundaryData().value(), 0.01);

    Foam::scalar t = runTime.time().value();
    Foam::scalar dt = runTime.deltaT().value();

    NeoN::Dictionary fvSchemesDict = FoamAdapter::convert(mesh.schemesDict());
    NeoN::Dictionary fvSolutionDict = FoamAdapter::convert(mesh.solutionDict());
    auto& solverDict = fvSolutionDict.get<NeoN::Dictionary>("solvers");

    SECTION("discreteMomentumFields " + execName)
    {
        nf::PDESolver<NeoN::Vec3> nfUEqn(
            dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
            nfU,
            rt
        );

        Foam::fvVectorMatrix ofUEqn(
            fvm::ddt(ofU) + fvm::div(ofPhi, ofU) - fvm::laplacian(ofNu, ofU)
        );

        SECTION("rAU")
        {
            Foam::volScalarField forAU("forAU", 1.0 / ofUEqn.A());
            nfUEqn.assemble();
            auto nfrAU = nf::computeRAU(nfUEqn);

            FoamAdapter::compare(nfrAU, forAU, ApproxScalar(1e-15), false);
        }

        SECTION("rAU modified U")
        {
            ofU.primitiveFieldRef() *= 2.5;
            ofU.correctBoundaryConditions();
            Foam::volScalarField forAU("forAU", 1.0 / ofUEqn.A());

            nfU.internalVector() *= 2.5;
            nfU.correctBoundaryConditions();
            nfUEqn.assemble();
            auto nfrAU = nf::computeRAU(nfUEqn);

            FoamAdapter::compare(nfrAU, forAU, ApproxScalar(1e-15), false);
        }

        SECTION("HbyA")
        {
            Foam::volScalarField forAU("forAU", 1.0 / ofUEqn.A());
            Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());

            nfUEqn.assemble();
            auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);
            auto hostnfHbyA = nfHbyA.internalVector().copyToHost();

            for (size_t celli = 0; celli < hostnfHbyA.size(); celli++)
            {
                REQUIRE(hostnfHbyA.view()[celli][0] == Catch::Approx(HbyA[celli][0]).margin(1e-14));
                REQUIRE(hostnfHbyA.view()[celli][1] == Catch::Approx(HbyA[celli][1]).margin(1e-14));
                REQUIRE(hostnfHbyA.view()[celli][2] == Catch::Approx(HbyA[celli][2]).margin(1e-14));
            }

            SECTION("constrainHbyA")
            {
                Foam::volVectorField ofConstrainHbyA(
                    "ofConstrainHbyA",
                    Foam::constrainHbyA(forAU * ofUEqn.H(), ofU, ofp)
                );
                nf::constrainHbyA(nfU, nfp, nfHbyA);
                auto hostBCnfHbyA = nfHbyA.boundaryData().value().copyToHost();

                forAll(ofConstrainHbyA.boundaryField(), patchi)
                {
                    REQUIRE(
                        ofConstrainHbyA.boundaryField()[patchi].size()
                        == nfHbyA.boundaryData().nBoundaryFaces(patchi)
                    );
                    const Foam::fvPatchVectorField& ofConstrainHbyAPatch =
                        ofConstrainHbyA.boundaryField()[patchi];
                    auto [start, end] = nfHbyA.boundaryData().range(patchi);

                    forAll(ofConstrainHbyAPatch, bfacei)
                    {
                        REQUIRE(
                            hostBCnfHbyA.view()[start + bfacei][0]
                            == Catch::Approx(ofConstrainHbyAPatch[bfacei][0]).margin(1e-14)
                        );
                        REQUIRE(
                            hostBCnfHbyA.view()[start + bfacei][1]
                            == Catch::Approx(ofConstrainHbyAPatch[bfacei][1]).margin(1e-14)
                        );
                        REQUIRE(
                            hostBCnfHbyA.view()[start + bfacei][2]
                            == Catch::Approx(ofConstrainHbyAPatch[bfacei][2]).margin(1e-14)
                        );
                    }
                }
            }
        }

        SECTION("HbyA modified U")
        {
            ofU.primitiveFieldRef() *= 2.5;
            ofU.correctBoundaryConditions();
            Foam::volScalarField forAU("forAU", 1.0 / ofUEqn.A());
            Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());

            nfU.internalVector() *= 2.5;
            nfU.correctBoundaryConditions();
            nfUEqn.assemble();
            auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);
            auto hostnfHbyA = nfHbyA.internalVector().copyToHost();

            for (size_t celli = 0; celli < hostnfHbyA.size(); celli++)
            {
                REQUIRE(hostnfHbyA.view()[celli][0] == Catch::Approx(HbyA[celli][0]).margin(1e-14));
                REQUIRE(hostnfHbyA.view()[celli][1] == Catch::Approx(HbyA[celli][1]).margin(1e-14));
                REQUIRE(hostnfHbyA.view()[celli][2] == Catch::Approx(HbyA[celli][2]).margin(1e-14));
            }
        }

        // SECTION("Solve steady state momentum without grad(p)")
        // {
        //     auto& solverDict = rt.fvSolutionDict.get<NeoN::Dictionary>("solvers");
        //     solverDict.get<NeoN::Dictionary>("nfU") =
        //         nf::mapFvSolution(solverDict.get<NeoN::Dictionary>("nfU"));

        //     // require fields to be initially the same
        //     auto hostnfU = nfU.internalVector().copyToHost();
        //     for (size_t celli = 0; celli < hostnfU.size(); celli++)
        //     {
        //         REQUIRE(hostnfU.view()[celli][0] == Catch::Approx(ofU[celli][0]).margin(1e-03));
        //         REQUIRE(hostnfU.view()[celli][1] == Catch::Approx(ofU[celli][1]).margin(1e-03));
        //         REQUIRE(hostnfU.view()[celli][2] == Catch::Approx(ofU[celli][2]).margin(1e-06));
        //     }

        //     Foam::fvVectorMatrix ofUEqn
        //     (
        //      //  Foam::fvm::ddt(ofU)
        //      // +
        //       Foam::fvm::div(ofPhi,ofU)
        //     - Foam::fvm::laplacian(ofNu, ofU)
        //     );

        //     Foam::solve(ofUEqn);

        //     nf::PDESolver<NeoN::Vec3> nfUEqn(
        //         //  dsl::imp::ddt(nfU)
        //          dsl::imp::div(nfPhi, nfU)
        //          - dsl::imp::laplacian(nfNu, nfU),
        //         nfU,
        //         rt
        //     );

        //     nfUEqn.solve();

        //     auto hostnfU2 = nfU.internalVector().copyToHost();
        //     for (size_t celli = 0; celli < hostnfU.size(); celli++)
        //     {
        //         REQUIRE(hostnfU2.view()[celli][0] == Catch::Approx(ofU[celli][0]).margin(1e-06));
        //         REQUIRE(hostnfU2.view()[celli][1] == Catch::Approx(ofU[celli][1]).margin(1e-06));
        //         REQUIRE(hostnfU2.view()[celli][2] == Catch::Approx(ofU[celli][2]).margin(1e-06));
        //     }
        // }
        SECTION("Solve transient momentum without grad(p)")
        {
            auto& solverDict = rt.fvSolutionDict.get<NeoN::Dictionary>("solvers");
            solverDict.get<NeoN::Dictionary>("nfU") =
                nf::mapFvSolution(solverDict.get<NeoN::Dictionary>("nfU"));

            // require fields to be initially the same
            auto hostnfU = nfU.internalVector().copyToHost();
            for (size_t celli = 0; celli < hostnfU.size(); celli++)
            {
                REQUIRE(hostnfU.view()[celli][0] == Catch::Approx(ofU[celli][0]).margin(1e-12));
                REQUIRE(hostnfU.view()[celli][1] == Catch::Approx(ofU[celli][1]).margin(1e-12));
                REQUIRE(hostnfU.view()[celli][2] == Catch::Approx(ofU[celli][2]).margin(1e-12));
            }

            Foam::fvVectorMatrix ofUEqn(
                Foam::fvm::ddt(ofU) + Foam::fvm::div(ofPhi, ofU) - Foam::fvm::laplacian(ofNu, ofU)
            );

            Foam::solve(ofUEqn);

            nf::PDESolver<NeoN::Vec3> nfUEqn(
                dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
                nfU,
                rt
            );

            nfUEqn.solve();

            auto hostnfU2 = nfU.internalVector().copyToHost();
            for (size_t celli = 0; celli < hostnfU.size(); celli++)
            {
                REQUIRE(hostnfU2.view()[celli][0] == Catch::Approx(ofU[celli][0]).margin(1e-12));
                REQUIRE(hostnfU2.view()[celli][1] == Catch::Approx(ofU[celli][1]).margin(1e-12));
                REQUIRE(hostnfU2.view()[celli][2] == Catch::Approx(ofU[celli][2]).margin(1e-12));
            }

            SECTION("HbyA modified U")
            {
                ofU.correctBoundaryConditions();
                Foam::volScalarField forAU("forAU", 1.0 / ofUEqn.A());
                Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());
                Foam::surfaceScalarField phiHbyA("phiHbyA", Foam::fvc::flux(HbyA));

                nfU.correctBoundaryConditions();
                auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);

                auto hostnfrAU = nfrAU.internalVector().copyToHost();
                for (size_t celli = 0; celli < hostnfrAU.size(); celli++)
                {
                    REQUIRE(hostnfrAU.view()[celli] == Catch::Approx(forAU[celli]).margin(1e-10));
                }

                auto hostnfHbyA = nfHbyA.internalVector().copyToHost();
                for (size_t celli = 0; celli < hostnfHbyA.size(); celli++)
                {
                    // std::cout << " nf " << hostnfHbyA.view()[celli][0] << " of " <<
                    // HbyA[celli][0]  << "\n";
                    REQUIRE(
                        hostnfHbyA.view()[celli][0] == Catch::Approx(HbyA[celli][0]).margin(1e-12)
                    );
                    REQUIRE(
                        hostnfHbyA.view()[celli][1] == Catch::Approx(HbyA[celli][1]).margin(1e-12)
                    );
                    REQUIRE(
                        hostnfHbyA.view()[celli][2] == Catch::Approx(HbyA[celli][2]).margin(1e-12)
                    );
                }

                auto nfPhiHbyA = nf::flux(nfHbyA);
                auto hostnfPhiHbyA = nfPhiHbyA.internalVector().copyToHost();
                for (size_t celli = 0; celli < hostnfHbyA.size(); celli++)
                {
                    REQUIRE(
                        hostnfPhiHbyA.view()[celli] == Catch::Approx(phiHbyA[celli]).margin(1e-6)
                    );
                }
            }
        }


        SECTION("Solve transient momentum with grad(p)")
        {
            auto& solverDict = rt.fvSolutionDict.get<NeoN::Dictionary>("solvers");
            solverDict.get<NeoN::Dictionary>("nfU") =
                nf::mapFvSolution(solverDict.get<NeoN::Dictionary>("nfU"));

            // require fields to be initially the same
            auto hostnfU = nfU.internalVector().copyToHost();
            for (size_t celli = 0; celli < hostnfU.size(); celli++)
            {
                REQUIRE(hostnfU.view()[celli][0] == Catch::Approx(ofU[celli][0]).margin(1e-12));
                REQUIRE(hostnfU.view()[celli][1] == Catch::Approx(ofU[celli][1]).margin(1e-12));
                REQUIRE(hostnfU.view()[celli][2] == Catch::Approx(ofU[celli][2]).margin(1e-12));
            }

            Foam::fvVectorMatrix ofUEqn(
                Foam::fvm::ddt(ofU) + Foam::fvm::div(ofPhi, ofU) - Foam::fvm::laplacian(ofNu, ofU)
                + Foam::fvc::grad(ofp)
            );

            Foam::solve(ofUEqn); // == -Foam::fvc::grad(ofp));

            nf::PDESolver<NeoN::Vec3> nfUEqn(
                dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU)
                    + dsl::exp::grad(nfp),
                nfU,
                rt
            );

            nfUEqn.solve();

            auto hostnfU2 = nfU.internalVector().copyToHost();
            for (size_t celli = 0; celli < hostnfU.size(); celli++)
            {
                REQUIRE(hostnfU2.view()[celli][0] == Catch::Approx(ofU[celli][0]).margin(1e-12));
                REQUIRE(hostnfU2.view()[celli][1] == Catch::Approx(ofU[celli][1]).margin(1e-12));
                REQUIRE(hostnfU2.view()[celli][2] == Catch::Approx(ofU[celli][2]).margin(1e-12));
            }

            SECTION("HbyA modified U")
            {
                ofU.correctBoundaryConditions();
                Foam::volScalarField forAU("forAU", 1.0 / ofUEqn.A());
                Foam::volVectorField HbyA("HbyA", forAU * ofUEqn.H());

                nfU.correctBoundaryConditions();
                auto [nfrAU, nfHbyA] = nf::computeRAUandHByA(nfUEqn);

                auto hostnfrAU = nfrAU.internalVector().copyToHost();
                for (size_t celli = 0; celli < hostnfrAU.size(); celli++)
                {
                    REQUIRE(hostnfrAU.view()[celli] == Catch::Approx(forAU[celli]).margin(1e-12));
                }

                auto hostnfHbyA = nfHbyA.internalVector().copyToHost();
                for (size_t celli = 0; celli < hostnfHbyA.size(); celli++)
                {
                    std::cout << " nf[0] " << hostnfHbyA.view()[celli][0] << " of[0] "
                              << HbyA[celli][0] << "\n";
                    std::cout << " nf[1] " << hostnfHbyA.view()[celli][1] << " of[1] "
                              << HbyA[celli][1] << "\n";
                }
                for (size_t celli = 0; celli < hostnfHbyA.size(); celli++)
                {
                    REQUIRE(
                        hostnfHbyA.view()[celli][0] == Catch::Approx(HbyA[celli][0]).margin(1e-8)
                    );
                    REQUIRE(
                        hostnfHbyA.view()[celli][1] == Catch::Approx(HbyA[celli][1]).margin(1e-8)
                    );
                    REQUIRE(
                        hostnfHbyA.view()[celli][2] == Catch::Approx(HbyA[celli][2]).margin(1e-8)
                    );
                }

                Foam::surfaceScalarField phiHbyA("phiHbyA", Foam::fvc::flux(HbyA));

                auto nfPhiHbyA = nf::flux(nfHbyA);
                auto hostnfPhiHbyA = nfPhiHbyA.internalVector().copyToHost();
                for (size_t celli = 0; celli < hostnfHbyA.size(); celli++)
                {
                    REQUIRE(
                        hostnfPhiHbyA.view()[celli] == Catch::Approx(phiHbyA[celli]).margin(1e-6)
                    );
                }
            }
        }


        SECTION("matrix flux")
        {
            // create rAUf
            Foam::surfaceScalarField forAUf(
                Foam::IOobject(
                    "forAUf",
                    runTime.timeName(),
                    mesh,
                    Foam::IOobject::NO_READ,
                    Foam::IOobject::NO_WRITE
                ),
                mesh,
                Foam::dimensionedScalar("forAUf", Foam::dimensionSet(0, 0, 1, 0, 0), 0.1)
            );
            auto nfrAUf = FoamAdapter::constructSurfaceField(rt.exec, rt.nfMesh, forAUf);
            nfrAUf.name = "nfrAUf";

            Foam::surfaceScalarField ofPhi0("ofPhi0", ofPhi * 0.0);
            auto nfPhi0 = FoamAdapter::constructSurfaceField(rt.exec, rt.nfMesh, ofPhi0);
            nfPhi0.name = "nfPhi0";

            nf::PDESolver<NeoN::scalar> pEqn(
                dsl::imp::laplacian(nfrAUf, nfp) - dsl::exp::div(nfPhi),
                nfp,
                rt
            );

            pEqn.assemble();

            nf::updateFaceVelocity(nfPhi, pEqn, nfPhi0);

            Foam::fvScalarMatrix ofpEqn(fvm::laplacian(forAUf, ofp) == fvc::div(ofPhi));

            ofPhi0 = ofPhi - ofpEqn.flux();

            auto hostPhi0 = nfPhi0.internalVector().copyToHost();
            for (size_t facei = 0; facei < rt.nfMesh.nInternalFaces(); facei++)
            {
                REQUIRE(hostPhi0.view()[facei] == Catch::Approx(ofPhi0[facei]).margin(1e-14));
            }

            auto hostBCPhi0 = nfPhi0.boundaryData().value().copyToHost();
            forAll(ofPhi0.boundaryField(), patchi)
            {
                REQUIRE(
                    ofPhi0.boundaryField()[patchi].size()
                    == nfPhi0.boundaryData().nBoundaryFaces(patchi)
                );
                const Foam::fvsPatchScalarField& ofPhi0Patch = ofPhi0.boundaryField()[patchi];
                auto [start, end] = nfPhi0.boundaryData().range(patchi);

                forAll(ofPhi0Patch, bfacei)
                {
                    REQUIRE(
                        hostBCPhi0.view()[start + bfacei]
                        == Catch::Approx(ofPhi0Patch[bfacei]).margin(1e-14)
                    );
                }
            }
        }
    }
}
