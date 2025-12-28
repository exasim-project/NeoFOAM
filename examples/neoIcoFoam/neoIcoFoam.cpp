// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 nf authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

#include "fvCFD.H"
#include "pisoControl.H"

#include <memory>

using Foam::Info;
using Foam::endl;
using Foam::nl;

// NOTE about namespace usage
// here the namespaces are used deliberately verbose to
// demonstrate where things are implemented
namespace fvc = Foam::fvc;
namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{
    NeoN::initialize(argc, argv);
    {
#include "addCheckCaseOptions.H"
#include "setRootCase.H"
#include "createTime.H"

        auto rt = nf::createAdapterRunTime(runTime);
        auto& mesh = rt.mesh;

        Foam::pisoControl piso(mesh);

#include "createFields.H"

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(rt.db, "VectorCollection");

        fvcc::VolumeField<NeoN::scalar>& p =
            vectorCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::volScalarField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
                    .foamField = ofp,
                    .name = "p"
                }
            );

        fvcc::VolumeField<NeoN::Vec3>& U =
            vectorCollection.registerVector<fvcc::VolumeField<NeoN::Vec3>>(
                nf::CreateFromFoamField<Foam::volVectorField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
                    .foamField = ofU,
                    .name = "U"
                }
            );

        auto nuBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh);
        fvcc::SurfaceField<NeoN::scalar> nu(rt.exec, "nu", rt.nfMesh, nuBCs);
        NeoN::fill(nu.internalVector(), viscosity.value());
        NeoN::fill(nu.boundaryData().value(), viscosity.value());

        NeoN::Logging::info("Creating phi");
	auto& phi = vectorCollection.registerVector<fvcc::SurfaceField<NeoN::scalar>>(
            NeoFOAM::CreateFromFoamField<Foam::surfaceScalarField>{
                .exec = rt.exec,
                .nfMesh = rt.nfMesh,
                .foamField = ofphi,
                .name = "phi"
            }
        );
        //auto phi = nf::constructFrom(rt.exec, rt.nfMesh, ofphi);

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        NeoN::Logging::info("Starting time loop");
        while (runTime.loop())
        {
            // Logging supports string formatting
            NeoN::Logging::info("Time = {}", rt.t);

            auto& oldU = fvcc::oldTime(U);
            oldU.internalVector() = U.internalVector();
	    auto& oldPhi = fvcc::oldTime(phi);
            oldPhi.internalVector() = phi.internalVector();

            auto [maxCoNum, meanCoNum] = fvcc::computeCoNum(phi, rt.dt);
            NeoN::Logging::info("Courant Number mean: {} max: {}", meanCoNum, maxCoNum);
            nf::syncRunTimes(runTime, rt, maxCoNum);

            // Momentum predictor
            nf::PDESolver<NeoN::Vec3> UEqn(
                dsl::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(nu, U),
                U,
                rt
            );

	    const auto* ddtScheme = UEqn.ddtScheme();

            if (piso.momentumPredictor())
            {
                // NOTE solve on a temporary clone of UEqn
                // TODO use a free function here
                UEqn.solve(-1.0 * dsl::exp::grad(p));
            }
            else
            {
                // NOTE since computing rAU and HbyA requires an assembled system matrix we
                // explicitly trigger assembly here.
                UEqn.assemble();
            }

            // --- PISO loop
            while (piso.correct())
            {
                NeoN::Logging::info("PISO loop");
                auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
                nf::constrainHbyA(U, p, hByA);

                nnfvcc::SurfaceField<NeoN::scalar> rAU =
                    fvcc::SurfaceInterpolation<NeoN::scalar>(
                        rt.exec,
                        rt.nfMesh,
                        NeoN::TokenList({std::string("linear")})
                    )
                        .interpolate(crAU);
                rAU.name = "rAUf";

                auto phiHbyA = nf::flux(hByA);
                // TODO: OpenFOAM typically also corrects phiHbyA with
                // + fvc::interpolate(rAU) * fvc::ddtCorr(U, phi);
                // for the first term we can use but fvc::ddtCorr is missing
                // NeoN::Input input = NeoN::TokenList({"linear"});
                // fvcc::SurfaceInterpolation<NeoN::scalar> surfInterpolation(rt.exec, rt.nfMesh,
                // input); auto surfRAU = surfInterpolation.interpolate(rAU);

                // TODO additionally missing
                // Foam::adjustPhi(phiHbyA, U, p);
                // Update the pressure BCs to ensure flux consistency
                // Foam::constrainPressure(p, U, phiHbyA, rAU);

                // Non-orthogonal pressure corrector loop
                while (piso.correctNonOrthogonal())
                {
                    // Pressure corrector
                    nf::PDESolver<NeoN::scalar> pEqn(
                        NeoN::dsl::imp::laplacian(rAU, p) - NeoN::dsl::exp::div(phiHbyA),
                        p,
                        rt
                    );

		    pEqn.enableDdtFluxCorr(*ddtScheme, U, phi);

                    if (ofp.needReference() && pRefCell >= 0)
                    {
                        pEqn.setReference(pRefCell, pRefValue);
                    }

                    auto stats = pEqn.solve();
                    p.correctBoundaryConditions();

                    if (piso.finalNonOrthogonalIter())
                    {
                        nf::updateFaceVelocity(phiHbyA, pEqn, phi);
                    }
                }
                // TODO: missing
                // #include "continuityErrs.H"

                nf::updateVelocity(hByA, crAU, p, U);
                U.correctBoundaryConditions();
            }

            runTime.write();
            if (runTime.outputTime())
            {
                NeoN::Logging::info("Writing p");
                write(p, mesh);
                NeoN::Logging::info("Writing U");
                write(U, mesh);
            }

            runTime.printExecutionTime(Info);
        }
    }
    NeoN::finalize();

    return 0;
}

// ************************************************************************* //
