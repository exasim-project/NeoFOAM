// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 nf authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

#include "fvCFD.H"
#include "pisoControl.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "wallDist.H"
#include "LESModel.H"

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

        fvcc::VolumeField<NeoN::scalar>& nuTilda =
            vectorCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::volScalarField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
                    .foamField = ofnuTilda,
                    .name = "nuTilda"
                }
            );

        auto surfCalcBCs =
            fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh);
        auto volCalcBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(rt.nfMesh);
        auto volCalcVecBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(rt.nfMesh);

        NeoN::Logging::info("Creating phi");
        auto& phi = vectorCollection.registerVector<fvcc::SurfaceField<NeoN::scalar>>(
            NeoFOAM::CreateFromFoamField<Foam::surfaceScalarField> {
                .exec = rt.exec,
                .nfMesh = rt.nfMesh,
                .foamField = ofphi,
                .name = "phi"
            }
        );
        // Turbulence model logic
        auto tnu = laminarTransport.nu();
        auto nu = nf::constructFrom(rt.exec, rt.nfMesh, tnu());
        Foam::wallDist y(mesh);
        auto wallDist = nf::constructFrom(rt.exec, rt.nfMesh, y.y());

        const Foam::incompressible::LESModel& lesModel =
            Foam::refCast<const Foam::incompressible::LESModel>(turbulence());
        auto delta = nf::constructFrom(rt.exec, rt.nfMesh, lesModel.delta());
        NeoN::turbulenceModels::SpalartAllmarasDDES saBase(rt.exec, rt.nfMesh);

        auto gradOp = nnfvcc::GaussGreenGrad(rt.exec, rt.nfMesh);
	fvcc::TensorVecField G{
            fvcc::VolumeField<NeoN::Vec3>(rt.exec, "gradUx", rt.nfMesh, volCalcVecBCs),
            fvcc::VolumeField<NeoN::Vec3>(rt.exec, "gradUy", rt.nfMesh, volCalcVecBCs),
            fvcc::VolumeField<NeoN::Vec3>(rt.exec, "gradUz", rt.nfMesh, volCalcVecBCs)
        };
        gradOp.grad(U,G);
        fvcc::VolumeField<NeoN::scalar>
            magSqrGradNuTilda(rt.exec, "magSqrGradNuTilda", rt.nfMesh, volCalcBCs);
        fvcc::VolumeField<NeoN::Vec3> gradNuTilda(rt.exec, "gradNuTilda", rt.nfMesh, volCalcVecBCs);
        fvcc::VolumeField<NeoN::scalar> production(rt.exec, "production", rt.nfMesh, volCalcBCs);
        fvcc::VolumeField<NeoN::scalar> spCoeff(rt.exec, "spCoeff", rt.nfMesh, volCalcBCs);
        fvcc::SurfaceField<NeoN::scalar> nuTildaEff(rt.exec, "nuTildaEff", rt.nfMesh, surfCalcBCs);
	fvcc::SurfaceField<NeoN::scalar> nuEff(rt.exec, "nuEff", rt.nfMesh, surfCalcBCs);
        fvcc::SurfaceField<NeoN::scalar> surfNu(rt.exec, "surfNu", rt.nfMesh, surfCalcBCs);
        fvcc::SurfaceField<NeoN::scalar> surfNut(rt.exec, "surfNut", rt.nfMesh, surfCalcBCs);
        fvcc::SurfaceField<NeoN::scalar> surfNuTilda(rt.exec, "surfNuTilda", rt.nfMesh, surfCalcBCs);
	auto surfInterpol = fvcc::SurfaceInterpolation<NeoN::scalar>(
            rt.exec,
            rt.nfMesh,
            NeoN::TokenList({std::string("linear")})
        );
	surfInterpol.interpolate(nu,surfNu);
        auto nut = nf::constructFrom(rt.exec, rt.nfMesh, ofnut);
	saBase.calcNuTildaDiffusionCoeff(nuTilda,surfNu,surfNuTilda, nuTildaEff);
        saBase.correctNut(nut,surfNut,nuEff, nuTilda, nu,surfNu);
        
	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        NeoN::Logging::info("Starting time loop");
        while (runTime.loop())
        {
            // Logging supports string formatting
            NeoN::Logging::info("Time = {}", rt.t);

            fvcc::rotateOldTimes(U);
            fvcc::rotateOldTimes(phi);
            fvcc::rotateOldTimes(nuTilda);

            auto [maxCoNum, meanCoNum] = fvcc::computeCoNum(phi, rt.dt);
            NeoN::Logging::info("Courant Number mean: {} max: {}", meanCoNum, maxCoNum);
            nf::syncRunTimes(runTime, rt, maxCoNum);

            // Momentum predictor
            nf::PDESolver<NeoN::Vec3> UEqn(
                dsl::imp::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(nuEff, U)
                    + dsl::exp::viscousStress(nu, nut, G),
                U,
                rt
            );

            const auto ddtScheme = UEqn.ddtScheme();

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

                nnfvcc::SurfaceField<NeoN::scalar> rAU = surfInterpol.interpolate(crAU);
                rAU.name = "rAUf";

                auto phiHbyA = nf::flux(hByA) + rAU * fvcc::ddtFluxCorr(U, phi, rt.dt, ddtScheme);

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
            // Turbulence calculations
            gradOp.grad(U,G);
	    gradOp.grad(nuTilda,gradNuTilda);
            saBase.calcMagSqrVec(magSqrGradNuTilda, gradNuTilda);
            saBase.computeProdSpDDES(
                production,
                spCoeff,
                nuTilda,
                nu,
		G.Tx,
		G.Ty,
		G.Tz,
                wallDist,
                delta,
                magSqrGradNuTilda
            );
            
	    nf::PDESolver<NeoN::scalar> nuTildaEqn(
                dsl::imp::ddt(nuTilda) + dsl::imp::div(phi, nuTilda)
                    - NeoN::dsl::imp::laplacian(nuTildaEff, nuTilda)
                    + dsl::imp::source(spCoeff, nuTilda) - dsl::exp::sourceU(production, nuTilda),
                nuTilda,
                rt
            );
            nuTildaEqn.solve();
	    saBase.calcNuTildaDiffusionCoeff(nuTilda,surfNu,surfNuTilda, nuTildaEff);
            saBase.correctNut(nut,surfNut,nuEff, nuTilda, nu,surfNu);

            runTime.write();
            if (runTime.outputTime())
            {
                NeoN::Logging::info("Writing p");
                write(p, mesh);
                NeoN::Logging::info("Writing U");
                write(U, mesh);
                NeoN::Logging::info("Writing nuTilda");
                write(nuTilda, mesh);
            }

            runTime.printExecutionTime(Info);
        }
    }
    NeoN::finalize();

    return 0;
}

// ************************************************************************* //
