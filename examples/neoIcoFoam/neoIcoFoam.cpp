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

	fvcc::VolumeField<NeoN::scalar>& nuTilda =
            vectorCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                nf::CreateFromFoamField<Foam::volScalarField> {
                    .exec = rt.exec,
                    .nfMesh = rt.nfMesh,
                    .foamField = ofnuTilda,
                    .name = "nuTilda"
                }
            );

//        auto nuBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(rt.nfMesh);
        auto volCalcBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(rt.nfMesh);
        fvcc::VolumeField<NeoN::scalar> nu(rt.exec, "nu", rt.nfMesh, volCalcBCs);
        NeoN::fill(nu.internalVector(), viscosity.value());
        NeoN::fill(nu.boundaryData().value(), viscosity.value());
        fvcc::VolumeField<NeoN::scalar> one(rt.exec, "one", rt.nfMesh, volCalcBCs);
        NeoN::fill(one.internalVector(), scalar(1));
        NeoN::fill(one.boundaryData().value(), scalar(1));

        NeoN::Logging::info("Creating phi");
        auto& phi = vectorCollection.registerVector<fvcc::SurfaceField<NeoN::scalar>>(
            NeoFOAM::CreateFromFoamField<Foam::surfaceScalarField> {
                .exec = rt.exec,
                .nfMesh = rt.nfMesh,
                .foamField = ofphi,
                .name = "phi"
            }
        );

	NeoN::turbulenceModels::DES::SpalartAllmarasBase saBase(rt.exec, rt.nfMesh);
	auto gradOp = nnfvcc::GaussGreenGrad(rt.exec, rt.nfMesh);
	auto gradUx = gradOp.grad(U.x());
	auto gradUy = gradOp.grad(U.y());
	auto gradUz = gradOp.grad(U.z());
	NeoN::Logging::info("after grad(U)");
	fvcc::VolumeField<NeoN::scalar> wallDist(rt.exec, "wallDist", rt.nfMesh, volCalcBCs);
	fvcc::VolumeField<NeoN::scalar> chi(rt.exec, "chi", rt.nfMesh, volCalcBCs);
	fvcc::VolumeField<NeoN::scalar> fv1(rt.exec, "fv1", rt.nfMesh, volCalcBCs);
	fvcc::VolumeField<NeoN::scalar> fv2(rt.exec, "fv2", rt.nfMesh, volCalcBCs);
	fvcc::VolumeField<NeoN::scalar> ft2(rt.exec, "ft2", rt.nfMesh, volCalcBCs);
	fvcc::VolumeField<NeoN::scalar> stilda(rt.exec, "stilda", rt.nfMesh, volCalcBCs);
	fvcc::VolumeField<NeoN::scalar> fw(rt.exec, "fw", rt.nfMesh, volCalcBCs);
	fvcc::VolumeField<NeoN::scalar> nut(rt.exec, "nut", rt.nfMesh, volCalcBCs);
	fvcc::VolumeField<NeoN::scalar> dTilda(rt.exec, "dTilda", rt.nfMesh, volCalcBCs);
	fvcc::VolumeField<NeoN::scalar> strainRate(rt.exec, "strainRate", rt.nfMesh, volCalcBCs);
	fvcc::VolumeField<NeoN::scalar> dNuTildaEff(rt.exec, "dNuTildaEff", rt.nfMesh, volCalcBCs);
	NeoN::turbulenceModels::DES::maxDeltaxyz deltaModel(rt.nfMesh);
        NeoN::turbulenceModels::DES::SpalartAllmarasDDES ddes(rt.exec);
	NeoN::Logging::info("wallDist");
	saBase.wallDistance(wallDist);
	NeoN::Logging::info("chi");
	saBase.chi(chi, nuTilda, nu);
        saBase.fv1(fv1, chi);
        saBase.fv2(fv2, chi, fv1);
        saBase.ft2(ft2, chi);
	NeoN::Logging::info("strainRate");
	saBase.strainRate(strainRate,gradUx, gradUy, gradUz); // from velocity gradients
	auto delta = deltaModel.delta();
        ddes.dTilda(dTilda, wallDist, nuTilda, nu, strainRate, delta);
        saBase.stilda(stilda, strainRate, nuTilda, dTilda, fv2);
        saBase.fw(fw, stilda, dTilda, nuTilda);
        saBase.dNuTildaEff(dNuTildaEff, nuTilda, nu);
	saBase.nut(nut, nuTilda, nu);
	nnfvcc::SurfaceField<NeoN::scalar> nuEff =
                    fvcc::SurfaceInterpolation<NeoN::scalar>(
                        rt.exec,
                        rt.nfMesh,
                        NeoN::TokenList({std::string("linear")})
                    )
                        .interpolate(nu+nut);
                nuEff.name = "nuEff";
	const auto& coeffs = saBase.coeffs();
        const auto Cb1 = coeffs.Cb1;
        const auto Cb2 = coeffs.Cb2;
        const auto kappa = coeffs.kappa;
        const auto sigmaNut = coeffs.sigmaNut;
	const auto Cw1 = saBase.cw1();
        const auto Cw2 = coeffs.Cw2;
        const auto Cw3 = coeffs.Cw3;
        //NeoN::turbulenceModels::ReynoldsStress reynolds(rt.exec, rt.nfMesh);

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
                dsl::imp::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(nuEff, U),
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

                nnfvcc::SurfaceField<NeoN::scalar> rAU =
                    fvcc::SurfaceInterpolation<NeoN::scalar>(
                        rt.exec,
                        rt.nfMesh,
                        NeoN::TokenList({std::string("linear")})
                    )
                        .interpolate(crAU);
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

	    deltaModel.update();
	    auto delta = deltaModel.delta();
            saBase.strainRate(strainRate,gradUx, gradUy, gradUz); // from velocity gradients
            ddes.dTilda(dTilda, wallDist, nuTilda, nu, strainRate, delta);
	    ddes.correct(dTilda, nut, saBase, wallDist, nuTilda, nu, strainRate, delta);
	    auto production = Cb1 * stilda * nuTilda * (one - ft2);
	    auto spCoeff = (Cw1 * fw - (Cb1 / (kappa * kappa)) * ft2) * nuTilda / (dTilda * dTilda);
	    auto nuTildaEff = NeoN::scalar(1/sigmaNut)*nuEff;
	    nf::PDESolver<NeoN::scalar> nuTildaEqn(
                        dsl::imp::ddt(nuTilda) + dsl::imp::div(phi, nuTilda) - NeoN::dsl::imp::laplacian(nuTildaEff, nuTilda)
			+ dsl::imp::source(spCoeff, nuTilda)
			- dsl::exp::source(production, nuTilda),
                        nuTilda,
                        rt
                    );
	    nuTildaEqn.solve();

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
