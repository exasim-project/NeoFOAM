// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 nf authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

#include "fvCFD.H"
#include "simpleControl.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "wallDist.H"

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
// Bring up OpenFOAM (and MPI) before NeoN, matching neoIcoFoam, so the rank is
// known when NeoN configures logging and the rank-0 muting engages immediately.
#include "addCheckCaseOptions.H"
#include "setRootCase.H"
#include "createTime.H"
    NeoN::initialize(argc, argv);
    {
        auto rt = nf::createAdapterRunTime(runTime);
        auto& mesh = rt.mesh;

        Foam::simpleControl simple(mesh);

#include "createFields.H"

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
        solverDict.subDict("k") = nf::mapFvSolution(solverDict.subDict("k"));
        solverDict.subDict("omega") = nf::mapFvSolution(solverDict.subDict("omega"));
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(rt.db, "VectorCollection");

        auto& p = nf::constructAndRegister(vectorCollection, rt, ofP, false);
        auto& U = nf::constructAndRegister(vectorCollection, rt, ofU, false);
        auto& k = nf::constructAndRegister(vectorCollection, rt, ofK, false);
        auto& omega = nf::constructAndRegister(vectorCollection, rt, ofOmega, false);
        auto& nut = nf::constructAndRegister(vectorCollection, rt, ofNut, false);

        NeoN::Logging::info("Creating phi");
        auto& phi = nf::constructAndRegister(vectorCollection, rt, ofPhi, false);

        // Turbulence model setup
        auto nu = nf::constructFrom(rt.exec, rt.nfMesh, tnu());
        auto wallDist = nf::constructFrom(rt.exec, rt.nfMesh, y.y());
        nf::KOmegaSST turb(rt.exec, rt.nfMesh, nu, wallDist);

        turb.validate(U, k, omega, nut);

        // TODO: surface interpolation also instantiated in turbulence model -> doubled?!
        auto surfInterpol = fvcc::SurfaceInterpolation<NeoN::scalar>(
            rt.exec,
            rt.nfMesh,
            NeoN::TokenList({std::string("linear")})
        );
        NeoN::scalar cumulativeContErr = 0.0;

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        NeoN::Logging::info("Starting time loop");
        while (runTime.loop())
        {
            rt.t = runTime.time().value();
            NeoN::Logging::info("Time = {}", rt.t);

            // Steady-state: no rotateOldTimes, no Courant number, no syncRunTimes

            // Momentum predictor (no ddt for steady-state SIMPLE)
            nf::PDESolver<NeoN::Vec3> UEqn(
                dsl::imp::div(phi, U) - dsl::imp::laplacian(turb.nuEff(), U)
                    + dsl::exp::viscousStress(nu, nut, turb.gradU()),
                U,
                rt
            );

            if (simple.momentumPredictor())
            {
                UEqn.solve(-1.0 * dsl::exp::grad(p));
            }
            else
            {
                UEqn.assemble();
            }

            // --- SIMPLE pressure-velocity coupling (single pass, no inner PISO loop)
            {
                auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
                nf::constrainHbyA(U, p, hByA);

                nnfvcc::SurfaceField<NeoN::scalar> rAU = surfInterpol.interpolate(crAU);
                rAU.name = "rAUf";

                // No ddtFluxCorr: SIMPLE is steady-state
                auto phiHbyA = nf::flux(hByA);

                // TODO additionally missing
                // Foam::adjustPhi(phiHbyA, U, p);
                // Foam::constrainPressure(p, U, phiHbyA, rAU);

                // Non-orthogonal pressure corrector loop
                while (simple.correctNonOrthogonal())
                {
                    nf::PDESolver<NeoN::scalar> pEqn(
                        NeoN::dsl::imp::laplacian(rAU, p) - NeoN::dsl::exp::div(phiHbyA),
                        p,
                        rt
                    );

                    if (ofP.needReference() && pRefCell >= 0)
                    {
                        pEqn.setReference(pRefCell, pRefValue);
                    }

                    pEqn.solve();
                    p.correctBoundaryConditions();

                    if (simple.finalNonOrthogonalIter())
                    {
                        nf::updateFaceVelocity(phiHbyA, pEqn, phi);
                    }
                }
                nf::reportContinuityError(phi, rt, cumulativeContErr);

                // TODO: p.relax() - explicit pressure under-relaxation not yet implemented

                nf::updateVelocity(hByA, crAU, p, U);
                U.correctBoundaryConditions();
            }

            // Turbulence update
            turb.correct(U, phi, k, omega, nut, rt);

            runTime.write();
            if (runTime.outputTime())
            {
                NeoN::Logging::info("Writing p");
                write(p, mesh);
                NeoN::Logging::info("Writing U");
                write(U, mesh);
                NeoN::Logging::info("Writing turbulence variables");
                write(k, mesh);
                write(omega, mesh);
                write(nut, mesh);
            }

            runTime.printExecutionTime(Info);
        }
    }
    NeoN::finalize();

    return 0;
}

// ************************************************************************* //
