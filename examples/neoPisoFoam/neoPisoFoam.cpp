// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 nf authors

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"
#include "NeoFOAM/turbulenceModels/turbulenceModel.hpp"

#include "fvCFD.H"
#include "pisoControl.H"
#include "singlePhaseTransportModel.H"

#include <memory>

using Foam::Info;
using Foam::endl;
using Foam::nl;

namespace fvc = Foam::fvc;
namespace dsl = NeoN::dsl;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{
    // Bring up OpenFOAM (and MPI) before NeoN, matching neoIcoFoam, so the rank is
    // known when NeoN configures logging and the rank-0 muting engages immediately.
    Foam::argList::addOption("executor", "word", "NeoN executor type (Serial/CPU/GPU/default)");
#include "addCheckCaseOptions.H"
#include "setRootCase.H"
    NeoN::initialize(argc, argv);
    {
// createTime.H is included INSIDE the NeoN init/finalize bracket so Foam::Time --
// which owns the controlDict function objects (e.g. neoForceCoeffs) and therefore any
// NeoN/Kokkos memory they hold -- is destroyed before NeoN::finalize() calls
// Kokkos::finalize(). Otherwise ~Time() runs after finalize and aborts with
// "Kokkos allocation ... deallocated after Kokkos::finalize was called".
#include "createTime.H"
        auto rt = nf::createAdapterRunTime(runTime, args);
        auto& mesh = rt.mesh;

        Foam::pisoControl piso(mesh);

#include "createFields.H"

        nf::createMappedFvSolutionDicts(rt);
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

        // grad(p) for the velocity reconstruction honours the configured gradSchemes.
        auto gradPOp = nf::makeGradOperator(rt.exec, rt.nfMesh, rt.fvSchemesDict, "grad(p)");

        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(rt.db, "VectorCollection");

        auto& p = nf::constructAndRegister(vectorCollection, rt, ofP, false);
        auto& U = nf::constructAndRegister(vectorCollection, rt, ofU, false);

        NeoN::Logging::info("Creating phi");
        auto& phi = nf::constructAndRegister(vectorCollection, rt, ofPhi, false);

        auto nu = nf::constructFrom(rt.exec, rt.nfMesh, tnu());

        auto turb = nf::TurbulenceModel::create(rt, nu);

        turb->validate(U);

        // TODO: surface interpolation also instantiated in turbulence model -> doubled?!
        auto surfInterpol = fvcc::SurfaceInterpolation<NeoN::scalar>(
            rt.exec,
            rt.nfMesh,
            NeoN::TokenList({std::string("linear")})
        );
        NeoN::scalar cumulativeContErr = 0.0;

        auto uSolver = nf::Solver(U, rt);
        auto pSolver = nf::Solver(p, rt);

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        NeoN::Logging::info("Starting time loop");
        while (runTime.loop())
        {
            // runTime.loop() has already advanced OpenFOAM to the current step. Mirror that into
            // the adapter time before logging: rt.t is set to startTime at construction and only
            // refreshed later in syncRunTimes, so logging it here would report the previous step's
            // time -- making NeoFOAM's time column start at 0 and lag OpenFOAM by one step when
            // comparing per-step timings. syncRunTimes still runs below for the dt adjustment.
            rt.t = runTime.time().value();
            NeoN::Logging::info("Time = {}", rt.t);

            fvcc::rotateOldTimes(U);
            fvcc::rotateOldTimes(phi);
            turb->rotateOldTimes();

            auto [maxCoNum, meanCoNum] = fvcc::computeCoNum(phi, rt.dt);
            NeoN::Logging::info("Courant Number mean: {} max: {}", meanCoNum, maxCoNum);
            nf::syncRunTimes(runTime, rt, maxCoNum);

            // Momentum predictor
            nf::PDE<NeoN::Vec3> UEqn(
                dsl::imp::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(turb->nuEff(), U)
                + dsl::exp::viscousStress(nu, turb->nut(), turb->gradU())
            );

            const auto ddtScheme = UEqn.ddtScheme();

            if (piso.momentumPredictor())
            {
                uSolver.solve(UEqn, -1.0 * dsl::exp::grad(p));
            }
            else
            {
                uSolver.assemble(UEqn);
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
                    nf::PDE<NeoN::scalar> pEqn(
                        NeoN::dsl::imp::laplacian(rAU, p) - NeoN::dsl::exp::div(phiHbyA),
                        p,
                        rt
                    );

                    // updateFaceVelocity reconstructs phi from this pressure system; keep the
                    // non-orthogonal faceFluxCorrection so the reconstruction can add it back.
                    // TODO find a more suitable spot
                    pEqn.linearSystem().keepFaceFluxCorrection(true);

                    if (ofP.needReference() && pRefCell >= 0)
                    {
                        pEqn.setReference(pRefCell, pRefValue);
                    }

                    pSolver.solve(pEqn);
                    p.correctBoundaryConditions();

                    if (piso.finalNonOrthogonalIter())
                    {
                        nf::updateFaceVelocity(phiHbyA, pEqn, phi);
                    }
                }
                nf::reportContinuityError(phi, rt, cumulativeContErr);

                nf::updateVelocity(hByA, crAU, p, U, *gradPOp);
                U.correctBoundaryConditions();
            }

            turb->correct(U, phi, rt);

            runTime.write();
            if (runTime.outputTime())
            {
                NeoN::Logging::info("Writing p");
                write(p, mesh);
                NeoN::Logging::info("Writing U");
                write(U, mesh);
                NeoN::Logging::info("Writing turbulence variables");
                turb->write(mesh);
            }

            runTime.printExecutionTime(Info);
        }
    }
    NeoN::finalize();

    return 0;
}

// ************************************************************************* //
