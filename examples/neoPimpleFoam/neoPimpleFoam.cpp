// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

/*---------------------------------------------------------------------------*\
Application
    neoPimpleFoam

Description
    Transient incompressible PIMPLE solver (PISO inner correctors inside an outer
    under-relaxed PIMPLE loop), the NeoFOAM/NeoN port of OpenFOAM's pimpleFoam.
\*---------------------------------------------------------------------------*/

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"
#include "NeoFOAM/solutionControl/pimpleControl.hpp"
#include "NeoFOAM/turbulenceModels/turbulenceModel.hpp"

#include "fvCFD.H"
#include "pisoControl.H"
#include "singlePhaseTransportModel.H"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>

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
    Foam::argList::addOption("executor", "word", "NeoN executor type (Serial/CPU/GPU/default)");
#include "addCheckCaseOptions.H"
#include "setRootCase.H"
#include "createTime.H"
    NeoN::initialize(argc, argv);
    {
        auto rt = nf::createAdapterRunTime(runTime, args);
        auto& mesh = rt.mesh;

        // Inner-loop counting (nCorrectors, nNonOrthogonalCorrectors) lives in the "PIMPLE"
        // subdict; read from there so a stock pimpleFoam case with no "PISO" block works.
        Foam::pisoControl piso(mesh, "PIMPLE");

#include "createFields.H"

        nf::createMappedFvSolutionDicts(rt);
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(rt.db, "VectorCollection");

        auto& p = nf::constructAndRegister(vectorCollection, rt, ofP, false);
        auto& U = nf::constructAndRegister(vectorCollection, rt, ofU, false);

        // Gradient operator for the explicit deviatoric viscous-stress term.
        // The full viscous term is div(nuEff*(grad(U) + grad(U)^T)) = laplacian(nuEff,U)
        // + div(nuEff*dev2(T(grad(U)))); the implicit laplacian alone is not sufficient.
        // grad(U) and grad(p) honour the configured gradSchemes (e.g. cellLimited).
        auto gradOp = nf::makeGradOperator(rt.exec, rt.nfMesh, rt.fvSchemesDict, "grad(U)");
        auto gradPOp = nf::makeGradOperator(rt.exec, rt.nfMesh, rt.fvSchemesDict, "grad(p)");

        NeoN::Logging::info("Creating phi");
        auto& phi = nf::constructAndRegister(vectorCollection, rt, ofPhi, false);

        auto nu = nf::constructFrom(rt.exec, rt.nfMesh, tnu());

        auto turb = nf::TurbulenceModel::create(rt, nu);

        turb->validate(U);

        // Hoisted reuse buffer for the 2nd+ outer correctors' current-U velocity gradient,
        // allocated ONCE here and refilled in place via gradTensor(U, localGradU) when needed —
        // so no VolumeField<Tensor> (~nCells*9 scalars) is allocated per outer corrector.
        fvcc::VolumeField<NeoN::Tensor> localGradU(
            rt.exec,
            "gradU",
            rt.nfMesh,
            fvcc::createCalculatedProcBCs<fvcc::VolumeBoundary<NeoN::Tensor>>(rt.nfMesh)
        );
        NeoN::fill(localGradU.internalVector(), NeoN::zero<NeoN::Tensor>());
        gradOp->gradTensor(U, localGradU, dsl::Coeff {});

        // Hoist the surface interpolation once to avoid re-constructing it per inner corrector.
        auto surfInterpol = fvcc::SurfaceInterpolation<NeoN::scalar>(
            rt.exec,
            rt.nfMesh,
            NeoN::TokenList({std::string("linear")})
        );

        NeoN::scalar cumulativeContErr = 0.0;

        auto uSolver = nf::Solver(U, rt);
        auto pSolver = nf::Solver(p, rt);

        nf::PimpleControl pimpleLoop(rt.fvSolutionDict);

        // Max-component reduction: reduce Ux/Uy/Uz entries to a single {init, final} pair.
        auto reduceU = [](const NeoN::la::SolverStats& s) -> std::pair<NeoN::scalar, NeoN::scalar>
        {
            NeoN::scalar mi = 0.0;
            NeoN::scalar mf = 0.0;
            for (const auto& e : s.entries)
            {
                mi = std::max(mi, e.initResNorm);
                mf = std::max(mf, e.finalResNorm);
            }
            return {mi, mf};
        };

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        NeoN::Logging::info("Starting time loop");
        while (runTime.loop())
        {
            NeoN::Logging::info("Time = {}", rt.t);

            fvcc::rotateOldTimes(U);
            fvcc::rotateOldTimes(phi);
            turb->rotateOldTimes();

            auto [maxCoNum, meanCoNum] = fvcc::computeCoNum(phi, rt.dt);
            NeoN::Logging::info("Courant Number mean: {} max: {}", meanCoNum, maxCoNum);
            nf::syncRunTimes(runTime, rt, maxCoNum);

            nf::ResidualMap residuals;
            while (pimpleLoop.loop(residuals))
            {
                NeoN::Logging::info("PIMPLE outer corrector");
                const bool finalIter = pimpleLoop.finalIter();

                // Snapshot p at the TOP of each outer corrector (deep copy of
                // the internal vector) so we can blend against it after the pressure solve.
                auto prevP = NeoN::dsl::fieldRelaxationSnapshot(p);

                // On the first outer corrector reuse turb->gradU() (already at U^n) to avoid
                // an extra grad(U) allocation; later correctors update localGradU in place.
                const fvcc::VolumeField<NeoN::Tensor>* gradUPtr = nullptr;
                if (pimpleLoop.firstIter())
                {
                    gradUPtr = &turb->gradU();
                }
                else
                {
                    gradOp->gradTensor(U, localGradU, dsl::Coeff {});
                    gradUPtr = &localGradU;
                }
                const auto& gradU = *gradUPtr;

                nf::PDE<NeoN::Vec3> UEqn(
                    dsl::imp::ddt(U) + dsl::imp::div(phi, U) - dsl::imp::laplacian(turb->nuEff(), U)
                    + dsl::exp::viscousStress(nu, turb->nut(), gradU)
                );

                const auto ddtScheme = UEqn.ddtScheme();
                NF_ASSERT(
                    ddtScheme != fvcc::DdtScheme::None,
                    "neoPimpleFoam: steadyState ddt unsupported in v1 (BDF1/BDF2 only)"
                );

                UEqn.setFinalIter(finalIter);

                if (piso.momentumPredictor())
                {
                    auto statsU = uSolver.solve(UEqn, -1.0 * dsl::exp::grad(p));
                    residuals["U"] = reduceU(statsU);
                }
                else
                {
                    // Relax unconditionally so computeRAUandHByA reads the relaxed diagonal
                    // regardless of whether the momentum predictor runs.
                    uSolver.assembleAndRelax(UEqn);
                }

                std::pair<NeoN::scalar, NeoN::scalar> pRes {};
                bool havePRes = false;

                // --- PISO inner-corrector loop (Foam::pisoControl)
                while (piso.correct())
                {
                    NeoN::Logging::info("PISO loop");
                    auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
                    nf::constrainHbyA(U, p, hByA);

                    nnfvcc::SurfaceField<NeoN::scalar> rAU = surfInterpol.interpolate(crAU);
                    rAU.name = "rAUf";

                    auto phiHbyA =
                        nf::flux(hByA) + rAU * fvcc::ddtFluxCorr(U, phi, rt.dt, ddtScheme);

                    // TODO additionally missing:
                    // TODO adjustPhi(phiHbyA, U, p);
                    // TODO constrainPressure(p, U, phiHbyA, rAU);

                    // Non-orthogonal pressure corrector loop
                    while (piso.correctNonOrthogonal())
                    {
                        // Pressure corrector
                        nf::PDE<NeoN::scalar> pEqn(
                            NeoN::dsl::imp::laplacian(rAU, p) - NeoN::dsl::exp::div(phiHbyA)
                        );

                        pEqn.setFinalIter(finalIter);

                        if (ofP.needReference() && pRefCell >= 0)
                        {
                            pEqn.setReference(pRefCell, pRefValue);
                        }

                        auto statsP = pSolver.solve(pEqn);
                        if (!havePRes)
                        {
                            pRes = {statsP.entries[0].initResNorm, statsP.entries[0].finalResNorm};
                            havePRes = true;
                        }
                        p.correctBoundaryConditions();

                        if (piso.finalNonOrthogonalIter())
                        {
                            nf::updateFaceVelocity(phiHbyA, pEqn, phi);
                        }
                    }

                    NeoN::dsl::applyFieldRelaxation(
                        p,
                        prevP,
                        nf::lookupFieldRelaxation(rt.fvSolutionDict, p.name, finalIter)
                            .value_or(1.0)
                    );
                    p.correctBoundaryConditions();

                    nf::reportContinuityError(phi, rt, cumulativeContErr);

                    nf::updateVelocity(hByA, crAU, p, U, *gradPOp);
                    U.correctBoundaryConditions();
                }

                if (havePRes)
                {
                    residuals["p"] = pRes;
                }
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
