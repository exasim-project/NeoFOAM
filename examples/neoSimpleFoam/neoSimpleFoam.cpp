// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

/*---------------------------------------------------------------------------*\
Application
    neoSimpleFoam

Description
    Steady-state incompressible SIMPLE/SIMPLEC solver with turbulence,
    the NeoFOAM/NeoN port of OpenFOAM's simpleFoam.
\*---------------------------------------------------------------------------*/

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"

#include "fvCFD.H"
#include "simpleControl.H"
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
#include "addCheckCaseOptions.H"
    Foam::argList::addOption("executor", "word", "NeoN executor type (Serial/CPU/GPU/default)");
#include "setRootCase.H"
#include "createTime.H"
    NeoN::initialize(argc, argv);
    {
        auto rt = nf::createAdapterRunTime(runTime, args);
        auto& mesh = rt.mesh;

        Foam::simpleControl simple(mesh);

#include "createFields.H"

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("p") = nf::mapFvSolution(solverDict.subDict("p"));
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));
        for (const auto& field : {"k", "omega", "epsilon"})
        {
            if (solverDict.contains(field))
                solverDict.subDict(field) = nf::mapFvSolution(solverDict.subDict(field));
        }
        auto& schemesDict = rt.fvSchemesDict;
        schemesDict = nf::mapFvSchemes(rt.fvSchemesDict);

        fvcc::VectorCollection& vectorCollection =
            fvcc::VectorCollection::instance(rt.db, "VectorCollection");

        auto& p = nf::constructAndRegister(vectorCollection, rt, ofP, false);
        auto& U = nf::constructAndRegister(vectorCollection, rt, ofU, false);

        // grad(p) honours the configured gradSchemes (e.g. cellLimited).
        auto gradPOp = nf::makeGradOperator(rt.exec, rt.nfMesh, rt.fvSchemesDict, "grad(p)");

        NeoN::Logging::info("Creating phi");
        auto& phi = nf::constructAndRegister(vectorCollection, rt, ofPhi, false);

        auto nu = nf::constructFrom(rt.exec, rt.nfMesh, tnu());

        auto turb = nf::TurbulenceModel::create(rt, nu);

        turb->validate(U);

        auto surfInterpol = fvcc::SurfaceInterpolation<NeoN::scalar>(
            rt.exec,
            rt.nfMesh,
            NeoN::TokenList({std::string("linear")})
        );
        NeoN::scalar cumulativeContErr = 0.0;

        NeoN::Logging::info("Starting time loop");
        while (runTime.loop())
        {
            rt.t = runTime.time().value();
            NeoN::Logging::info("Time = {}", rt.t);

            nf::PDESolver<NeoN::Vec3> UEqn(
                dsl::imp::div(phi, U) - dsl::imp::laplacian(turb->nuEff(), U)
                    + dsl::exp::viscousStress(nu, turb->nut(), turb->gradU()),
                U,
                rt
            );

            if (simple.momentumPredictor())
            {
                UEqn.solve(-1.0 * dsl::exp::grad(p));
            }
            else
            {
                UEqn.assembleAndRelax();
            }

            // SIMPLE / SIMPLEC pressure-velocity coupling (single pass, no inner PISO loop)
            {
                auto [crAU, hByA] = nf::computeRAUandHByA(UEqn);
                nf::constrainHbyA(U, p, hByA);

                // SIMPLEC: rAtU absorbs off-diagonal coupling; for plain SIMPLE crAtU == crAU.
                const bool consistent = simple.consistent();
                fvcc::VolumeField<NeoN::scalar> crAtU =
                    consistent ? nf::computeRAtU(UEqn, crAU) : crAU;

                fvcc::SurfaceField<NeoN::scalar> rAU = surfInterpol.interpolate(crAtU);
                rAU.name = "rAUf";

                auto phiHbyA = nf::flux(hByA);

                if (consistent)
                {
                    nf::addConsistentFluxCorrection(phiHbyA, crAU, crAtU, p);
                    nf::subtractConsistentHbyA(hByA, crAU, crAtU, p);
                }

                auto pPrev = NeoN::dsl::fieldRelaxationSnapshot(p);

                while (simple.correctNonOrthogonal())
                {
                    nf::PDESolver<NeoN::scalar> pEqn(
                        NeoN::dsl::imp::laplacian(rAU, p) - NeoN::dsl::exp::div(phiHbyA),
                        p,
                        rt
                    );

                    // Required so updateFaceVelocity can subtract the non-orthogonal
                    // faceFluxCorrection and keep phi consistent with the corrected snGrad.
                    pEqn.linearSystem().keepFaceFluxCorrection(true);

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

                NeoN::dsl::applyFieldRelaxation(
                    p,
                    pPrev,
                    nf::lookupFieldRelaxation(rt.fvSolutionDict, "p", false)
                        .value_or(NeoN::scalar(1))
                );
                p.correctBoundaryConditions();

                nf::updateVelocity(hByA, crAtU, p, U, *gradPOp);
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
