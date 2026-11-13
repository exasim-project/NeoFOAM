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
// Included explicitly rather than relying on the generated NeoFOAM.hpp: that header is
// produced by a `file(GLOB_RECURSE ... *.hpp)` in include/CMakeLists.txt WITHOUT
// CONFIGURE_DEPENDS, so a newly added header is invisible until someone re-runs cmake.
#include "NeoFOAM/algorithms/velocityDamping.hpp"

#include "fvCFD.H"
#include "simpleControl.H"
#include "singlePhaseTransportModel.H"

#include <memory>
#include <string>
#include <vector>

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

        // ---- fvOptions: velocityDampingConstraint ------------------------------------------
        // NeoFOAM has no fvOptions layer. Rather than let a case that ships
        // system/fvOptions run silently without its constraints, read the one
        // constraint this solver supports and apply it through PDE::setPostRelaxConstraint.
        // Anything else in the file is reported as unsupported instead of ignored.
        struct VelocityDamping
        {
            std::string name;
            NeoN::scalar UMax;
            NeoN::scalar C;
        };
        std::vector<VelocityDamping> dampings;
        {
            Foam::IOobject fvOptionsHeader(
                "fvOptions",
                runTime.system(),
                mesh,
                Foam::IOobject::READ_IF_PRESENT,
                Foam::IOobject::NO_WRITE
            );

            if (fvOptionsHeader.typeHeaderOk<Foam::IOdictionary>(false))
            {
                Foam::IOdictionary fvOptionsDict(fvOptionsHeader);
                for (const Foam::entry& e : fvOptionsDict)
                {
                    if (!e.isDict()) continue;
                    const Foam::dictionary& sub = e.dict();
                    const Foam::word type = sub.getOrDefault<Foam::word>("type", Foam::word::null);

                    if (type != "velocityDampingConstraint")
                    {
                        NeoN::Logging::warn(
                            "fvOptions '{}' has type '{}', which NeoFOAM does not implement -- "
                            "it is NOT applied (OpenFOAM would apply it).",
                            std::string(e.keyword()),
                            std::string(type)
                        );
                        continue;
                    }

                    // fv::option::active_, read from the option dict itself.
                    if (!sub.getOrDefault<Foam::Switch>("active", true))
                    {
                        NeoN::Logging::info(
                            "fvOptions '{}': velocityDampingConstraint inactive, skipped",
                            std::string(e.keyword())
                        );
                        continue;
                    }

                    // fv::option puts coefficients in <type>Coeffs when present, else the
                    // dict itself.
                    const Foam::dictionary& coeffs =
                        sub.optionalSubDict(type + "Coeffs", Foam::keyType::LITERAL);

                    const Foam::word mode = coeffs.getOrDefault<Foam::word>("selectionMode", "all");
                    if (mode != "all")
                    {
                        NeoN::Logging::warn(
                            "fvOptions '{}': selectionMode '{}' is not supported (only 'all'); "
                            "the constraint is NOT applied.",
                            std::string(e.keyword()),
                            std::string(mode)
                        );
                        continue;
                    }

                    // OpenFOAM applies every option, so their diagonal contributions accumulate.
                    dampings.push_back(
                        {std::string(e.keyword()),
                         coeffs.get<Foam::scalar>("UMax"),
                         coeffs.getOrDefault<Foam::scalar>("C", 1.0)}
                    );
                    const VelocityDamping& d = dampings.back();
                    NeoN::Logging::info(
                        "fvOptions '{}': velocityDampingConstraint active, UMax = {}, C = {}",
                        d.name,
                        d.UMax,
                        d.C
                    );
                }
            }
        }

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

            // simpleFoam/UEqn.H: UEqn.relax()
            UEqn.relax();

            // simpleFoam/UEqn.H: fvOptions.constrain(UEqn), applied to the relaxed system before
            // -grad(p) is added and before the solve. Registered per step because UEqn is
            // rebuilt each iteration.
            if (!dampings.empty())
            {
                UEqn.setPostRelaxConstraint(
                    [&](auto& ls)
                    {
                        for (const auto& d : dampings)
                        {
                            if (d.UMax <= 0.0) continue;
                            Foam::label nDamped = static_cast<Foam::label>(
                                nf::applyVelocityDamping(ls, U, d.UMax, d.C)
                            );
                            Foam::reduce(nDamped, Foam::sumOp<Foam::label>());
                            NeoN::Logging::info(
                                "{}: velocityDampingConstraint damped {} cells, with max limit {}",
                                d.name,
                                nDamped,
                                d.UMax
                            );
                        }
                    }
                );
            }

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
                    nf::addConsistentFluxCorrection(phiHbyA, crAU, crAtU, p, rt.fvSchemesDict);
                    nf::subtractConsistentHbyA(hByA, crAU, crAtU, p);
                }

                auto pPrev = NeoN::dsl::fieldRelaxationSnapshot(p);

                // totalPressure needs phi and U to subtract the dynamic head on inflow
                // faces; without them it degenerates to a fixedValue at p0.
                fvcc::BoundaryContext pCtx;
                pCtx.insert("phi", phi);
                pCtx.insert("U", U);

                while (simple.correctNonOrthogonal())
                {
                    // Refresh the pressure boundary BEFORE assembly. OpenFOAM's fvMatrix
                    // constructor calls psi.boundaryField().updateCoeffs(), so the
                    // total-pressure value built from the current phi/U is what the laplacian
                    // discretises; correcting only after the solve would assemble every
                    // iteration against the previous one's boundary value.
                    p.correctBoundaryConditions(pCtx);

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
                    // Mirrors the psi.correctBoundaryConditions() that closes
                    // fvMatrix::solveSegregated, so the boundary tracks the new solution.
                    p.correctBoundaryConditions(pCtx);

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
                p.correctBoundaryConditions(pCtx);

                nf::updateVelocity(hByA, crAtU, p, U, rt);
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
