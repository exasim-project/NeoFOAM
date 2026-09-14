// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

#include <cstddef>
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

#include "gaussConvectionScheme.H"

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object


TEST_CASE("fvSolution")
{
    Foam::Time& runTime = *timePtr;
    Foam::argList& args = *argsPtr;

    std::string execName = "Serial";
    auto exec = NeoN::SerialExecutor {};

    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    NeoN::Dictionary fvSolutionDict = NeoFOAM::convert(mesh.solutionDict());
    NeoN::Dictionary& solverDict = fvSolutionDict.subDict("solvers");
    NeoN::Dictionary& solver1 = solverDict.subDict("T");

    SECTION("updateSolver")
    {
        SECTION("PCG")
        {
            solver1.insert("solver", std::string("PCG"));
            NeoFOAM::updateSolver(solver1);
            REQUIRE(solver1.get<std::string>("solver") == "Ginkgo");
            REQUIRE(solver1.get<std::string>("type") == "solver::Cg");
        }
        SECTION("PBiCG")
        {
            solver1.insert("solver", std::string("PBiCG"));
            NeoFOAM::updateSolver(solver1);
            REQUIRE(solver1.get<std::string>("solver") == "Ginkgo");
            REQUIRE(solver1.get<std::string>("type") == "solver::Bicg");
        }
        SECTION("PBiCGStab")
        {
            solver1.insert("solver", std::string("PBiCGStab"));
            NeoFOAM::updateSolver(solver1);
            REQUIRE(solver1.get<std::string>("solver") == "Ginkgo");
            REQUIRE(solver1.get<std::string>("type") == "solver::Bicgstab");
        }
        // GAMG has no dictionary-level mapping: Ginkgo's Multigrid needs mg_level /
        // coarse_solver entries this mapper cannot synthesise. It must be rejected here
        // rather than reaching Ginkgo, which fails with an opaque config error instead.
        SECTION("GAMG is rejected")
        {
            solver1.insert("solver", std::string("GAMG"));
            REQUIRE_THROWS_AS(NeoFOAM::updateSolver(solver1), std::runtime_error);
        }
    }

    SECTION("updatePreconditioner")
    {
        SECTION("diagonal")
        {
            solver1.insert("preconditioner", std::string("diagonal"));
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Jacobi");
            REQUIRE(preconditionerDict.get<int>("max_block_size") == 1);
        }
        SECTION("DIC")
        {
            solver1.insert("preconditioner", std::string("DIC"));
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ic");
        }
        SECTION("DILU")
        {
            solver1.insert("preconditioner", std::string("DILU"));
            NeoFOAM::updatePreconditioner(solver1);
            auto& preconditionerDict = solver1.subDict("preconditioner");
            REQUIRE(preconditionerDict.get<std::string>("type") == "preconditioner::Ilu");
            REQUIRE(
                preconditionerDict.subDict("factorization").get<std::string>("type")
                == "factorization::ParIlu"
            );
        }
    }

    // The mapped dictionary carries a "reportName" label (Ginkgo
    // preconditioner+solver) that the per-solve residual report prints. These
    // run serially, so the serial (non-Schwarz) preconditioner names apply.
    SECTION("mapFvSolution reportName")
    {
        SECTION("DIC + PCG -> Ic+Cg")
        {
            solver1.insert("solver", std::string("PCG"));
            solver1.insert("preconditioner", std::string("DIC"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "Ic+Cg");
        }
        SECTION("DILU + PBiCGStab -> Ilu+Bicgstab")
        {
            solver1.insert("solver", std::string("PBiCGStab"));
            solver1.insert("preconditioner", std::string("DILU"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "Ilu+Bicgstab");
        }
        SECTION("diagonal + PCG -> Jacobi+Cg")
        {
            solver1.insert("solver", std::string("PCG"));
            solver1.insert("preconditioner", std::string("diagonal"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "Jacobi+Cg");
        }
        SECTION("configFile -> configFile")
        {
            solver1.insert("configFile", std::string("mySolver.json"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "configFile");
        }
    }
}


// mapFvSchemes rewrites the scheme dictionaries so NeoN's factories see specs they can resolve.
// Two rewrites need covering, both of which silently degrade the convection term when missing:
// linearUpwind's reconstruction gradient arrives as a *key* into gradSchemes, and limitedLinear's
// limiter gradient is not in its spec at all (OpenFOAM looks it up as grad(<field>)).
TEST_CASE("fvSchemes")
{
    using NeoN::TokenList;

    // Renders a token list as space-separated words/numbers so a spec can be asserted in one go.
    auto spell = [](const TokenList& tl)
    {
        std::string out;
        // Named copy: tokens() hands back a reference into the list, so binding it straight from
        // a temporary would dangle before the loop body runs.
        TokenList copy(tl);
        for (const auto& tok : copy.tokens())
        {
            if (!out.empty()) out += " ";
            if (tok.type() == typeid(std::string)) out += std::any_cast<const std::string&>(tok);
            else if (tok.type() == typeid(NeoN::scalar))
                out += std::to_string(std::any_cast<NeoN::scalar>(tok));
            else if (tok.type() == typeid(NeoN::label))
                out += std::to_string(std::any_cast<NeoN::label>(tok));
            else
                out += "?";
        }
        return out;
    };

    auto mapDiv = [&](const TokenList& div,
                      const std::string& divKey,
                      const TokenList& grad,
                      const std::string& gradKey)
    {
        NeoN::Dictionary divSchemes;
        divSchemes.insert(divKey, div);
        NeoN::Dictionary gradSchemes;
        gradSchemes.insert(gradKey, grad);

        NeoN::Dictionary schemes;
        schemes.insert("divSchemes", divSchemes);
        schemes.insert("gradSchemes", gradSchemes);

        auto mapped = NeoFOAM::mapFvSchemes(schemes);
        return spell(mapped.subDict("divSchemes").get<TokenList>(divKey));
    };

    const TokenList cellLimitedGrad {
        std::string("cellLimited"),
        std::string("Gauss"),
        std::string("linear"),
        NeoN::scalar(1)
    };
    const TokenList plainGrad {std::string("Gauss"), std::string("linear")};

    SECTION("linearUpwind gradient key expands to its gradSchemes definition")
    {
        const TokenList div {
            std::string("bounded"),
            std::string("Gauss"),
            std::string("linearUpwind"),
            std::string("limited")
        };
        REQUIRE(
            mapDiv(div, "div(phi,U)", cellLimitedGrad, "limited")
            == "bounded Gauss linearUpwind cellLimited Gauss linear 1.000000"
        );
    }

    SECTION("linearUpwind gradient key falls back to default when it has no entry")
    {
        // OpenFOAM's schemesLookup::lookupDetail::lookup returns the "default" entry when the
        // named one is absent, so "linearUpwind grad(U)" against a gradSchemes that only
        // defines "default" must still resolve. Left unexpanded, NeoN sees the bare key
        // "grad(U)", fails to recognise it as cellLimited, and silently uses its unlimited
        // Gauss-Green gradient -- the very defect this mapping exists to prevent.
        const TokenList div {
            std::string("bounded"),
            std::string("Gauss"),
            std::string("linearUpwind"),
            std::string("grad(U)")
        };
        REQUIRE(
            mapDiv(div, "div(phi,U)", cellLimitedGrad, "default")
            == "bounded Gauss linearUpwind cellLimited Gauss linear 1.000000"
        );
    }

    SECTION("bounded survives the rewrite")
    {
        // BoundedDiv implements the prefix; stripping it dropped the Sp(div(phi), psi) term that
        // compensates a steady run's continuity error.
        const TokenList div {
            std::string("bounded"),
            std::string("Gauss"),
            std::string("linearUpwind"),
            std::string("limited")
        };
        REQUIRE(mapDiv(div, "div(phi,U)", cellLimitedGrad, "limited").rfind("bounded", 0) == 0);
    }

    SECTION("limitedLinear is marked cellLimited after a cellLimited grad(<field>)")
    {
        const TokenList div {
            std::string("bounded"),
            std::string("Gauss"),
            std::string("limitedLinear"),
            NeoN::scalar(1)
        };
        REQUIRE(
            mapDiv(div, "div(phi,k)", cellLimitedGrad, "grad(k)")
            == "bounded Gauss limitedLinear 1.000000 cellLimited"
        );
    }

    SECTION("limitedLinear is left alone after an unlimited grad(<field>)")
    {
        const TokenList div {
            std::string("bounded"),
            std::string("Gauss"),
            std::string("limitedLinear"),
            NeoN::scalar(1)
        };
        REQUIRE(
            mapDiv(div, "div(phi,epsilon)", plainGrad, "grad(epsilon)")
            == "bounded Gauss limitedLinear 1.000000"
        );
    }

    SECTION("a field without its own grad entry falls back to default")
    {
        const TokenList div {
            std::string("bounded"),
            std::string("Gauss"),
            std::string("limitedLinear"),
            NeoN::scalar(1)
        };
        REQUIRE(
            mapDiv(div, "div(phi,k)", cellLimitedGrad, "default")
            == "bounded Gauss limitedLinear 1.000000 cellLimited"
        );
    }

    SECTION("a div key that names no transported field is left alone")
    {
        const TokenList div {std::string("Gauss"), std::string("limitedLinear"), NeoN::scalar(1)};
        REQUIRE(
            mapDiv(div, "div((nuEff*dev2(T(grad(U)))))", cellLimitedGrad, "grad(U)")
            == "Gauss limitedLinear 1.000000"
        );
    }
}
