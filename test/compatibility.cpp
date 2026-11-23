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
        SECTION("smoothSolver")
        {
            solver1.insert("solver", std::string("smoothSolver"));
            NeoFOAM::updateSolver(solver1);
            REQUIRE(solver1.get<std::string>("solver") == "Ginkgo");
            REQUIRE(solver1.get<std::string>("type") == "solver::Ir");
        }
        // GAMG maps to a CG outer iteration preconditioned by Ginkgo's Pgm algebraic
        // multigrid: the solver entry becomes Ginkgo/solver::Cg and the multigrid is
        // synthesised into the preconditioner sub-dict, replacing the OpenFOAM one.
        SECTION("GAMG")
        {
            solver1.insert("solver", std::string("GAMG"));
            NeoFOAM::updateSolver(solver1);
            REQUIRE(solver1.get<std::string>("solver") == "Ginkgo");
            REQUIRE(solver1.get<std::string>("type") == "solver::Cg");

            auto& multigrid = solver1.subDict("preconditioner");
            REQUIRE(multigrid.get<std::string>("type") == "solver::Multigrid");
            REQUIRE(multigrid.subDict("mg_level").get<std::string>("type") == "multigrid::Pgm");
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
        SECTION("GaussSeidel smoothSolver -> Jacobi+Ir")
        {
            solver1.remove("preconditioner"); // the case's T entry has DIC
            solver1.insert("solver", std::string("smoothSolver"));
            solver1.insert("smoother", std::string("GaussSeidel"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "Jacobi+Ir");
        }
        SECTION("smoothSolver without smoother -> Jacobi+Ir")
        {
            solver1.remove("preconditioner"); // the case's T entry has DIC
            solver1.insert("solver", std::string("smoothSolver"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "Jacobi+Ir");
        }
        SECTION("configFile -> configFile")
        {
            solver1.insert("configFile", std::string("mySolver.json"));
            auto mapped = NeoFOAM::mapFvSolution(solver1);
            REQUIRE(mapped.get<std::string>("reportName") == "configFile");
        }
    }
}

TEST_CASE("fvSolution keyMatches")
{
    SECTION("exact literal key")
    {
        REQUIRE(NeoFOAM::keyMatches("U", "U"));
        REQUIRE_FALSE(NeoFOAM::keyMatches("U", "p"));
    }

    SECTION("quoted key is a regular expression")
    {
        REQUIRE(NeoFOAM::keyMatches("\"(U|k)\"", "U"));
        REQUIRE(NeoFOAM::keyMatches("\"(U|k)\"", "k"));
        REQUIRE_FALSE(NeoFOAM::keyMatches("\"(U|k)\"", "p"));
    }

    SECTION("unquoted pattern is not a regular expression")
    {
        REQUIRE_FALSE(NeoFOAM::keyMatches("(U|k)", "U"));
        // ... it is only ever an identical keyword
        REQUIRE(NeoFOAM::keyMatches("(U|k)", "(U|k)"));
    }

    SECTION("the pattern must match the whole name")
    {
        REQUIRE_FALSE(NeoFOAM::keyMatches("\"(U|k)\"", "UFinal"));
        REQUIRE_FALSE(NeoFOAM::keyMatches("\"U\"", "Uy"));
        REQUIRE(NeoFOAM::keyMatches("\"(U|k).*\"", "UFinal"));
    }
}

TEST_CASE("fvSolution matchKey")
{
    SECTION("an exact key wins over a regex key that also matches")
    {
        NeoN::Dictionary dict(
            {{std::string("U"), std::string("exact")},
             {std::string("\"(U|k)\""), std::string("pattern")}}
        );
        const auto key = NeoFOAM::matchKey(dict, "U");
        REQUIRE(key.has_value());
        REQUIRE(*key == "U");
    }

    SECTION("a regex key selects a field without its own entry")
    {
        NeoN::Dictionary dict({{std::string("\"(U|k)\""), std::string("pattern")}});
        const auto key = NeoFOAM::matchKey(dict, "k");
        REQUIRE(key.has_value());
        REQUIRE(*key == "\"(U|k)\"");
    }

    SECTION("nothing matches")
    {
        NeoN::Dictionary dict({{std::string("\"(U|k)\""), std::string("pattern")}});
        REQUIRE_FALSE(NeoFOAM::matchKey(dict, "p").has_value());
    }

    SECTION("two regex keys matching the same field are ambiguous")
    {
        NeoN::Dictionary dict(
            {{std::string("\"(U|k)\""), std::string("first")},
             {std::string("\"(U|epsilon)\""), std::string("second")}}
        );
        REQUIRE_THROWS_MATCHES(
            NeoFOAM::matchKey(dict, "U"),
            std::runtime_error,
            Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(
                "Several fvSolution keys match the field 'U': \"(U|epsilon)\", \"(U|k)\""
            ))
        );
    }
}

TEST_CASE("fvSolution solverSettings")
{
    NeoN::Dictionary solvers(
        {{std::string("p"), NeoN::Dictionary({{std::string("solver"), std::string("PCG")}})},
         {std::string("\"(U|k|epsilon)\""),
          NeoN::Dictionary({{std::string("solver"), std::string("PBiCGStab")}})}}
    );

    SECTION("resolves through a regex key")
    {
        REQUIRE(NeoFOAM::hasSolverSettings(solvers, "k"));
        REQUIRE(NeoFOAM::solverSettings(solvers, "k").get<std::string>("solver") == "PBiCGStab");
    }

    SECTION("resolves an exact key")
    {
        REQUIRE(NeoFOAM::hasSolverSettings(solvers, "p"));
        REQUIRE(NeoFOAM::solverSettings(solvers, "p").get<std::string>("solver") == "PCG");
    }

    SECTION("a matching key holding a non-dictionary is no settings entry")
    {
        NeoN::Dictionary notADict({{std::string("U"), std::string("PCG")}});
        REQUIRE_FALSE(NeoFOAM::hasSolverSettings(notADict, "U"));
    }

    SECTION("no matching key")
    {
        REQUIRE_FALSE(NeoFOAM::hasSolverSettings(solvers, "T"));
        REQUIRE_THROWS_MATCHES(
            NeoFOAM::solverSettings(solvers, "T"),
            NeoFOAM::FvSolutionKeyNotFound,
            Catch::Matchers::Message("No entry for field 'T' in system/fvSolution/solvers; "
                                     "available keys: \"(U|k|epsilon)\", p")
        );
    }

    SECTION("the exception carries the field and the available keys")
    {
        try
        {
            NeoFOAM::solverSettings(solvers, "T");
            FAIL("solverSettings did not throw");
        }
        catch (const NeoFOAM::FvSolutionKeyNotFound& e)
        {
            REQUIRE(e.field() == "T");
            REQUIRE(e.dictName() == "system/fvSolution/solvers");
            REQUIRE(
                e.availableKeys()
                == std::vector<std::string> {"\"(U|k|epsilon)\"", std::string("p")}
            );
        }
    }
}

TEST_CASE("fvSolution relaxation lookup")
{
    SECTION("equations and fields are independent")
    {
        NeoN::Dictionary fvSolution(
            {{std::string("relaxationFactors"),
              NeoN::Dictionary(
                  {{std::string("equations"),
                    NeoN::Dictionary({{std::string("U"), NeoN::scalar(0.7)}})},
                   {std::string("fields"), NeoN::Dictionary({{std::string("p"), NeoN::scalar(0.3)}})
                   }}
              )}}
        );
        REQUIRE(NeoFOAM::lookupEqnRelaxation(fvSolution, "U", false) == NeoN::scalar(0.7));
        REQUIRE_FALSE(NeoFOAM::lookupFieldRelaxation(fvSolution, "U", false).has_value());
        REQUIRE(NeoFOAM::lookupFieldRelaxation(fvSolution, "p", false) == NeoN::scalar(0.3));
        REQUIRE_FALSE(NeoFOAM::lookupEqnRelaxation(fvSolution, "p", false).has_value());
    }

    SECTION("the Final key wins on the final iteration")
    {
        NeoN::Dictionary fvSolution(
            {{std::string("relaxationFactors"),
              NeoN::Dictionary(
                  {{std::string("equations"),
                    NeoN::Dictionary(
                        {{std::string("U"), NeoN::scalar(0.7)},
                         {std::string("UFinal"), NeoN::scalar(0.9)}}
                    )}}
              )}}
        );
        REQUIRE(NeoFOAM::lookupEqnRelaxation(fvSolution, "U", true) == NeoN::scalar(0.9));
        REQUIRE(NeoFOAM::lookupEqnRelaxation(fvSolution, "U", false) == NeoN::scalar(0.7));
    }

    SECTION("the final iteration falls back to the base key")
    {
        NeoN::Dictionary fvSolution(
            {{std::string("relaxationFactors"),
              NeoN::Dictionary(
                  {{std::string("equations"),
                    NeoN::Dictionary({{std::string("U"), NeoN::scalar(0.7)}})}}
              )}}
        );
        REQUIRE(NeoFOAM::lookupEqnRelaxation(fvSolution, "U", true) == NeoN::scalar(0.7));
    }

    SECTION("an int entry is coerced to scalar")
    {
        NeoN::Dictionary fvSolution(
            {{std::string("relaxationFactors"),
              NeoN::Dictionary(
                  {{std::string("fields"), NeoN::Dictionary({{std::string("pFinal"), 1}})}}
              )}}
        );
        REQUIRE(NeoFOAM::lookupFieldRelaxation(fvSolution, "p", true) == NeoN::scalar(1));
    }

    SECTION("a regex key resolves both the base and the Final name")
    {
        NeoN::Dictionary fvSolution(
            {{std::string("relaxationFactors"),
              NeoN::Dictionary(
                  {{std::string("equations"),
                    NeoN::Dictionary({{std::string("\"(k|omega|epsilon).*\""), NeoN::scalar(0.7)}})}
                  }
              )}}
        );
        REQUIRE(NeoFOAM::lookupEqnRelaxation(fvSolution, "epsilon", false) == NeoN::scalar(0.7));
        REQUIRE(NeoFOAM::lookupEqnRelaxation(fvSolution, "epsilon", true) == NeoN::scalar(0.7));
        REQUIRE_FALSE(NeoFOAM::lookupEqnRelaxation(fvSolution, "U", true).has_value());
    }

    SECTION("no relaxationFactors entry")
    {
        NeoN::Dictionary fvSolution({{std::string("solvers"), NeoN::Dictionary()}});
        REQUIRE_FALSE(NeoFOAM::lookupEqnRelaxation(fvSolution, "U", false).has_value());
        REQUIRE_FALSE(NeoFOAM::lookupFieldRelaxation(fvSolution, "U", false).has_value());
    }

    SECTION("relaxationFactors is not a dictionary")
    {
        NeoN::Dictionary fvSolution({{std::string("relaxationFactors"), std::string("0.7")}});
        REQUIRE_FALSE(NeoFOAM::lookupEqnRelaxation(fvSolution, "U", false).has_value());
        REQUIRE_FALSE(NeoFOAM::lookupFieldRelaxation(fvSolution, "U", false).has_value());
    }

    SECTION("the equations sub-dict is not a dictionary")
    {
        NeoN::Dictionary fvSolution(
            {{std::string("relaxationFactors"),
              NeoN::Dictionary({{std::string("equations"), std::string("0.7")}})}}
        );
        REQUIRE_FALSE(NeoFOAM::lookupEqnRelaxation(fvSolution, "U", false).has_value());
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

namespace
{

using GkoCg = gko::solver::Cg<NeoN::scalar>;
using GkoMultigrid = gko::solver::Multigrid;
using GkoPgm = gko::multigrid::Pgm<NeoN::scalar, int>;

NeoN::Dictionary rawGamgBlock()
{
    NeoN::Dictionary block;
    block.insert("solver", std::string("GAMG"));
    block.insert("tolerance", NeoN::scalar(1e-6));
    block.insert("relTol", NeoN::scalar(0.01));
    return block;
}

// The form a typed Multigrid block reaches the mapper in.
NeoN::Dictionary gamgBlock()
{
    NeoN::Dictionary block;
    block.insert("solver", std::string("Ginkgo"));
    block.insert("type", std::string("solver::Cg"));
    block.insert("preconditioner", std::string("GAMG"));
    block.insert("tolerance", NeoN::scalar(1e-6));
    block.insert("relTol", NeoN::scalar(0.01));
    return block;
}

std::shared_ptr<const gko::LinOpFactory> ginkgoFactory(const NeoN::Dictionary& mapped)
{
    gko::config::registry registry(NeoN::la::ginkgo::mergedPgmConfigMap<NeoN::scalar>());
    return gko::config::parse(
               NeoN::la::ginkgo::parse(mapped),
               registry,
               gko::config::make_type_descriptor<NeoN::scalar>()
    )
        .on(gko::ReferenceExecutor::create());
}

std::shared_ptr<const GkoMultigrid::Factory> multigridOf(const NeoN::Dictionary& mapped)
{
    const auto cg = gko::as<GkoCg::Factory>(ginkgoFactory(mapped));
    return gko::as<GkoMultigrid::Factory>(cg->get_parameters().preconditioner);
}

gko::size_type smootherSweeps(const std::shared_ptr<const gko::LinOpFactory>& smoother)
{
    const auto& criteria = gko::as<GkoCg::Factory>(smoother)->get_parameters().criteria;
    return gko::as<gko::stop::Iteration::Factory>(criteria.front())->get_parameters().max_iters;
}

} // namespace

TEST_CASE("fvSolution Ginkgo tuning options")
{
    SECTION("GAMG without options equals GAMG with every default spelled out")
    {
        NeoN::Dictionary spelled = gamgBlock();
        spelled.insert("max_levels", 10);
        spelled.insert("cycle", std::string("v"));
        spelled.insert("nCoarsestSweeps", 8);
        spelled.insert("coarsening", NeoN::Dictionary());

        REQUIRE(
            NeoN::la::ginkgo::parse(NeoFOAM::mapFvSolution(gamgBlock()))
            == NeoN::la::ginkgo::parse(NeoFOAM::mapFvSolution(spelled))
        );
    }

    SECTION("multigrid hierarchy and sweeps reach Ginkgo")
    {
        NeoN::Dictionary block = gamgBlock();
        block.insert("max_levels", 7);
        block.insert("cycle", std::string("f"));
        block.insert("nCoarsestSweeps", 12);
        const auto multigrid = multigridOf(NeoFOAM::mapFvSolution(block));

        REQUIRE(multigrid->get_parameters().max_levels == 7);
        REQUIRE(multigrid->get_parameters().cycle == gko::solver::multigrid::cycle::f);
        REQUIRE(smootherSweeps(multigrid->get_parameters().coarsest_solver.front()) == 12);
    }

    SECTION("Pgm numeric options stay in the dictionary")
    {
        NeoN::Dictionary coarsening;
        coarsening.insert("max_iterations", 20);
        coarsening.insert("max_unassigned_ratio", NeoN::scalar(0.1));
        coarsening.insert("deterministic", std::string("false"));
        NeoN::Dictionary block = gamgBlock();
        block.insert("coarsening", coarsening);

        const auto mapped = NeoFOAM::mapFvSolution(block);
        const auto& level = mapped.subDict("preconditioner").subDict("mg_level");
        REQUIRE_FALSE(level.contains("configFile"));

        const auto multigrid = multigridOf(mapped);
        const auto& pgm = gko::as<GkoPgm::Factory>(multigrid->get_parameters().mg_level.front())
                              ->get_parameters();
        REQUIRE(pgm.max_iterations == 20);
        REQUIRE(pgm.max_unassigned_ratio == Catch::Approx(0.1));
        REQUIRE_FALSE(pgm.deterministic);
    }

    SECTION("true Pgm switches reach Ginkgo")
    {
        NeoN::Dictionary coarsening;
        coarsening.insert("max_iterations", 20);
        coarsening.insert("max_unassigned_ratio", NeoN::scalar(0.1));
        coarsening.insert("deterministic", std::string("true"));
        coarsening.insert("skip_sorting", true);
        NeoN::Dictionary block = gamgBlock();
        block.insert("coarsening", coarsening);

        const auto multigrid = multigridOf(NeoFOAM::mapFvSolution(block));
        const auto& pgm = gko::as<GkoPgm::Factory>(multigrid->get_parameters().mg_level.front())
                              ->get_parameters();
        REQUIRE(pgm.max_iterations == 20);
        REQUIRE(pgm.max_unassigned_ratio == Catch::Approx(0.1));
        REQUIRE(pgm.deterministic);
        REQUIRE(pgm.skip_sorting);
    }

    SECTION("krylov_dim reaches Gmres beside a GAMG preconditioner")
    {
        NeoN::Dictionary block;
        block.insert("solver", std::string("Ginkgo"));
        block.insert("type", std::string("solver::Gmres"));
        block.insert("preconditioner", std::string("GAMG"));
        block.insert("krylov_dim", 7);

        const auto gmres = gko::as<gko::solver::Gmres<NeoN::scalar>::Factory>(
            ginkgoFactory(NeoFOAM::mapFvSolution(block))
        );
        REQUIRE(gmres->get_parameters().krylov_dim == 7);
    }

    SECTION("krylov_dim on the GAMG solver is still dropped")
    {
        NeoN::Dictionary block = rawGamgBlock();
        block.insert("krylov_dim", 7);
        REQUIRE_FALSE(NeoFOAM::mapFvSolution(block).contains("krylov_dim"));
    }

    SECTION("raw solver GAMG keeps dropping the Multigrid tuning keys")
    {
        NeoN::Dictionary coarsening;
        coarsening.insert("max_iterations", 20);
        coarsening.insert("deterministic", true);
        NeoN::Dictionary block = rawGamgBlock();
        block.insert("max_levels", 8);
        block.insert("cycle", std::string("f"));
        block.insert("nCoarsestSweeps", 12);
        block.insert("coarsening", coarsening);

        const auto mapped = NeoFOAM::mapFvSolution(block);
        for (const std::string key : {"max_levels", "cycle", "nCoarsestSweeps", "coarsening"})
        {
            REQUIRE_FALSE(mapped.contains(key));
        }
        REQUIRE(
            NeoN::la::ginkgo::parse(mapped)
            == NeoN::la::ginkgo::parse(NeoFOAM::mapFvSolution(rawGamgBlock()))
        );
    }

    SECTION("relaxation_factor reaches Ir, also as a whole number")
    {
        const auto relaxation = GENERATE(as<double> {}, 0.5, 1.0);
        NeoN::Dictionary block;
        block.insert("solver", std::string("smoothSolver"));
        block.insert("smoother", std::string("symGaussSeidel"));
        if (relaxation == 1.0) block.insert("relaxation_factor", 1);
        else
            block.insert("relaxation_factor", NeoN::scalar(relaxation));

        const auto ir = gko::as<gko::solver::Ir<NeoN::scalar>::Factory>(
            ginkgoFactory(NeoFOAM::mapFvSolution(block))
        );
        REQUIRE(ir->get_parameters().relaxation_factor == Catch::Approx(relaxation));
    }

    SECTION("max_block_size reaches Jacobi")
    {
        NeoN::Dictionary block;
        block.insert("solver", std::string("PCG"));
        block.insert("preconditioner", std::string("diagonal"));
        block.insert("max_block_size", 4);

        const auto cg = gko::as<GkoCg::Factory>(ginkgoFactory(NeoFOAM::mapFvSolution(block)));
        const auto jacobi = gko::as<gko::preconditioner::Jacobi<NeoN::scalar, int>::Factory>(
            cg->get_parameters().preconditioner
        );
        REQUIRE(jacobi->get_parameters().max_block_size == 4);
    }

    SECTION("invalid options fail at mapping")
    {
        NeoN::Dictionary badCycle = gamgBlock();
        badCycle.insert("cycle", std::string("x"));
        REQUIRE_THROWS_AS(NeoFOAM::mapFvSolution(badCycle), std::runtime_error);

        NeoN::Dictionary unknownEntry;
        unknownEntry.insert("merge_levels", 2);
        NeoN::Dictionary badCoarsening = gamgBlock();
        badCoarsening.insert("coarsening", unknownEntry);
        REQUIRE_THROWS_AS(NeoFOAM::mapFvSolution(badCoarsening), std::runtime_error);

        NeoN::Dictionary badWord;
        badWord.insert("deterministic", std::string("maybe"));
        NeoN::Dictionary badSwitch = gamgBlock();
        badSwitch.insert("coarsening", badWord);
        REQUIRE_THROWS_AS(NeoFOAM::mapFvSolution(badSwitch), std::runtime_error);

        NeoN::Dictionary wrongPreconditioner;
        wrongPreconditioner.insert("solver", std::string("PCG"));
        wrongPreconditioner.insert("preconditioner", std::string("DIC"));
        wrongPreconditioner.insert("max_block_size", 4);
        REQUIRE_THROWS_AS(NeoFOAM::mapFvSolution(wrongPreconditioner), std::runtime_error);
    }
}
