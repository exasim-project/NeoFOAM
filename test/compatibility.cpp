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
