// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/DSL/eqnSystem.hpp"

#include "testOperator.hpp"

namespace dsl = NeoFOAM::DSL;
namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

NeoFOAM::scalar getField(const NeoFOAM::Field<NeoFOAM::scalar>& source)
{
    auto sourceField = source.copyToHost();
    return sourceField.span()[0];
}

TEST_CASE("DSL")
{
    auto exec = NeoFOAM::SerialExecutor();
    size_t nCells = 1;
    NeoFOAM::scalar value = 1;
    dsl::EqnTerm<NeoFOAM::scalar> lapTerm =
        Laplacian(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells, value);

    REQUIRE("Laplacian" == lapTerm.display());

    dsl::EqnTerm<NeoFOAM::scalar> divTerm =
        Divergence(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells, value);

    REQUIRE("Divergence" == divTerm.display());
    {
        NeoFOAM::Field<NeoFOAM::scalar> source(exec, 1);
        NeoFOAM::fill(source, 0.0);

        lapTerm.explicitOperation(source);
        divTerm.explicitOperation(source);

        REQUIRE(getField(source) == 2.0);
    }

    {
        dsl::EqnSystem eqnSys = lapTerm + divTerm;
        REQUIRE(eqnSys.size() == 2);
        REQUIRE(getField(eqnSys.explicitOperation()) == 2);
    }
    BENCHMARK("Creation from EqnTerm")
    {
        dsl::EqnSystem eqnSys = lapTerm + divTerm;
        return eqnSys;
    };

    {
        dsl::EqnSystem eqnSys(lapTerm + lapTerm + divTerm + divTerm);
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(getField(eqnSys.explicitOperation()) == 4.0);
    }
    BENCHMARK("Creation from multiple terms")
    {
        dsl::EqnSystem eqnSys2(lapTerm + lapTerm + divTerm + divTerm);
        return eqnSys2;
    };

    {
        dsl::EqnSystem eqnSys = lapTerm - divTerm;
        REQUIRE(eqnSys.size() == 2);
        REQUIRE(getField(eqnSys.explicitOperation()) == 0.0);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm - lapTerm - divTerm - divTerm);
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(getField(eqnSys.explicitOperation()) == -2.0);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm + lapTerm + divTerm + divTerm);
        dsl::EqnSystem eqnSys2(lapTerm + lapTerm + divTerm + divTerm);
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(getField(eqnSys.explicitOperation()) == 4.0);
        dsl::EqnSystem combinedEqnSys = eqnSys + eqnSys2;
        REQUIRE(combinedEqnSys.size() == 8);
        REQUIRE(getField(combinedEqnSys.explicitOperation()) == 8.0);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm + lapTerm - divTerm - divTerm);
        dsl::EqnSystem eqnSys2(lapTerm - lapTerm - divTerm - divTerm);
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(getField(eqnSys.explicitOperation()) == 0.0);
        REQUIRE(eqnSys2.size() == 4);
        REQUIRE(getField(eqnSys2.explicitOperation()) == -2.0);

        SECTION("multiplying eqnSys by 2")
        {
            dsl::EqnSystem multiplyEqnSys = 2.0 * eqnSys2;
            REQUIRE(multiplyEqnSys.size() == 4);
            REQUIRE(getField(multiplyEqnSys.explicitOperation()) == -4.0);
        }

        SECTION("adding eqnSys to eqnSys2")
        {
            dsl::EqnSystem addEqnSys = eqnSys2 + eqnSys;
            REQUIRE(addEqnSys.size() == 8);
            REQUIRE(getField(addEqnSys.explicitOperation()) == -2.0);
        }
        SECTION("subtracting eqnSys from eqnSys2")
        {
            dsl::EqnSystem subtractEqnSys = eqnSys - eqnSys2;
            REQUIRE(subtractEqnSys.size() == 8);
            REQUIRE(getField(subtractEqnSys.explicitOperation()) == 2.0);
        }
    }
    // profiling
    // with different number of terms
    BENCHMARK("Creation from 2 terms")
    {
        dsl::EqnSystem eqnSys(divTerm + lapTerm);

        return eqnSys;
    };

    BENCHMARK("Creation from 4 terms")
    {
        dsl::EqnSystem eqnSys(divTerm + lapTerm + lapTerm + lapTerm);

        return eqnSys;
    };

    BENCHMARK("Creation from 8 terms")
    {
        dsl::EqnSystem eqnSys(
            divTerm + lapTerm + lapTerm + lapTerm + lapTerm + lapTerm + lapTerm + lapTerm
        );

        return eqnSys;
    };

    BENCHMARK("Creation from 16 terms")
    {
        dsl::EqnSystem eqnSys(
            divTerm + lapTerm + lapTerm + lapTerm + lapTerm + lapTerm + lapTerm + lapTerm + lapTerm
            + lapTerm + lapTerm + lapTerm + lapTerm + lapTerm + lapTerm
        );

        return eqnSys;
    };
}


auto createLaplacian(const NeoFOAM::Executor& exec, size_t nCells)
{
    return Laplacian(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells);
}

auto createDivergence(const NeoFOAM::Executor& exec, size_t nCells)
{
    return Divergence(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells);
}


TEST_CASE("build eqnsystem from unevaluated terms")
{
    auto exec = NeoFOAM::SerialExecutor();
    size_t nCells = 1;
    NeoFOAM::Dictionary dict {{"value", 1.0}};

    SECTION("addition")
    {
        dsl::EqnSystem eqnSys(
            createLaplacian(exec, nCells) + createLaplacian(exec, nCells)
            + createDivergence(exec, nCells) + createDivergence(exec, nCells)
        );
        eqnSys.build(dict);
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(getField(eqnSys.explicitOperation()) == 4.0);
    }

    SECTION("subtraction")
    {
        dsl::EqnSystem eqnSys(
            createLaplacian(exec, nCells) - createLaplacian(exec, nCells)
            - createDivergence(exec, nCells) - createDivergence(exec, nCells)
        );
        eqnSys.build(dict);
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(getField(eqnSys.explicitOperation()) == 1.0 - 1.0 - 1.0 - 1.0);
    }

    SECTION("with scaling")
    {
        dsl::EqnSystem eqnSys(
            4.0 * createLaplacian(exec, nCells) - 2 * createLaplacian(exec, nCells)
            - createDivergence(exec, nCells) - createDivergence(exec, nCells)
        );
        eqnSys.build(dict);
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(getField(eqnSys.explicitOperation()) == 4.0 - 2.0 - 1.0 - 1.0);
    }

    SECTION("with scaling")
    {
        dsl::EqnSystem eqnSys(
            4.0 * createLaplacian(exec, nCells) - 2 * createLaplacian(exec, nCells)
            - createDivergence(exec, nCells) - createDivergence(exec, nCells)
        );
        eqnSys.build(dict);
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(getField(eqnSys.explicitOperation()) == 4.0 - 2.0 - 1.0 - 1.0);
    }
}
