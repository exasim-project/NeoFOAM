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
#include "NeoFOAM/DSL/eqnTermBuilder.hpp"

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
            std::cout << "subtracting eqnSys from eqnSys2" << std::endl;
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

dsl::eqnTermBuilder<NeoFOAM::scalar> createLaplacian(NeoFOAM::scalar value)
{
    size_t nCells = 1;

    dsl::eqnTermBuilder<NeoFOAM::scalar> builder;
    builder.push_back(
        [&](const NeoFOAM::Input&)
        {
            return Laplacian(
                dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit,
                NeoFOAM::SerialExecutor(),
                nCells,
                value
            );
        }
    );
    return builder;
}

dsl::eqnTermBuilder<NeoFOAM::scalar> createDivergence(NeoFOAM::scalar value)
{
    size_t nCells = 1;

    dsl::eqnTermBuilder<NeoFOAM::scalar> builder;
    builder.push_back(
        [&](const NeoFOAM::Input&)
        {
            return Divergence(
                dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit,
                NeoFOAM::SerialExecutor(),
                nCells,
                value
            );
        }
    );
    return builder;
}

TEST_CASE("build EqnSystem from Terms")
{
    SECTION("test basics")
    {
        size_t nCells = 1;
        NeoFOAM::scalar value = 1;

        auto exec = NeoFOAM::SerialExecutor();
        dsl::EqnTerm<NeoFOAM::scalar> lapTerm =
            Laplacian(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells, value);
        dsl::eqnTermBuilder<NeoFOAM::scalar> builder;
        NeoFOAM::Input input;

        // Add some build functions
        builder.push_back(
            [&](const NeoFOAM::Input&) {
                return Laplacian(
                    dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells, value
                );
            }
        );
        builder.push_back(
            [&](const NeoFOAM::Input&) {
                return Laplacian(
                    dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells, value
                );
            }
        );
        builder.push_back(
            [&](const NeoFOAM::Input&) {
                return Laplacian(
                    dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells, value
                );
            }
        );

        std::vector<dsl::EqnTerm<NeoFOAM::scalar>> terms;
        for (auto& buildFunction : builder)
        {
            auto term = buildFunction(input);
            std::string display = term.display();
            REQUIRE(display == "Laplacian");
            terms.push_back(term);
        }

        REQUIRE(terms.size() == 3);
    }

    SECTION("Construction of EqnSystem from EqnTermBuilder")
    {
        NeoFOAM::Input input;
        dsl::eqnTermBuilder<NeoFOAM::scalar> builder;
        auto lapBuilder = createLaplacian(1.0);
        auto divBuilder = createDivergence(2.0);
        builder.push_back(lapBuilder);
        builder.push_back(divBuilder);

        std::vector<dsl::EqnTerm<NeoFOAM::scalar>> terms;
        for (auto& buildFunction : builder)
        {
            auto term = buildFunction(input);
            std::string display = term.display();
            REQUIRE((display == "Laplacian" || display == "Divergence"));
            terms.push_back(term);
        }

        REQUIRE(terms.size() == 2);

        SECTION("addition")
        {
            auto testBuilder = createLaplacian(3.0) + createDivergence(4.0);
            REQUIRE(terms.size() == 2);
        }

        SECTION("subtraction")
        {
            auto testBuilder = createLaplacian(3.0) - createDivergence(4.0);
            REQUIRE(terms.size() == 2);
        }
    }
}
