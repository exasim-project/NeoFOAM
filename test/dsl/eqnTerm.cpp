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
    const auto hostField = source.copyToHost();
    return hostField.span()[0];
}

NeoFOAM::Field<NeoFOAM::scalar> evaluateTerm(dsl::EqnTerm<NeoFOAM::scalar>& term)
{
    NeoFOAM::Field<NeoFOAM::scalar> source(term.exec(), term.nCells());
    NeoFOAM::fill(source, 0.0);
    term.explicitOperation(source);
    return source;
}


TEST_CASE("EqnTerm")
{
    // size_t nCells = GENERATE(1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7);
    size_t nCells = 1e3;

    // NeoFOAM::Executor exec = NeoFOAM::SerialExecutor();
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    NeoFOAM::Field<NeoFOAM::scalar> a(exec, nCells);
    NeoFOAM::fill(a, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> b(exec, nCells);
    NeoFOAM::fill(b, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> c(exec, nCells);
    NeoFOAM::fill(c, 3.0);
    NeoFOAM::Field<NeoFOAM::scalar> d(exec, nCells);
    NeoFOAM::fill(d, 4.0);

    SECTION("evaluateTerm_" + execName + "_nCells_" + std::to_string(nCells))
    {
        NeoFOAM::scalar value = 1;
        dsl::EqnTerm<NeoFOAM::scalar> lapTerm =
            Laplacian(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells, value);


        auto results = evaluateTerm(lapTerm);

        REQUIRE(getField(results) == 1.0);

        auto lapTerm_neg = -1.0 * lapTerm;
        auto res_neg = evaluateTerm(lapTerm_neg);


        REQUIRE(getField(res_neg) == -1.0);

        BENCHMARK("Explicit operation")
        {
            evaluateTerm(lapTerm);
            return 0;
        };

        BENCHMARK("Explicit operation negative")
        {
            evaluateTerm(lapTerm_neg);
            return 0;
        };
    }

    SECTION("Scale Scalar")
    {
        NeoFOAM::scalar value = 1;
        dsl::EqnTerm<NeoFOAM::scalar> lapTerm =
            Laplacian(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells, value);

        auto scaledTerm = 2.0 * lapTerm;
        REQUIRE(!scaledTerm.scaleField());
        auto res = evaluateTerm(scaledTerm);
        REQUIRE(getField(res) == 2.0);

        BENCHMARK("Scale Scalar")
        {
            auto res = evaluateTerm(scaledTerm);
            return 0;
        };
    }

    SECTION("Scale Field")
    {
        NeoFOAM::Field<NeoFOAM::scalar> scale(exec, nCells);
        NeoFOAM::fill(scale, 2.0);
        NeoFOAM::scalar value = 1;
        dsl::EqnTerm<NeoFOAM::scalar> lapTerm =
            Laplacian(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells, value);

        {
            auto scaledTerm = scale * lapTerm;
            REQUIRE(scaledTerm.scaleField());
            auto res = evaluateTerm(scaledTerm);
            REQUIRE(getField(res) == 2.0);
        }

        {
            auto scaledTerm = (scale + scale) * lapTerm;
            REQUIRE(scaledTerm.scaleField());
            auto res = evaluateTerm(scaledTerm);
            REQUIRE(getField(res) == 4.0);
        }

        {
            auto scaledTerm = (scale + scale + scale + scale) * lapTerm;
            REQUIRE(scaledTerm.scaleField());
            auto res = evaluateTerm(scaledTerm);
            REQUIRE(getField(res) == 8.0);
        }

        {
            auto span = scale.span();
            auto scaledTerm =
                (KOKKOS_LAMBDA(const NeoFOAM::size_t i) { return 4.0 * span[i]; }) * lapTerm;
            auto res = evaluateTerm(scaledTerm);
            REQUIRE(getField(res) == 8.0);
        }
    }

    SECTION("Performance")
    {
        NeoFOAM::Field<NeoFOAM::scalar> scale(exec, nCells);
        NeoFOAM::fill(scale, 2.0);
        NeoFOAM::scalar value = 1;
        dsl::EqnTerm<NeoFOAM::scalar> lapTerm =
            Laplacian(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells, value);

        BENCHMARK("Scale no Field")
        {
            // auto scaledTerm = scale * lapTerm;
            auto res = evaluateTerm(lapTerm);
            return 0;
        };

        BENCHMARK("Scale scaled Field")
        {
            auto scaledTerm = 1.0 * lapTerm;
            auto res = evaluateTerm(scaledTerm);
            return 0;
        };

        BENCHMARK("Scale Field 1 ")
        {
            auto scaledTerm = scale * lapTerm;
            auto res = evaluateTerm(scaledTerm);
            return 0;
        };

        BENCHMARK("Scale Field 2 ")
        {
            auto scaledTerm = (scale + scale) * lapTerm;
            auto res = evaluateTerm(scaledTerm);
            return 0;
        };

        BENCHMARK("Scale Field 4 ")
        {
            auto scaledTerm = (scale + scale + scale + scale) * lapTerm;
            auto res = evaluateTerm(scaledTerm);
            return 0;
        };

        BENCHMARK("Scale Field lambda ")
        {
            auto span = scale.span();
            auto scaledTerm =
                (KOKKOS_LAMBDA(const NeoFOAM::size_t i) { return 4.0 * span[i]; }) * lapTerm;
            auto res = evaluateTerm(scaledTerm);
            return 0;
        };
    }
}

dsl::EqnTerm<NeoFOAM::scalar> createLaplacian(const NeoFOAM::Executor& exec, size_t nCells)
{
    return Laplacian(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells);
}

TEST_CASE("term create from function")
{
    NeoFOAM::scalar nCells = 1;
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    NeoFOAM::Dictionary dict {{"value", 1.0}};

    SECTION("Scale Field")
    {
        NeoFOAM::Field<NeoFOAM::scalar> scale(exec, nCells, 2.0);

        {
            auto scaledTerm = scale * createLaplacian(exec, nCells);
            scaledTerm.build(dict);
            REQUIRE(scaledTerm.scaleField());
            auto res = evaluateTerm(scaledTerm);
            REQUIRE(getField(res) == 2.0);
        }

        {
            auto scaledTerm = (scale + scale) * createLaplacian(exec, nCells);
            scaledTerm.build(dict);
            REQUIRE(scaledTerm.scaleField());
            auto res = evaluateTerm(scaledTerm);
            REQUIRE(getField(res) == 4.0);
        }

        {
            auto scaledTerm = (scale + scale + scale + scale) * createLaplacian(exec, nCells);
            scaledTerm.build(dict);
            REQUIRE(scaledTerm.scaleField());
            auto res = evaluateTerm(scaledTerm);
            REQUIRE(getField(res) == 8.0);
        }

        {
            auto span = scale.span();
            auto scaledTerm = (KOKKOS_LAMBDA(const NeoFOAM::size_t i) { return 4.0 * span[i]; })
                            * createLaplacian(exec, nCells);
            scaledTerm.build(dict);
            auto res = evaluateTerm(scaledTerm);
            REQUIRE(getField(res) == 8.0);
        }
    }
}
