// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/DSL/coeff.hpp"

using Field = NeoFOAM::Field<NeoFOAM::scalar>;
using Coeff = NeoFOAM::DSL::Coeff;
namespace detail = NeoFOAM::DSL::detail;


TEST_CASE("Coeff")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("Coefficient evaluation on " + execName)
    {
        Field fA(exec, 3, 2.0);
        Field res(exec, 1);

        Coeff a {};
        Coeff b {2.0};
        Coeff c = 2 * a * b;
        REQUIRE(c[0] == 4.0);

        Coeff d {3.0, fA};
        detail::toField(d, res);
        auto hostResD = res.copyToHost();
        REQUIRE(hostResD.data()[0] == 6.0);
        REQUIRE(hostResD.data()[1] == 6.0);
        REQUIRE(hostResD.data()[2] == 6.0);

        Coeff e = d * b;
        detail::toField(e, res);
        auto hostResE = res.copyToHost();
        REQUIRE(hostResE.data()[0] == 12.0);
        REQUIRE(hostResE.data()[1] == 12.0);
        REQUIRE(hostResE.data()[2] == 12.0);

        Coeff f = b * d;
        detail::toField(f, res);
        auto hostResF = res.copyToHost();
        REQUIRE(hostResF.data()[0] == 12.0);
        REQUIRE(hostResF.data()[1] == 12.0);
        REQUIRE(hostResF.data()[2] == 12.0);
    }
}
