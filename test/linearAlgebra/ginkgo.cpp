// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
#include "catch2_common.hpp"

#include "NeoFOAM/NeoFOAM.hpp"


#if NF_WITH_GINKGO

using NeoFOAM::Executor;
using NeoFOAM::Dictionary;
using NeoFOAM::scalar;
using NeoFOAM::localIdx;
using NeoFOAM::Field;
using NeoFOAM::la::LinearSystem;
using NeoFOAM::la::CSRMatrix;
using NeoFOAM::la::Solver;

TEST_CASE("Dictionary Parsing - Ginkgo")
{
    SECTION("String")
    {
        NeoFOAM::Dictionary dict {{{"key", std::string("value")}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {"value"}}});
        CHECK(node == expected);
    }
    SECTION("Const Char *")
    {
        NeoFOAM::Dictionary dict {{{"key", "value"}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {"value"}}});
        CHECK(node == expected);
    }
    SECTION("Int")
    {
        NeoFOAM::Dictionary dict {{{"key", 10}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {10}}});
        CHECK(node == expected);
    }
    SECTION("Double")
    {
        NeoFOAM::Dictionary dict {{{"key", 1.0}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0}}});
        CHECK(node == expected);
    }
    SECTION("Float")
    {
        NeoFOAM::Dictionary dict {{{"key", 1.0f}}};

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected({{"key", gko::config::pnode {1.0f}}});
        CHECK(node == expected);
    }
    SECTION("Dict")
    {
        NeoFOAM::Dictionary dict;
        dict.insert("key", NeoFOAM::Dictionary {{"key", "value"}});

        auto node = NeoFOAM::la::ginkgo::parse(dict);

        gko::config::pnode expected(
            {{"key", gko::config::pnode({{"key", gko::config::pnode {"value"}}})}}
        );
        CHECK(node == expected);
    }
    SECTION("Throws")
    {
        NeoFOAM::Dictionary dict({{"key", std::pair<int*, std::vector<double>> {}}});

        REQUIRE_THROWS_AS(NeoFOAM::la::ginkgo::parse(dict), NeoFOAM::NeoFOAMException);
    }
}

TEST_CASE("MatrixAssembly - Ginkgo")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    gko::matrix_data<double, int> expected {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};

    SECTION("Solve linear system " + execName)
    {

        Field<scalar> values(exec, {1.0, -0.1, -0.1, 1.0, -0.1, -0.1, 1.0});
        Field<localIdx> colIdx(exec, {0, 1, 0, 1, 2, 1, 2});
        Field<localIdx> rowPtrs(exec, {0, 2, 5, 7});
        CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowPtrs);

        Field<scalar> rhs(exec, {1.0, 2.0, 3.0});
        LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);
        Field<scalar> x(exec, {0.0, 0.0, 0.0});

        Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Cg"},
             {"criteria", Dictionary {{{"iteration", 3}, {"relative_residual_norm", 1e-7}}}}}
        };

        // Create solver
        auto solver = NeoFOAM::la::Solver(exec, solverDict);

        // Solve system
        solver.solve(linearSystem, x);

        auto hostX = x.copyToHost();
        auto hostXS = hostX.span();
        REQUIRE((hostXS[0]) == Catch::Approx(1.24489796).margin(1e-8));
        REQUIRE((hostXS[1]) == Catch::Approx(2.44897959).margin(1e-8));
        REQUIRE((hostXS[2]) == Catch::Approx(3.24489796).margin(1e-8));
    }
}
#endif
