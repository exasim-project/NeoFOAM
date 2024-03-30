// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/fields/Field.hpp"
#include "NeoFOAM/fields/FieldOperations.hpp"
#include "NeoFOAM/fields/FieldTypeDefs.hpp"

#include "NeoFOAM/fields/boundaryFields.hpp"
#include "NeoFOAM/fields/domainField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/fields/fvccVolField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/fvccBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarFixedValueBoundaryField.hpp"
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/vol/scalar/fvccScalarZeroGradientBoundaryField.hpp"

#include "NeoFOAM/fields/comparisions/fieldComparision.hpp"

int main(int argc, char* argv[])
{

    // Initialize Catch2
    Kokkos::initialize(argc, argv);
    Catch::Session session;

    // Specify command line options
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    int result = session.run();

    // Run benchmarks if there are any
    Kokkos::finalize();

    return result;
}

TEST_CASE("Field Operations")
{
    NeoFOAM::executor exec = GENERATE(
        NeoFOAM::executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::executor(NeoFOAM::OMPExecutor {}),
        NeoFOAM::executor(NeoFOAM::GPUExecutor {})
    );
    std::string exec_name = std::visit([](auto e) { return e.print(); },exec);

    SECTION("Field_" + exec_name)
    {
        int N = 10;
        NeoFOAM::Field<NeoFOAM::scalar> a(exec, N);
        auto s_a = a.field();
        NeoFOAM::fill(a, 5.0);

        REQUIRE(compare(a, 5.0));

        NeoFOAM::Field<NeoFOAM::scalar> b(exec, N + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.field().size() == N + 2);
        REQUIRE(compare(a, b));

        add(a, b);
        REQUIRE(a.field().size() == N + 2);
        REQUIRE(compare(a, 20.0));

        a = a + b;
        REQUIRE(compare(a, 30.0));

        a = a - b;
        REQUIRE(compare(a, 20.0));

        a = a * 0.1;
        REQUIRE(compare(a, 2.0));

        a = a * b;
        REQUIRE(compare(a, 20.0));

        auto s_b = b.field();
        a.apply(KOKKOS_LAMBDA(int i) { return 2 * s_b[i]; });
        REQUIRE(compare(a, 20.0));
    };
    
}

TEST_CASE("Primitives")
{
    SECTION("Vector")
    {
        SECTION("CPU")
        {
            NeoFOAM::Vector a(1.0, 2.0, 3.0);
            REQUIRE(a(0) == 1.0);
            REQUIRE(a(1) == 2.0);
            REQUIRE(a(2) == 3.0);

            NeoFOAM::Vector b(1.0, 2.0, 3.0);
            REQUIRE(a == b);

            NeoFOAM::Vector c(2.0, 4.0, 6.0);

            REQUIRE(a + b == c);

            REQUIRE((a - b) == NeoFOAM::Vector(0.0, 0.0, 0.0));

            a += b;
            REQUIRE(a == c);

            a -= b;
            REQUIRE(a == b);
            a *= 2;
            REQUIRE(a == c);
            a = b;

            REQUIRE(a == b);

            NeoFOAM::Vector d(4.0, 8.0, 12.0);
            REQUIRE((a + a + a + a) == d);
            REQUIRE((4 * a) == d);
            REQUIRE((a * 4) == d);
            REQUIRE((a + 3 * a) == d);
            REQUIRE((a + 2 * a + a) == d);
        };
    };
};

TEST_CASE("Boundaries")
{

    NeoFOAM::executor exec = GENERATE(
        NeoFOAM::executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::executor(NeoFOAM::OMPExecutor {}),
        NeoFOAM::executor(NeoFOAM::GPUExecutor {})
    );

    std::string exec_name = std::visit([](auto e) { return e.print(); },exec);
    SECTION("domainField_" + exec_name)
    {

        NeoFOAM::domainField<double> a(exec, 1000, 100, 10);

        NeoFOAM::fill(a.internalField(), 2.0);
        REQUIRE(compare(a.internalField(), 2.0));
    }

    SECTION("boundaryFields_" + exec_name)
    {

        NeoFOAM::boundaryFields<double> BCs(exec, 100, 10);

        NeoFOAM::fill(BCs.value(), 2.0);
        REQUIRE(compare(BCs.value(), 2.0));

        NeoFOAM::fill(BCs.refValue(), 2.0);
        REQUIRE(compare(BCs.refValue(), 2.0));

        NeoFOAM::fill(BCs.refGrad(), 2.0);
        REQUIRE(compare(BCs.refGrad(), 2.0));

        NeoFOAM::fill(BCs.valueFraction(), 2.0);
        REQUIRE(compare(BCs.valueFraction(), 2.0));
    }

    // SECTION("fvccBoundaryField")
    // {

    //     std::vector<std::unique_ptr<NeoFOAM::fvccBoundaryField<double>>> bcs;
    //     bcs.push_back(std::make_unique<NeoFOAM::fvccScalarFixedValueBoundaryField>(0, 10, 1.0));
    //     bcs.push_back(std::make_unique<NeoFOAM::fvccScalarFixedValueBoundaryField>(10, 20, 2.0));

    //     NeoFOAM::fvccVolField<NeoFOAM::scalar> volField(
    //         1000,
    //         20,
    //         2,
    //         std::move(bcs),
    //         exec
    //     );

    //     NeoFOAM::boundaryFields<NeoFOAM::scalar>& bField = volField.boundaryField();

    //     auto& volBCs = volField.boundaryConditions();

    //     REQUIRE(volBCs.size() == 2.0);

    //     volField.correctBoundaryConditions();

    //     auto& bIn = bField.value();
    //     auto& bRefIn = bField.refValue();

    //     for (int i = 0; i < 10; i++)
    //     {
    //         REQUIRE(bIn.field()[i] == 1.0);
    //         REQUIRE(bRefIn.field()[i] == 1.0);
    //     }

    //     for (int i = 10; i < 20; i++)
    //     {
    //         REQUIRE(bIn.field()[i] == 2.0);
    //         REQUIRE(bRefIn.field()[i] == 2.0);
    //     }
    // }
}
