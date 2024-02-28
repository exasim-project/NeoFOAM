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
#include "NeoFOAM/cellCentredFiniteVolume/bcFields/scalar/fvccScalarFixedValueBoundaryField.hpp"

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

    SECTION("CPU")
    {
        int N = 10;
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, N);
        auto s_a = a.field();
        NeoFOAM::fill(a, 5.0);

        for (int i = 0; i < N; i++)
        {
            REQUIRE(s_a[i] == 5.0);
        }
        NeoFOAM::Field<NeoFOAM::scalar> b(cpuExec, N + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.field().size() == N + 2);

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 10.0);
        }

        add(a, b);
        REQUIRE(a.field().size() == N + 2);

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a + b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 30.0);
        }

        a = a - b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a * 0.1;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 2.0);
        }

        a = a * b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        auto s_b = b.field();
        a.apply(KOKKOS_LAMBDA(int i) { return 2 * s_b[i]; });

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }
    };

    SECTION("OpenMP")
    {
        int N = 10;
        NeoFOAM::OMPExecutor OMPExec {};

        NeoFOAM::Field<NeoFOAM::scalar> a(OMPExec, N);
        auto s_a = a.field();
        NeoFOAM::fill(a, 5.0);

        for (int i = 0; i < N; i++)
        {
            REQUIRE(s_a[i] == 5.0);
        }
        NeoFOAM::Field<NeoFOAM::scalar> b(OMPExec, N + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.field().size() == N + 2);
        ;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 10.0);
        }

        add(a, b);
        REQUIRE(a.field().size() == N + 2);

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a + b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 30.0);
        }

        a = a - b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        a = a * 0.1;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 2.0);
        }

        a = a * b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }

        auto s_b = b.field();
        a.apply(KOKKOS_LAMBDA(int i) { return 2 * s_b[i]; });

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.field()[i] == 20.0);
        }
    };

    SECTION("GPU")
    {
        int N = 10;
        NeoFOAM::GPUExecutor gpuExec {};
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> GPUa(gpuExec, N);
        NeoFOAM::fill(GPUa, 5.0);

        NeoFOAM::Field<NeoFOAM::scalar> CPUa(cpuExec, N);
        NeoFOAM::fill(CPUa, 10.0);
        for (int i = 0; i < N; i++)
        {
            REQUIRE(CPUa.field()[i] == 10.0);
        }
        CPUa = GPUa.copyToHost();

        for (int i = 0; i < N; i++)
        {
            REQUIRE(CPUa.field()[i] == 5.0);
        }

        NeoFOAM::Field<NeoFOAM::scalar> a(gpuExec, N);
        auto s_a = a.field();
        NeoFOAM::fill(a, 5.0);

        NeoFOAM::Field<NeoFOAM::scalar> b(gpuExec, N + 2);
        NeoFOAM::fill(b, 10.0);

        a = b;
        REQUIRE(a.field().size() == N + 2);

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 10.0);
        }

        add(a, b);
        REQUIRE(a.field().size() == N + 2);

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }

        a = a + b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 30.0);
        }

        a = a - b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }

        a = a * 0.1;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 2.0);
        }

        a = a * b;

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }

        auto s_b = b.field();
        a.apply(KOKKOS_LAMBDA(int i) { return 2 * s_b[i]; });

        for (int i = 0; i < N + 2; i++)
        {
            REQUIRE(a.copyToHost().field()[i] == 20.0);
        }
    };
}

TEST_CASE("Boundaries")
{

    // NeoFOAM::CPUExecutor cpuExec{};
    // GENERATE(NeoFOAM::CPUExecutor{}, NeoFOAM::ompExecutor{}, NeoFOAM::GPUExecutor{});
    NeoFOAM::executor exec = NeoFOAM::CPUExecutor {};

    SECTION("domainField")
    {

        NeoFOAM::domainField<double> a(1000, 100, 10, exec);
        // auto& aIn = a.internalField();

        NeoFOAM::fill(a.internalField(), 2.0);

        for (int i = 0; i < a.internalField().size(); i++)
        {
            REQUIRE(a.internalField().field()[i] == 2.0);
        }
    }

    SECTION("boundaryFields")
    {

        NeoFOAM::boundaryFields<double> BCs(100, 10, exec);

        NeoFOAM::fill(BCs.value(), 2.0);

        for (int i = 0; i < BCs.value().size(); i++)
        {
            REQUIRE(BCs.value().field()[i] == 2.0);
        }

        NeoFOAM::fill(BCs.refValue(), 2.0);

        for (int i = 0; i < BCs.refValue().size(); i++)
        {
            REQUIRE(BCs.refValue().field()[i] == 2.0);
        }

        NeoFOAM::fill(BCs.refGrad(), 2.0);

        for (int i = 0; i < BCs.refGrad().size(); i++)
        {
            REQUIRE(BCs.refGrad().field()[i] == 2.0);
        }

        NeoFOAM::fill(BCs.valueFraction(), 2.0);

        for (int i = 0; i < BCs.valueFraction().size(); i++)
        {
            REQUIRE(BCs.valueFraction().field()[i] == 2.0);
        }
    }

    SECTION("fvccBoundaryField")
    {

        std::vector<std::unique_ptr<NeoFOAM::fvccBoundaryField<double>>> bcs;
        bcs.push_back(std::make_unique<NeoFOAM::fvccScalarFixedValueBoundaryField>(0, 10, 1.0));
        bcs.push_back(std::make_unique<NeoFOAM::fvccScalarFixedValueBoundaryField>(10, 20, 2.0));

        NeoFOAM::fvccVolField<NeoFOAM::scalar> volField(
            1000,
            20,
            2,
            std::move(bcs),
            exec
        );

        NeoFOAM::boundaryFields<NeoFOAM::scalar>& bField = volField.boundaryField();

        auto& volBCs = volField.boundaryConditions();

        REQUIRE(volBCs.size() == 2.0);

        volField.correctBoundaryConditions();

        auto& bIn = bField.value();
        auto& bRefIn = bField.refValue();

        for (int i = 0; i < 10; i++)
        {
            REQUIRE(bIn.field()[i] == 1.0);
            REQUIRE(bRefIn.field()[i] == 1.0);
        }

        for (int i = 10; i < 20; i++)
        {
            REQUIRE(bIn.field()[i] == 2.0);
            REQUIRE(bRefIn.field()[i] == 2.0);
        }
    }
}
