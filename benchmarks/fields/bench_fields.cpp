// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <vector>
#include <iostream>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/fields/operations/operationsMacros.hpp"
#include "NeoFOAM/fields/fieldTypeDefs.hpp"


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

template<typename SpanA, typename SpanB, typename T>
struct kernelLambda_1
{
    SpanA spanA;
    SpanB spanB;
    T val;

    KOKKOS_FUNCTION void operator()(const size_t i) const { spanA[i] = spanB[i] + val; }
};

template<typename SpanA, typename SpanB, typename T>
struct kernelLambdaPlusB
{
    SpanA spanA;
    SpanB spanB;

    KOKKOS_FUNCTION T operator()(const size_t i) const { return spanA[i] + spanB[i]; }
};

TEST_CASE("Vector addition [benchmark]")
{

    auto size =
        GENERATE(8, 64, 512, 4096, 32768, 262144, 1048576, 1048576 * 4, 1048576 * 16, 1048576 * 64);

    CAPTURE(size); // Capture the value of size

    // capture the value of size as section name
    DYNAMIC_SECTION("" << size) {{NeoFOAM::SerialExecutor cpuExec {};
    NeoFOAM::Field<NeoFOAM::scalar> cpuA(cpuExec, size);
    NeoFOAM::fill(cpuA, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> cpuB(cpuExec, size);
    NeoFOAM::fill(cpuB, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> cpuC(cpuExec, size);
    NeoFOAM::fill(cpuC, 0.0);

    BENCHMARK("Field<CPU> addition") { return (cpuC = cpuA + cpuB); };
}

{
    NeoFOAM::CPUExecutor ompExec {};
    NeoFOAM::Field<NeoFOAM::scalar> ompA(ompExec, size);
    NeoFOAM::fill(ompA, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> ompB(ompExec, size);
    NeoFOAM::fill(ompB, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> ompC(ompExec, size);
    NeoFOAM::fill(ompC, 0.0);

    BENCHMARK("Field<omp> addition") { return (ompC = ompA + ompB); };
}

{
    NeoFOAM::GPUExecutor gpuExec {};
    NeoFOAM::Field<NeoFOAM::scalar> gpuA(gpuExec, size);
    NeoFOAM::fill(gpuA, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> gpuB(gpuExec, size);
    NeoFOAM::fill(gpuB, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> gpuC(gpuExec, size);
    NeoFOAM::fill(gpuC, 0.0);

    BENCHMARK("Field<GPU> addition")
    {
        gpuC = gpuA + gpuB;
        return Kokkos::fence();
    };
}

{
    NeoFOAM::GPUExecutor gpuExec {};
    NeoFOAM::Field<NeoFOAM::scalar> gpuA(gpuExec, size);
    NeoFOAM::fill(gpuA, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> gpuB(gpuExec, size);
    NeoFOAM::fill(gpuB, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> gpuC(gpuExec, size);
    NeoFOAM::fill(gpuC, 0.0);

    auto sGpuB = gpuB.span();
    auto sGpuC = gpuC.span();
    kernelLambdaPlusB<decltype(sGpuB), decltype(sGpuC), NeoFOAM::scalar> kernel {sGpuB, sGpuC};
    BENCHMARK("Field<GPU> addition no allocation")
    {
        gpuA.apply(kernel);
        return Kokkos::fence();
        // return GPUa;
    };
}

{
    NeoFOAM::CPUExecutor ompExec {};
    NeoFOAM::Field<NeoFOAM::scalar> ompA(ompExec, size);
    NeoFOAM::fill(ompA, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> ompB(ompExec, size);
    NeoFOAM::fill(ompB, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> ompC(ompExec, size);
    NeoFOAM::fill(ompC, 0.0);

    auto sompB = ompB.span();
    auto sompC = ompC.span();
    kernelLambdaPlusB<decltype(sompB), decltype(sompC), NeoFOAM::scalar> kernel {sompB, sompC};
    BENCHMARK("Field<OMP> addition no allocation") { ompA.apply(kernel); };
}
}
;
}
