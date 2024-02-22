// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/fields/Field.hpp"
#include "NeoFOAM/fields/FieldOperations.hpp"
#include "NeoFOAM/fields/FieldTypeDefs.hpp"
#include <vector>

#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>

#include <iostream>

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

TEST_CASE("Vector addition [benchmark]")
{

    auto N = GENERATE(8, 64, 512, 4096, 32768, 262144, 1048576, 1048576 * 4, 1048576 * 16, 1048576 * 64);

    CAPTURE(N); // Capture the value of N

    // capture the value of N as section name
    DYNAMIC_SECTION("" << N) {{NeoFOAM::CPUExecutor cpuExec {};
    NeoFOAM::Field<NeoFOAM::scalar> CPUa(cpuExec, N);
    NeoFOAM::fill(CPUa, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> CPUb(cpuExec, N);
    NeoFOAM::fill(CPUb, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> CPUc(cpuExec, N);
    NeoFOAM::fill(CPUc, 0.0);

    BENCHMARK("Field<CPU> addition") { return (CPUc = CPUa + CPUb); };
}

{
    NeoFOAM::OMPExecutor ompExec {};
    NeoFOAM::Field<NeoFOAM::scalar> ompa(ompExec, N);
    NeoFOAM::fill(ompa, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> ompb(ompExec, N);
    NeoFOAM::fill(ompb, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> ompc(ompExec, N);
    NeoFOAM::fill(ompc, 0.0);

    BENCHMARK("Field<omp> addition") { return (ompc = ompa + ompb); };
}

{
    NeoFOAM::GPUExecutor GPUExec {};
    NeoFOAM::Field<NeoFOAM::scalar> GPUa(GPUExec, N);
    NeoFOAM::fill(GPUa, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> GPUb(GPUExec, N);
    NeoFOAM::fill(GPUb, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> GPUc(GPUExec, N);
    NeoFOAM::fill(GPUc, 0.0);

    BENCHMARK("Field<GPU> addition")
    {
        GPUc = GPUa + GPUb;
        return Kokkos::fence();
    };
}

{
    NeoFOAM::GPUExecutor GPUExec {};
    NeoFOAM::Field<NeoFOAM::scalar> GPUa(GPUExec, N);
    NeoFOAM::fill(GPUa, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> GPUb(GPUExec, N);
    NeoFOAM::fill(GPUb, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> GPUc(GPUExec, N);
    NeoFOAM::fill(GPUc, 0.0);

    auto s_GPUb = GPUb.field();
    auto s_GPUc = GPUc.field();
    BENCHMARK("Field<GPU> addition no allocation")
    {
        GPUa.apply(KOKKOS_LAMBDA(const int i) { return s_GPUb[i] + s_GPUc[i]; });
        return Kokkos::fence();
        // return GPUa;
    };
}

{
    NeoFOAM::OMPExecutor OMPExec {};
    NeoFOAM::Field<NeoFOAM::scalar> OMPa(OMPExec, N);
    NeoFOAM::fill(OMPa, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> OMPb(OMPExec, N);
    NeoFOAM::fill(OMPb, 2.0);
    NeoFOAM::Field<NeoFOAM::scalar> OMPc(OMPExec, N);
    NeoFOAM::fill(OMPc, 0.0);

    auto s_OMPb = OMPb.field();
    auto s_OMPc = OMPc.field();
    BENCHMARK("Field<OMP> addition no allocation")
    {
        OMPa.apply(KOKKOS_LAMBDA(const int i) { return s_OMPb[i] + s_OMPc[i]; });
    };
}
}
;
}
