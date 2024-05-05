// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/fields/Field.hpp"
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

    auto size = GENERATE(8, 64, 512, 4096, 32768, 262144, 1048576, 1048576 * 4, 1048576 * 16, 1048576 * 64);

    CAPTURE(size); // Capture the value of size

    auto runAdditionBenchmark = [](Executor exec, int size)
    {
        NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, size);
        NeoFOAM::fill(a, 1.0);
        NeoFOAM::Field<NeoFOAM::scalar> b(cpuExec, size);
        NeoFOAM::fill(b, 2.0);
        NeoFOAM::Field<NeoFOAM::scalar> c(cpuExec, size);
        NeoFOAM::fill(c, 0.0);

        BENCHMARK("Addition " + exec.name())
        {
            c = a + b;
            return Kokkos::fence();
        };
    }
    // capture the value of size as section name
    DYNAMIC_SECTION("" << size)
    {
        runAdditionBenchmark(NeoFOAM::CPUExecutor {}, size);

        runAdditionBenchmark(NeoFOAM::OMPExecutor {}, size);

        runAdditionBenchmark(NeoFOAM::GPUExecutor {}, size);

        {
            NeoFOAM::GPUExecutor execGPU {};
            NeoFOAM::Field<NeoFOAM::scalar> a(execGPU, size);
            NeoFOAM::fill(GPUa, 1.0);
            NeoFOAM::Field<NeoFOAM::scalar> b(execGPU, size);
            NeoFOAM::fill(GPUb, 2.0);
            NeoFOAM::Field<NeoFOAM::scalar> c(execGPU, size);
            NeoFOAM::fill(GPUc, 0.0);

            auto sGPUb = b.field();
            auto sGPUc = c.field();
            BENCHMARK("Field<GPU> addition no allocation")
            {
                GPUa.apply(KOKKOS_LAMBDA(const int i) { return sGPUb[i] + sGPUc[i]; });
                return Kokkos::fence();
            };
        }

        {
            NeoFOAM::OMPExecutor OMPExec {};
            NeoFOAM::Field<NeoFOAM::scalar> OMPa(OMPExec, size);
            NeoFOAM::fill(OMPa, 1.0);
            NeoFOAM::Field<NeoFOAM::scalar> OMPb(OMPExec, size);
            NeoFOAM::fill(OMPb, 2.0);
            NeoFOAM::Field<NeoFOAM::scalar> OMPc(OMPExec, size);
            NeoFOAM::fill(OMPc, 0.0);

            auto sOMPb = OMPb.field();
            auto sOMPc = OMPc.field();
            BENCHMARK("Field<OMP> addition no allocation")
            {
                OMPa.apply(KOKKOS_LAMBDA(const int i) { return sOMPb[i] + sOMPc[i]; });
            };
        }
    };
}
