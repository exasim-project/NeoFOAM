// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <vector>
#include "NeoFOAM/blas/fields.hpp"
#include "NeoFOAM/blas/Field.hpp"
#include "NeoFOAM/blas/FieldOperations.hpp"

#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <iostream>

int main(int argc, char* argv[]) {
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


void serial_scalarField_addition(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c)
{
    for (int i = 0; i < a.size(); ++i)
    {
        c[i] = a[i] + b[i];
    }
}

void GPU_scalarField_addition(NeoFOAM::scalarField& a, NeoFOAM::scalarField& b, NeoFOAM::scalarField& c)
{
    c.apply(KOKKOS_LAMBDA(int i) { return a(i) + b(i); });
    Kokkos::fence();
}

TEST_CASE("Vector addition [benchmark]") {

    auto N = GENERATE(8, 64, 512, 4096, 32768, 262144, 1048576, 1048576*4, 1048576*16, 1048576*64);

    CAPTURE(N);  // Capture the value of N
    
    // capture the value of N as section name
    DYNAMIC_SECTION( "" << N ) {
        {
            std::vector<double> CPUa(N, 1.0);
            std::vector<double> CPUb(N, 2.0);
            std::vector<double> CPUc(N, 0.0);
            
            BENCHMARK("std::vector addition no allocation") {
                return serial_scalarField_addition(CPUa, CPUb, CPUc);
            };
        }

        {
            NeoFOAM::scalarField GPUa("a", N);
            Kokkos::parallel_for(
                N, KOKKOS_LAMBDA(const int i) {
                    GPUa(i) = 1;
                });
            NeoFOAM::scalarField GPUb("b", N);
            Kokkos::parallel_for(
                N, KOKKOS_LAMBDA(const int i) {
                    GPUb(i) = 2;
                });
            NeoFOAM::scalarField GPUc("c", N);

            BENCHMARK("GPU vector addition no allocation") {
                return GPU_scalarField_addition(GPUa, GPUb, GPUc);
            };
        }

        {
            NeoFOAM::CPUExecutor cpuExec{};
            NeoFOAM::Field<NeoFOAM::scalar> CPUa(N, cpuExec);
            NeoFOAM::fill(CPUa, 1.0);
            NeoFOAM::Field<NeoFOAM::scalar> CPUb(N, cpuExec);
            NeoFOAM::fill(CPUb, 2.0);
            NeoFOAM::Field<NeoFOAM::scalar> CPUc(N, cpuExec);
            NeoFOAM::fill(CPUc, 0.0);

            
            BENCHMARK("Field<CPU> addition") {
                return (CPUc = CPUa + CPUb);
            };
        }

        {
            NeoFOAM::ompExecutor ompExec{};
            NeoFOAM::Field<NeoFOAM::scalar> ompa(N, ompExec);
            NeoFOAM::fill(ompa, 1.0);
            NeoFOAM::Field<NeoFOAM::scalar> ompb(N, ompExec);
            NeoFOAM::fill(ompb, 2.0);
            NeoFOAM::Field<NeoFOAM::scalar> ompc(N, ompExec);
            NeoFOAM::fill(ompc, 0.0);

            BENCHMARK("Field<omp> addition") {
                return (ompc = ompa + ompb);
            };
        }

        {
            NeoFOAM::GPUExecutor GPUExec{};
            NeoFOAM::Field<NeoFOAM::scalar> GPUa(N, GPUExec);
            NeoFOAM::fill(GPUa, 1.0);
            NeoFOAM::Field<NeoFOAM::scalar> GPUb(N, GPUExec);
            NeoFOAM::fill(GPUb, 2.0);
            NeoFOAM::Field<NeoFOAM::scalar> GPUc(N, GPUExec);
            NeoFOAM::fill(GPUc, 0.0);

            
            BENCHMARK("Field<GPU> addition") {
                GPUc = GPUa + GPUb;
                return Kokkos::fence();
            };
        }

        {
            NeoFOAM::GPUExecutor GPUExec{};
            NeoFOAM::Field<NeoFOAM::scalar> GPUa(N, GPUExec);
            NeoFOAM::fill(GPUa, 1.0);
            NeoFOAM::Field<NeoFOAM::scalar> GPUb(N, GPUExec);
            NeoFOAM::fill(GPUb, 2.0);
            NeoFOAM::Field<NeoFOAM::scalar> GPUc(N, GPUExec);
            NeoFOAM::fill(GPUc, 0.0);

            auto s_GPUb = GPUb.field();
            auto s_GPUc = GPUc.field();
            BENCHMARK("Field<GPU> addition no allocation") {
                GPUa.apply(KOKKOS_LAMBDA(const int i) { return s_GPUb[i] + s_GPUc[i]; });
                return Kokkos::fence();
                // return GPUa;
            };
        }

        {
            NeoFOAM::ompExecutor OMPExec{};
            NeoFOAM::Field<NeoFOAM::scalar> OMPa(N, OMPExec);
            NeoFOAM::fill(OMPa, 1.0);
            NeoFOAM::Field<NeoFOAM::scalar> OMPb(N, OMPExec);
            NeoFOAM::fill(OMPb, 2.0);
            NeoFOAM::Field<NeoFOAM::scalar> OMPc(N, OMPExec);
            NeoFOAM::fill(OMPc, 0.0);

            auto s_OMPb = OMPb.field();
            auto s_OMPc = OMPc.field();
            BENCHMARK("Field<OMP> addition no allocation") {
                OMPa.apply(KOKKOS_LAMBDA(const int i) { return s_OMPb[i] + s_OMPc[i]; });
            };
        }

    };
}
