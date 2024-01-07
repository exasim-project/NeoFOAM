#include <benchmark/benchmark.h>
#include "NeoFOAM/blas/field.hpp"

static void serial_scalarField_addition(benchmark::State &state)
{

    int N = state.range(0);
    std::vector<double> a(N, 1.0);
    std::vector<double> b(N, 2.0);
    std::vector<double> c(N, 0.0);

    // add a + b to c
    for (auto _ : state)
    {
        for (int i = 0; i < N; ++i)
        {
            c[i] = a[i] + b[i];
        }
    }
}

static void scalarField_add(benchmark::State &state)
{
    int N = state.range(0);
    NeoFOAM::scalarField a("a", N);
    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            a(i) = 1;
        });
    NeoFOAM::scalarField b("b", N);
    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            b(i) = 2;
        });
    NeoFOAM::scalarField c("c", N); // preallocate does not help only with apply
    for (auto _ : state)
    {
        // NeoFOAM::scalarField c = a + b; // needs two allocations: + and = operators
        c.apply(KOKKOS_LAMBDA(int i) { return a(i) + b(i); }); // no allocations 3 times faster
    }
}

// Register the function as a benchmark
BENCHMARK(serial_scalarField_addition)->RangeMultiplier(8)->Range(8, 1 << 20); // from 8 to 2^20 elements
BENCHMARK(scalarField_add)->RangeMultiplier(8)->Range(8, 1 << 20);        // from 8 to 2^20 elements

int main(int argc, char **argv)
{
    Kokkos::initialize(argc, argv);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;

    // Run the benchmarks
    ::benchmark::RunSpecifiedBenchmarks();

    // Custom teardown (if needed)
    Kokkos::finalize();

    return 0;
}
