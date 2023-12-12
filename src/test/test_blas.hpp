// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <gtest/gtest.h>
#include "NeoFOAM/blas/deviceAdjacency.hpp"
#include "NeoFOAM/blas/fields.hpp"
#include <Kokkos_Core.hpp>

template <typename T, typename primitive>
void fill_field(T &a, primitive value)
{
    Kokkos::parallel_for(
        a.size(), KOKKOS_LAMBDA(const int i) {
            a.operator()(i) = value;
        });
}

template <typename T, typename primitive>
void copy_and_check_EQ(T &a, primitive value)
{
    Kokkos::View<primitive *, Kokkos::HostSpace> testview("testview", a.size());
    Kokkos::deep_copy(testview, a.field());
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OpenMP>(0, a.size()), KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(testview(i), value);
        });
}

void test_scalar(int N)
{

    Kokkos::View<double *, Kokkos::HostSpace> testview("testview", N);
    NeoFOAM::scalarField a("a", N);

    fill_field(a, 1.0);

    copy_and_check_EQ(a, 1.0);

    NeoFOAM::scalarField b("b", N);

    fill_field(b, 2.0);

    copy_and_check_EQ(b, 2.0);

    auto c = a + b;

    copy_and_check_EQ(c, 3.0);

    c = c - b;

    copy_and_check_EQ(c, 1.0);

    c = c * 2;

    copy_and_check_EQ(c, 2.0);

    NeoFOAM::scalarField scale("scalar_a", N);

    fill_field(scale, 0.0);

    c = c * scale;

    copy_and_check_EQ(c, 0.0);
}

void test_vector(int N)
{

    NeoFOAM::vectorField a("a", N);

    fill_field(a, NeoFOAM::vector(1.0, 1.0, 1.0));

    copy_and_check_EQ(a, NeoFOAM::vector(1.0, 1.0, 1.0));

    NeoFOAM::vectorField b("b", N);

    fill_field(b, NeoFOAM::vector(2.0, 2.0, 2.0));

    copy_and_check_EQ(b, NeoFOAM::vector(2.0, 2.0, 2.0));

    auto c = a + b;

    copy_and_check_EQ(c, NeoFOAM::vector(3.0, 3.0, 3.0));

    c = c - b;

    copy_and_check_EQ(c, NeoFOAM::vector(1.0, 1.0, 1.0));

    c = c * 2;

    copy_and_check_EQ(c, NeoFOAM::vector(2.0, 2.0, 2.0));

    NeoFOAM::scalarField scale("scalar_a", N);

    fill_field(scale, 0.0);

    c = c * scale;

    copy_and_check_EQ(c, NeoFOAM::vector(0.0, 0.0, 0.0));
}

TEST(BLAS, scalar_ops)
{
    int N = 10;

    test_scalar(N);
}

TEST(BLAS, vector_ops)
{
    int N = 10;

    test_vector(N);
}

TEST(BLAS, deviceAdjacency_name) {
    EXPECT_TRUE(false);
}