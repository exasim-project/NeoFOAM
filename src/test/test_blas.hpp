// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <gtest/gtest.h>
#include "NeoFOAM/blas/fields.hpp"
#include <Kokkos_Core.hpp>

#include "NeoFOAM/blas/fields.hpp"
#include "NeoFOAM/blas/Field.hpp"
#include "NeoFOAM/blas/FieldOperations.hpp"


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


void test_Field(int N)
{
    NeoFOAM::CPUExecutor cpuExec{};
    NeoFOAM::GPUExecutor GPUExec{};
    {
        NeoFOAM::Field<NeoFOAM::scalar> a(N, cpuExec);
        auto s_a = a.field();
        NeoFOAM::fill(a, 5.0);

        for (int i = 0; i < N; i++){
            EXPECT_EQ(s_a[i], 5.0);
        }
        NeoFOAM::Field<NeoFOAM::scalar> b(N+2, cpuExec);
        auto s_b = a.field();
        NeoFOAM::fill(b, 10.0);
        // setField(a, b);
        // s_a = a.field();
        a = b;
        EXPECT_EQ(a.field().size(), N+2);

        for (int i = 0; i < N+2; i++){
            EXPECT_EQ(a.field()[i], 10.0);
        }

        add(a, b);
        EXPECT_EQ(a.field().size(), N+2);

        for (int i = 0; i < N+2; i++){
            EXPECT_EQ(a.field()[i], 20.0);
        }

        a = a + b;

        for (int i = 0; i < N+2; i++){
            EXPECT_EQ(a.field()[i], 30.0);
        }

        a = a - b;

        for (int i = 0; i < N+2; i++){
            EXPECT_EQ(a.field()[i], 20.0);
        }

        a = a * 0.1;

        for (int i = 0; i < N+2; i++){
            EXPECT_EQ(a.field()[i], 2.0);
        }

        a = a * b;

        for (int i = 0; i < N+2; i++){
            EXPECT_EQ(a.field()[i], 20.0);
        }
    }

    {
        NeoFOAM::Field<NeoFOAM::scalar> c(N, GPUExec);
    }

    NeoFOAM::Field<NeoFOAM::scalar> b(N, GPUExec);
    NeoFOAM::fill(b, 5.0);
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

TEST(BLAS, field_ops)
{
    int N = 10;

    test_Field(N);
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