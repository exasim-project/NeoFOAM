#pragma once
#include <gtest/gtest.h>
#include "NeoFOAM/blas/fields.hpp"

TEST(BLAS, scalar_ops)
{
    int N = 10;

    NeoFOAM::scalarField a("a", N);

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(a(i), 0);
        });

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            a(i) = 1;
        });

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(a(i), 1);
        });

    NeoFOAM::scalarField b("b", N);

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            b(i) = 2;
        });

    auto c = a + b;

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(c(i), 3);
        });

    c = c - b;

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(c(i), 1);
        });

    c = c * 2;

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(c(i, 0), 2);
        });

    NeoFOAM::scalarField scale("scalar_a", N);

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            scale(i) = 0;
        });

    c = c * scale;

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(c(i, 0), 0);
        });
}

TEST(BLAS, vector_ops)
{
    int N = 10;
    // initialize a vector
    NeoFOAM::vectorField vec_a("vec_a", N, true);

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(vec_a(i, 0), 0);
            EXPECT_EQ(vec_a(i, 1), 0);
            EXPECT_EQ(vec_a(i, 2), 0);
        });

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            vec_a(i, 0) = 1;
            vec_a(i, 1) = 1;
            vec_a(i, 2) = 1;
        });

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(vec_a(i, 0), 1);
            EXPECT_EQ(vec_a(i, 1), 1);
            EXPECT_EQ(vec_a(i, 2), 1);
        });

    NeoFOAM::vectorField vec_b("vec_b", N);

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            vec_b(i, 0) = 2;
            vec_b(i, 1) = 2;
            vec_b(i, 2) = 2;
        });

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(vec_b(i, 0), 2);
            EXPECT_EQ(vec_b(i, 1), 2);
            EXPECT_EQ(vec_b(i, 2), 2);
        });

    auto vec_c = vec_a + vec_b;

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(vec_c(i, 0), 3);
            EXPECT_EQ(vec_c(i, 1), 3);
            EXPECT_EQ(vec_c(i, 2), 3);
        });

    vec_c = vec_c - vec_b;

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(vec_c(i, 0), 1);
            EXPECT_EQ(vec_c(i, 1), 1);
            EXPECT_EQ(vec_c(i, 2), 1);
        });

    vec_c = vec_c * 2;

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(vec_c(i, 0), 2);
            EXPECT_EQ(vec_c(i, 1), 2);
            EXPECT_EQ(vec_c(i, 2), 2);
        });

    NeoFOAM::scalarField scale("scalar_a", N);

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            scale(i) = 0;
        });

    vec_c = vec_c * scale;

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
            EXPECT_EQ(vec_c(i, 0), 0);
            EXPECT_EQ(vec_c(i, 1), 0);
            EXPECT_EQ(vec_c(i, 2), 0);
        });
}