// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "NeoFOAM/blas/adjacency.hpp"
#include "NeoFOAM/blas/fields.hpp"

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

TEST(BLAS, field_constructor_initialiser_list)
{
    std::string name = "test";
    NeoFOAM::scalarField field(name, {1.0, 2.0, 3.0});

    EXPECT_EQ(field(0), 1.0);
    EXPECT_EQ(field(1), 2.0);
    EXPECT_EQ(field(2), 3.0);
    EXPECT_EQ(field.name(), name);
}


TEST(BLAS, deviceAdjacency_name) {

    std::string name = "test";
    NeoFOAM::deviceField<NeoFOAM::localIdx> adjacency("adjacency", {0, 1, 2});
    NeoFOAM::deviceField<NeoFOAM::localIdx> offset("offset", {0, 2});
    NeoFOAM::localAdjacency<false> test_adjacency(name, adjacency, offset);
      
    std::cout<<"\n"<<test_adjacency(0)(0);
    std::cout<<"\n"<<test_adjacency(0)(1);

    test_adjacency.insert(Kokkos::pair(0, 2));

    EXPECT_TRUE(false);
}

TEST(BLAS, deviceAdjacency_contains) {

}

TEST(BLAS, deviceAdjacency_insert) {
  NeoFOAM::localAdjacency<false> graph;

  // insert a pentagram out of order, with some edges flipped
  EXPECT_TRUE(graph.insert({0, 1}));
  EXPECT_TRUE(graph.insert({3, 4}));
  EXPECT_TRUE(graph.insert({0, 4}));

  EXPECT_TRUE(graph.insert({2, 3}));
  EXPECT_TRUE(graph.insert({1, 2}));
    std::cout<<"hee4"<<std::endl;
exit(1);
  // Add two additional vertices, as an 'appendage'.
  EXPECT_TRUE(graph.insert({3, 6}));
  EXPECT_TRUE(graph.insert({5, 6}));

  // check ascending order has been maintained.
  //EXPECT_EQ(graph.size_edge(), 7);
  //EXPECT_EQ(graph.size_vertex(), 7);

  // Attempt to insert an existing edge
  EXPECT_FALSE(graph.insert({1, 2}));

  // Connect across
  EXPECT_TRUE(graph.insert({0, 3}));
  EXPECT_TRUE(graph.insert({0, 2}));

  // Check graph
  //EXPECT_EQ(graph.size_edge(), 9);
  //EXPECT_EQ(graph.size_vertex(), 7);
  EXPECT_EQ(graph(0).size(), 4);
  EXPECT_EQ(graph(1).size(), 2);
  EXPECT_EQ(graph(2).size(), 3);
  EXPECT_EQ(graph(3).size(), 4);
  EXPECT_EQ(graph(4).size(), 2);
  EXPECT_EQ(graph(5).size(), 1);
  EXPECT_EQ(graph(6).size(), 2);

  EXPECT_EQ(graph(0)(0), 1);
  EXPECT_EQ(graph(0)(1), 2);
  EXPECT_EQ(graph(0)(2), 3);
  EXPECT_EQ(graph(0)(3), 4);
  EXPECT_EQ(graph(1)(0), 0);
  EXPECT_EQ(graph(1)(1), 2);
  EXPECT_EQ(graph(2)(0), 0);
  EXPECT_EQ(graph(2)(1), 1);
  EXPECT_EQ(graph(2)(2), 3);
  EXPECT_EQ(graph(3)(0), 0);
  EXPECT_EQ(graph(3)(1), 2);
  EXPECT_EQ(graph(3)(2), 4);
  EXPECT_EQ(graph(3)(3), 6);
  EXPECT_EQ(graph(4)(0), 0);
  EXPECT_EQ(graph(4)(1), 3);
  EXPECT_EQ(graph(5)(0), 6);
  EXPECT_EQ(graph(6)(0), 3);
  EXPECT_EQ(graph(6)(1), 5);

  // death test - invalid edge
  EXPECT_DEATH(graph.insert({1, 1}), ".*");
}
