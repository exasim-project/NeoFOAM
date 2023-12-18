// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "NeoFOAM/blas/adjacency.hpp"

TEST(DeviceAdjacency, name) {

    std::string name = "test";
    NeoFOAM::deviceField<NeoFOAM::localIdx> adjacency("adjacency", {0, 1, 2});
    NeoFOAM::deviceField<NeoFOAM::localIdx> offset("offset", {0, 2});
    NeoFOAM::localAdjacency<false> test_adjacency(name);
    EXPECT_STREQ(test_adjacency.name().c_str(), name.c_str());
}

TEST(DeviceAdjacency, InsertPentagonGraph_Directed) {
    NeoFOAM::localAdjacency<true> graph;

    // Insert the edges of the 'pentagon' graph
    EXPECT_TRUE(graph.insert({0, 1}));
    EXPECT_TRUE(graph.insert({3, 4}));
    EXPECT_TRUE(graph.insert({2, 3}));
    EXPECT_TRUE(graph.insert({1, 2}));
    EXPECT_TRUE(graph.insert({0, 4}));

    // Check the size of the graph
    EXPECT_EQ(graph.size(), 4);

    // Check the adjacency of each vertex
    EXPECT_EQ(graph(0).size(), 2);
    EXPECT_EQ(graph(1).size(), 1);
    EXPECT_EQ(graph(2).size(), 1);
    EXPECT_EQ(graph(3).size(), 1);
    
    // Check the connections of each vertex
    EXPECT_EQ(graph(0)(0), 1);
    EXPECT_EQ(graph(0)(1), 4);
    EXPECT_EQ(graph(1)(0), 2);
    EXPECT_EQ(graph(2)(0), 3);
    EXPECT_EQ(graph(3)(0), 4);
}

TEST(DeviceAdjacency, InsertPentagonGraph_Undirected) {
    NeoFOAM::localAdjacency<false> graph;

    // Insert the edges of the 'pentagon' graph
    EXPECT_TRUE(graph.insert({0, 1}));
    EXPECT_TRUE(graph.insert({1, 2}));
    EXPECT_TRUE(graph.insert({2, 3}));
    EXPECT_TRUE(graph.insert({3, 4}));
    EXPECT_TRUE(graph.insert({0, 4}));

    // Check the size of the graph
    EXPECT_EQ(graph.size(), 5);

    // Check the adjacency of each vertex
    EXPECT_EQ(graph(0).size(), 2);
    EXPECT_EQ(graph(1).size(), 2);
    EXPECT_EQ(graph(2).size(), 2);
    EXPECT_EQ(graph(3).size(), 2);
    EXPECT_EQ(graph(4).size(), 2);

    // Check the connections of each vertex
    EXPECT_EQ(graph(0)(0), 1);
    EXPECT_EQ(graph(0)(1), 4);
    EXPECT_EQ(graph(1)(0), 0);
    EXPECT_EQ(graph(1)(1), 2);
    EXPECT_EQ(graph(2)(0), 1);
    EXPECT_EQ(graph(2)(1), 3);
    EXPECT_EQ(graph(3)(0), 2);
    EXPECT_EQ(graph(3)(1), 4);
    EXPECT_EQ(graph(4)(0), 0);
    EXPECT_EQ(graph(4)(1), 3);
}

TEST(DeviceAdjacency, insert) {
  NeoFOAM::localAdjacency<false> graph;

  // insert a pentagram out of order, with some edges flipped
  EXPECT_TRUE(graph.insert({0, 1}));
  EXPECT_TRUE(graph.insert({3, 4}));
  EXPECT_TRUE(graph.insert({0, 4}));

  EXPECT_TRUE(graph.insert({2, 3}));
  EXPECT_TRUE(graph.insert({1, 2}));
    
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
  //EXPECT_DEATH(graph.insert({1, 1}), ".*");
}

TEST(DeviceAdjacency, ResizeGraph_Directed) {
    NeoFOAM::deviceField<NeoFOAM::localIdx> adjacency("adjacency", {10, 10, 20, 30, 40});
    NeoFOAM::deviceField<NeoFOAM::localIdx> offset("offset", {0, 1, 3, 4, 5});
    NeoFOAM::localAdjacency<true> graph("test_graph", adjacency, offset);

    // Resize the graph to a larger size
    graph.resize(10);

    // Check the size of the graph
    EXPECT_EQ(graph.size(), 10);

    // Check the adjacency of each vertex
    EXPECT_EQ(graph(0).size(), 1);
    EXPECT_EQ(graph(1).size(), 2);
    EXPECT_EQ(graph(2).size(), 1);
    EXPECT_EQ(graph(3).size(), 1);
    EXPECT_EQ(graph(4).size(), 0);
    EXPECT_EQ(graph(5).size(), 0);
    EXPECT_EQ(graph(6).size(), 0);
    EXPECT_EQ(graph(7).size(), 0);
    EXPECT_EQ(graph(8).size(), 0);
    EXPECT_EQ(graph(9).size(), 0);

    // Check the connections of each vertex
    EXPECT_EQ(graph(0)(0), 10);
    EXPECT_EQ(graph(1)(0), 10);
    EXPECT_EQ(graph(1)(1), 20);
    EXPECT_EQ(graph(2)(0), 30);
    EXPECT_EQ(graph(3)(0), 40);

    // Resize the graph to a smaller size
    graph.resize(2);

    // Check the size of the graph
    EXPECT_EQ(graph.size(), 2);

    // Check the adjacency of each vertex
    EXPECT_EQ(graph(0).size(), 1);
    EXPECT_EQ(graph(1).size(), 2);

    // Check the connections of each vertex
    EXPECT_EQ(graph(0)(0), 10);
    EXPECT_EQ(graph(1)(0), 10);
    EXPECT_EQ(graph(1)(1), 20);
}

TEST(DeviceAdjacency, ContainsUndirectedEdge) {
    NeoFOAM::localAdjacency<false> graph;

    // Insert some edges
    graph.insert({0, 1});
    graph.insert({1, 2});
    graph.insert({2, 3});

    // Test contains function
    EXPECT_TRUE(graph.contains({0, 1}));
    EXPECT_TRUE(graph.contains({1, 0}));
    EXPECT_TRUE(graph.contains({1, 2}));
    EXPECT_TRUE(graph.contains({2, 1}));
    EXPECT_TRUE(graph.contains({2, 3}));
    EXPECT_TRUE(graph.contains({3, 2}));

    // Test non-existent edges
    EXPECT_FALSE(graph.contains({0, 2}));
    EXPECT_FALSE(graph.contains({2, 0}));
    EXPECT_FALSE(graph.contains({1, 3}));
    EXPECT_FALSE(graph.contains({3, 1}));
}

TEST(DeviceAdjacency, ContainsDirectedEdge) {
    // connection 0 -> 1, 1 -> 2, 2 -> 3
    NeoFOAM::deviceField<NeoFOAM::localIdx> adjacency("adjacency", {1, 2, 3});
    NeoFOAM::deviceField<NeoFOAM::localIdx> offset("offset", {0, 1, 2, 3});
    NeoFOAM::localAdjacency<true> graph("test_graph", adjacency, offset);

    // Test contains function
    EXPECT_TRUE(graph.contains({0, 1}));
    EXPECT_FALSE(graph.contains({1, 0}));
    EXPECT_TRUE(graph.contains({1, 2}));
    EXPECT_FALSE(graph.contains({2, 1}));
    EXPECT_TRUE(graph.contains({2, 3}));
    EXPECT_FALSE(graph.contains({3, 2}));

    // Test non-existent edges
    EXPECT_FALSE(graph.contains({0, 2}));
    EXPECT_FALSE(graph.contains({2, 0}));
    EXPECT_FALSE(graph.contains({1, 3}));
    EXPECT_FALSE(graph.contains({3, 1}));
}

TEST(DeviceAdjacency, ContainsEmptyGraph) {
    NeoFOAM::localAdjacency<false> graph;

    // Test contains function on empty graph
    EXPECT_FALSE(graph.contains({0, 1}));
    EXPECT_FALSE(graph.contains({1, 0}));
    EXPECT_FALSE(graph.contains({1, 2}));
    EXPECT_FALSE(graph.contains({2, 1}));
    EXPECT_FALSE(graph.contains({2, 3}));
    EXPECT_FALSE(graph.contains({3, 2}));
}
