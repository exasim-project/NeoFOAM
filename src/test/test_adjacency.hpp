// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "NeoFOAM/blas/adjacency.hpp"

TEST(DeviceAdjacency, DefaultConstructor) {
    NeoFOAM::localAdjacency<true> graph;
    EXPECT_TRUE(graph.empty());
    EXPECT_EQ(graph.size(), 0);
}

TEST(DeviceAdjacency, NameConstructor) {
    std::string name = "test_graph";
    NeoFOAM::localAdjacency<true> graph(name);
    EXPECT_TRUE(graph.empty());
    EXPECT_EQ(graph.size(), 0);
    EXPECT_EQ(graph.name(), name);
}

TEST(DeviceAdjacency, CopyConstructor) {
    NeoFOAM::deviceField<NeoFOAM::localIdx> adjacency("adjacency", {0, 1, 2});
    NeoFOAM::deviceField<NeoFOAM::localIdx> offset("offset", {0, 2});
    NeoFOAM::localAdjacency<true> graph1("test_graph_1", adjacency, offset);
    NeoFOAM::localAdjacency<true> graph2(graph1);
    EXPECT_EQ(graph2.size(), graph1.size());
    EXPECT_EQ(graph2.name(), graph1.name());
    // TODO: Add more assertions to compare the adjacency and offset data
}

TEST(DeviceAdjacency, name) {

    std::string name = "test";
    NeoFOAM::deviceField<NeoFOAM::localIdx> adjacency("adjacency", {0, 1, 2});
    NeoFOAM::deviceField<NeoFOAM::localIdx> offset("offset", {0, 2});
    NeoFOAM::localAdjacency<false> test_adjacency(name);
    EXPECT_STREQ(test_adjacency.name().c_str(), name.c_str());
}

TEST(DeviceAdjacency, EmptyGraph) {
    NeoFOAM::deviceField<NeoFOAM::localIdx> adjacency("adjacency", {});
    NeoFOAM::deviceField<NeoFOAM::localIdx> offset("offset", {});
    NeoFOAM::localAdjacency<true> graph("test_graph", adjacency, offset);

    // Check if the graph is empty
    EXPECT_TRUE(graph.empty());
}

TEST(DeviceAdjacency, NonEmptyGraph) {
    NeoFOAM::deviceField<NeoFOAM::localIdx> adjacency("adjacency", {0, 1, 2});
    NeoFOAM::deviceField<NeoFOAM::localIdx> offset("offset", {0, 2});
    NeoFOAM::localAdjacency<true> graph("test_graph", adjacency, offset);

    // Check if the graph is not empty
    EXPECT_FALSE(graph.empty());
}

TEST(DeviceAdjacency, SizeNonEmpty) {
    NeoFOAM::deviceField<NeoFOAM::localIdx> adjacency("adjacency", {10, 10, 20, 30, 40});
    NeoFOAM::deviceField<NeoFOAM::localIdx> offset("offset", {0, 1, 3, 4, 5});
    NeoFOAM::localAdjacency<true> graph("test_graph",adjacency, offset);

    // Check the size of the graph
    EXPECT_EQ(graph.size(), 4);
}

TEST(DeviceAdjacency, SizeEmpty) {
    NeoFOAM::localAdjacency<true> graph("test_graph");

    // Check the size of the graph
    EXPECT_EQ(graph.size(), 0);
}

TEST(DeviceAdjacency, InsertGraphDirected) {
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

TEST(DeviceAdjacency, InsertGraphUndirected) {
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
