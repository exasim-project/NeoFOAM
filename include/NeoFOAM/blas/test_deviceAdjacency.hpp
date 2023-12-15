#include <gtest/gtest.h>
#include <NeoFOAM/blas/deviceAdjacency.hpp>

TEST(DeviceAdjacency, InsertPentagonGraph_Directed) {
    NeoFOAM::deviceAdjacency<true> graph;

    // Insert the edges of the 'pentagon' graph
    EXPECT_TRUE(graph.insert({0, 1}));
    EXPECT_TRUE(graph.insert({1, 2}));
    EXPECT_TRUE(graph.insert({2, 3}));
    EXPECT_TRUE(graph.insert({3, 4}));
    EXPECT_TRUE(graph.insert({0, 4}));

    // Check the size of the graph
    EXPECT_EQ(graph.size(), 5);

    // Check the adjacency of each vertex
    EXPECT_EQ(graph(0).size(), 1);
    EXPECT_EQ(graph(1).size(), 1);
    EXPECT_EQ(graph(2).size(), 1);
    EXPECT_EQ(graph(3).size(), 1);
    EXPECT_EQ(graph(4).size(), 1);

    // Check the connections of each vertex
    EXPECT_EQ(graph(0)(0), 1);
    EXPECT_EQ(graph(1)(0), 2);
    EXPECT_EQ(graph(2)(0), 3);
    EXPECT_EQ(graph(3)(0), 4);
    EXPECT_EQ(graph(4)(0), 0);
}

TEST(DeviceAdjacency, InsertPentagonGraph_Undirected) {
    NeoFOAM::deviceAdjacency<false> graph;

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