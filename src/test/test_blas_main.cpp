// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

// #include "test_adjacency.hpp"
#include "test_field.hpp"
#include "test_boundaryField.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
