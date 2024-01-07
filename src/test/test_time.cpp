// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/NeoFOAM.hpp"
#include "gtest/gtest.h"

TEST(HelloWord, check_return_value) {
  NeoFOAM::Time time{};
  EXPECT_EQ(time.timeName(), "0.000000");
}
