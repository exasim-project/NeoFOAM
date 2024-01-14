// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "gtest/gtest.h"
#include "NeoFOAM/NeoFOAM.hpp"

TEST(HelloWord, check_return_value) {
  NeoFOAM::Time time{};
  EXPECT_EQ(time.timeName(), "0.000000");
}
