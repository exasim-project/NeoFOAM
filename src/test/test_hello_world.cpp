// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "gtest/gtest.h"

import NeoFOAM;

TEST(HelloWord, check_return_value){
	EXPECT_EQ(NeoFOAM::hello_world(), 1);
}
