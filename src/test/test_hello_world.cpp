#include "gtest/gtest.h"

import NeoFOAM;

TEST(HelloWord, check_return_value){
	EXPECT_EQ(NeoFOAM::hello_world(), 1);
}
