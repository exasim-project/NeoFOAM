#include "neofoam/hello_world.hpp"
#include "gtest/gtest.h"

TEST(HelloWord, check_return_value){
	EXPECT_EQ(hello_world(), 1);
}
