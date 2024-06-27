// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoFOAM/core/runtimeSelectionFactory.hpp"

class BaseClass : public NeoFOAM::RuntimeSelectionFactory<BaseClass>
{

public:

    static std::string name() { return "BaseClass"; }
};

class BaseClass2 : public NeoFOAM::RuntimeSelectionFactory<BaseClass2>
{
    BaseClass2() {}

public:

    static std::string name() { return "BaseClass2"; }
};

class DerivedClass : public BaseClass::Register<DerivedClass>
{
public:

    DerivedClass() {}

    static std::string name() { return "DerivedClass"; }
};

// class DerivedClass2 : public BaseClass2::Register<DerivedClass2>
// {
// public:

//     DerivedClass2()
//     {

//     }

//     static std::string name()
//     {
//         return "DerivedClass";
//     }
// };

TEST_CASE("Register")
{
    std::cout << "Table size: " << NeoFOAM::runTimeSelectionManager::table().size() << std::endl;
    std::cout << "Table size: " << NeoFOAM::runTimeSelectionManager::doc("BaseClass2") << std::endl;

    REQUIRE(BaseClass::table().size() == 2);
}
