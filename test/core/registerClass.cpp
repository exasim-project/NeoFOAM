// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoFOAM/core/runtimeSelectionFactory.hpp"

class BaseClass : public NeoFOAM::RuntimeSelectionFactory<BaseClass, NeoFOAM::Parameters<>>
{

public:

    BaseClass() {}

    static std::string name() { return "BaseClass"; }
};

class BaseClass2 : public NeoFOAM::RuntimeSelectionFactory<BaseClass2, NeoFOAM::Parameters<>>
{

public:

    BaseClass2() {}

    static std::string name() { return "BaseClass2"; }
};

class DerivedClass : public BaseClass::Register<DerivedClass>
{
public:

    DerivedClass() {}

    static std::string name() { return "DerivedClass"; }

    static std::string doc() { return "DerivedClass documentation"; }

    static std::string schema() { return "DerivedClass schema"; }
};

class DerivedClass2 : public BaseClass2::Register<DerivedClass2>
{
public:

    DerivedClass2() {}

    static std::string name() { return "DerivedClass2"; }

    static std::string doc() { return "DerivedClass2 documentation"; }

    static std::string schema() { return "DerivedClass2 schema"; }
};

TEST_CASE("Register")
{
    std::cout << "Table size: " << NeoFOAM::BaseClassDocumentation::docTable().size() << std::endl;

    CHECK(NeoFOAM::BaseClassDocumentation::docTable().size() == 2);
    for (const auto& it : NeoFOAM::BaseClassDocumentation::docTable())
    {
        std::string baseClassName = it.first;
        std::cout << "baseClassName " << baseClassName << std::endl;
        auto entries = NeoFOAM::BaseClassDocumentation::entries(baseClassName);
        for (const auto& derivedClass : entries)
        {
            std::cout << "   - " << derivedClass << std::endl;
            std::cout << "     doc: "
                      << NeoFOAM::BaseClassDocumentation::doc(baseClassName, derivedClass)
                      << std::endl;
            std::cout << "     schema: "
                      << NeoFOAM::BaseClassDocumentation::schema(baseClassName, derivedClass)
                      << std::endl;
            CHECK(!NeoFOAM::BaseClassDocumentation::doc(baseClassName, derivedClass).empty());
            CHECK(!NeoFOAM::BaseClassDocumentation::schema(baseClassName, derivedClass).empty());
        }
    }

    CHECK(BaseClass::table().size() == 1);
    CHECK(BaseClass2::table().size() == 1);
}
