// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include <set>

#include "testRegister.hpp"

TEST_CASE("Register")
{
    std::cout << "Table size: " << NeoFOAM::BaseClassDocumentation::docTable().size() << std::endl;

    std::set<std::string> customClasses = {BaseClass::name(), BaseClass2::name()};

    int presentCustomClasses = 0;
    for (const auto& it : NeoFOAM::BaseClassDocumentation::docTable())
        presentCustomClasses += customClasses.contains(it.first);

    bool customClassesArePresent = presentCustomClasses == customClasses.size();

    CHECK(customClassesArePresent);
    if (!customClassesArePresent)
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
                CHECK(!NeoFOAM::BaseClassDocumentation::schema(baseClassName, derivedClass).empty()
                );
            }
        }

    CHECK(BaseClass::table().size() == 1);
    CHECK(BaseClass2::table().size() == 1);
}
