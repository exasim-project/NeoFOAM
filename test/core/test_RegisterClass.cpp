// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include "NeoFOAM/core/registerClass.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include <iostream>


class TestBaseClass
{
public:

    // entries required runtime discovery
    using regTestBaseClass = NeoFOAM::RegisterClassManager<TestBaseClass, std::string, double>;

    template<typename derivedClass>
    using TestBaseClassReg =
        NeoFOAM::RegisterClass<derivedClass, TestBaseClass, std::string, double>;

    template<typename derivedClass>
    void registerClass()
    {
        TestBaseClassReg<derivedClass> reg;
    }

    static int size() { return regTestBaseClass::classMap.size(); }

    // standard base class entries
    virtual ~TestBaseClass() = default;

    virtual std::string testString() = 0;

    virtual double testValue() = 0;


private:
};


class TestDerivedClass : public TestBaseClass
{

public:

    TestDerivedClass(std::string name, double test)
        : TestBaseClass(), testString_(name), testValue_(test)
    {
        registerClass<TestDerivedClass>();
    }

    static std::unique_ptr<TestBaseClass> create(std::string name, double test)
    {
        return std::make_unique<TestDerivedClass>(name, test);
    }

    static std::string name() { return "TestDerivedClass"; }

    virtual std::string testString() override { return testString_; };

    virtual double testValue() override { return testValue_; };

private:

    std::string testString_;
    double testValue_;
};

class TestDerivedClass2 : public TestBaseClass
{

public:

    TestDerivedClass2(std::string name, double test)
        : TestBaseClass(), testString_(name), testValue_(test)
    {
        registerClass<TestDerivedClass2>();
    }

    static std::unique_ptr<TestBaseClass> create(std::string name, double test)
    {
        return std::make_unique<TestDerivedClass2>(name, test);
    }

    static std::string name() { return "TestDerivedClass2"; }

    virtual std::string testString() override { return testString_; };

    virtual double testValue() override { return testValue_ * 2; };

private:

    std::string testString_;
    double testValue_;
};


TEST_CASE("Register Class")
{
    std::cout << "Number of registered classes: " << TestBaseClass::size() << std::endl;
    REQUIRE(TestBaseClass::size() == 2);

    std::unique_ptr<TestBaseClass> testDerived =
        TestBaseClass::regTestBaseClass::create("TestDerivedClass", "FirstDerived", 1.0);
    std::cout << "TestBaseClass testValue: " << testDerived->testValue() << std::endl;
    std::cout << "TestBaseClass testString: " << testDerived->testString() << std::endl;
    REQUIRE(testDerived->testString() == "FirstDerived");
    REQUIRE(testDerived->testValue() == 1.0);

    std::unique_ptr<TestBaseClass> testDerived2 =
        TestBaseClass::regTestBaseClass::create("TestDerivedClass2", "SecondDerived", 1.0);

    std::cout << "TestBaseClass testValue: " << testDerived2->testValue() << std::endl;
    std::cout << "TestBaseClass testString: " << testDerived2->testString() << std::endl;
    REQUIRE(testDerived2->testString() == "SecondDerived");
    REQUIRE(
        testDerived2->testValue() == 2.0
    ); // multiplied by 2 (see implementation of TestDerivedClass2)
}
