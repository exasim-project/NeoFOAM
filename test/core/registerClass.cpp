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

// forward declaration so we can use it to define the create function and the class manager
class TestBaseClass;

// define the create function use to instantiate the derived classes
using CreateFunc = std::function<std::unique_ptr<TestBaseClass>(std::string, double)>;

// define the class manager to register the classes
using BaseClassRegistry = NeoFOAM::BaseClassRegistry<TestBaseClass, CreateFunc>;


class TestBaseClass : public BaseClassRegistry
{
public:

    MAKE_CLASS_A_RUNTIME_FACTORY(TestBaseClass, BaseClassRegistry, CreateFunc)

    virtual ~TestBaseClass() = default;

    virtual std::string testString() = 0;

    virtual double testValue() = 0;
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
    std::cout << "Number of registered classes: " << TestBaseClass::nRegistered() << std::endl;
    REQUIRE(TestBaseClass::nRegistered() == 2);

    std::unique_ptr<TestBaseClass> testDerived =
        TestBaseClass::create("TestDerivedClass", "FirstDerived", 1.0);

    std::cout << "TestBaseClass testValue: " << testDerived->testValue() << std::endl;
    std::cout << "TestBaseClass testString: " << testDerived->testString() << std::endl;

    REQUIRE(testDerived->testString() == "FirstDerived");
    REQUIRE(testDerived->testValue() == 1.0);

    std::unique_ptr<TestBaseClass> testDerived2 =
        TestBaseClass::create("TestDerivedClass2", "SecondDerived", 1.0);

    std::cout << "TestBaseClass testValue: " << testDerived2->testValue() << std::endl;
    std::cout << "TestBaseClass testString: " << testDerived2->testString() << std::endl;

    REQUIRE(testDerived2->testString() == "SecondDerived");
    REQUIRE(
        testDerived2->testValue() == 2.0
    ); // multiplied by 2 (see implementation of TestDerivedClass2)
}
