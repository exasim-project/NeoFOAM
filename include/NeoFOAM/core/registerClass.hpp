// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

// additional information see:
// https://stackoverflow.com/questions/52354538/derived-class-discovery-at-compile-time
// https://stackoverflow.com/questions/10332725/how-to-automatically-register-a-class-on-creation

#pragma once

#include <functional>
#include <iostream>

namespace NeoFOAM
{

// Type trait to check if a type is a std::function
template<typename T>
struct is_std_function : std::false_type
{
};

template<typename Ret, typename... Args>
struct is_std_function<std::function<Ret(Args...)>> : std::true_type
{
};

// Concept that uses the type trait
template<typename T>
concept StdFunction = is_std_function<T>::value;

/**
 * @brief Template struct for managing class registration.
 *
 * This struct provides a mechanism for registering classes with a given base class and a create
 * function. It maintains a map of class names to create functions, allowing for dynamic class
 * instantiation.
 *
 * @tparam baseClass The base class type.
 * @tparam CreateFunction The type of the create function.
 */
template<typename baseClass, StdFunction CreateFunction>
class RegisterClassManager // needs to be a class otherwise breathe will not document it
{
public:

    /**
     * @brief Registers a class with the given name and create function.
     *
     * @param name The name of the class.
     * @param createFunc The create function for the class.
     * @return true if the class was successfully registered, false if the class name already exists
     * in the map.
     * @throws std::runtime_error if the class name already exists in the map.
     */
    static bool registerClass(const std::string name, CreateFunction createFunc)
    {
        auto result = classMap.insert({name, createFunc});
        if (!result.second)
        {
            throw std::runtime_error("Insertion failed: Key already exists.");
        }
        return result.second;
    }

    /**
     * @brief Gets the number of registered classes.
     *
     * @return The number of registered classes.
     */
    static size_t size() { return classMap.size(); }

    /**
     * @brief A container that maps strings to create functions.
     *
     * This unordered map is used to store a mapping between strings and create functions.
     * The keys are strings, and the values are functions that create objects
     *
     * @tparam Key The type of the keys in the map (std::string).
     * @tparam T The type of the values in the map (CreateFunction)
     */
    static inline std::unordered_map<std::string, CreateFunction> classMap;
};


/**
 * @brief Template struct for registering a derived class with a base class.
 *
 * This struct provides a mechanism for registering a derived class with a base class.
 *
 * @tparam derivedClass The derived class to be registered.
 * @tparam baseClass The base class that the derived class inherits from.
 * @tparam CreateFunction The function pointer type for creating an instance of the derived class.
 */
template<typename derivedClass, typename baseClass, StdFunction CreateFunction>
class RegisterClass
{
public:

    /**
     * @brief Constructor for the RegisterClass struct.
     *
     * reg needs to be called in the constructor to ensure that the class is registered.
     */
    RegisterClass() { reg; };

    /**
     * @brief Static flag indicating if the class has been registered.
     */
    static bool reg;

    /**
     * @brief Initializes the registration of the derived class with the base class.
     *
     * This function registers the derived class with the base class by calling the
     * RegisterClassManager::registerClass() function with the derived class's name and
     * create function.
     *
     * @return True if the registration is successful, false otherwise.
     */
    static bool init()
    {
        RegisterClassManager<baseClass, CreateFunction>::registerClass(
            derivedClass::name(), derivedClass::create
        );
        return true;
    }
};


/**
 * @brief Template specialization for registering a derived class with a base class in NeoFOAM.
 *
 * This template class is used to register a derived class with a base class in NeoFOAM.
 * It provides a static member variable `reg` which is initialized to `true` when the class is
 * instantiated
 *
 * @tparam derivedClass The derived class to be registered.
 * @tparam baseClass The base class with which the derived class is registered.
 * @tparam CreateFunction The function pointer type for creating an instance of the derived class.
 */
template<typename derivedClass, typename baseClass, StdFunction CreateFunction>
bool NeoFOAM::RegisterClass<derivedClass, baseClass, CreateFunction>::reg =
    NeoFOAM::RegisterClass<derivedClass, baseClass, CreateFunction>::init();

} // namespace NeoFOAM
