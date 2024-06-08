// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <functional>
#include <iostream>

namespace NeoFOAM
{


template<typename baseClass, typename... Args>
struct registerClassManager
{
    using createFunction = std::function<std::unique_ptr<baseClass>(Args... args)>;

    static bool registerClass(const std::string name, createFunction funcCreate)
    {
        auto result = s_methods.insert({name, funcCreate});
        if (!result.second)
        {
            throw std::runtime_error("Insertion failed: Key already exists.");
        }
        return result.second;
    }

    static std::unique_ptr<baseClass> Create(const std::string& name, Args... args)
    {
        try
        {
            auto func = s_methods.at(name);
            return func(args...);
        }
        catch (const std::out_of_range& e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return nullptr;
        }
    }

    static inline std::unordered_map<std::string, createFunction> s_methods;

protected:
};


template<typename derivedClass, typename baseClass, typename... Args>
struct registerClass
{

    using createFunction = std::function<std::unique_ptr<baseClass>(Args... args)>;
    registerClass() { reg; };

    static bool reg;

    static bool init()
    {
        registerClassManager<baseClass, Args...>::registerClass(
            derivedClass::name(), derivedClass::create
        );
        return true;
    }
};

template<typename derivedClass, typename baseClass, typename... Args>
bool NeoFOAM::registerClass<derivedClass, baseClass, Args...>::reg =
    NeoFOAM::registerClass<derivedClass, baseClass, Args...>::init();

} // namespace NeoFOAM
