// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 AMR Wind Authors
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
// ##############################################################################
// # Original design taken from amr wind                                        #
// # from here                                                                  #
// # https://github.com/Exawind/amr-wind/blob/v2.1.0/amr-wind/core/Factory.H    #
// ##############################################################################

#pragma once

#include <memory>
#include <unordered_map>
#include <iostream>
#include <functional>


namespace NeoFOAM
{

class BaseClassData
{
public:

    std::function<std::string(const std::string&)> doc;
    std::function<std::string(const std::string&)> schema;
};

class BaseClassDocumentation
{
public:

    using LookupTable = std::unordered_map<std::string, BaseClassData>;

    static void registerClass(
        std::string name,
        std::function<std::string(const std::string&)> doc,
        std::function<std::string(const std::string&)> schema
    )
    {
        // if not already registered
        docTable()[name] = BaseClassData {doc, schema};
    }

    static std::string doc(const std::string& baseClassName, const std::string& derivedClassName)
    {
        return docTable().at(baseClassName).doc(derivedClassName);
    }

    static std::string schema(const std::string& baseClassName, const std::string& derivedClassName)
    {
        // get the schema of the derived class
        return docTable().at(baseClassName).schema(derivedClassName);
    }


    static LookupTable& docTable()
    {
        static LookupTable tbl;
        return tbl;
    }
};


template<class T>
struct RegisterDocumentation
{
    RegisterDocumentation()
    {
        reg; // force specialization
    }
    static bool reg;
    static bool init()
    {
        BaseClassDocumentation::registerClass(T::name(), T::doc, T::schema);
        return true;
    }
};

template<class T>
bool RegisterDocumentation<T>::reg = RegisterDocumentation<T>::init();

class DerivedClassDocumentation
{
public:

    std::function<std::string()> doc;
    std::function<std::string()> schema;
};

// Parameters helper type
template<typename... Args>
struct Parameters
{
};

// Primary template declaration
template<typename Base, typename Params>
class RuntimeSelectionFactory;

// Partial specialization for Parameters
template<typename Base, typename... Args>
class RuntimeSelectionFactory<Base, Parameters<Args...>> : public RegisterDocumentation<Base>
{
public:

    using CreatorFunc = std::function<std::unique_ptr<Base>(Args...)>;
    using LookupTable = std::unordered_map<std::string, CreatorFunc>;
    using ClassDocTable = std::unordered_map<std::string, DerivedClassDocumentation>;

    static std::string doc(const std::string& derivedClassName)
    {
        // get the documentation of the derived class
        return docTable().at(derivedClassName).doc();
    }

    static std::string schema(const std::string& derivedClassName)
    {
        // get the schema of the derived class
        return docTable().at(derivedClassName).schema();
    }


    static std::unique_ptr<Base> create(const std::string& key, Args... args)
    {
        key_exists_or_error(key);
        auto ptr = table().at(key)(std::forward<Args>(args)...);
        return ptr;
    }


    static void print(std::ostream& os)
    {
        const auto& tbl = table();
        os << Base::name() << " " << tbl.size() << std::endl;
        for (const auto& it : tbl)
        {
            os << " - " << it.first << std::endl;
        }
    }

    template<class T>
    class Register : public Base
    {
    public:

        friend T;
        static bool registered;

        static bool add_sub_type()
        {
            CreatorFunc func = [](Args... args) -> std::unique_ptr<Base>
            { return std::unique_ptr<Base>(new T(std::forward<Args>(args)...)); };
            RuntimeSelectionFactory::table()[T::name()] = func;

            DerivedClassDocumentation childData;
            childData.doc = []() -> std::string { return T::doc(); };
            childData.schema = []() -> std::string { return T::schema(); };
            RuntimeSelectionFactory::docTable()[T::name()] = childData;

            return true;
        }

        ~Register() override
        {
            if (registered)
            {
                const auto& tbl = RuntimeSelectionFactory::table();
                const auto it = tbl.find(T::name());
                registered = (it != tbl.end());
            }
        }

    private:

        Register()
        {
            // avoid unused variable warning
            bool tmp = registered;
        }
    };

    virtual ~RuntimeSelectionFactory() = default;
    friend Base;

    static std::size_t size() { return table().size(); }

    static LookupTable& table()
    {
        static LookupTable tbl;
        return tbl;
    }

    static ClassDocTable& docTable()
    {
        static ClassDocTable tbl;
        return tbl;
    }

private:

    static void key_exists_or_error(const std::string& key)
    {
        const auto& tbl = table();
        if (tbl.find(key) == tbl.end())
        {
            std::cout << "Cannot find instance: " << key << std::endl;
            std::cout << "Valid options are: " << std::endl;
            print(std::cout);
        }
    }

    RuntimeSelectionFactory() = default;
};

template<class Base, class... Args>
template<class T>
bool RuntimeSelectionFactory<Base, Parameters<Args...>>::Register<T>::registered =
    RuntimeSelectionFactory<Base, Parameters<Args...>>::Register<T>::add_sub_type();

}; // namespace NeoFOAM
