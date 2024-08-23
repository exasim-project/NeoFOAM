// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <vector>
#include <any>
#include <string>
#include <iostream>

#include "NeoFOAM/core/demangle.hpp"

namespace NeoFOAM
{


class TokenList
{
public:

    TokenList() = default;

    TokenList(const std::vector<std::any>& data);

    TokenList(const std::initializer_list<std::any>& initList);

    void insert(const std::any& value);

    void remove(size_t index);

    [[nodiscard]] bool empty() const;

    [[nodiscard]] size_t size() const;


    template<typename T>
    [[nodiscard]] T& get(const size_t& idx)
    {
        try
        {
            return std::any_cast<T&>(data_.at(idx));
        }
        catch (const std::bad_any_cast& e)
        {
            std::cerr << "Caught a bad_any_cast exception: \n"
                      << "requested type " << demangle(typeid(T).name()) << "\n"
                      << "actual type " << demangle(data_.at(idx).type().name()) << "\n"
                      << e.what() << std::endl;
            throw e;
        }
    }


    template<typename T>
    [[nodiscard]] const T& get(const size_t& idx) const
    {
        try
        {
            return std::any_cast<const T&>(data_.at(idx));
        }
        catch (const std::bad_any_cast& e)
        {
            std::cerr << "Caught a bad_any_cast exception: \n"
                      << "requested type " << demangle(typeid(T).name()) << "\n"
                      << "actual type " << demangle(data_.at(idx).type().name()) << "\n"
                      << e.what() << std::endl;
            throw e;
        }
    }

    [[nodiscard]] std::vector<std::any>& tokens();


private:

    std::vector<std::any> data_;
};

} // namespace NeoFOAM
