// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <any>
#include <string>

namespace NeoFOAM
{

class StencilDataBase
{
public:

    StencilDataBase() = default;

    template<typename T>
    void insert(const std::string& key, T value)
    {
        stencilDB_.emplace(key, value);
    }

    std::any& operator[](const std::string& key);
    const std::any& operator[](const std::string& key) const;

    template<typename T>
    T& get(const std::string& key)
    {
        return std::any_cast<T&>(stencilDB_.at(key));
    }

    template<typename T>
    const T& get(const std::string& key) const
    {
        return std::any_cast<const T&>(stencilDB_.at(key));
    }

    bool contains(const std::string& key) const;

private:
    std::unordered_map<std::string, std::any> stencilDB_;

};

} // namespace NeoFOAM