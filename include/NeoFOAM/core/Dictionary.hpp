// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <any>
#include <string>

namespace NeoFOAM
{

class Dictionary
{
public:
    Dictionary() = default;
    // Dictionary(Dictionary&);
    // Dictionary(std::unordered_map<std::string, std::any> data);

    void insert(const std::string& key, const std::any& value);
    std::any& operator[](const std::string& key);
    const std::any& operator[](const std::string& key) const;

    template <typename T>
    T& get(const std::string& key)
    {
        return std::any_cast<T&>(data_.at(key));
    }

    template <typename T>
    const T& get(const std::string& key) const
    {
        return std::any_cast<T&>(data_.at(key));
    }

    Dictionary& subDict(const std::string& key);

private:
    std::unordered_map<std::string, std::any> data_;
};

} // namespace NeoFOAM