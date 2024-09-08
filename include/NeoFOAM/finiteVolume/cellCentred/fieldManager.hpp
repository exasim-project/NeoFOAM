// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <any>
#include <string>

namespace NeoFOAM
{

enum class FieldCategory
{
    Iteration,
    oldTime,
    newTime,
    cacheField
};

template<typename ValueType>
struct FieldComponent
{
    std::shared_ptr<ValueType> field;
    std::string timeIndex;
    FieldCategory category;
};

class FieldManager
{

public:

    FieldManager();

    template<typename T>
    void insert(const std::string& key, T value)
    {
        fieldDB_.emplace(key, value);
    }

    template<typename T>
    T& get(const std::string& key)
    {
        return std::any_cast<T&>(fieldDB_.at(key));
    }

private:

    std::unordered_map<int, std::any> fieldDB_;
    std::vector<std::string> fieldNames_;

};  

} // namespace NeoFOAM
