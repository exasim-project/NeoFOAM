// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <any>
#include <string>
#include <memory>
#include <vector>

#include "NeoFOAM/core/demangle.hpp"
#include "NeoFOAM/core/error.hpp"

#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

enum class FieldCategory
{
    Iteration,
    oldTime,
    newTime,
    cacheField
};

template<typename Type>
struct FieldComponent
{
    std::shared_ptr<Type> field;
    std::size_t timeIndex;
    std::size_t iterationIndex;
    FieldCategory category;
};

class SolutionFields
{

public:

    SolutionFields();

    template<typename T>
    void insert(const int& key, T value)
    {
        fieldDB_.emplace(key, value);
    }

    template<typename T>
    T& get(const int& key)
    {
        try
        {
            return std::any_cast<T&>(fieldDB_.at(key));
        }
        catch (const std::bad_any_cast& e)
        {
            logBadAnyCast<T>(e, key, fieldDB_);
            throw e;
        }
    }

    template<typename T>
    const T& get(const int& key) const
    {
        try
        {
            return std::any_cast<const T&>(fieldDB_.at(key));
        }
        catch (const std::bad_any_cast& e)
        {
            logBadAnyCast<T>(e, key, fieldDB_);
            throw e;
        }
    }

    std::string fieldName;
private:

    std::unordered_map<int, std::any> fieldDB_;

};  

namespace operations
{

template <typename T>
SolutionFields newFieldEntity(const T& field)
{
    SolutionFields SolutionFields;
    SolutionFields.fieldName = field.name;
    FieldComponent<T> fieldComponent{
        .field = std::make_shared<T>(field),
        .timeIndex = 0,
        .iterationIndex = 0,
        .category = FieldCategory::newTime
    };
    SolutionFields.insert(0, fieldComponent);
    return SolutionFields;
}

} // namespace operations

} // namespace NeoFOAM
