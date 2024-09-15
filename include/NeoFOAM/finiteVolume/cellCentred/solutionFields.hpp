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

struct fieldMetadata
{
    std::size_t timeIndex;
    std::size_t iterationIndex;
    FieldCategory category;
};

template<typename Type>
struct FieldComponent
{
    std::shared_ptr<Type> field;
    fieldMetadata metaData;
};

class SolutionFields
{

public:

    SolutionFields(
        const VolumeField<NeoFOAM::scalar>& field
    );

    template<typename T>
    size_t insert(T value)
    {
        fieldDB_.push_back(value);
        return fieldDB_.size() - 1;
    }

    template<typename T>
    T& get(const size_t key)
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
    const T& get(const size_t key) const
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
    std::string name() const { return fieldName_; }

    VolumeField<NeoFOAM::scalar> field;

    
private:

    std::string fieldName_;
    std::vector<std::any> fieldDB_;

};  

namespace operations
{

// template <typename T>
// SolutionFields newFieldEntity(const T& field)
// {
//     SolutionFields SolutionFields;
//     SolutionFields.fieldName = field.name;
//     FieldComponent<T> fieldComponent{
//         .field = std::make_shared<T>(field),
//         .timeIndex = 0,
//         .iterationIndex = 0,
//         .category = FieldCategory::newTime
//     };
//     SolutionFields.insert(0, fieldComponent);
//     return SolutionFields;
// }

template <typename T>
VolumeField<T>& oldTime(VolumeField<T>& field)
{
    SolutionFields& solutionFields = field.solField();
    FieldComponent<VolumeField<T>> fieldComponent{
        .field = std::make_shared<VolumeField<T>>(field),
        .metaData = {
            .timeIndex = 0,
            .iterationIndex = 0,
            .category = FieldCategory::oldTime
        }
    };
    size_t key = solutionFields.insert(fieldComponent);
    auto& fc = solutionFields.get<FieldComponent<VolumeField<T>>>(key); 
    VolumeField<T>& volField = *fc.field;
    volField.name = field.name + "_0";
    return volField;
    // return fieldComponent.get;
}

} // namespace operations

} // namespace NeoFOAM
