// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <any>
#include <string>
#include <memory>
#include <vector>
#include <ranges> // for std::ranges::find
#include <iterator> // for std::distance

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

struct FieldMetaData
{
    std::size_t timeIndex;
    std::size_t iterationIndex;
    FieldCategory category;

    bool operator==(const FieldMetaData& other) const
    {
        return timeIndex == other.timeIndex && iterationIndex == other.iterationIndex && category == other.category;
    }
};

template<typename Type>
struct FieldComponent
{
    std::shared_ptr<Type> field;
    FieldMetaData metaData;
};

template<class GeoField>
class SolutionFields
{

public:

    SolutionFields(const GeoField& field)  {
        auto mainField = field;
        mainField.solutionFieldKey = 0;
        fieldDB_.push_back(mainField);
        fieldMetaData.push_back(FieldMetaData{.timeIndex = 0, .iterationIndex = 0, .category = FieldCategory::newTime});
    }

    size_t insert(const GeoField& field, FieldMetaData metaData)
    {
        fieldDB_.push_back(field);
        fieldMetaData.push_back(metaData);
        return fieldDB_.size() - 1;
    }

    GeoField& getField(const size_t key)
    {
        return fieldDB_.at(key);
    }

    const GeoField& getField(const size_t key) const
    {
        return fieldDB_.at(key);
    }

    int32_t findField(const GeoField& field)
    {
        int32_t key = -1;
        std::cout << "address findField: " << &field << std::endl;
        for ( int32_t i = 0; i < fieldDB_.size(); i++)
        {
            std::cout << "fieldDB_ " << &fieldDB_[i] << std::endl;
            if (&field == &fieldDB_[i])
            {
                key = i;
                break;
            }
        }
        return key;
    }

    std::string name() const { return fieldDB_[0].name; }

    GeoField& field() { return fieldDB_[0]; }

    const GeoField& field() const { return fieldDB_[0]; }

    std::size_t size() const { return fieldDB_.size(); }

    std::vector<FieldMetaData> fieldMetaData;

private:


    std::vector<GeoField> fieldDB_;
};

namespace operations
{

template<typename T>
VolumeField<T>& oldTime(VolumeField<T>& field)
{
    auto& solutionFields = field.solField().get();
    size_t fieldKey = field.solutionFieldKey.value();

    FieldMetaData oldTimeMetaData = solutionFields.fieldMetaData[fieldKey];
    oldTimeMetaData.category = FieldCategory::oldTime;
    oldTimeMetaData.timeIndex = oldTimeMetaData.timeIndex - 1;

    auto it = std::ranges::find(solutionFields.fieldMetaData, oldTimeMetaData);
    bool found = (it != solutionFields.fieldMetaData.end());
    size_t key = std::distance(solutionFields.fieldMetaData.begin(), it);

    if (!found)
    {
        VolumeField<T> oldTimeField = field;
        oldTimeField.name = field.name + "_0";
        oldTimeField.setSolField(solutionFields);
        key = solutionFields.insert(oldTimeField, oldTimeMetaData);
        VolumeField<T>& volField = solutionFields.getField(key);
        volField.solutionFieldKey = key;
    }
    
    VolumeField<T>& volField = solutionFields.getField(key);
    return volField;
}


} // namespace operations

} // namespace NeoFOAM
