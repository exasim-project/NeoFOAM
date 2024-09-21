// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <any>
#include <string>
#include <functional>
#include "NeoFOAM/core/demangle.hpp"
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/solutionFields.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


template<typename GeoField>
using CreateFunction = std::function<GeoField()>;
/**
 * @class FieldDatabase
 * @brief A class that represents a field database.
 *
 * The FieldDatabase class provides a container for storing field data. 
 */
class FieldDatabase
{
public:

    /**
     * @brief Default constructor for FieldDatabase.
     */
    FieldDatabase();

    /**
     * @brief Inserts a value into the stencil database.
     *
     * @tparam T The type of the value to be inserted.
     * @param key The key associated with the value.
     * @param value The value to be inserted.
     */
    template<typename T>
    void insert(const std::string& key, T value)
    {
        fieldDB_.emplace(key, value);
    }

        /**
     * @brief Retrieves a reference to the value associated with the given key.
     *
     * @tparam T The type of the value to be retrieved.
     * @param key The key associated with the value.
     * @return A reference to the value associated with the key.
     * @throws std::bad_any_cast if the value cannot be cast to type T.
     */
    template<typename T>
    T& get(const std::string& key)
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

    /**
     * @brief Retrieves a const reference to the value associated with the given
     * key.
     *
     * @tparam T The type of the value to be retrieved.
     * @param key The key associated with the value.
     * @return A const reference to the value associated with the key.
     * @throws std::bad_any_cast if the value cannot be cast to type T.
     */
    template<typename T>
    const T& get(const std::string& key) const
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

    template<typename GeoField>
    SolutionFields<GeoField>& createSolutionField(CreateFunction<GeoField> creatFunc)
    {
        SolutionFields<GeoField> solutionField(creatFunc());
        fieldDB_.emplace(solutionField.name(), solutionField);
        auto& solField = std::any_cast<SolutionFields<GeoField>&>(fieldDB_.at(solutionField.name()));
        solField.field().setSolField(solField);
        return solField;
    }

private:

    // /**
    //  * @brief The field database to register field data.
    //  */
    std::unordered_map<std::string, std::any> fieldDB_;
};

} // namespace NeoFOAM
