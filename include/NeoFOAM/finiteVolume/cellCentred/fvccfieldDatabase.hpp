// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <any>
#include <string>

namespace NeoFOAM
{

/**
 * @class fieldDataBase
 * @brief A class that represents a field database.
 *
 * The fieldDataBase class provides a container for storing field data. 
 */
class fieldDataBase
{
public:

    /**
     * @brief Default constructor for fieldDataBase.
     */
    fieldDataBase();

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
        return std::any_cast<T&>(fieldDB_.at(key));
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
        return std::any_cast<const T&>(fieldDB_.at(key));
    }

private:

    // /**
    //  * @brief The field database to register field data.
    //  */
    std::unordered_map<std::string, std::any> fieldDB_;
};

} // namespace NeoFOAM
