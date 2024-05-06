// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <any>
#include <string>

namespace NeoFOAM
{

/**
 * @class Dictionary
 * @brief A class representing a dictionary that stores key-value pairs.
 *
 * The Dictionary class provides a way to store and retrieve values using string.
 * It supports inserting key-value pairs, accessing values using the subscript operator,
 * and retrieving values of specific types using the `get` function.
 * It also supports storing sub-dictionaries, which can be accessed using the `subDict` function.
 * The values are stored using `std::any`, which allows storing values of any type.
 */
class Dictionary
{
public:

    Dictionary() = default;

    /**
     * @brief Inserts a key-value pair into the dictionary.
     * @param key The key to insert.
     * @param value The value to insert.
     */
    void insert(const std::string& key, const std::any& value);

    /**
     * @brief Checks if the given key is present in the dictionary.
     * @param key The key to check.
     * @return True if the key is present, false otherwise.
     */
    [[nodiscard]] bool found(const std::string& key) const;

    /**
     * @brief Removes an entry from the dictionary based on the specified key.
     *
     * This function removes the entry with the specified key from the dictionary.
     *
     * @param key The key of the entry to be removed.
     */
    void remove(const std::string& key);

    /**
     * @brief Accesses the value associated with the given key.
     * @param key The key to access.
     * @return A reference to the value associated with the key.
     */
    [[nodiscard]] std::any& operator[](const std::string& key);

    /**
     * @brief Accesses the value associated with the given key.
     * @param key The key to access.
     * @return A const reference to the value associated with the key.
     */
    [[nodiscard]] const std::any& operator[](const std::string& key) const;

    /**
     * @brief Retrieves the value associated with the given key, casting it to the specified type.
     * @tparam T The type to cast the value to.
     * @param key The key to retrieve the value for.
     * @return A reference to the value associated with the key, casted to type T.
     */
    template<typename T>
    [[nodiscard]] T& get(const std::string& key)
    {
        return std::any_cast<T&>(data_.at(key));
    }

    /**
     * @brief Retrieves the value associated with the given key, casting it to the specified type.
     * @tparam T The type to cast the value to.
     * @param key The key to retrieve the value for.
     * @return A const reference to the value associated with the key, casted to type T.
     */
    template<typename T>
    [[nodiscard]] const T& get(const std::string& key) const
    {
        return std::any_cast<const T&>(data_.at(key));
    }

    /**
     * @brief Retrieves a sub-dictionary associated with the given key.
     * @param key The key to retrieve the sub-dictionary for.
     * @return A reference to the sub-dictionary associated with the key.
     */
    Dictionary& subDict(const std::string& key);

    /**
     * @brief Retrieves the underlying unordered map of the dictionary.
     * @return A reference to the underlying unordered map.
     */
    std::unordered_map<std::string, std::any>& getMap();

    /**
     * @brief Retrieves the underlying unordered map of the dictionary.
     * @return A const reference to the underlying unordered map.
     */
    const std::unordered_map<std::string, std::any>& getMap() const;

private:

    std::unordered_map<std::string, std::any> data_;
};

} // namespace NeoFOAM
