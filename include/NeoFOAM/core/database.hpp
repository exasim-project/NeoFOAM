// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <unordered_map>
#include <string>
#include <any>
#include <memory>
#include "NeoFOAM/core/collection.hpp"

namespace NeoFOAM
{


/**
 * @class Database
 * @brief A class representing a database that stores multiple collections.
 *
 * The Database class provides a way to store and retrieve collections.
 */
class Database
{
public:

    Collection& insert(const std::string& key, const Collection& col);

    bool contains(const std::string& name) const;

    bool remove(const std::string& name);

    Collection& getCollection(const std::string& name);
    const Collection& getCollection(const std::string& name) const;

    template<typename CollectionType>
    CollectionType& get(const std::string& name)
    {
        Collection& collection = getCollection(name);
        return collection.as<CollectionType>();
    }

    std::size_t size() const;

private:

    std::unordered_map<std::string, Collection> collections_;
};

} // namespace NeoFOAM
