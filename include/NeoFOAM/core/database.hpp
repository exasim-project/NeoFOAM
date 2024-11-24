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
// forward declaration
// class Collection;

using key = std::string;

/**
 * @class Database
 * @brief A class representing a database that stores multiple collections.
 *
 * The Database class provides a way to store and retrieve collections.
 */
class Database
{
public:

    Collection& createCollection(const key& name, Collection col);

    bool foundCollection(const key& name) const;

    Collection& getCollection(const key& name);
    const Collection& getCollection(const key& name) const;

    template<typename CollectionType>
    CollectionType& get(const key& name)
    {
        Collection& collection = getCollection(name);
        return collection.as<CollectionType>();
    }

private:

    std::unordered_map<key, Collection> collections_;
};

} // namespace NeoFOAM
