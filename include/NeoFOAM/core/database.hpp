// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <string>
#include <any>
#include <memory>

namespace NeoFOAM
{
// forward declaration
class Collection;

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

    void createCollection(key name, std::string type);
    Collection& getCollection(const key& name);
    const Collection& getCollection(const key& name) const;

    std::shared_ptr<Collection> getCollectionPtr(const key& name);
    

private:

    std::unordered_map<key, std::shared_ptr<Collection>> collections_;
};

} // namespace NeoFOAM