// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <string>
#include <any>

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

    void createCollection(std::string name, std::string type);
    Collection& getCollection(const std::string& name);
    const Collection& getCollection(const std::string& name) const;

    

private:

    std::unordered_map<key, Collection> collections_;
};

} // namespace NeoFOAM