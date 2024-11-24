// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include <stdexcept>

#include "NeoFOAM/core/database.hpp"
#include "NeoFOAM/core/collection.hpp"

namespace NeoFOAM
{

Collection& Database::createCollection(const std::string& name, Collection col)
{
    collections_.emplace(name, col);
    return collections_.at(name);
}

bool Database::foundCollection(const std::string& name) const
{
    return collections_.contains(name);
}

Collection& Database::getCollection(const std::string& name)
{
    auto it = collections_.find(name);
    if (it != collections_.end())
    {
        return it->second;
    }
    throw std::runtime_error("Collection not found");
}

const Collection& Database::getCollection(const std::string& name) const
{
    auto it = collections_.find(name);
    if (it != collections_.end())
    {
        return it->second;
    }
    throw std::runtime_error("Collection not found");
}

} // namespace NeoFOAM
