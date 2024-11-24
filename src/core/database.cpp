// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include <stdexcept>

#include "NeoFOAM/core/database.hpp"
#include "NeoFOAM/core/collection.hpp"

namespace NeoFOAM
{

Collection& Database::insert(const std::string& name, const Collection& col)
{
    collections_.emplace(name, col);
    return collections_.at(name);
}

bool Database::contains(const std::string& name) const { return collections_.contains(name); }

Collection& Database::getCollection(const std::string& name) { return collections_.at(name); }

const Collection& Database::getCollection(const std::string& name) const
{
    return collections_.at(name);
}

std::size_t Database::size() const { return collections_.size(); }

bool Database::remove(const std::string& name) { return collections_.erase(name) > 0; }

} // namespace NeoFOAM
