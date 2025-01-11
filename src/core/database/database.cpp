// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/core/database/database.hpp"
#include "NeoFOAM/core/database/collection.hpp"

namespace NeoFOAM
{

Collection& Database::insert(const std::string& name, const Collection& col)
{
    return collections_.emplace(name, col).first->second;
}

bool Database::contains(const std::string& name) const { return collections_.contains(name); }

Collection& Database::at(const std::string& name) { return collections_.at(name); }

const Collection& Database::at(const std::string& name) const { return collections_.at(name); }

std::size_t Database::size() const { return collections_.size(); }

bool Database::remove(const std::string& name) { return collections_.erase(name) > 0; }

} // namespace NeoFOAM
