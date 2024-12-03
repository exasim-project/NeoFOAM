// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/core/database/collection.hpp"


namespace NeoFOAM
{

Collection::Collection(const Collection& other) : impl_(other.impl_->clone()) {}

Document& Collection::doc(const std::string& id) { return impl_->doc(id); }

const Document& Collection::doc(const std::string& id) const { return impl_->doc(id); }

std::vector<std::string> Collection::find(const std::function<bool(const Document&)>& predicate) const
{
    return impl_->find(predicate);
}

size_t Collection::size() const { return impl_->size(); }

std::string Collection::type() const { return impl_->type(); }

std::string Collection::name() const { return impl_->name(); }

Database& Collection::db() { return impl_->db(); }

const Database& Collection::db() const { return impl_->db(); }

} // namespace NeoFOAM
