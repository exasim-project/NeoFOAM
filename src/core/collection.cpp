// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/core/collection.hpp"


namespace NeoFOAM
{


Document& Collection::get(const key& id) { return impl_->get(id); }

const Document& Collection::get(const key& id) const { return impl_->get(id); }

std::vector<key> Collection::find(const std::function<bool(const Document&)>& predicate) const
{
    return impl_->find(predicate);
}

size_t Collection::size() const { return impl_->size(); }

std::string Collection::type() const { return impl_->type(); }

std::string Collection::name() const { return impl_->name(); }

Database& Collection::db() { return impl_->db(); }

const Database& Collection::db() const { return impl_->db(); }

} // namespace NeoFOAM
