// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/tokenList.hpp"

namespace NeoFOAM
{

TokenList::TokenList(const std::initializer_list<std::any>& initList) : data_(initList) {}

void TokenList::insert(const std::any& value) { data_.push_back(value); }

void TokenList::remove(size_t index)
{
    data_.erase(data_.begin() + static_cast<std::vector<double>::difference_type>(index));
}

[[nodiscard]] bool TokenList::empty() const { return data_.empty(); }

[[nodiscard]] size_t TokenList::size() const { return data_.size(); }

[[nodiscard]] std::vector<std::any>& TokenList::tokens() { return data_; }

} // namespace NeoFOAM
