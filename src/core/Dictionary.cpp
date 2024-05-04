// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/Dictionary.hpp"

void NeoFOAM::Dictionary::insert(const std::string& key, const std::any& value)
{
    data_[key] = value;
}

bool NeoFOAM::Dictionary::found(const std::string& key) const
{
    return data_.find(key) != data_.end();
}

std::any& NeoFOAM::Dictionary::operator[](const std::string& key)
{
    return data_.at(key);
}

const std::any& NeoFOAM::Dictionary::operator[](const std::string& key) const
{
    return data_.at(key);
}

NeoFOAM::Dictionary& NeoFOAM::Dictionary::subDict(const std::string& key)
{
    return std::any_cast<NeoFOAM::Dictionary&>(data_.at(key));
}
