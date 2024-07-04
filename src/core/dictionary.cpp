// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/dictionary.hpp"

void NeoFOAM::Dictionary::insert(const std::string& key, const std::any& value)
{
    data_[key] = value;
}

void NeoFOAM::Dictionary::remove(const std::string& key) { data_.erase(key); }

bool NeoFOAM::Dictionary::found(const std::string& key) const
{
    return data_.find(key) != data_.end();
}

std::any& NeoFOAM::Dictionary::operator[](const std::string& key) { return data_.at(key); }

const std::any& NeoFOAM::Dictionary::operator[](const std::string& key) const
{
    return data_.at(key);
}

NeoFOAM::Dictionary& NeoFOAM::Dictionary::subDict(const std::string& key)
{
    return std::any_cast<NeoFOAM::Dictionary&>(data_.at(key));
}

std::unordered_map<std::string, std::any>& NeoFOAM::Dictionary::getMap() { return data_; }

const std::unordered_map<std::string, std::any>& NeoFOAM::Dictionary::getMap() const
{
    return data_;
}
