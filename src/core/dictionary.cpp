// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/dictionary.hpp"

namespace NeoFOAM
{

Dictionary::Dictionary(const std::unordered_map<std::string, std::any>& keyValuePairs)
    : data_(keyValuePairs)
{}

Dictionary::Dictionary(const std::initializer_list<std::pair<std::string, std::any>>& initList)
{
    for (const auto& pair : initList)
    {
        data_.insert(pair);
    }
}

void Dictionary::insert(const std::string& key, const std::any& value) { data_[key] = value; }

void Dictionary::remove(const std::string& key) { data_.erase(key); }

bool Dictionary::found(const std::string& key) const { return data_.find(key) != data_.end(); }

std::any& Dictionary::operator[](const std::string& key) { return data_.at(key); }

const std::any& Dictionary::operator[](const std::string& key) const { return data_.at(key); }

Dictionary& Dictionary::subDict(const std::string& key)
{
    return std::any_cast<Dictionary&>(data_.at(key));
}

// get keys of the dictionary
std::vector<std::string> Dictionary::keys() const
{
    std::vector<std::string> keys;
    for (const auto& pair : data_)
    {
        keys.push_back(pair.first);
    }
    return keys;
}

std::unordered_map<std::string, std::any>& Dictionary::getMap() { return data_; }

const std::unordered_map<std::string, std::any>& Dictionary::getMap() const { return data_; }

} // namespace NeoFOAM
