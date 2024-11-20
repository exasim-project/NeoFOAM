// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <functional>
#include <string>
#include "NeoFOAM/core/dictionary.hpp"

namespace NeoFOAM
{

std::unordered_map<std::string, Dictionary>& singletonLookupTable();

template<typename Base, typename... Args>
class LookupTable
{
public:

    using CreatorFunc = std::function<std::unique_ptr<Base>(Args...)>;

    LookupTable() : baseDictionary(singletonLookupTable()[Base::name()]) {};

    std::size_t size() const { return baseDictionary.getMap().size(); }

    CreatorFunc get(const std::string& key) const
    {
        return std::any_cast<const CreatorFunc&>(baseDictionary[key]);
    }

    void set(const std::string& key, CreatorFunc value) { baseDictionary.insert(key, value); }

    std::vector<std::string> entries() const { return baseDictionary.keys(); }

    bool contains(const std::string& key) const { return baseDictionary.contains(key); }

    const Dictionary& getBaseDictionary() const { return baseDictionary; }

private:

    Dictionary& baseDictionary;
};

} // namespace NeoFOAM
