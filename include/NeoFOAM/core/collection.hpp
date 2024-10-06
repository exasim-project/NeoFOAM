// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <string>
#include "NeoFOAM/core/document.hpp"

namespace NeoFOAM
{

using key = std::string;


// forward declaration
class Database;

/**
 * @class Collection
 * @brief A class representing a collection of documents.
 *
 * The Collection class provides a way to store and retrieve documents.
 */
class Collection
{
public:

    Collection(std::string type, std::string name, Database& db);


    key insert(Document doc);

    Document& get(const key& id);

    const Document& get(const key& id) const;

    void update(const key& id, const Document& doc);

    void update(const Document& doc);

    std::vector<key> find(const std::function<bool(const Document&)>& predicate) const;
    size_t size() const;

    std::string type() const;

    std::string name() const;

private:

    std::unordered_map<key, Document> documents_;
    std::string type_;
    std::string name_;
    Database& db_;
};


} // namespace NeoFOAM