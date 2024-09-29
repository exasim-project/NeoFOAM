// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <unordered_map>
#include <string>
#include <any>
#include <optional>
#include <functional>
#include <atomic>
#include "NeoFOAM/core/dictionary.hpp"

namespace NeoFOAM
{

using key = std::string;

bool hasId(Dictionary doc);

class Document : public Dictionary
{
public:

    Document();

    Document(const Dictionary& dict, std::function<bool(Dictionary)> validator = hasId);

    void validate();

    std::string id() const { return get<std::string>("id"); }

private:

    static std::string generateID()
    {
        static std::atomic<int> counter {0};
        return "doc_" + std::to_string(counter++);
    }
    std::string id_;
    std::function<bool(Dictionary)> validator_;
};

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

    Collection(std::string type, Database& db);


    key insert(Document doc);

    Document& get(const key& id);

    const Document& get(const key& id) const;

    void update(const key& id, const Document& doc);

    void update(const Document& doc);

    std::vector<key> find(const std::function<bool(const Document&)>& predicate) const;
    size_t size() const;

    std::string type() const;

private:

    std::unordered_map<key, Document> documents_;
    std::string type_;
    Database& db_;
};


/**
 * @class Database
 * @brief A class representing a database that stores multiple collections.
 *
 * The Database class provides a way to store and retrieve collections.
 */
class Database
{
public:

    void createCollection(std::string name, std::string type);
    Collection& getCollection(const std::string& name);
    const Collection& getCollection(const std::string& name) const;

private:

    std::unordered_map<key, Collection> collections_;
};

} // namespace NeoFOAM