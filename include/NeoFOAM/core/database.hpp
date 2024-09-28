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

//using Document = Dictionary;
auto hasId = [](Dictionary doc) { return doc.contains("id"); };

class Document
:
public Dictionary
{
    public:

    Document();

    Document(
        const Dictionary& dict,
        std::function<bool(Dictionary)> validator = hasId
    );



    
    void validate();

    private:

        static std::string generateID() {
            static std::atomic<int> counter{0};
            return "doc_" + std::to_string(counter++);
        }
        std::string id_;
        std::function<bool(Dictionary)> validator_;
};


/**
 * @class Collection
 * @brief A class representing a collection of documents.
 *
 * The Collection class provides a way to store and retrieve documents.
 */
class Collection {
public:
    std::string insert(Document doc);
    std::optional<Document> getDocument(const std::string& id) const;
    std::vector<Document> find(const std::function<bool(const Document&)>& predicate) const;
    size_t size() const;

private:
    std::unordered_map<key, Document> documents_;
};

/**
 * @class Database
 * @brief A class representing a database that stores multiple collections.
 *
 * The Database class provides a way to store and retrieve collections.
 */
class Database {
public:
    void createCollection(const std::string& name);
    std::optional<Collection> getCollection(const std::string& name) const;

private:
    std::unordered_map<key, Collection> collections_;
};

} // namespace NeoFOAM