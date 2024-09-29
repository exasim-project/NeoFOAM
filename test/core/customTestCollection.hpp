// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <optional>
#include <string>
#include <NeoFOAM/core/database.hpp>

struct CustomTestDocument
{

    CustomTestDocument(const NeoFOAM::Document& doc)
        : name(doc.get<std::string>("name")), value(doc.get<double>("value")), id_(doc.get<std::string>("id"))
    {
    }

    CustomTestDocument(std::string name, double value)
        : name(name), value(value), id_(std::nullopt)
    {
    }
    
    std::string name;
    double value;

    // convert to Document
    operator NeoFOAM::Document() const
    {
        // document with define the id
        NeoFOAM::Document doc;
        if (id_.has_value())
        {
            doc.insert("id", id_.value());
        }
        doc.insert("name", name);
        doc.insert("value", value);
        return doc;
    }

    std::optional<std::string> id() const
    {
        return id_;
    }

    private:
       std::optional<std::string> id_;
};

class CustomTestCollection
{

public:

    CustomTestCollection(std::string name,NeoFOAM::Database& db)
        : name_(name), db_(db), collection_(db.getCollection(name))
    {

    }
    static void registerCollection(std::string name, NeoFOAM::Database& db)
    {
        db.createCollection(name, CustomTestCollection::typeName);
    }

    static CustomTestCollection getCollection(std::string name, NeoFOAM::Database& db)
    {
        return CustomTestCollection(name,db);
    }

    std::string name() const
    {
        return name_;
    }

    CustomTestDocument get(const std::string& id)
    {
        NeoFOAM::Document& doc = collection_.get(id);
        return CustomTestDocument(doc);
    }

    size_t size() const
    {
        return collection_.size();
    }

    auto insert(CustomTestDocument doc)
    {
        return collection_.insert(doc);
    }

    void update(const std::string& id, const CustomTestDocument& doc)
    {
        collection_.update(id, doc);
    }

    void update(const CustomTestDocument& doc)
    {
        collection_.update(doc);
    }

    std::vector<NeoFOAM::key> find(const std::function<bool(const NeoFOAM::Document&)>& predicate) const
    {
        return collection_.find(predicate);
    }

private:
    static const std::string typeName;
    std::string name_;
    NeoFOAM::Database& db_;
    NeoFOAM::Collection& collection_;

};

const std::string CustomTestCollection::typeName = "CustomTestCollection";