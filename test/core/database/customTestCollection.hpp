// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <optional>
#include <string>

#include "NeoFOAM/core.hpp"


bool validateCustomDoc(const NeoFOAM::Document& doc)
{
    return doc.contains("name") && doc.contains("testValue");
}

class CustomDocument
{
public:

    CustomDocument() : doc_(NeoFOAM::Document({{"name", ""}, {"testValue", 0}}, validateCustomDoc))
    {}

    CustomDocument(const NeoFOAM::Document& doc) : doc_(doc) {}

    CustomDocument(const std::string& name, const double& testValue)
        : doc_(NeoFOAM::Document({{"name", name}, {"testValue", testValue}}, validateCustomDoc))
    {}

    std::string& name() { return doc_.get<std::string>("name"); }

    const std::string& name() const { return doc_.get<std::string>("name"); }

    double testValue() const { return doc_.get<double>("testValue"); }

    double& testValue() { return doc_.get<double>("testValue"); }

    NeoFOAM::Document& doc() { return doc_; }

    const NeoFOAM::Document& doc() const { return doc_; }

    std::string id() const { return doc_.id(); }

    static std::string typeName() { return "CustomDocument"; }

private:

    NeoFOAM::Document doc_;
};

class CustomCollection : public NeoFOAM::CollectionMixin<CustomDocument>
{
public:

    CustomCollection(NeoFOAM::Database& db, std::string name)
        : NeoFOAM::CollectionMixin<CustomDocument>(db, name)
    {}

    bool contains(const std::string& id) const { return docs_.contains(id); }

    bool insert(const CustomDocument& cc)
    {
        std::string id = cc.id();
        if (contains(id))
        {
            return false;
        }
        docs_.emplace(id, cc);
        return true;
    }

    static CustomCollection& instance(NeoFOAM::Database& db, std::string name)
    {
        NeoFOAM::Collection& col = db.insert(name, CustomCollection(db, name));
        return col.as<CustomCollection>();
    }
};
