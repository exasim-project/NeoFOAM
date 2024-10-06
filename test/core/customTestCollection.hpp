// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <optional>
#include <string>
#include <NeoFOAM/core/database.hpp>
#include "NeoFOAM/core/document.hpp"

// auto 

struct CustomTestDocument
{
    NeoFOAM::DocumentValidator customValidator = [](const NeoFOAM::Document& doc)
    {
        return doc.contains("name") && doc.contains("value") && hasId(doc);
    };

    static NeoFOAM::Document create(std::string name, double value)
    {
        return NeoFOAM::Document({{"name", name}, {"value", value}});
    }

};

const double& value(const NeoFOAM::Document& doc)
{
    return doc.get<double>("value");
}

double& value(NeoFOAM::Document& doc)
{
    return doc.get<double>("value");
}

class CustomTestCollection
{
public:
    static const std::string typeName()
    {
        return "CustomTestCollection";
    }

    static void create(std::string name, NeoFOAM::Database& db)
    {
        db.createCollection(name, CustomTestCollection::typeName());
    }

    static NeoFOAM::Collection& getCollection(NeoFOAM::Database& db, std::string name)
    {
        return db.getCollection(name);
    }

};


//     CustomTestDocument(const NeoFOAM::Document& doc)
//         : doc_(doc)
//     {
//     }

//     CustomTestDocument(std::string name, double value)
//         : doc_(NeoFOAM::Document({{"name", name}, {"value", value}}))
//     {
//     }
    
//     double value() const
//     {
//         return doc_.get<double>("value");
//     }

//     double& value()
//     {
//         return doc_.get<double>("value");
//     }

//     const std::string& name() const
//     {
//         return doc_.get<std::string>("name");
//     }

//     std::string& name()
//     {
//         return doc_.get<std::string>("name");
//     }

//     std::string id() const
//     {
//         return doc_.get<std::string>("id");
//     }

//     const NeoFOAM::Document& doc() const
//     {
//         return doc_;
//     }

//     NeoFOAM::Document& doc()
//     {
//         return doc_;
//     }

//     private:
//        std::optional<std::string> id_;
//        NeoFOAM::Document doc_;
// };

// class CustomTestCollection
// {

// public:

//     CustomTestCollection(std::string name,NeoFOAM::Database& db)
//         : name_(name), db_(db)
//     {

//     }
//     static void registerCollection(std::string name, NeoFOAM::Database& db)
//     {
//         db.createCollection(name, CustomTestCollection::typeName);
//     }

//     static CustomTestCollection getCollection(std::string name, NeoFOAM::Database& db)
//     {
//         return CustomTestCollection(name,db);
//     }

//     std::string name() const
//     {
//         return name_;
//     }

//     CustomTestDocument& get(const std::string& id)
//     {
//         auto it = documents_.find(id);
//         if (it != documents_.end())
//         {
//             return it->second;
//         }
//         throw std::runtime_error("Document not found");
//     }

//     const CustomTestDocument& get(const std::string& id) const
//     {
//         auto it = documents_.find(id);
//         if (it != documents_.end())
//         {
//             return it->second;
//         }
//         throw std::runtime_error("Document not found");
//     }

//     size_t size() const
//     {
//         return documents_.size();
//     }

//     std::string insert(CustomTestDocument doc)
//     {
//         documents_.insert({doc.id(), doc});
//         return doc.id();
//     }

//     // void update(const std::string& id, const CustomTestDocument& doc)
//     // {
//     //     documents_.update(id, doc);
//     // }

//     // void update(const CustomTestDocument& doc)
//     // {
//     //     documents_.update(doc);
//     // }

//     std::vector<std::string> find(const std::function<bool(const NeoFOAM::Document&)>& predicate) const
//     {
//         std::vector<std::string> result;
//         for (const auto& [key, doc] : documents_)
//         {
//             if (predicate(doc.doc()))
//             {
//                 result.push_back(doc.id());
//             }
//         }
//         return result;
//     }

// private:
//     static const std::string typeName;
//     std::string name_;
//     NeoFOAM::Database& db_;
//     std::unordered_map<std::string, CustomTestDocument> documents_;

// };

// const std::string CustomTestCollection::typeName = "CustomTestCollection";