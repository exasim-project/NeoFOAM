// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/core/database/oldTimeCollection.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

OldTimeDocument::OldTimeDocument(const Document& doc)
    : doc_(doc)
{}

OldTimeDocument::OldTimeDocument(
    std::string nextTime,
    std::string previousTime,
    std::string currentTime,
    int32_t level
)
    : doc_(
          Document(
              {
                  {"nextTime", nextTime},
                  {"previousTime", previousTime},
                  {"currentTime", currentTime},
                  {"level", level}
              }
          )
      )
{}

std::string& OldTimeDocument::nextTime()
{
    return doc_.get<std::string>("nextTime");
}

const std::string& OldTimeDocument::nextTime() const
{
    return doc_.get<std::string>("nextTime");
}

std::string& OldTimeDocument::previousTime()
{
    return doc_.get<std::string>("previousTime");
}

const std::string& OldTimeDocument::previousTime() const
{
    return doc_.get<std::string>("previousTime");
}

std::string& OldTimeDocument::currentTime()
{
    return doc_.get<std::string>("currentTime");
}

const std::string& OldTimeDocument::currentTime() const
{
    return doc_.get<std::string>("currentTime");
}

int32_t& OldTimeDocument::level()
{
    return doc_.get<int32_t>("level");
}

const int32_t& OldTimeDocument::level() const
{
    return doc_.get<int32_t>("level");
}

Document& OldTimeDocument::doc()
{
    return doc_;
}

const Document& OldTimeDocument::doc() const
{
    return doc_;   
}

std::string OldTimeDocument::id() const
{
    return doc_.id();

}

std::string OldTimeDocument::typeName()
{
    return "OldTimeDocument";
}

OldTimeCollection::OldTimeCollection(Database& db, std::string name, std::string fieldCollectionName)
    : CollectionMixin<OldTimeDocument>(db, name), fieldCollectionName_(fieldCollectionName)
{}

bool OldTimeCollection::contains(const std::string& id) const
{
    return docs_.contains(id);;
}

bool OldTimeCollection::insert(const OldTimeDocument& otd)
{
    std::string id = otd.id();
    if (contains(id))
    {
        return false;
    }
    docs_.emplace(id, otd);
    return true;
}

std::string OldTimeCollection::findNextTime(std::string id) const
{
    auto keys = find(
        [id](const Document& doc)
        {
            return doc.get<std::string>("nextTime") == id;
        }
    );
    if (keys.size() == 1)
    {
        return keys[0];
    }
    return "";
}

std::string OldTimeCollection::findPreviousTime(std::string id) const
{
    auto keys = find(
        [id](const Document& doc)
        {
            return doc.get<std::string>("previousTime") == id;
        }
    );
    if (keys.size() == 1)
    {
        return keys[0];
    }
    return "";
}

OldTimeCollection& OldTimeCollection::instance(Database& db, std::string name, std::string fieldCollectionName)
{
    Collection& col = db.insert(name, OldTimeCollection(db, name,fieldCollectionName));
    return col.as<OldTimeCollection>();
}

const OldTimeCollection& OldTimeCollection::instance(const Database& db, std::string name)
{
    const Collection& col = db.getCollection(name);
    return col.as<OldTimeCollection>();
}

OldTimeCollection& OldTimeCollection::instance(FieldCollection& fieldCollection)
{
    std::string name = fieldCollection.name() + "_oldTime";
    return instance(fieldCollection.db(), name, fieldCollection.name());
}

const OldTimeCollection& OldTimeCollection::instance(const FieldCollection& fieldCollection)
{
    std::string name = fieldCollection.name() + "_oldTime";
    return instance(fieldCollection.db(), name);
}




} // namespace NeoFOAM::finiteVolume::cellCentred
