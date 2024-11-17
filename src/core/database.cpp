#include <stdexcept>

#include "NeoFOAM/core/database.hpp"
#include "NeoFOAM/core/collection.hpp"

namespace NeoFOAM
{

void Database::createCollection(std::string name, std::string type)
{
    collections_.emplace(name, std::make_shared<Collection>(type, name, *this));
}

Collection& Database::getCollection(const std::string& name)
{
    auto it = collections_.find(name);
    if (it != collections_.end())
    {
        return *(it->second);
    }
    throw std::runtime_error("Collection not found");
}



const Collection& Database::getCollection(const std::string& name) const
{
    auto it = collections_.find(name);
    if (it != collections_.end())
    {
        return *(it->second);
    }
    throw std::runtime_error("Collection not found");
}

std::shared_ptr<Collection> Database::getCollectionPtr(const std::string& name)
{
    auto it = collections_.find(name);
    if (it != collections_.end())
    {
        return it->second;
    }
    throw std::runtime_error("Collection not found");
}


} // namespace NeoFOAM