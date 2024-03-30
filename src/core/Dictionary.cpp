#include "NeoFOAM/core/Dictionary.hpp"



// NeoFOAM::Dictionary::Dictionary(std::unordered_map<std::string, std::any> data)
//     : data_(data)
// {
// }

// NeoFOAM::Dictionary::Dictionary(NeoFOAM::Dictionary& other)
//     : data_(other.data_)
// {
// }

void NeoFOAM::Dictionary::insert(const std::string& key, const std::any& value)
{
    data_[key] = value;
}

std::any& NeoFOAM::Dictionary::operator[](const std::string& key)
{
    return data_.at(key);
}

const std::any& NeoFOAM::Dictionary::operator[](const std::string& key) const
{
    return data_.at(key);
}

NeoFOAM::Dictionary& NeoFOAM::Dictionary::subDict(const std::string& key)
{
    return std::any_cast<NeoFOAM::Dictionary&>(data_.at(key));
}
